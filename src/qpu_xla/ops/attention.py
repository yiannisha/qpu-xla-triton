"""Exact unnormalized INT32 attention core composed from tiled GEMM stages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue

_SMUL24_LIMIT = 1 << 23


def _round_up(value: int, tile: int) -> int:
    """Round a positive dimension up to the tiled GEMM alignment."""
    return (value + tile - 1) // tile * tile


def _max_abs(values: npt.NDArray[np.int32]) -> int:
    """Return an overflow-safe absolute bound for one INT32 tensor."""
    return int(np.max(np.abs(values.astype(np.int64, copy=False)), initial=0))


def _validate_range(query: Tensor, key: Tensor, value: Tensor) -> None:
    """Validate the two `smul24` stages and their final INT32 accumulation."""
    query_values = cast(npt.NDArray[np.int32], query.numpy())
    key_values = cast(npt.NDArray[np.int32], key.numpy())
    value_values = cast(npt.NDArray[np.int32], value.numpy())
    max_query, max_key, max_value = map(_max_abs, (query_values, key_values, value_values))
    if max(max_query, max_key, max_value) >= _SMUL24_LIMIT:
        raise ValueError("attention_int32 uses smul24; query, key, and value must fit the signed 24-bit range")

    depth = query.shape[1]
    key_len = key.shape[0]
    score_bound = depth * max_query * max_key
    if score_bound >= _SMUL24_LIMIT:
        raise ValueError("attention score values may exceed the signed 24-bit range required by the value stage")
    output_bound = key_len * score_bound * max_value
    if output_bound > np.iinfo(np.int32).max:
        raise ValueError("attention output may exceed the signed int32 accumulation range")


def attention_int32(
    destination: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute exact unnormalized attention ``(query @ key.T) @ value``.

    The v0 operator accepts INT32 rank-2 tensors with shapes ``(M, D)``,
    ``(N, D)``, ``(N, V)``, and destination ``(M, V)``.  It deliberately
    does not apply scale, causal masking, or softmax; those belong to a later
    normalized-attention operator.  Both GEMM stages use the packaged tiled
    QPU kernel when its alignment and range contracts are met.
    """
    tensors = (query, key, value, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("attention tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.int32) for tensor in tensors):
        raise ValueError("attention_int32 requires int32 query, key, value, and destination tensors")
    if any(len(tensor.shape) != 2 for tensor in tensors):
        raise ValueError("attention_int32 requires rank-2 query, key, value, and destination tensors")

    query_len, depth = query.shape
    key_len, key_depth = key.shape
    value_rows, value_dim = value.shape
    if key_depth != depth or value_rows != key_len:
        raise ValueError("attention query, key, and value dimensions do not align")
    if destination.shape != (query_len, value_dim):
        raise ValueError("attention destination shape does not align")
    _validate_range(query, key, value)

    padded_query_len = _round_up(query_len, 16)
    padded_key_len = _round_up(key_len, 16)
    padded_depth = _round_up(depth, 4)
    padded_value_dim = _round_up(value_dim, 16)
    device = destination.buffer.device
    padded_query = device.tensor((padded_query_len, padded_depth), np.int32)
    padded_key_t = device.tensor((padded_depth, padded_key_len), np.int32)
    padded_value = device.tensor((padded_key_len, padded_value_dim), np.int32)
    scores = device.tensor((padded_query_len, padded_key_len), np.int32)
    padded_result = device.tensor((padded_query_len, padded_value_dim), np.int32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def prepare() -> None:
        padded_query.numpy().fill(0)
        padded_key_t.numpy().fill(0)
        padded_value.numpy().fill(0)
        padded_query.numpy()[:query_len, :depth] = query.numpy()
        padded_key_t.numpy()[:depth, :key_len] = key.numpy().T
        padded_value.numpy()[:key_len, :value_dim] = value.numpy()

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            query.access(AccessMode.READ),
            key.access(AccessMode.READ),
            value.access(AccessMode.READ),
            padded_query.access(AccessMode.WRITE),
            padded_key_t.access(AccessMode.WRITE),
            padded_value.access(AccessMode.WRITE),
        ),
    )
    score_event = matmul(scores, padded_query, padded_key_t, queue=selected_queue, wait_for=(prepare_event,))
    value_event = matmul(padded_result, scores, padded_value, queue=selected_queue, wait_for=(score_event,))

    def finish() -> None:
        result = cast(npt.NDArray[np.int32], padded_result.numpy()[:query_len, :value_dim])
        destination.numpy()[:] = result

    result_event = selected_queue.host_task(
        finish,
        wait_for=(value_event,),
        buffers=(padded_result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
    )
    if close_queue:
        result_event.wait()
        selected_queue.close()
    return result_event


def attention_fp32(
    destination: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute unnormalized FP32 attention ``(query @ key.T) @ value``.

    This v0 core has the same deliberate scope as :func:`attention_int32`:
    rank-2 matrices only, no scale, masking, or softmax. Compatible padded
    dimensions use the packaged FP32 tiled GEMM specialization.
    """
    tensors = (query, key, value, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("attention tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("attention_fp32 requires float32 query, key, value, and destination tensors")
    if any(len(tensor.shape) != 2 for tensor in tensors):
        raise ValueError("attention_fp32 requires rank-2 query, key, value, and destination tensors")
    query_len, depth = query.shape
    key_len, key_depth = key.shape
    value_rows, value_dim = value.shape
    if key_depth != depth or value_rows != key_len:
        raise ValueError("attention query, key, and value dimensions do not align")
    if destination.shape != (query_len, value_dim):
        raise ValueError("attention destination shape does not align")

    padded_query_len = _round_up(query_len, 16)
    padded_key_len = _round_up(key_len, 16)
    padded_depth = _round_up(depth, 4)
    padded_value_dim = _round_up(value_dim, 16)
    device = destination.buffer.device
    padded_query = device.tensor((padded_query_len, padded_depth), np.float32)
    padded_key_t = device.tensor((padded_depth, padded_key_len), np.float32)
    padded_value = device.tensor((padded_key_len, padded_value_dim), np.float32)
    scores = device.tensor((padded_query_len, padded_key_len), np.float32)
    padded_result = device.tensor((padded_query_len, padded_value_dim), np.float32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def prepare() -> None:
        padded_query.numpy().fill(0.0)
        padded_key_t.numpy().fill(0.0)
        padded_value.numpy().fill(0.0)
        padded_query.numpy()[:query_len, :depth] = query.numpy()
        padded_key_t.numpy()[:depth, :key_len] = key.numpy().T
        padded_value.numpy()[:key_len, :value_dim] = value.numpy()

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            query.access(AccessMode.READ),
            key.access(AccessMode.READ),
            value.access(AccessMode.READ),
            padded_query.access(AccessMode.WRITE),
            padded_key_t.access(AccessMode.WRITE),
            padded_value.access(AccessMode.WRITE),
        ),
        name="attention_fp32.prepare",
    )
    score_event = matmul(scores, padded_query, padded_key_t, queue=selected_queue, wait_for=(prepare_event,))
    value_event = matmul(padded_result, scores, padded_value, queue=selected_queue, wait_for=(score_event,))

    def finish() -> None:
        destination.numpy()[:] = padded_result.numpy()[:query_len, :value_dim]

    result_event = selected_queue.host_task(
        finish,
        wait_for=(value_event,),
        buffers=(padded_result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="attention_fp32.finish",
    )
    if close_queue:
        result_event.wait()
        selected_queue.close()
    return result_event
