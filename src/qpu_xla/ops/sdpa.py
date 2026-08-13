"""Scaled dot-product FP32 attention with QPU GEMM stages and CPU softmax."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.attention import _round_up
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue


def scaled_dot_product_attention_fp32(
    destination: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
    causal_offset: int = 0,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute FP32 softmax attention with optional causal masking.

    Inputs are rank-2 ``(query_len, depth)``, ``(key_len, depth)``, and
    ``(key_len, value_dim)`` matrices. Scores and value aggregation use the
    packaged tiled FP32 GEMM path when padded dimensions permit it; numerically
    sensitive row-wise softmax stays as an explicit CPU queue task. The default
    scale is ``1 / sqrt(depth)`` and causal masking permits keys
    ``j <= i + causal_offset``. The offset supports queries against a prefix
    held in a KV cache.
    """
    tensors = (query, key, value, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("attention tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("scaled_dot_product_attention_fp32 requires float32 tensors")
    if any(len(tensor.shape) != 2 for tensor in tensors):
        raise ValueError("scaled_dot_product_attention_fp32 requires rank-2 tensors")
    query_len, depth = query.shape
    key_len, key_depth = key.shape
    value_rows, value_dim = value.shape
    if key_depth != depth or value_rows != key_len:
        raise ValueError("attention query, key, and value dimensions do not align")
    if destination.shape != (query_len, value_dim):
        raise ValueError("attention destination shape does not align")
    if causal_offset < 0:
        raise ValueError("causal_offset must be non-negative")
    scale_value = float(1.0 / np.sqrt(depth)) if scale is None else float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("attention scale must be finite and positive")

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
        name="sdpa_fp32.prepare",
    )
    score_event = matmul(scores, padded_query, padded_key_t, queue=selected_queue, wait_for=(prepare_event,))

    def softmax() -> None:
        active = scores.numpy()[:query_len, :key_len]
        np.multiply(active, np.float32(scale_value), out=active)
        if causal:
            columns = np.arange(key_len)[None, :]
            rows = np.arange(query_len)[:, None]
            active[columns > rows + causal_offset] = -np.inf
        row_max = np.max(active, axis=1, keepdims=True)
        np.subtract(active, row_max, out=active)
        np.exp(active, out=active)
        row_sum = np.sum(active, axis=1, keepdims=True, dtype=np.float32)
        np.divide(active, row_sum, out=active)
        scores.numpy()[query_len:, :].fill(0.0)
        scores.numpy()[:, key_len:].fill(0.0)

    softmax_event = selected_queue.host_task(
        softmax,
        wait_for=(score_event,),
        buffers=(scores.access(AccessMode.READ_WRITE),),
        name="sdpa_fp32.softmax",
    )
    value_event = matmul(padded_result, scores, padded_value, queue=selected_queue, wait_for=(softmax_event,))

    def finish() -> None:
        destination.numpy()[:] = padded_result.numpy()[:query_len, :value_dim]

    result_event = selected_queue.host_task(
        finish,
        wait_for=(value_event,),
        buffers=(padded_result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="sdpa_fp32.finish",
    )
    if close_queue:
        result_event.wait()
        selected_queue.close()
    return result_event
