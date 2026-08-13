"""Numerically sensitive TinyLlama FP32 primitives kept on the CPU queue path."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue


def rms_norm_fp32(
    destination: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    epsilon: float = 1e-5,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply row-wise FP32 RMSNorm with a learned hidden-dimension weight."""
    tensors = (source, weight, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("RMSNorm tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("rms_norm_fp32 requires float32 tensors")
    if len(source.shape) != 2 or len(destination.shape) != 2 or len(weight.shape) != 1:
        raise ValueError("rms_norm_fp32 requires rank-2 source/destination and rank-1 weight")
    if destination.shape != source.shape or weight.shape != (source.shape[1],):
        raise ValueError("RMSNorm tensor shapes do not align")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and RMSNorm tensors must belong to the same device")

    def apply() -> None:
        values = cast(npt.NDArray[np.float32], source.numpy())
        scale = cast(npt.NDArray[np.float32], weight.numpy())
        mean_square = np.mean(values * values, axis=1, keepdims=True, dtype=np.float32)
        destination.numpy()[:] = values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon))) * scale

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            weight.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.rms_norm_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def rope_fp32(
    destination: Tensor,
    source: Tensor,
    positions: Tensor,
    *,
    base: float = 10_000.0,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply standard FP32 rotary position embedding to even/odd feature pairs."""
    tensors = (source, positions, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("RoPE tensors must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("rope_fp32 requires float32 source and destination tensors")
    if positions.dtype != np.dtype(np.int32):
        raise ValueError("rope_fp32 requires int32 positions")
    if len(source.shape) != 2 or destination.shape != source.shape or positions.shape != (source.shape[0],):
        raise ValueError("RoPE tensor shapes do not align")
    if source.shape[1] % 2:
        raise ValueError("rope_fp32 requires an even feature dimension")
    if not np.isfinite(base) or base <= 1:
        raise ValueError("RoPE base must be finite and greater than one")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and RoPE tensors must belong to the same device")

    def apply() -> None:
        values = np.array(source.numpy(), copy=True)
        feature_dim = values.shape[1]
        frequencies = np.power(np.float32(base), -np.arange(0, feature_dim, 2, dtype=np.float32) / feature_dim)
        angles = positions.numpy().astype(np.float32, copy=False)[:, None] * frequencies[None, :]
        cosine, sine = np.cos(angles), np.sin(angles)
        destination.numpy()[:, 0::2] = values[:, 0::2] * cosine - values[:, 1::2] * sine
        destination.numpy()[:, 1::2] = values[:, 0::2] * sine + values[:, 1::2] * cosine

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            positions.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.rope_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def embedding_lookup_fp32(
    destination: Tensor,
    token_ids: Tensor,
    table: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Look up FP32 token embeddings through an explicit CPU queue task."""
    tensors = (token_ids, table, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("embedding tensors must belong to the same device")
    if (
        token_ids.dtype != np.dtype(np.int32)
        or table.dtype != np.dtype(np.float32)
        or destination.dtype != np.dtype(np.float32)
    ):
        raise ValueError("embedding lookup requires int32 token_ids and float32 table/destination")
    if len(token_ids.shape) != 1 or len(table.shape) != 2 or destination.shape != (token_ids.shape[0], table.shape[1]):
        raise ValueError("embedding lookup tensor shapes do not align")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and embedding tensors must belong to the same device")

    def lookup() -> None:
        ids = cast(npt.NDArray[np.int32], token_ids.numpy())
        if np.any(ids < 0) or np.any(ids >= table.shape[0]):
            raise ValueError("embedding token ids are outside the vocabulary range")
        destination.numpy()[:] = table.numpy()[ids]

    event = selected_queue.host_task(
        lookup,
        wait_for=wait_for,
        buffers=(
            token_ids.access(AccessMode.READ),
            table.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.embedding_lookup_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def silu_gated_fp32(
    destination: Tensor,
    gate: Tensor,
    up: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute ``silu(gate) * up`` for the gated TinyLlama MLP activation."""
    tensors = (gate, up, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("gated activation tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("silu_gated_fp32 requires float32 tensors")
    if len(gate.shape) != 2 or up.shape != gate.shape or destination.shape != gate.shape:
        raise ValueError("gated activation tensors must have equal rank-2 shapes")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and gated activation tensors must belong to the same device")

    def apply() -> None:
        gate_values = cast(npt.NDArray[np.float32], np.array(gate.numpy(), dtype=np.float32, copy=True))
        up_values = cast(npt.NDArray[np.float32], up.numpy())
        destination.numpy()[:] = (gate_values / (np.float32(1.0) + np.exp(-gate_values))) * up_values

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(
            gate.access(AccessMode.READ),
            up.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.silu_gated_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def residual_add_fp32(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Add equal-shaped FP32 activations for a transformer residual path."""
    tensors = (left, right, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("residual tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("residual_add_fp32 requires float32 tensors")
    if len(left.shape) != 2 or right.shape != left.shape or destination.shape != left.shape:
        raise ValueError("residual tensors must have equal rank-2 shapes")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and residual tensors must belong to the same device")

    event = selected_queue.host_task(
        lambda: np.add(left.numpy(), right.numpy(), out=destination.numpy()),
        wait_for=wait_for,
        buffers=(
            left.access(AccessMode.READ),
            right.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.residual_add_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def greedy_sample_fp32(
    destination: Tensor,
    logits: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Select the highest-logit token for every rank-2 FP32 logits row."""
    if logits.buffer.device is not destination.buffer.device:
        raise DependencyError("sampling tensors must belong to the same device")
    if logits.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.int32):
        raise ValueError("greedy_sample_fp32 requires float32 logits and int32 destination")
    if len(logits.shape) != 2 or destination.shape != (logits.shape[0],):
        raise ValueError("sampling requires rank-2 logits and one output token per row")
    if logits.shape[1] <= 0 or logits.shape[1] > np.iinfo(np.int32).max:
        raise ValueError("sampling vocabulary size must fit signed int32")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and sampling tensors must belong to the same device")

    def sample() -> None:
        logits_values = cast(npt.NDArray[np.float32], logits.numpy())
        destination.numpy()[:] = np.argmax(logits_values, axis=1).astype(np.int32)

    event = selected_queue.host_task(
        sample,
        wait_for=wait_for,
        buffers=(logits.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="tinyllama.greedy_sample_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
