"""Gemma-style split-half rotary embeddings with cached FP32 tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.rope_split_half import (
    ROPE_SPLIT_HALF_FP32_KERNEL,
    supports_rope_split_half_fp32,
)
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def split_half_rope_tables_fp32(
    positions: npt.NDArray[np.int32],
    *,
    heads: int,
    head_dim: int,
    base: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build tables laid out as one contiguous row per token/head pair."""
    if positions.ndim != 1 or positions.dtype != np.dtype(np.int32):
        raise ValueError("split-half RoPE positions must be rank-1 int32")
    if heads <= 0 or head_dim <= 0 or head_dim % 32 or not np.isfinite(base) or base <= 1:
        raise ValueError("split-half RoPE requires heads, a 32-aligned width, and base > 1")
    half = head_dim // 2
    exponents = np.float32(2.0 / head_dim) * np.arange(half, dtype=np.float32)
    timescale = np.power(np.float32(base), exponents)
    angles = positions.astype(np.float32)[:, None] / timescale[None, :]
    cosine = np.repeat(np.cos(angles)[:, None, :], heads, axis=1).reshape(-1, half)
    sine = np.repeat(np.sin(angles)[:, None, :], heads, axis=1).reshape(-1, half)
    return np.ascontiguousarray(cosine), np.ascontiguousarray(sine)


def apply_split_half_rope_tables_fp32(
    destination: Tensor,
    source: Tensor,
    cosine: Tensor,
    sine: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply cached split-half rotations to flattened token/head rows."""
    tensors = (source, cosine, sine, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("split-half RoPE tensors must belong to one device")
    if (
        any(tensor.dtype != np.dtype(np.float32) for tensor in tensors)
        or len(source.shape) != 2
        or destination.shape != source.shape
        or cosine.shape != (source.shape[0], source.shape[1] // 2)
        or sine.shape != cosine.shape
    ):
        raise ValueError("split-half RoPE tensors do not meet the flattened head-row contract")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected.device is not destination.buffer.device:
        raise DependencyError("split-half RoPE queue belongs to a different device")

    def apply(src: Tensor, cos: Tensor, sin: Tensor, dst: Tensor) -> None:
        half = src.shape[1] // 2
        source_values = cast(npt.NDArray[np.float32], src.numpy())
        cosine_values = cast(npt.NDArray[np.float32], cos.numpy())
        sine_values = cast(npt.NDArray[np.float32], sin.numpy())
        destination_values = cast(npt.NDArray[np.float32], dst.numpy())
        first = source_values[:, :half]
        second = source_values[:, half:]
        destination_values[:, :half] = first * cosine_values - second * sine_values
        destination_values[:, half:] = second * cosine_values + first * sine_values

    supported = supports_rope_split_half_fp32(source, cosine, sine, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("split-half RoPE QPU placement requires contiguous 32-aligned head rows")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid split-half RoPE requires distinct CPU and QPU queues")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid split-half RoPE qpu_rows must leave two non-empty partitions")
        qpu_tensors = tuple(tensor.slice((slice(0, qpu_rows), slice(None))) for tensor in tensors)
        if not supports_rope_split_half_fp32(
            qpu_tensors[0], qpu_tensors[1], qpu_tensors[2], qpu_tensors[3], selected.device.backend
        ):
            raise ValueError("split-half RoPE QPU partition does not satisfy the kernel contract")
        qpu_event = selected.submit(
            ROPE_SPLIT_HALF_FP32_KERNEL,
            qpu_tensors,
            grid=(qpu_rows, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_tensors[0].access(AccessMode.READ),
                qpu_tensors[1].access(AccessMode.READ),
                qpu_tensors[2].access(AccessMode.READ),
                qpu_tensors[3].access(AccessMode.WRITE),
            ),
        )
        cpu_tensors = tuple(tensor.slice((slice(qpu_rows, rows), slice(None))) for tensor in tensors)
        cpu_event = cpu_queue.host_task(
            lambda: apply(*cpu_tensors),
            wait_for=wait_for,
            buffers=(
                cpu_tensors[0].access(AccessMode.READ),
                cpu_tensors[1].access(AccessMode.READ),
                cpu_tensors[2].access(AccessMode.READ),
                cpu_tensors[3].access(AccessMode.WRITE),
            ),
            name="rope_split_half_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(qpu_event, cpu_event),
            name="rope_split_half_fp32.hybrid_join",
        )
    elif placement is Placement.QPU:
        event = selected.submit(
            ROPE_SPLIT_HALF_FP32_KERNEL,
            tensors,
            grid=(source.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                cosine.access(AccessMode.READ),
                sine.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    else:
        event = selected.host_task(
            lambda: apply(*tensors),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                cosine.access(AccessMode.READ),
                sine.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="rope_split_half_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["apply_split_half_rope_tables_fp32", "split_half_rope_tables_fp32"]
