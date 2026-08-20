"""Cached-table rotary position embedding operator."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.rope import ROPE_FP32_KERNEL, _workgroup_count, supports_rope_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def rope_tables_fp32(
    positions: npt.NDArray[np.int32],
    feature_dim: int,
    *,
    base: float = 10_000.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build repeat-per-pair cosine and signed-sine tables suitable for caching."""
    if positions.ndim != 1 or positions.dtype != np.dtype(np.int32):
        raise ValueError("RoPE positions must be a rank-1 int32 array")
    if feature_dim <= 0 or feature_dim % 2 or not np.isfinite(base) or base <= 1:
        raise ValueError("RoPE requires a positive even feature dimension and base greater than one")
    frequencies = np.power(
        np.float32(base),
        -np.arange(0, feature_dim, 2, dtype=np.float32) / feature_dim,
    )
    angles = positions.astype(np.float32, copy=False)[:, None] * frequencies[None, :]
    cosine_pairs = np.cos(angles).astype(np.float32)
    sine_pairs = np.sin(angles).astype(np.float32)
    cosine = np.repeat(cosine_pairs, 2, axis=1)
    signed_sine = np.empty_like(cosine)
    signed_sine[:, 0::2] = -sine_pairs
    signed_sine[:, 1::2] = sine_pairs
    return np.ascontiguousarray(cosine), np.ascontiguousarray(signed_sine)


def apply_rope_tables_fp32(
    destination: Tensor,
    source: Tensor,
    cosine: Tensor,
    signed_sine: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply already prepared RoPE tables without charging trig generation."""
    tensors = (source, cosine, signed_sine, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("RoPE tensors must belong to one device")
    if any(tensor.dtype != np.dtype(np.float32) or tensor.shape != source.shape for tensor in tensors):
        raise ValueError("RoPE requires equal FP32 source, table, and destination tensors")
    if len(source.shape) != 2 or source.shape[1] % 2:
        raise ValueError("RoPE requires rank-2 tensors with an even feature dimension")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("RoPE queue belongs to a different device")
    supported = supports_rope_fp32(source, cosine, signed_sine, destination, selected_queue.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("RoPE QPU placement requires complete contiguous 16-value vectors")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected_queue or cpu_queue.device is not selected_queue.device:
            raise DependencyError("hybrid RoPE requires distinct CPU and QPU queues on the same device")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid RoPE qpu_rows must leave non-empty CPU and QPU row partitions")
        qpu_tensors = tuple(tensor.slice((slice(0, qpu_rows), slice(None))) for tensor in tensors)
        qpu_source, qpu_cosine, qpu_sine, qpu_destination = qpu_tensors
        if not supports_rope_fp32(
            qpu_source,
            qpu_cosine,
            qpu_sine,
            qpu_destination,
            selected_queue.device.backend,
        ):
            raise ValueError("RoPE QPU partition does not satisfy the cached-table kernel contract")
        vectors = qpu_source.nbytes // 64
        qpu_event = selected_queue.submit(
            ROPE_FP32_KERNEL,
            qpu_tensors,
            grid=(_workgroup_count(vectors), 1, 1),
            wait_for=wait_for,
            buffers=tuple(tensor.access(AccessMode.READ) for tensor in qpu_tensors[:3])
            + (qpu_destination.access(AccessMode.WRITE),),
        )
        cpu_source, cpu_cosine, cpu_sine, cpu_destination = tuple(
            tensor.slice((slice(qpu_rows, rows), slice(None))) for tensor in tensors
        )

        def cpu_tail() -> None:
            values = np.array(cpu_source.numpy(), copy=True)
            adjacent = values.reshape(-1, 2)[:, ::-1].reshape(values.shape)
            cpu_destination.numpy()[:] = values * cpu_cosine.numpy() + adjacent * cpu_sine.numpy()

        cpu_event = cpu_queue.host_task(
            cpu_tail,
            wait_for=wait_for,
            buffers=(
                cpu_source.access(AccessMode.READ),
                cpu_cosine.access(AccessMode.READ),
                cpu_sine.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="rope_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_event),
            name="rope_fp32.hybrid_join",
        )
        if close_queue:
            event.wait()
            selected_queue.close()
        return event
    if placement is Placement.QPU:
        vectors = source.nbytes // 64
        event = selected_queue.submit(
            ROPE_FP32_KERNEL,
            (source, cosine, signed_sine, destination),
            grid=(_workgroup_count(vectors), 1, 1),
            wait_for=wait_for,
            buffers=tuple(tensor.access(AccessMode.READ) for tensor in tensors[:3])
            + (destination.access(AccessMode.WRITE),),
        )
    else:

        def reference() -> None:
            values = np.array(source.numpy(), copy=True)
            adjacent = values.reshape(-1, 2)[:, ::-1].reshape(values.shape)
            destination.numpy()[:] = values * cosine.numpy() + adjacent * signed_sine.numpy()

        event = selected_queue.host_task(
            reference,
            wait_for=wait_for,
            buffers=tuple(tensor.access(AccessMode.READ) for tensor in tensors[:3])
            + (destination.access(AccessMode.WRITE),),
            name="rope_fp32.cached_table_reference",
        )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


__all__ = ["apply_rope_tables_fp32", "rope_tables_fp32"]
