"""FP32 RMSNorm with CPU, QPU, and concurrent row partitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.rms_norm import RMS_NORM_FP32_KERNEL, supports_rms_norm_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def rms_norm_fp32(
    destination: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    epsilon: float,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply row-wise RMS normalization under an explicit placement."""
    if any(tensor.buffer.device is not destination.buffer.device for tensor in (source, weight)):
        raise DependencyError("RMSNorm tensors must belong to one device")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    if (
        source.dtype != np.dtype(np.float32)
        or weight.dtype != source.dtype
        or destination.dtype != source.dtype
        or len(source.shape) != 2
        or destination.shape != source.shape
        or weight.shape != (source.shape[1],)
    ):
        raise ValueError("RMSNorm requires FP32 rows and a matching weight vector")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected.device is not destination.buffer.device:
        raise DependencyError("RMSNorm queue belongs to a different device")

    def apply(src: Tensor, dst: Tensor) -> None:
        values = cast(npt.NDArray[np.float32], src.numpy())
        destination_values = cast(npt.NDArray[np.float32], dst.numpy())
        weight_values = cast(npt.NDArray[np.float32], weight.numpy())
        mean_square = np.mean(values * values, axis=1, keepdims=True, dtype=np.float32)
        destination_values[:] = values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon))) * weight_values

    supported = supports_rms_norm_fp32(source, weight, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("RMSNorm QPU placement requires contiguous FP32 rows with a 16-aligned width")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid RMSNorm requires distinct CPU and QPU queues on one device")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid RMSNorm qpu_rows must leave two non-empty partitions")
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_rms_norm_fp32(qpu_source, weight, qpu_destination, selected.device.backend):
            raise ValueError("RMSNorm QPU row partition does not satisfy the kernel contract")
        qpu_event = selected.submit(
            RMS_NORM_FP32_KERNEL,
            (qpu_source, weight, qpu_destination, float(epsilon)),
            grid=(qpu_rows, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                qpu_destination.access(AccessMode.WRITE),
            ),
        )
        cpu_source = source.slice((slice(qpu_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, rows), slice(None)))
        cpu_event = cpu_queue.host_task(
            lambda: apply(cpu_source, cpu_destination),
            wait_for=wait_for,
            buffers=(
                cpu_source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="rms_norm_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(qpu_event, cpu_event),
            name="rms_norm_fp32.hybrid_join",
        )
    elif placement is Placement.QPU:
        event = selected.submit(
            RMS_NORM_FP32_KERNEL,
            (source, weight, destination, float(epsilon)),
            grid=(source.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    else:
        event = selected.host_task(
            lambda: apply(source, destination),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="rms_norm_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["rms_norm_fp32"]
