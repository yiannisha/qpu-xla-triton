"""FP32 scalar scale and column bias with heterogeneous row placement."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.affine import AFFINE_FP32_KERNEL, supports_affine_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def affine_fp32(
    destination: Tensor,
    source: Tensor,
    bias: Tensor,
    *,
    scale: float = 1.0,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply ``source * scale + bias`` over rank-2 FP32 rows."""
    if any(tensor.buffer.device is not destination.buffer.device for tensor in (source, bias)):
        raise DependencyError("affine tensors must belong to one device")
    if not np.isfinite(scale):
        raise ValueError("affine scale must be finite")
    if (
        source.dtype != np.dtype(np.float32)
        or bias.dtype != source.dtype
        or destination.dtype != source.dtype
        or len(source.shape) != 2
        or destination.shape != source.shape
        or bias.shape != (source.shape[1],)
    ):
        raise ValueError("affine requires FP32 rows and a matching bias vector")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected.device is not destination.buffer.device:
        raise DependencyError("affine queue belongs to a different device")

    def apply(src: Tensor, dst: Tensor) -> None:
        source_values = cast(npt.NDArray[np.float32], src.numpy())
        destination_values = cast(npt.NDArray[np.float32], dst.numpy())
        bias_values = cast(npt.NDArray[np.float32], bias.numpy())
        np.multiply(source_values, np.float32(scale), out=destination_values)
        np.add(destination_values, bias_values, out=destination_values)

    supported = supports_affine_fp32(source, bias, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("affine QPU placement requires contiguous FP32 rows with a 16-aligned width")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid affine requires distinct CPU and QPU queues")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid affine qpu_rows must leave two non-empty partitions")
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_affine_fp32(qpu_source, bias, qpu_destination, selected.device.backend):
            raise ValueError("affine QPU row partition does not satisfy the kernel contract")
        qpu_event = selected.submit(
            AFFINE_FP32_KERNEL,
            (qpu_source, bias, qpu_destination, float(scale)),
            grid=(qpu_rows, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_source.access(AccessMode.READ),
                bias.access(AccessMode.READ),
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
                bias.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="affine_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(qpu_event, cpu_event),
            name="affine_fp32.hybrid_join",
        )
    elif placement is Placement.QPU:
        event = selected.submit(
            AFFINE_FP32_KERNEL,
            (source, bias, destination, float(scale)),
            grid=(source.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                bias.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    else:
        event = selected.host_task(
            lambda: apply(source, destination),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                bias.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="affine_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["affine_fp32"]
