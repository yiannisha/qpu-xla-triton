"""Row-wise FP32 softmax with CPU, QPU, and row-split execution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.softmax import SOFTMAX_FP32_KERNEL, supports_softmax_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def softmax_fp32(
    destination: Tensor,
    source: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply stable row-wise softmax without implicit masking or scaling."""
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("softmax tensors must belong to one device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != source.dtype:
        raise ValueError("softmax requires FP32 tensors")
    if len(source.shape) != 2 or destination.shape != source.shape:
        raise ValueError("softmax requires equal rank-2 tensors")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("softmax queue belongs to a different device")

    def apply(cpu_source: Tensor = source, cpu_destination: Tensor = destination) -> None:
        values = cast(npt.NDArray[np.float32], cpu_source.numpy())
        output = cast(npt.NDArray[np.float32], cpu_destination.numpy())
        np.subtract(values, np.max(values, axis=1, keepdims=True), out=output)
        np.exp(output, out=output)
        output /= np.sum(output, axis=1, keepdims=True)

    supported = supports_softmax_fp32(source, destination, selected_queue.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("softmax QPU placement requires contiguous FP32 rows with 16-aligned width")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected_queue or cpu_queue.device is not selected_queue.device:
            raise DependencyError("hybrid softmax requires distinct CPU and QPU queues on the same device")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid softmax qpu_rows must leave non-empty CPU and QPU row partitions")
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_softmax_fp32(qpu_source, qpu_destination, selected_queue.device.backend):
            raise ValueError("softmax QPU partition does not satisfy the row kernel contract")
        qpu_event = selected_queue.submit(
            SOFTMAX_FP32_KERNEL,
            (qpu_source, qpu_destination),
            grid=(qpu_rows, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_source.access(AccessMode.READ),
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
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="softmax_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_event),
            name="softmax_fp32.hybrid_join",
        )
    elif placement is Placement.QPU:
        event = selected_queue.submit(
            SOFTMAX_FP32_KERNEL,
            (source, destination),
            grid=(source.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    else:
        event = selected_queue.host_task(
            apply,
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="softmax_fp32.reference",
        )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


__all__ = ["softmax_fp32"]
