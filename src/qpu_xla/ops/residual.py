"""FP32 residual addition with CPU, QPU, and row-split execution."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.residual import RESIDUAL_ADD_FP32_KERNEL, supports_residual_add_fp32
from qpu_xla.kernels.swiglu import _workgroup_count
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def residual_add_fp32(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Add FP32 activations with explicit CPU, QPU, or row-split placement."""
    if any(t.buffer.device is not destination.buffer.device for t in (left, right)):
        raise DependencyError("residual tensors must belong to the same device")
    if (
        any(t.dtype != np.dtype(np.float32) for t in (left, right, destination))
        or left.shape != right.shape
        or left.shape != destination.shape
    ):
        raise ValueError("residual add requires equal FP32 tensors")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    accesses = (left.access(AccessMode.READ), right.access(AccessMode.READ), destination.access(AccessMode.WRITE))
    supported = supports_residual_add_fp32(left, right, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("residual QPU placement requires contiguous tensors aligned to 16 values")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid residual add requires distinct CPU and QPU queues")
        if len(left.shape) != 2 or qpu_rows is None or qpu_rows <= 0 or qpu_rows >= left.shape[0]:
            raise ValueError("hybrid residual add requires a non-empty rank-2 row split")
        qpu_tensors = tuple(t.slice((slice(0, qpu_rows), slice(None))) for t in (left, right, destination))
        if not supports_residual_add_fp32(*qpu_tensors, selected.device.backend):
            raise ValueError("residual QPU partition does not satisfy vector alignment")
        vectors = qpu_tensors[2].nbytes // 64
        qpu_event = selected.submit(
            RESIDUAL_ADD_FP32_KERNEL,
            qpu_tensors,
            grid=(_workgroup_count(vectors), 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_tensors[0].access(AccessMode.READ),
                qpu_tensors[1].access(AccessMode.READ),
                qpu_tensors[2].access(AccessMode.WRITE),
            ),
        )
        cpu_tensors = tuple(t.slice((slice(qpu_rows, left.shape[0]), slice(None))) for t in (left, right, destination))
        cpu_event = cpu_queue.host_task(
            lambda: np.add(cpu_tensors[0].numpy(), cpu_tensors[1].numpy(), out=cpu_tensors[2].numpy()),
            wait_for=wait_for,
            buffers=(
                cpu_tensors[0].access(AccessMode.READ),
                cpu_tensors[1].access(AccessMode.READ),
                cpu_tensors[2].access(AccessMode.WRITE),
            ),
            name="residual_add_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None, wait_for=(qpu_event, cpu_event), name="residual_add_fp32.hybrid_join"
        )
    elif supported and placement is Placement.QPU:
        vectors = destination.nbytes // 64
        event = selected.submit(
            RESIDUAL_ADD_FP32_KERNEL,
            (left, right, destination),
            grid=(_workgroup_count(vectors), 1, 1),
            wait_for=wait_for,
            buffers=accesses,
        )
    else:
        event = selected.host_task(
            lambda: np.add(left.numpy(), right.numpy(), out=destination.numpy()),
            wait_for=wait_for,
            buffers=accesses,
            name="residual_add_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["residual_add_fp32"]
