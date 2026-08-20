"""Fused SiLU-gated activation with a VideoCore VII specialization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.swiglu import SWIGLU_FP32_KERNEL, _workgroup_count, supports_swiglu_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def swiglu_fp32(
    destination: Tensor,
    gate: Tensor,
    up: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply ``destination = silu(gate) * up`` through one queued operation."""
    if gate.buffer.device is not destination.buffer.device or up.buffer.device is not destination.buffer.device:
        raise DependencyError("SwiGLU tensors must belong to the same device")
    if gate.dtype != np.dtype(np.float32) or up.dtype != gate.dtype or destination.dtype != gate.dtype:
        raise ValueError("SwiGLU requires FP32 tensors")
    if gate.shape != destination.shape or up.shape != destination.shape:
        raise ValueError("SwiGLU tensors must have equal shapes")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and SwiGLU tensors must belong to the same device")
    accesses = (
        gate.access(AccessMode.READ),
        up.access(AccessMode.READ),
        destination.access(AccessMode.WRITE),
    )

    def reference() -> None:
        gate_values = cast(npt.NDArray[np.float32], gate.numpy())
        up_values = cast(npt.NDArray[np.float32], up.numpy())
        destination.numpy()[:] = (gate_values / (np.float32(1.0) + np.exp(-gate_values))) * up_values

    supported = supports_swiglu_fp32(gate, up, destination, selected_queue.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("SwiGLU QPU placement requires equal contiguous FP32 tensors aligned to 16 values")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected_queue or cpu_queue.device is not selected_queue.device:
            raise DependencyError("hybrid SwiGLU requires distinct CPU and QPU queues on the same device")
        if len(gate.shape) != 2:
            raise ValueError("hybrid SwiGLU currently partitions rank-2 tensors by rows")
        rows = gate.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid SwiGLU qpu_rows must leave non-empty CPU and QPU row partitions")
        qpu_gate = gate.slice((slice(0, qpu_rows), slice(None)))
        qpu_up = up.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_swiglu_fp32(qpu_gate, qpu_up, qpu_destination, selected_queue.device.backend):
            raise ValueError("SwiGLU QPU partition does not satisfy the vector kernel contract")
        vector_count = qpu_destination.nbytes // (16 * np.dtype(np.float32).itemsize)
        qpu_event = selected_queue.submit(
            SWIGLU_FP32_KERNEL,
            (qpu_gate, qpu_up, qpu_destination),
            grid=(_workgroup_count(vector_count), 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_gate.access(AccessMode.READ),
                qpu_up.access(AccessMode.READ),
                qpu_destination.access(AccessMode.WRITE),
            ),
        )
        cpu_gate = gate.slice((slice(qpu_rows, rows), slice(None)))
        cpu_up = up.slice((slice(qpu_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, rows), slice(None)))

        def cpu_tail() -> None:
            gate_values = cast(npt.NDArray[np.float32], cpu_gate.numpy())
            up_values = cast(npt.NDArray[np.float32], cpu_up.numpy())
            cpu_destination.numpy()[:] = (
                gate_values / (np.float32(1.0) + np.exp(-gate_values))
            ) * up_values

        cpu_event = cpu_queue.host_task(
            cpu_tail,
            wait_for=wait_for,
            buffers=(
                cpu_gate.access(AccessMode.READ),
                cpu_up.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="swiglu_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_event),
            name="swiglu_fp32.hybrid_join",
        )
    elif supported and placement is not Placement.CPU:
        vector_count = destination.nbytes // (16 * np.dtype(np.float32).itemsize)
        event = selected_queue.submit(
            SWIGLU_FP32_KERNEL,
            (gate, up, destination),
            grid=(_workgroup_count(vector_count), 1, 1),
            wait_for=wait_for,
            buffers=accesses,
        )
    else:
        event = selected_queue.host_task(reference, wait_for=wait_for, buffers=accesses, name="swiglu_fp32.reference")
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


__all__ = ["swiglu_fp32"]
