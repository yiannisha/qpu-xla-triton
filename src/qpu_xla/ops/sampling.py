"""Greedy FP32 sampling with CPU, QPU, and row-split execution."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.argmax import ARGMAX_FP32_KERNEL
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def greedy_sample_fp32(
    destination: Tensor,
    logits: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Select the first maximum index per row using the requested placement."""
    if logits.buffer.device is not destination.buffer.device:
        raise DependencyError("sampling tensors must belong to one device")
    if (
        logits.dtype != np.dtype(np.float32)
        or destination.dtype != np.dtype(np.int32)
        or len(logits.shape) != 2
        or destination.shape != (logits.shape[0],)
    ):
        raise ValueError("greedy sampling requires rank-2 FP32 logits and one INT32 result per row")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    qpu_supported = logits.shape[1] % 16 == 0 and logits.numpy().flags.c_contiguous
    if placement is Placement.QPU and not qpu_supported:
        raise ValueError("QPU argmax requires contiguous logits with a 16-aligned vocabulary")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid argmax requires distinct CPU and QPU queues")
        rows = logits.shape[0]
        units = rows // 2 if qpu_rows is None else qpu_rows
        if units <= 0 or units >= rows:
            raise ValueError("hybrid argmax must leave non-empty row partitions")
    else:
        units = logits.shape[0]
    if placement in (Placement.QPU, Placement.HYBRID):
        qpu_logits = logits.slice((slice(0, units), slice(None)))
        scratch = destination.buffer.device.tensor((units, 16), np.int32)
        qpu_event = selected.submit(
            ARGMAX_FP32_KERNEL,
            (qpu_logits, scratch),
            grid=(units, 1, 1),
            wait_for=wait_for,
            buffers=(qpu_logits.access(AccessMode.READ), scratch.access(AccessMode.WRITE)),
        )

        def finish_qpu() -> None:
            destination.numpy()[:units] = scratch.numpy()[:, 0]
            scratch.buffer.close()

        qpu_finish = selected.host_task(
            finish_qpu,
            wait_for=(qpu_event,),
            buffers=(scratch.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="greedy_sample_fp32.qpu_finish",
        )
        if placement is Placement.HYBRID:
            assert cpu_queue is not None
            cpu_logits = logits.slice((slice(units, logits.shape[0]), slice(None)))
            cpu_destination = destination.slice((slice(units, logits.shape[0]),))
            cpu_event = cpu_queue.host_task(
                lambda: np.copyto(cpu_destination.numpy(), np.argmax(cpu_logits.numpy(), axis=1).astype(np.int32)),
                wait_for=wait_for,
                buffers=(cpu_logits.access(AccessMode.READ), cpu_destination.access(AccessMode.WRITE)),
                name="greedy_sample_fp32.hybrid_cpu_tail",
            )
            event = cpu_queue.host_task(
                lambda: None, wait_for=(qpu_finish, cpu_event), name="greedy_sample_fp32.hybrid_join"
            )
        else:
            event = qpu_finish
    else:
        event = selected.host_task(
            lambda: np.copyto(destination.numpy(), np.argmax(logits.numpy(), axis=1).astype(np.int32)),
            wait_for=wait_for,
            buffers=(logits.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="greedy_sample_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["greedy_sample_fp32"]
