"""FP32 embedding lookup with CPU, QPU, and token-split execution."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.embedding import EMBEDDING_LOOKUP_FP32_KERNEL, supports_embedding_lookup_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def embedding_lookup_fp32(
    destination: Tensor,
    token_ids: Tensor,
    table: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_tokens: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Gather token rows with explicit CPU, QPU, or token-split placement."""
    if any(t.buffer.device is not destination.buffer.device for t in (token_ids, table)):
        raise DependencyError("embedding tensors must belong to one device")
    if (
        token_ids.dtype != np.dtype(np.int32)
        or table.dtype != np.dtype(np.float32)
        or destination.dtype != np.dtype(np.float32)
        or len(token_ids.shape) != 1
        or len(table.shape) != 2
        or destination.shape != (token_ids.shape[0], table.shape[1])
    ):
        raise ValueError("embedding lookup shapes or dtypes do not align")
    ids = token_ids.numpy()
    if np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ValueError("embedding token ids are outside the vocabulary range")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    supported = supports_embedding_lookup_fp32(token_ids, table, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("QPU embedding lookup requires a contiguous 16-aligned embedding width")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid embedding lookup requires distinct CPU and QPU queues")
        tokens = token_ids.shape[0]
        units = tokens // 2 if qpu_tokens is None else qpu_tokens
        if units <= 0 or units >= tokens:
            raise ValueError("hybrid embedding lookup must leave non-empty token partitions")
        qpu_ids = token_ids.slice((slice(0, units),))
        qpu_destination = destination.slice((slice(0, units), slice(None)))
        if not supports_embedding_lookup_fp32(qpu_ids, table, qpu_destination, selected.device.backend):
            raise ValueError("embedding QPU partition does not satisfy the kernel contract")
        qpu_event = selected.submit(
            EMBEDDING_LOOKUP_FP32_KERNEL,
            (qpu_ids, table, qpu_destination),
            grid=(units, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_ids.access(AccessMode.READ),
                table.access(AccessMode.READ),
                qpu_destination.access(AccessMode.WRITE),
            ),
        )
        cpu_ids = token_ids.slice((slice(units, tokens),))
        cpu_destination = destination.slice((slice(units, tokens), slice(None)))
        cpu_event = cpu_queue.host_task(
            lambda: np.copyto(cpu_destination.numpy(), table.numpy()[cpu_ids.numpy()]),
            wait_for=wait_for,
            buffers=(
                cpu_ids.access(AccessMode.READ),
                table.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="embedding_lookup_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None, wait_for=(qpu_event, cpu_event), name="embedding_lookup_fp32.hybrid_join"
        )
    elif supported and placement is Placement.QPU:
        event = selected.submit(
            EMBEDDING_LOOKUP_FP32_KERNEL,
            (token_ids, table, destination),
            grid=(token_ids.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                token_ids.access(AccessMode.READ),
                table.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    else:
        event = selected.host_task(
            lambda: np.copyto(destination.numpy(), table.numpy()[token_ids.numpy()]),
            wait_for=wait_for,
            buffers=(
                token_ids.access(AccessMode.READ),
                table.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="embedding_lookup_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["embedding_lookup_fp32"]
