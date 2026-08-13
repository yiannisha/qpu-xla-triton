"""Elementwise operator contracts with queue-integrated CPU references."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernel import Kernel
from qpu_xla.kernels.copy import WORD_COPY_KERNEL, supports_word_copy
from qpu_xla.kernels.minmax import MAXIMUM_WORD_KERNEL, MINIMUM_WORD_KERNEL, supports_word_minmax
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue


def _select_queue(destination: Tensor, queue: Queue | None) -> tuple[Queue, bool]:
    """Use a supplied queue or create a short-lived queue for one operation."""
    if queue is None:
        return destination.buffer.device.queue(), True
    if queue.device is not destination.buffer.device:
        raise DependencyError("queue and destination tensor must belong to the same device")
    return queue, False


def _validate_binary(destination: Tensor, left: Tensor, right: Tensor) -> None:
    """Validate the no-broadcast v0 contract shared by min and max."""
    tensors = (destination, left, right)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("elementwise tensors must belong to the same device")
    if any(tensor.shape != destination.shape for tensor in tensors):
        raise ValueError("elementwise v0 requires equal tensor shapes")
    if any(tensor.dtype != destination.dtype for tensor in tensors):
        raise ValueError("elementwise v0 requires equal tensor dtypes")


def copy(
    destination: Tensor,
    source: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Copy one equal-shaped tensor into another through a runtime queue.

    This is the CPU reference implementation for the initial packaged operator
    surface. Its declared accesses make it safe to replace with a QPU kernel
    specialization without changing caller synchronization semantics.
    """
    _validate_binary(destination, source, source)
    selected_queue, close_queue = _select_queue(destination, queue)
    accesses = (destination.access(AccessMode.WRITE), source.access(AccessMode.READ))
    if supports_word_copy(source, destination, selected_queue.device.backend):
        event = selected_queue.submit(WORD_COPY_KERNEL, (source, destination), wait_for=wait_for, buffers=accesses)
    else:
        event = selected_queue.host_task(
            lambda: np.copyto(destination.numpy(), source.numpy(), casting="no"),
            wait_for=wait_for,
            buffers=accesses,
        )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def minimum(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Store the elementwise minimum without broadcasting."""
    return _binary_op(
        destination,
        left,
        right,
        np.minimum,
        qpu_kernel=MINIMUM_WORD_KERNEL,
        queue=queue,
        wait_for=wait_for,
    )


def maximum(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Store the elementwise maximum without broadcasting."""
    return _binary_op(
        destination,
        left,
        right,
        np.maximum,
        qpu_kernel=MAXIMUM_WORD_KERNEL,
        queue=queue,
        wait_for=wait_for,
    )


def _binary_op(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    operation: Callable[..., object],
    *,
    qpu_kernel: Kernel,
    queue: Queue | None,
    wait_for: Iterable[Event],
) -> Event:
    """Submit one binary CPU reference operation with declared tensor accesses."""
    _validate_binary(destination, left, right)
    selected_queue, close_queue = _select_queue(destination, queue)

    def action() -> None:
        operation(left.numpy(), right.numpy(), out=destination.numpy())

    accesses = (
        destination.access(AccessMode.WRITE),
        left.access(AccessMode.READ),
        right.access(AccessMode.READ),
    )
    if supports_word_minmax(left, right, destination, selected_queue.device.backend):
        event = selected_queue.submit(qpu_kernel, (left, right, destination), wait_for=wait_for, buffers=accesses)
    else:
        event = selected_queue.host_task(action, wait_for=wait_for, buffers=accesses)
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
