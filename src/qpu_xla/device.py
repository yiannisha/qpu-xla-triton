"""Device lifetime and allocation entry points for QPU-XLA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self
from weakref import WeakSet

import numpy.typing as npt

from qpu_xla.backend import Backend, BackendAllocation, FakeBackend, PyVideoCore7Backend, _validate_allocation_request
from qpu_xla.errors import DeviceClosedError
from qpu_xla.memory import Buffer, Tensor, _AllocationState

if TYPE_CHECKING:
    from qpu_xla.queue import Queue


class Device:
    """An owned QPU-XLA device context backed by one low-level backend."""

    def __init__(self: Self, backend: Backend) -> None:
        """Create a device over a supplied backend implementation."""
        self._backend = backend
        self._closed = False
        self._allocations: list[_AllocationState] = []
        self._free_allocations: list[BackendAllocation] = []
        self._queues: WeakSet[Queue] = WeakSet()

    @classmethod
    def fake(cls: type[Device]) -> Device:
        """Create a CPU-only device for tests and API development."""
        return cls(FakeBackend())

    @classmethod
    def open(cls: type[Device], **driver_kwargs: object) -> Device:
        """Open a hardware-backed device using the existing VideoCore driver."""
        return cls(PyVideoCore7Backend(**driver_kwargs))

    @property
    def backend(self: Self) -> Backend:
        """Return the internal backend for kernel adapters, if still open."""
        self._require_open()
        return self._backend

    @property
    def closed(self: Self) -> bool:
        """Whether the device has been closed."""
        return self._closed

    def _require_open(self: Self) -> None:
        """Raise if the device can no longer create or submit work."""
        if self._closed:
            raise DeviceClosedError("device is closed")

    def allocate(self: Self, nbytes: int, *, alignment: int = 1) -> Buffer:
        """Allocate or reuse a QPU-addressable byte buffer owned by this device."""
        self._require_open()
        _validate_allocation_request(nbytes, alignment)
        allocation = self._take_free_allocation(nbytes, alignment)
        state = _AllocationState(allocation)
        self._allocations.append(state)
        return Buffer(self, state, nbytes=nbytes)

    def _take_free_allocation(self: Self, nbytes: int, alignment: int) -> BackendAllocation:
        """Reuse the smallest compatible released allocation or request a new one."""
        compatible = [
            (index, allocation)
            for index, allocation in enumerate(self._free_allocations)
            if allocation.array.nbytes >= nbytes and allocation.address % alignment == 0
        ]
        if not compatible:
            return self._backend.allocate(nbytes, alignment=alignment)
        index, allocation = min(compatible, key=lambda candidate: candidate[1].array.nbytes)
        del self._free_allocations[index]
        return allocation

    def _release_allocation(self: Self, state: _AllocationState) -> None:
        """Return one closed allocation to the device-local reusable pool."""
        if self._closed:
            return
        self._free_allocations.append(state.allocation)

    def tensor(
        self: Self,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        *,
        alignment: int = 1,
    ) -> Tensor:
        """Allocate a contiguous typed tensor without a host/device copy."""
        import numpy as np

        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("tensor shape must contain only positive dimensions")
        nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        return self.allocate(nbytes, alignment=alignment).tensor(shape, dtype)

    def queue(self: Self) -> Queue:
        """Create an in-order submission queue associated with this device."""
        from qpu_xla.queue import Queue

        self._require_open()
        queue = Queue(self)
        self._queues.add(queue)
        return queue

    def close(self: Self) -> None:
        """Wait for submitted work, close queues, and invalidate allocations."""
        if self._closed:
            return
        for queue in tuple(self._queues):
            queue.close()
        for state in self._allocations:
            state.closed = True
        self._free_allocations.clear()
        self._backend.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter a context-managed device lifetime."""
        self._require_open()
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the device at context exit."""
        self.close()
