"""Narrow backend protocol used by the QPU-XLA runtime."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import AllocationError, DeviceClosedError


@dataclass(frozen=True, slots=True)
class BackendAllocation:
    """A contiguous, host-mapped allocation with a QPU-visible address."""

    array: npt.NDArray[np.uint8]
    address: int

    def __post_init__(self: Self) -> None:
        """Validate the allocation invariant shared by all backends."""
        if self.array.dtype != np.dtype(np.uint8) or self.array.ndim != 1:
            raise AllocationError("backend allocations must be one-dimensional uint8 arrays")
        if not 0 <= self.address <= np.iinfo(np.uint32).max:
            raise AllocationError("QPU addresses must fit in the unsigned 32-bit range")


class Backend(Protocol):
    """Protocol isolating the runtime from the low-level driver."""

    def allocate(self: Self, nbytes: int, *, alignment: int) -> BackendAllocation:
        """Allocate host-mapped memory with a QPU-visible base address."""

    def close(self: Self) -> None:
        """Release all backend resources."""


def _validate_allocation_request(nbytes: int, alignment: int) -> None:
    """Validate common allocation arguments before reaching a backend."""
    if nbytes <= 0:
        raise AllocationError("allocation size must be positive")
    if alignment <= 0 or alignment & (alignment - 1):
        raise AllocationError("alignment must be a positive power of two")


class FakeBackend:
    """CPU-only backend used for deterministic runtime and scheduler tests."""

    def __init__(self: Self, *, address_base: int = 0x1000) -> None:
        """Create a fake device with deterministic virtual QPU addresses."""
        if address_base < 0 or address_base > np.iinfo(np.uint32).max:
            raise AllocationError("address_base must fit in the unsigned 32-bit range")
        self._address = address_base
        self._closed = False
        self._allocations: list[npt.NDArray[np.uint8]] = []

    def allocate(self: Self, nbytes: int, *, alignment: int) -> BackendAllocation:
        """Allocate aligned host memory and assign it a fake device address."""
        _validate_allocation_request(nbytes, alignment)
        if self._closed:
            raise DeviceClosedError("backend is closed")
        address = (self._address + alignment - 1) & -alignment
        end_address = address + nbytes
        if end_address - 1 > np.iinfo(np.uint32).max:
            raise AllocationError("fake QPU address space is exhausted")
        storage = np.empty(nbytes + alignment - 1, dtype=np.uint8)
        offset = (-int(storage.ctypes.data)) % alignment
        array = storage[offset : offset + nbytes]
        self._allocations.append(storage)
        self._address = end_address
        return BackendAllocation(array=array, address=address)

    def close(self: Self) -> None:
        """Invalidate the fake backend and release its retained allocations."""
        self._closed = True
        self._allocations.clear()


class PyVideoCore7Backend:
    """Backend adapter over the existing synchronous :mod:`videocore7` driver."""

    def __init__(self: Self, **driver_kwargs: Any) -> None:
        """Open one low-level driver instance for runtime-owned allocations."""
        from videocore7.driver import Driver

        self._driver = Driver(**driver_kwargs)
        self._closed = False
        self._driver_lock = RLock()

    @property
    def raw_driver(self: Self) -> Any:
        """Expose the driver only for internal backend-specific kernel adapters."""
        if self._closed:
            raise DeviceClosedError("backend is closed")
        return self._driver

    @contextmanager
    def driver_session(self: Self) -> Any:
        """Serialize direct driver use for cached low-level kernel adapters."""
        if self._closed:
            raise DeviceClosedError("backend is closed")
        with self._driver_lock:
            yield self._driver

    def allocate(self: Self, nbytes: int, *, alignment: int) -> BackendAllocation:
        """Allocate memory from the existing driver-owned buffer object.

        The underlying driver is a linear allocator and currently cannot insert
        arbitrary alignment padding while retaining its address-aware ndarray
        subclass. It can safely satisfy alignments up to its allocation start
        alignment and the element alignment used here (one byte); stricter
        placement is intentionally rejected rather than silently misreported.
        """
        _validate_allocation_request(nbytes, alignment)
        if self._closed:
            raise DeviceClosedError("backend is closed")
        if alignment > 1:
            raise AllocationError(
                "PyVideoCore7Backend currently supports byte-aligned allocations only; "
                "use an aligned allocator backend before requesting a larger alignment"
            )
        raw_array: Any = self._driver.alloc((nbytes,), dtype=np.uint8)
        array = cast(npt.NDArray[np.uint8], np.asarray(raw_array))
        return BackendAllocation(array=array, address=int(raw_array.addresses()[0]))

    def close(self: Self) -> None:
        """Close the wrapped driver exactly once."""
        if not self._closed:
            self._driver.close()
            self._closed = True


KernelCallable = Callable[[Backend, tuple[Any, ...], tuple[int, int, int]], None]
