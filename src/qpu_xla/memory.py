"""Owned buffer and tensor views over backend allocations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

from qpu_xla.backend import BackendAllocation
from qpu_xla.errors import AllocationError, BufferClosedError

if TYPE_CHECKING:
    from qpu_xla.device import Device


class AccessMode(Enum):
    """Declared access mode for a buffer region used by a queue submission."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"

    @property
    def writes(self: Self) -> bool:
        """Whether this access can modify the referenced region."""
        return self is not AccessMode.READ


@dataclass(slots=True)
class _AllocationState:
    """Shared close state retained by buffers and all of their tensor views."""

    allocation: BackendAllocation
    closed: bool = False

    def require_open(self: Self) -> None:
        """Raise if this allocation has been invalidated."""
        if self.closed:
            raise BufferClosedError("buffer is closed")


@dataclass(frozen=True, slots=True)
class BufferAccess:
    """A declared range and access mode used to construct safe queue dependencies."""

    buffer: Buffer
    mode: AccessMode
    offset: int = 0
    nbytes: int | None = None

    def __post_init__(self: Self) -> None:
        """Validate the requested range while the buffer is still live."""
        self.buffer._state.require_open()
        nbytes = self.buffer.nbytes - self.offset if self.nbytes is None else self.nbytes
        if self.offset < 0 or nbytes <= 0 or self.offset + nbytes > self.buffer.nbytes:
            raise AllocationError("buffer access range is outside the buffer")
        object.__setattr__(self, "nbytes", nbytes)

    @property
    def absolute_offset(self: Self) -> int:
        """Return the access offset relative to the root allocation."""
        return self.buffer.offset + self.offset

    def conflicts_with(self: Self, other: BufferAccess) -> bool:
        """Return whether two declared accesses overlap and at least one writes."""
        if self.buffer._state is not other.buffer._state or not (self.mode.writes or other.mode.writes):
            return False
        assert self.nbytes is not None
        assert other.nbytes is not None
        return (
            self.absolute_offset < other.absolute_offset + other.nbytes
            and other.absolute_offset < self.absolute_offset + self.nbytes
        )


class Buffer:
    """A runtime-owned, QPU-addressable byte range."""

    def __init__(
        self: Self,
        device: Device,
        state: _AllocationState,
        *,
        offset: int = 0,
        nbytes: int | None = None,
    ) -> None:
        """Create a buffer or sub-buffer backed by one allocation state."""
        state.require_open()
        total_bytes = state.allocation.array.nbytes
        nbytes = total_bytes - offset if nbytes is None else nbytes
        if offset < 0 or nbytes <= 0 or offset + nbytes > total_bytes:
            raise AllocationError("buffer range is outside its allocation")
        self._device = device
        self._state = state
        self._offset = offset
        self._nbytes = nbytes

    @property
    def device(self: Self) -> Device:
        """Return the owning device."""
        return self._device

    @property
    def offset(self: Self) -> int:
        """Return the byte offset from the allocation base."""
        return self._offset

    @property
    def nbytes(self: Self) -> int:
        """Return the number of accessible bytes."""
        self._state.require_open()
        return self._nbytes

    @property
    def address(self: Self) -> int:
        """Return the QPU-visible address of this buffer range."""
        self._state.require_open()
        return self._state.allocation.address + self._offset

    def close(self: Self) -> None:
        """Invalidate this allocation and every view derived from it."""
        if self._state.closed:
            return
        self._state.closed = True
        self._device._release_allocation(self._state)

    def slice(self: Self, offset: int, nbytes: int | None = None) -> Buffer:
        """Create a QPU-addressable sub-buffer that retains the parent allocation."""
        self._state.require_open()
        nbytes = self._nbytes - offset if nbytes is None else nbytes
        return Buffer(self._device, self._state, offset=self._offset + offset, nbytes=nbytes)

    def access(
        self: Self,
        mode: AccessMode,
        *,
        offset: int = 0,
        nbytes: int | None = None,
    ) -> BufferAccess:
        """Declare an access range for a queue operation."""
        return BufferAccess(self, mode, offset, nbytes)

    def tensor(
        self: Self,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        *,
        offset: int = 0,
        strides: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Create a typed tensor view into this buffer without copying."""
        return Tensor(self, shape, dtype, offset=offset, strides=strides)


class Tensor:
    """A typed, strided tensor view that shares one QPU allocation with NumPy."""

    def __init__(
        self: Self,
        buffer: Buffer,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        *,
        offset: int = 0,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        """Create a validated tensor view into ``buffer``."""
        buffer._state.require_open()
        if not shape or any(dimension <= 0 for dimension in shape):
            raise AllocationError("tensor shape must contain only positive dimensions")
        dtype = np.dtype(dtype)
        if offset < 0 or offset % dtype.itemsize:
            raise AllocationError("tensor offset must be non-negative and aligned to the dtype")
        strides = self._contiguous_strides(shape, dtype.itemsize) if strides is None else strides
        if len(strides) != len(shape) or any(stride <= 0 for stride in strides):
            raise AllocationError("tensor strides must be positive byte strides for every dimension")
        required_nbytes = self._required_nbytes(shape, dtype.itemsize, strides)
        if offset + required_nbytes > buffer.nbytes:
            raise AllocationError("tensor view exceeds its buffer")
        self._buffer = buffer
        self._shape = shape
        self._dtype = dtype
        self._offset = offset
        self._strides = strides

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
        """Return C-order byte strides for a positive-dimensional shape."""
        strides: list[int] = []
        stride = itemsize
        for dimension in reversed(shape):
            strides.append(stride)
            stride *= dimension
        return tuple(reversed(strides))

    @staticmethod
    def _required_nbytes(shape: tuple[int, ...], itemsize: int, strides: tuple[int, ...]) -> int:
        """Return the backing span needed by a positive-stride tensor."""
        return itemsize + sum((dimension - 1) * stride for dimension, stride in zip(shape, strides, strict=True))

    @property
    def buffer(self: Self) -> Buffer:
        """Return the buffer containing this tensor."""
        return self._buffer

    @property
    def shape(self: Self) -> tuple[int, ...]:
        """Return the tensor shape."""
        return self._shape

    @property
    def dtype(self: Self) -> np.dtype[np.generic]:
        """Return the tensor element dtype."""
        return self._dtype

    @property
    def strides(self: Self) -> tuple[int, ...]:
        """Return tensor byte strides."""
        return self._strides

    @property
    def nbytes(self: Self) -> int:
        """Return the backing span occupied by this tensor."""
        self._buffer._state.require_open()
        return self._required_nbytes(self._shape, self._dtype.itemsize, self._strides)

    @property
    def address(self: Self) -> int:
        """Return the QPU-visible address of tensor element ``[0, ...]``."""
        self._buffer._state.require_open()
        return self._buffer.address + self._offset

    def numpy(self: Self) -> npt.NDArray[np.generic]:
        """Return a zero-copy NumPy view while the owning buffer remains open."""
        self._buffer._state.require_open()
        return np.ndarray(
            self._shape,
            dtype=self._dtype,
            buffer=self._buffer._state.allocation.array,
            offset=self._buffer.offset + self._offset,
            strides=self._strides,
        )

    def slice(self: Self, slices: tuple[slice, ...]) -> Tensor:
        """Return a positive-step tensor slice without copying."""
        if len(slices) != len(self._shape):
            raise AllocationError("one slice is required for each tensor dimension")
        shape: list[int] = []
        offset = self._offset
        for dimension, stride, item in zip(self._shape, self._strides, slices, strict=True):
            start, stop, step = item.indices(dimension)
            if step != 1:
                raise AllocationError("only unit-step tensor slices are currently supported")
            if stop <= start:
                raise AllocationError("tensor slices must not be empty")
            shape.append(stop - start)
            offset += start * stride
        return Tensor(self._buffer, tuple(shape), self._dtype, offset=offset, strides=self._strides)

    def access(self: Self, mode: AccessMode) -> BufferAccess:
        """Declare access to the tensor's occupied backing range."""
        return self._buffer.access(mode, offset=self._offset, nbytes=self.nbytes)
