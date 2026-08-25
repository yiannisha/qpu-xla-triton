"""Prepared FP32 tensor permutations backed by a QPU address table."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.gather import INDEXED_GATHER_FP32_KERNEL, supports_indexed_gather_fp32
from qpu_xla.kernels.swiglu import _workgroup_count
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


class PreparedIndexedGatherFP32:
    """Cache a source-index/address table for a fixed FP32 permutation."""

    def __init__(
        self: Self,
        source: Tensor,
        destination: Tensor,
        source_indices: npt.NDArray[np.integer],
    ) -> None:
        """Validate tensors and upload the immutable gather address table."""
        if source.buffer.device is not destination.buffer.device:
            raise DependencyError("prepared gather tensors must belong to one device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("prepared gather requires FP32 source and destination")
        if source_indices.shape != destination.shape or not np.issubdtype(source_indices.dtype, np.integer):
            raise ValueError("prepared gather needs one integer source index per destination value")
        indices = np.ascontiguousarray(source_indices, dtype=np.uint32)
        if np.any(source_indices < 0) or np.any(source_indices >= int(np.prod(source.shape))):
            raise ValueError("prepared gather source index is out of range")
        self.device = source.buffer.device
        self.source = source
        self.destination = destination
        self._indices = indices
        self.metadata = self.device.tensor((indices.size,), np.uint32)
        self.metadata.numpy()[:] = np.uint32(source.address) + indices.ravel() * np.uint32(4)
        self._last_event: Event | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter the prepared plan lifetime."""
        if self._closed:
            raise RuntimeError("prepared gather is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release prepared metadata."""
        self.close()

    def close(self: Self) -> None:
        """Wait for pending work and release the QPU address table."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.wait()
        self.metadata.buffer.close()
        self._closed = True

    def execute(
        self: Self,
        *,
        queue: Queue,
        cpu_queue: Queue | None = None,
        placement: Placement = Placement.QPU,
        qpu_rows: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Execute CPU, QPU, or concurrent row-partitioned gather."""
        if self._closed:
            raise RuntimeError("prepared gather is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared gather already has an in-flight invocation")
        if queue.device is not self.device or (cpu_queue is not None and cpu_queue.device is not self.device):
            raise DependencyError("prepared gather queues must belong to its device")
        supported = supports_indexed_gather_fp32(self.source, self.metadata, self.destination, self.device.backend)
        if placement is Placement.QPU and not supported:
            raise ValueError("prepared QPU gather requires a 16-aligned contiguous destination")

        def cpu_all() -> None:
            self.destination.numpy()[:] = self.source.numpy().reshape(-1)[self._indices]

        if placement is Placement.HYBRID:
            if cpu_queue is None or cpu_queue is queue:
                raise DependencyError("hybrid gather requires distinct CPU and QPU queues")
            if len(self.destination.shape) != 2:
                raise ValueError("hybrid gather partitions rank-2 output rows")
            rows, width = self.destination.shape
            if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
                raise ValueError("hybrid gather qpu_rows must leave two non-empty partitions")
            qpu_destination = self.destination.slice((slice(0, qpu_rows), slice(None)))
            qpu_metadata = self.metadata.slice((slice(0, qpu_rows * width),))
            if not supports_indexed_gather_fp32(self.source, qpu_metadata, qpu_destination, self.device.backend):
                raise ValueError("gather QPU row partition does not satisfy vector alignment")
            vectors = qpu_destination.nbytes // 64
            qpu_event = queue.submit(
                INDEXED_GATHER_FP32_KERNEL,
                (self.source, qpu_metadata, qpu_destination),
                grid=(_workgroup_count(vectors), 1, 1),
                wait_for=wait_for,
                buffers=(
                    self.source.access(AccessMode.READ),
                    qpu_metadata.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
            )
            cpu_destination = self.destination.slice((slice(qpu_rows, rows), slice(None)))
            cpu_indices = self._indices[qpu_rows:]
            cpu_event = cpu_queue.host_task(
                lambda: cpu_destination.numpy().__setitem__(slice(None), self.source.numpy().reshape(-1)[cpu_indices]),
                wait_for=wait_for,
                buffers=(
                    self.source.access(AccessMode.READ),
                    cpu_destination.access(AccessMode.WRITE),
                ),
                name="indexed_gather_fp32.hybrid_cpu_tail",
            )
            event = cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="indexed_gather_fp32.hybrid_join",
            )
        elif placement is Placement.QPU:
            vectors = self.destination.nbytes // 64
            event = queue.submit(
                INDEXED_GATHER_FP32_KERNEL,
                (self.source, self.metadata, self.destination),
                grid=(_workgroup_count(vectors), 1, 1),
                wait_for=wait_for,
                buffers=(
                    self.source.access(AccessMode.READ),
                    self.metadata.access(AccessMode.READ),
                    self.destination.access(AccessMode.WRITE),
                ),
            )
        else:
            event = queue.host_task(
                cpu_all,
                wait_for=wait_for,
                buffers=(self.source.access(AccessMode.READ), self.destination.access(AccessMode.WRITE)),
                name="indexed_gather_fp32.cpu",
            )
        self._last_event = event
        return event


def patchify_indices(channels: int, height: int, width: int, patch_size: int) -> npt.NDArray[np.uint32]:
    """Return Conv2D-compatible non-overlapping CHW patch indices."""
    if min(channels, height, width, patch_size) <= 0 or height % patch_size or width % patch_size:
        raise ValueError("patchify geometry must be positive and divisible by patch_size")
    indexes = np.arange(channels * height * width, dtype=np.uint32).reshape(channels, height, width)
    return np.ascontiguousarray(
        indexes.reshape(
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .transpose(1, 3, 0, 2, 4)
        .reshape((height // patch_size) * (width // patch_size), channels * patch_size**2)
    )


def pixel_shuffle_indices(
    token_count: int,
    hidden_size: int,
    scale_factor: int,
) -> npt.NDArray[np.uint32]:
    """Return indices matching ``SmolVLMConnector.pixel_shuffle`` exactly."""
    side = int(np.sqrt(token_count))
    if side * side != token_count or scale_factor <= 0 or side % scale_factor:
        raise ValueError("pixel shuffle requires a square token grid divisible by scale_factor")
    indexes = np.arange(token_count * hidden_size, dtype=np.uint32).reshape(side, side, hidden_size)
    indexes = indexes.reshape(side, side // scale_factor, hidden_size * scale_factor)
    indexes = indexes.transpose(1, 0, 2)
    indexes = indexes.reshape(
        side // scale_factor,
        side // scale_factor,
        hidden_size * scale_factor**2,
    )
    indexes = indexes.transpose(1, 0, 2)
    return np.ascontiguousarray(indexes.reshape(token_count // scale_factor**2, -1))


class PreparedPatchifyFP32(PreparedIndexedGatherFP32):
    """Fixed patch-embedding input gather."""

    def __init__(self: Self, source: Tensor, destination: Tensor, *, patch_size: int) -> None:
        """Prepare the exact CHW-to-patch permutation."""
        if len(source.shape) != 3 or len(destination.shape) != 2:
            raise ValueError("patchify requires CHW source and a rank-2 destination")
        indexes = patchify_indices(*source.shape, patch_size)
        if destination.shape != indexes.shape:
            raise ValueError("patchify destination shape does not match the patch grid")
        super().__init__(source, destination, indexes)


class PreparedPixelShuffleFP32(PreparedIndexedGatherFP32):
    """Fixed SmolVLM token pixel-shuffle gather."""

    def __init__(self: Self, source: Tensor, destination: Tensor, *, scale_factor: int) -> None:
        """Prepare the exact SmolVLM connector permutation."""
        if len(source.shape) != 2 or len(destination.shape) != 2:
            raise ValueError("pixel shuffle requires rank-2 source and destination")
        indexes = pixel_shuffle_indices(source.shape[0], source.shape[1], scale_factor)
        if destination.shape != indexes.shape:
            raise ValueError("pixel-shuffle destination shape does not match the requested factor")
        super().__init__(source, destination, indexes)


__all__ = [
    "PreparedIndexedGatherFP32",
    "PreparedPatchifyFP32",
    "PreparedPixelShuffleFP32",
    "patchify_indices",
    "pixel_shuffle_indices",
]
