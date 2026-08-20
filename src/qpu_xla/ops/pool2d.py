"""Exact INT32 NCHW 2x2/stride-2 max and average pooling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.pool2d import (
    AVGPOOL2D_FP32_KERNEL,
    AVGPOOL2D_INT32_KERNEL,
    MAXPOOL2D_FP32_KERNEL,
    MAXPOOL2D_INT32_KERNEL,
    PoolMode,
    supports_pool2d_fp32,
    supports_pool2d_int32,
)
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue


def _trunc_div4(values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Divide signed values by four with truncation toward zero."""
    return np.where(values < 0, -((-values) // 4), values // 4)


def _fill_pool2d_metadata(metadata: Tensor, source: Tensor, destination: Tensor) -> None:
    """Populate source addresses for one fixed 2x2/stride-2 tensor pair."""
    batch, channels, height, width = source.shape
    output_height, output_width = destination.shape[2:]
    indexes = np.arange(metadata.shape[0], dtype=np.uint64).reshape(destination.shape)
    plane, channel_plane = output_height * output_width, height * width
    batch_index = indexes // (channels * plane)
    channel_index = (indexes // plane) % channels
    spatial_index = indexes % plane
    input_index = (
        batch_index * channels * channel_plane
        + channel_index * channel_plane
        + (spatial_index // output_width * 2) * width
        + (spatial_index % output_width * 2)
    )
    metadata.numpy()[:] = np.uint32(source.address) + (input_index * source.dtype.itemsize).astype(np.uint32).ravel()


class PreparedPool2DFP32:
    """Cache FP32 pooling metadata for repeated execution on fixed tensors."""

    def __init__(self: Self, source: Tensor, destination: Tensor) -> None:
        """Create a plan for one fixed source/destination tensor pair."""
        if source.buffer.device is not destination.buffer.device:
            raise DependencyError("prepared pooling tensors must belong to the same device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("prepared FP32 pooling requires float32 tensors")
        if len(source.shape) != 4 or len(destination.shape) != 4:
            raise ValueError("prepared FP32 pooling requires NCHW rank-4 tensors")
        batch, channels, height, width = source.shape
        if height % 2 or width % 2 or destination.shape != (batch, channels, height // 2, width // 2):
            raise ValueError("prepared FP32 pooling requires aligned 2x2/stride-2 geometry")
        self.device = source.buffer.device
        self.source = source
        self.destination = destination
        self.metadata = self.device.tensor((int(np.prod(destination.shape)),), np.uint32)
        _fill_pool2d_metadata(self.metadata, source, destination)
        if not supports_pool2d_fp32(source, destination, self.metadata, self.device.backend):
            self.metadata.buffer.close()
            raise ValueError("prepared FP32 pooling requires a 16-aligned hardware-supported output")
        self._queue: Queue | None = None
        self._last_event: Event | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Return this open plan as a context manager."""
        if self._closed:
            raise RuntimeError("prepared FP32 pooling plan is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Wait for pending work and release cached metadata."""
        self.close()

    def close(self: Self) -> None:
        """Release cached metadata after the last submitted invocation."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.wait()
        self.metadata.buffer.close()
        self._closed = True

    def execute(
        self: Self,
        *,
        mode: PoolMode,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Submit one max/average pool using cached source-address metadata."""
        if mode not in ("max", "avg"):
            raise ValueError("pool mode must be 'max' or 'avg'")
        if self._closed:
            raise RuntimeError("prepared FP32 pooling plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared FP32 pooling plan already has an in-flight invocation")
        if self._queue is None:
            self._queue = queue
        elif self._queue is not queue:
            raise DependencyError("prepared FP32 pooling plan is bound to its first queue")
        if queue.device is not self.device:
            raise DependencyError("prepared FP32 pooling queue belongs to a different device")
        kernel = MAXPOOL2D_FP32_KERNEL if mode == "max" else AVGPOOL2D_FP32_KERNEL
        event = queue.submit(
            kernel,
            (self.source, self.destination, self.metadata),
            wait_for=wait_for,
            buffers=(
                self.source.access(AccessMode.READ),
                self.metadata.access(AccessMode.READ),
                self.destination.access(AccessMode.WRITE),
            ),
        )
        self._last_event = event
        return event


def pool2d_int32(
    destination: Tensor,
    source: Tensor,
    *,
    mode: PoolMode,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Pool NCHW INT32 input using a 2x2 window and stride two."""
    if mode not in ("max", "avg"):
        raise ValueError("pool mode must be 'max' or 'avg'")
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("pooling tensors must belong to the same device")
    if source.dtype != np.dtype(np.int32) or destination.dtype != np.dtype(np.int32):
        raise ValueError("pool2d_int32 requires int32 source and destination tensors")
    if len(source.shape) != 4 or len(destination.shape) != 4:
        raise ValueError("pool2d_int32 requires NCHW rank-4 source and destination tensors")
    batch, channels, height, width = source.shape
    if height % 2 or width % 2:
        raise ValueError("pool2d_int32 requires even input height and width")
    if destination.shape != (batch, channels, height // 2, width // 2):
        raise ValueError("pooling destination shape does not align")
    if mode == "avg":
        maximum = int(np.max(np.abs(source.numpy().astype(np.int64, copy=False)), initial=0))
        if 4 * maximum > np.iinfo(np.int32).max:
            raise ValueError("average pooling sum can overflow signed int32")

    device = destination.buffer.device
    metadata = device.tensor((int(np.prod(destination.shape)),), np.uint32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def cpu_reference() -> None:
        values = cast(npt.NDArray[np.int32], source.numpy())
        x00, x01 = values[:, :, 0::2, 0::2], values[:, :, 0::2, 1::2]
        x10, x11 = values[:, :, 1::2, 0::2], values[:, :, 1::2, 1::2]
        output = destination.numpy()
        if mode == "max":
            np.maximum(np.maximum(x00, x01), np.maximum(x10, x11), out=output)
        else:
            summed = x00.astype(np.int64) + x01.astype(np.int64) + x10.astype(np.int64) + x11.astype(np.int64)
            output[:] = _trunc_div4(summed).astype(np.int32)

    def prepare_metadata() -> None:
        _fill_pool2d_metadata(metadata, source, destination)

    if not supports_pool2d_int32(source, destination, metadata, device.backend):
        event = selected_queue.host_task(
            cpu_reference,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name=f"pool2d_int32.{mode}.cpu",
        )
    else:
        metadata_event = selected_queue.host_task(
            prepare_metadata,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), metadata.access(AccessMode.WRITE)),
            name="pool2d_int32.prepare_metadata",
        )
        kernel = MAXPOOL2D_INT32_KERNEL if mode == "max" else AVGPOOL2D_INT32_KERNEL
        event = selected_queue.submit(
            kernel,
            (source, destination, metadata),
            wait_for=(metadata_event,),
            buffers=(
                source.access(AccessMode.READ),
                metadata.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def pool2d_fp32(
    destination: Tensor,
    source: Tensor,
    *,
    mode: PoolMode,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Pool FP32 NCHW input using a 2x2 window and stride two."""
    if mode not in ("max", "avg"):
        raise ValueError("pool mode must be 'max' or 'avg'")
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("pooling tensors must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("pool2d_fp32 requires float32 source and destination tensors")
    if len(source.shape) != 4 or len(destination.shape) != 4:
        raise ValueError("pool2d_fp32 requires NCHW rank-4 source and destination tensors")
    batch, channels, height, width = source.shape
    if height % 2 or width % 2:
        raise ValueError("pool2d_fp32 requires even input height and width")
    if destination.shape != (batch, channels, height // 2, width // 2):
        raise ValueError("pooling destination shape does not align")

    device = destination.buffer.device
    metadata = device.tensor((int(np.prod(destination.shape)),), np.uint32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def cpu_reference() -> None:
        values = cast(npt.NDArray[np.float32], source.numpy())
        x00, x01 = values[:, :, 0::2, 0::2], values[:, :, 0::2, 1::2]
        x10, x11 = values[:, :, 1::2, 0::2], values[:, :, 1::2, 1::2]
        output = destination.numpy()
        if mode == "max":
            np.maximum(np.maximum(x00, x01), np.maximum(x10, x11), out=output)
        else:
            output[:] = ((x00 + x01) + (x10 + x11)) * np.float32(0.25)

    def prepare_metadata() -> None:
        _fill_pool2d_metadata(metadata, source, destination)

    if not supports_pool2d_fp32(source, destination, metadata, device.backend):
        event = selected_queue.host_task(
            cpu_reference,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name=f"pool2d_fp32.{mode}.cpu",
        )
    else:
        metadata_event = selected_queue.host_task(
            prepare_metadata,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), metadata.access(AccessMode.WRITE)),
            name="pool2d_fp32.prepare_metadata",
        )
        kernel = MAXPOOL2D_FP32_KERNEL if mode == "max" else AVGPOOL2D_FP32_KERNEL
        event = selected_queue.submit(
            kernel,
            (source, destination, metadata),
            wait_for=(metadata_event,),
            buffers=(
                source.access(AccessMode.READ),
                metadata.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
