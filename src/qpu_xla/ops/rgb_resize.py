"""Persistent RGB bilinear letterbox/normalization plan."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.rgb_resize import RGB_RESIZE_NORM_FP32_KERNEL, supports_rgb_resize_norm_fp32
from qpu_xla.kernels.swiglu import _workgroup_count
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def _bilinear_metadata(
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
    ratio = max(source_width / target_width, source_height / target_height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    output_y, output_x = np.indices((target_height, target_width), dtype=np.int64)
    active = (output_y >= pad_height) & (output_x >= pad_width)
    local_y = output_y - pad_height
    local_x = output_x - pad_width
    source_y = (local_y.astype(np.float32) + np.float32(0.5)) * (
        np.float32(source_height) / np.float32(resized_height)
    ) - np.float32(0.5)
    source_x = (local_x.astype(np.float32) + np.float32(0.5)) * (
        np.float32(source_width) / np.float32(resized_width)
    ) - np.float32(0.5)
    y0_unclipped, x0_unclipped = np.floor(source_y).astype(np.int64), np.floor(source_x).astype(np.int64)
    y1_unclipped, x1_unclipped = y0_unclipped + 1, x0_unclipped + 1
    wy, wx = source_y - y0_unclipped, source_x - x0_unclipped
    y0 = np.clip(y0_unclipped, 0, source_height - 1)
    y1 = np.clip(y1_unclipped, 0, source_height - 1)
    x0 = np.clip(x0_unclipped, 0, source_width - 1)
    x1 = np.clip(x1_unclipped, 0, source_width - 1)
    spatial_offsets = (
        (y0 * source_width + x0) * 3,
        (y0 * source_width + x1) * 3,
        (y1 * source_width + x0) * 3,
        (y1 * source_width + x1) * 3,
    )
    spatial_weights = (
        (np.float32(1.0) - wy) * (np.float32(1.0) - wx),
        (np.float32(1.0) - wy) * wx,
        wy * (np.float32(1.0) - wx),
        wy * wx,
    )
    output_count = 3 * target_height * target_width
    offsets = np.empty((4, output_count), dtype=np.uint32)
    weights = np.empty((4, output_count), dtype=np.float32)
    for tap, (spatial_offset, spatial_weight) in enumerate(zip(spatial_offsets, spatial_weights, strict=True)):
        offsets[tap] = np.concatenate(tuple(((spatial_offset + channel) * 4).ravel() for channel in range(3))).astype(
            np.uint32
        )
        masked_weight = np.where(active, spatial_weight, np.float32(0.0)).astype(np.float32)
        weights[tap] = np.tile(masked_weight.ravel(), 3)
    return offsets, weights


class PreparedRGBResizeNormFP32:
    """Own source/destination buffers and reusable bilinear metadata.

    The input boundary remains ``uint8[H,W,3]``.  Upload converts bytes to a
    shared FP32 HWC tensor; the expensive bilinear resize, top/left padding,
    CHW permutation, and ``[0,1] -> [-1,1]`` normalization run on the selected
    CPU/QPU partitions.
    """

    def __init__(
        self: Self,
        source: Tensor,
        destination: Tensor,
    ) -> None:
        """Precompute and upload exact bilinear sampling metadata."""
        if source.buffer.device is not destination.buffer.device:
            raise DependencyError("RGB resize tensors must belong to one device")
        if (
            source.dtype != np.dtype(np.float32)
            or len(source.shape) != 3
            or source.shape[2] != 3
            or destination.dtype != np.dtype(np.float32)
            or len(destination.shape) != 3
            or destination.shape[0] != 3
        ):
            raise ValueError("RGB resize requires FP32 HWC source and FP32 CHW destination")
        self.device = source.buffer.device
        self.source = source
        self.destination = destination
        self.source_height, self.source_width = source.shape[:2]
        _, self.target_height, self.target_width = destination.shape
        offsets, weights = _bilinear_metadata(
            self.source_height,
            self.source_width,
            self.target_height,
            self.target_width,
        )
        self._offsets = offsets
        self._weights = weights
        self.metadata = self.device.tensor((8, offsets.shape[1]), np.uint32)
        self.metadata.numpy()[:4] = offsets
        self.metadata.numpy()[4:] = weights.view(np.uint32)
        self._last_event: Event | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter the prepared resize-plan lifetime."""
        if self._closed:
            raise RuntimeError("prepared RGB resize is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the resize metadata."""
        self.close()

    def close(self: Self) -> None:
        """Release prepared metadata after outstanding work completes."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.wait()
        self.metadata.buffer.close()
        self._closed = True

    def upload(
        self: Self,
        image: npt.NDArray[np.uint8],
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Queue conversion from the public uint8 HWC boundary to FP32 HWC."""
        if self._closed:
            raise RuntimeError("prepared RGB resize is closed")
        if image.dtype != np.dtype(np.uint8) or image.shape != self.source.shape or not image.flags.c_contiguous:
            raise ValueError("RGB upload image does not match the prepared uint8 HWC boundary")
        if queue.device is not self.device:
            raise DependencyError("RGB upload queue belongs to a different device")
        return queue.host_task(
            lambda: np.multiply(image, np.float32(1.0 / 255.0), out=self.source.numpy(), casting="unsafe"),
            wait_for=wait_for,
            buffers=(self.source.access(AccessMode.WRITE),),
            name="rgb_resize_fp32.upload_uint8",
        )

    def execute(
        self: Self,
        *,
        queue: Queue,
        cpu_queue: Queue | None = None,
        placement: Placement = Placement.QPU,
        qpu_rows: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Resize with CPU, QPU, or simultaneous disjoint output-row work."""
        if self._closed:
            raise RuntimeError("prepared RGB resize is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared RGB resize already has an in-flight invocation")
        if queue.device is not self.device or (cpu_queue is not None and cpu_queue.device is not self.device):
            raise DependencyError("RGB resize queues must belong to its device")
        flat_destination = self.destination.buffer.tensor(
            (3 * self.target_height, self.target_width),
            np.float32,
            offset=self.destination.address - self.destination.buffer.address,
        )

        def cpu_range(start_row: int, stop_row: int) -> None:
            start = start_row * self.target_width
            stop = stop_row * self.target_width
            source = self.source.numpy().reshape(-1)
            output = np.zeros((stop - start,), dtype=np.float32)
            for tap in range(4):
                output += source[self._offsets[tap, start:stop] // np.uint32(4)] * self._weights[tap, start:stop]
            output *= np.float32(2.0)
            output -= np.float32(1.0)
            flat_destination.numpy()[start_row:stop_row] = output.reshape(stop_row - start_row, self.target_width)

        supported = supports_rgb_resize_norm_fp32(self.source, self.metadata, self.destination, self.device.backend)
        if placement is Placement.QPU and not supported:
            raise ValueError("RGB resize QPU placement requires a 16-aligned contiguous output")
        if placement is Placement.HYBRID:
            if cpu_queue is None or cpu_queue is queue:
                raise DependencyError("hybrid RGB resize requires distinct CPU and QPU queues")
            rows = flat_destination.shape[0]
            if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
                raise ValueError("hybrid RGB resize qpu_rows must leave two non-empty output partitions")
            qpu_destination = flat_destination.slice((slice(0, qpu_rows), slice(None)))
            if not supports_rgb_resize_norm_fp32(self.source, self.metadata, qpu_destination, self.device.backend):
                raise ValueError("RGB resize QPU partition does not satisfy vector alignment")
            vectors = qpu_destination.nbytes // 64
            qpu_event = queue.submit(
                RGB_RESIZE_NORM_FP32_KERNEL,
                (self.source, self.metadata, qpu_destination),
                grid=(_workgroup_count(vectors), 1, 1),
                wait_for=wait_for,
                buffers=(
                    self.source.access(AccessMode.READ),
                    self.metadata.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
            )
            cpu_destination = flat_destination.slice((slice(qpu_rows, rows), slice(None)))
            cpu_event = cpu_queue.host_task(
                lambda: cpu_range(qpu_rows, rows),
                wait_for=wait_for,
                buffers=(
                    self.source.access(AccessMode.READ),
                    cpu_destination.access(AccessMode.WRITE),
                ),
                name="rgb_resize_fp32.hybrid_cpu_tail",
            )
            event = cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="rgb_resize_fp32.hybrid_join",
            )
        elif placement is Placement.QPU:
            vectors = self.destination.nbytes // 64
            event = queue.submit(
                RGB_RESIZE_NORM_FP32_KERNEL,
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
                lambda: cpu_range(0, flat_destination.shape[0]),
                wait_for=wait_for,
                buffers=(self.source.access(AccessMode.READ), self.destination.access(AccessMode.WRITE)),
                name="rgb_resize_fp32.cpu",
            )
        self._last_event = event
        return event


__all__ = ["PreparedRGBResizeNormFP32"]
