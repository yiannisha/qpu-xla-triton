"""Event-safe fixed-capacity NV12 frame ring for capture/preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from qpu_xla.device import Device
from qpu_xla.errors import DeviceClosedError
from qpu_xla.memory import Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.video.preprocess import ColorRange, preprocess_nv12_to_nchw_fp32


@dataclass(slots=True)
class Nv12FrameSlot:
    """One reusable capture input and model-ready output allocation group."""

    index: int
    y_plane: Tensor
    uv_plane: Tensor
    destination: Tensor
    completion: Event | None = None


class Nv12FrameRing:
    """Serialize reuse of two or more NV12 frame slots through completion events."""

    def __init__(
        self: Self,
        device: Device,
        *,
        capacity: int,
        frame_height: int,
        frame_width: int,
        output_height: int,
        output_width: int,
    ) -> None:
        """Allocate fixed NV12 and NCHW tensors owned by the caller's device."""
        if capacity < 2:
            raise ValueError("NV12 frame rings require at least two slots")
        if min(frame_height, frame_width, output_height, output_width) <= 0 or frame_height % 2 or frame_width % 2:
            raise ValueError("NV12 frame dimensions must be positive and even")
        self._device = device
        self._slots = [
            Nv12FrameSlot(
                index,
                device.tensor((frame_height, frame_width), np.uint8),
                device.tensor((frame_height // 2, frame_width), np.uint8),
                device.tensor((1, 3, output_height, output_width), np.float32),
            )
            for index in range(capacity)
        ]
        self._next = 0
        self._closed = False

    @property
    def slots(self: Self) -> tuple[Nv12FrameSlot, ...]:
        """Expose immutable slot ordering while retaining ring ownership."""
        self._require_open()
        return tuple(self._slots)

    def _require_open(self: Self) -> None:
        """Reject capture/projection work after the ring releases its buffers."""
        if self._closed:
            raise DeviceClosedError("NV12 frame ring is closed")

    def acquire(self: Self, *, timeout: float | None = None) -> Nv12FrameSlot:
        """Return the next slot only after its prior preprocessing event finishes."""
        self._require_open()
        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        if slot.completion is not None:
            slot.completion.wait(timeout)
        return slot

    def submit(
        self: Self,
        slot: Nv12FrameSlot,
        *,
        queue: Queue,
        crop: tuple[int, int, int, int] | None = None,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        std: Sequence[float] = (1.0, 1.0, 1.0),
        color_range: ColorRange = "limited",
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Preprocess one acquired frame and record its completion before reuse."""
        self._require_open()
        if not any(slot is owned_slot for owned_slot in self._slots):
            raise ValueError("frame slot does not belong to this ring")
        if queue.device is not self._device:
            raise ValueError("frame ring queue must belong to the ring device")
        dependencies = tuple(wait_for)
        if slot.completion is not None and slot.completion not in dependencies:
            dependencies += (slot.completion,)
        event = preprocess_nv12_to_nchw_fp32(
            slot.destination,
            slot.y_plane,
            slot.uv_plane,
            crop=crop,
            mean=mean,
            std=std,
            color_range=color_range,
            queue=queue,
            wait_for=dependencies,
        )
        slot.completion = event
        return event

    def close(self: Self) -> None:
        """Close every owned allocation without closing the shared device."""
        if self._closed:
            return
        for slot in self._slots:
            slot.y_plane.buffer.close()
            slot.uv_plane.buffer.close()
            slot.destination.buffer.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter a context-managed ring lifetime."""
        self._require_open()
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release all ring allocations at context exit."""
        self.close()
