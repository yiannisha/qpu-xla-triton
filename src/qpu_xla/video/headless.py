"""Headless NV12 preprocessing runner with deterministic output and latency reporting."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Self

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.video.preprocess import ColorRange
from qpu_xla.video.ring import Nv12FrameRing


@dataclass(frozen=True, slots=True)
class Nv12Frame:
    """One host NV12 frame consumed by the headless preprocessing runner."""

    y_plane: npt.NDArray[np.uint8]
    uv_plane: npt.NDArray[np.uint8]

    def __post_init__(self: Self) -> None:
        """Reject incomplete or incompatible planar NV12 host input."""
        if self.y_plane.ndim != 2 or self.uv_plane.shape != (self.y_plane.shape[0] // 2, self.y_plane.shape[1]):
            raise ValueError("NV12 frame planes must have (height, width) and (height // 2, width) shapes")
        height, width = self.y_plane.shape
        if height <= 0 or width <= 0 or height % 2 or width % 2:
            raise ValueError("NV12 frame dimensions must be positive and even")
        if self.y_plane.dtype != np.dtype(np.uint8) or self.uv_plane.dtype != np.dtype(np.uint8):
            raise ValueError("NV12 frame planes must have uint8 dtype")


@dataclass(frozen=True, slots=True)
class HeadlessPreprocessReport:
    """Model-ready NCHW outputs and complete per-frame end-to-end latency samples."""

    outputs: tuple[npt.NDArray[np.float32], ...]
    latency_seconds: tuple[float, ...]

    @property
    def frame_count(self: Self) -> int:
        """Return the number of completed frames."""
        return len(self.outputs)

    @property
    def p50_seconds(self: Self) -> float:
        """Return the median end-to-end preprocessing latency."""
        return float(np.percentile(self.latency_seconds, 50))

    @property
    def p95_seconds(self: Self) -> float:
        """Return the p95 end-to-end preprocessing latency."""
        return float(np.percentile(self.latency_seconds, 95))


def run_headless_nv12_preprocessing(
    device: Device,
    frames: Iterable[Nv12Frame],
    *,
    output_height: int,
    output_width: int,
    ring_capacity: int = 2,
    mean: Sequence[float] = (0.0, 0.0, 0.0),
    std: Sequence[float] = (1.0, 1.0, 1.0),
    color_range: ColorRange = "limited",
) -> HeadlessPreprocessReport:
    """Run a finite same-sized NV12 stream through the event-safe preprocessing ring."""
    input_frames = tuple(frames)
    if not input_frames:
        raise ValueError("headless preprocessing requires at least one frame")
    first = input_frames[0]
    if any(
        frame.y_plane.shape != first.y_plane.shape or frame.uv_plane.shape != first.uv_plane.shape
        for frame in input_frames
    ):
        raise ValueError("headless preprocessing requires frames with identical NV12 shapes")
    height, width = first.y_plane.shape
    outputs: list[npt.NDArray[np.float32]] = []
    latencies: list[float] = []
    with (
        device.queue() as queue,
        Nv12FrameRing(
            device,
            capacity=ring_capacity,
            frame_height=height,
            frame_width=width,
            output_height=output_height,
            output_width=output_width,
        ) as ring,
    ):
        for frame in input_frames:
            slot = ring.acquire()
            slot.y_plane.numpy()[:] = frame.y_plane
            slot.uv_plane.numpy()[:] = frame.uv_plane
            started = perf_counter()
            ring.submit(slot, queue=queue, mean=mean, std=std, color_range=color_range).wait()
            latencies.append(perf_counter() - started)
            outputs.append(np.array(slot.destination.numpy(), copy=True))
    return HeadlessPreprocessReport(tuple(outputs), tuple(latencies))
