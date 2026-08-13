"""CPU reference implementation for an NV12 camera preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue

ColorRange = Literal["full", "limited"]


def _validate_channel_values(values: Sequence[float], name: str, *, positive: bool = False) -> npt.NDArray[np.float32]:
    """Validate the three per-channel normalization values."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} values must be positive")
    return array


def _validate_crop(crop: tuple[int, int, int, int] | None, height: int, width: int) -> tuple[int, int, int, int]:
    """Return a chroma-aligned crop inside an NV12 frame."""
    if crop is None:
        return 0, 0, width, height
    x, y, crop_width, crop_height = crop
    if any(not isinstance(value, int) for value in crop):
        raise ValueError("crop coordinates and dimensions must be integers")
    if x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0 or x + crop_width > width or y + crop_height > height:
        raise ValueError("crop must lie inside the source frame")
    if x % 2 or y % 2:
        raise ValueError("NV12 crop x and y offsets must be even for chroma alignment")
    return x, y, crop_width, crop_height


def _bilinear_coordinates(
    source_length: int,
    destination_length: int,
    offset: int,
    crop_length: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float32]]:
    """Compute half-pixel bilinear coordinates clipped to one source axis."""
    positions = (
        (np.arange(destination_length, dtype=np.float32) + np.float32(0.5))
        * (np.float32(crop_length) / np.float32(destination_length))
        + np.float32(offset)
        - np.float32(0.5)
    )
    lower = np.floor(positions).astype(np.intp)
    upper = lower + 1
    fraction = cast(npt.NDArray[np.float32], positions - lower.astype(np.float32))
    if offset < 0 or offset + crop_length > source_length:
        raise ValueError("crop lies outside the source axis")
    np.clip(lower, offset, offset + crop_length - 1, out=lower)
    np.clip(upper, offset, offset + crop_length - 1, out=upper)
    return lower, upper, fraction


def preprocess_nv12_to_nchw_fp32(
    destination: Tensor,
    y_plane: Tensor,
    uv_plane: Tensor,
    *,
    crop: tuple[int, int, int, int] | None = None,
    mean: Sequence[float] = (0.0, 0.0, 0.0),
    std: Sequence[float] = (1.0, 1.0, 1.0),
    color_range: ColorRange = "limited",
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Convert one NV12 frame to normalized FP32 NCHW with bilinear resize.

    ``y_plane`` has shape ``(height, width)`` and ``uv_plane`` has interleaved
    Cb/Cr bytes with shape ``(height // 2, width)``. ``destination`` is a
    batch-one ``(1, 3, output_height, output_width)`` tensor. Conversion uses
    BT.601 coefficients and either the standard video-range or full-range
    convention. The current implementation is an explicit CPU queue task,
    preserving the synchronization contract for a future QPU specialization.
    """
    tensors = (destination, y_plane, uv_plane)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("video preprocessing tensors must belong to the same device")
    if y_plane.dtype != np.dtype(np.uint8) or uv_plane.dtype != np.dtype(np.uint8):
        raise ValueError("NV12 planes must have uint8 dtype")
    if destination.dtype != np.dtype(np.float32):
        raise ValueError("video preprocessing destination must have float32 dtype")
    if len(y_plane.shape) != 2 or len(uv_plane.shape) != 2:
        raise ValueError("NV12 planes must be rank-2 tensors")
    height, width = y_plane.shape
    if height < 2 or width < 2 or height % 2 or width % 2 or uv_plane.shape != (height // 2, width):
        raise ValueError("NV12 requires even source dimensions and a (height // 2, width) UV plane")
    if len(destination.shape) != 4 or destination.shape[0:2] != (1, 3):
        raise ValueError("destination must have batch-one NCHW shape (1, 3, height, width)")
    output_height, output_width = destination.shape[2:]
    if output_height <= 0 or output_width <= 0:
        raise ValueError("destination spatial dimensions must be positive")
    crop_x, crop_y, crop_width, crop_height = _validate_crop(crop, height, width)
    mean_values = _validate_channel_values(mean, "mean")
    std_values = _validate_channel_values(std, "std", positive=True)
    if color_range not in ("full", "limited"):
        raise ValueError("color_range must be 'full' or 'limited'")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and video preprocessing tensors must belong to the same device")

    def convert() -> None:
        y_values = cast(npt.NDArray[np.uint8], y_plane.numpy()).astype(np.float32)
        uv_values = cast(npt.NDArray[np.uint8], uv_plane.numpy()).astype(np.float32)
        y0, y1, fy = _bilinear_coordinates(height, output_height, crop_y, crop_height)
        x0, x1, fx = _bilinear_coordinates(width, output_width, crop_x, crop_width)
        top = (
            y_values[y0[:, None], x0[None, :]] * (np.float32(1.0) - fx)[None, :]
            + y_values[y0[:, None], x1[None, :]] * fx[None, :]
        )
        bottom = (
            y_values[y1[:, None], x0[None, :]] * (np.float32(1.0) - fx)[None, :]
            + y_values[y1[:, None], x1[None, :]] * fx[None, :]
        )
        luma = top * (np.float32(1.0) - fy)[:, None] + bottom * fy[:, None]
        chroma_y = np.minimum((y0 + y1) // 4, height // 2 - 1)
        chroma_x = (x0 // 2) * 2
        cb = uv_values[chroma_y[:, None], chroma_x[None, :]] - np.float32(128.0)
        cr = uv_values[chroma_y[:, None], chroma_x[None, :] + 1] - np.float32(128.0)
        if color_range == "limited":
            scaled_luma = np.maximum(luma - np.float32(16.0), np.float32(0.0)) * np.float32(1.164383)
            red = scaled_luma + np.float32(1.596027) * cr
            green = scaled_luma - np.float32(0.391762) * cb - np.float32(0.812968) * cr
            blue = scaled_luma + np.float32(2.017232) * cb
        else:
            red = luma + np.float32(1.402) * cr
            green = luma - np.float32(0.344136) * cb - np.float32(0.714136) * cr
            blue = luma + np.float32(1.772) * cb
        channels = np.stack((red, green, blue), axis=0)
        np.clip(channels, 0.0, 255.0, out=channels)
        normalized = (channels / np.float32(255.0) - mean_values[:, None, None]) / std_values[:, None, None]
        destination.numpy()[0] = normalized

    event = selected_queue.host_task(
        convert,
        wait_for=wait_for,
        buffers=(
            y_plane.access(AccessMode.READ),
            uv_plane.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="video.preprocess_nv12_to_nchw_fp32.cpu",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
