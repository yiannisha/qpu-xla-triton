from __future__ import annotations

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.video import Nv12Frame, Nv12FrameRing, preprocess_nv12_to_nchw_fp32, run_headless_nv12_preprocessing


def test_nv12_preprocessing_full_range_neutral_chroma_matches_rgb_and_normalization() -> None:
    y_values = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    uv_values = np.full((1, 2), 128, dtype=np.uint8)
    mean = (0.1, 0.2, 0.3)
    std = (0.5, 0.25, 2.0)
    grayscale = y_values.astype(np.float32) / 255.0
    expected = np.stack(
        (
            (grayscale - mean[0]) / std[0],
            (grayscale - mean[1]) / std[1],
            (grayscale - mean[2]) / std[2],
        ),
        axis=0,
    )[None]
    with Device.fake() as device, device.queue() as queue:
        y_plane = device.tensor(y_values.shape, np.uint8)
        uv_plane = device.tensor(uv_values.shape, np.uint8)
        destination = device.tensor((1, 3, 2, 2), np.float32)
        y_plane.numpy()[:] = y_values
        uv_plane.numpy()[:] = uv_values

        preprocess_nv12_to_nchw_fp32(
            destination,
            y_plane,
            uv_plane,
            color_range="full",
            mean=mean,
            std=std,
            queue=queue,
        ).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_nv12_preprocessing_crop_and_resize_preserves_constant_frame() -> None:
    y_values = np.full((4, 4), 100, dtype=np.uint8)
    uv_values = np.array([[90, 180, 90, 180], [90, 180, 90, 180]], dtype=np.uint8)
    with Device.fake() as device, device.queue() as queue:
        y_plane = device.tensor(y_values.shape, np.uint8)
        uv_plane = device.tensor(uv_values.shape, np.uint8)
        destination = device.tensor((1, 3, 3, 5), np.float32)
        y_plane.numpy()[:] = y_values
        uv_plane.numpy()[:] = uv_values

        preprocess_nv12_to_nchw_fp32(destination, y_plane, uv_plane, crop=(2, 0, 2, 4), queue=queue).wait()

        expected_rgb = np.array(
            [
                1.164383 * (100 - 16) + 1.596027 * (180 - 128),
                1.164383 * (100 - 16) - 0.391762 * (90 - 128) - 0.812968 * (180 - 128),
                1.164383 * (100 - 16) + 2.017232 * (90 - 128),
            ],
            dtype=np.float32,
        )
        expected = np.broadcast_to(np.clip(expected_rgb, 0.0, 255.0)[:, None, None] / 255.0, (3, 3, 5))
        np.testing.assert_allclose(destination.numpy()[0], expected, atol=1e-6, rtol=1e-6)


def test_nv12_preprocessing_rejects_invalid_chroma_crop_and_plane_shapes() -> None:
    with Device.fake() as device:
        y_plane = device.tensor((4, 4), np.uint8)
        uv_plane = device.tensor((2, 4), np.uint8)
        destination = device.tensor((1, 3, 2, 2), np.float32)
        with pytest.raises(ValueError, match="offsets must be even"):
            preprocess_nv12_to_nchw_fp32(destination, y_plane, uv_plane, crop=(1, 0, 2, 2))
        wrong_uv = device.tensor((2, 2), np.uint8)
        with pytest.raises(ValueError, match="UV plane"):
            preprocess_nv12_to_nchw_fp32(destination, y_plane, wrong_uv)


def test_nv12_frame_ring_waits_before_reusing_a_slot_and_retains_its_output() -> None:
    with (
        Device.fake() as device,
        device.queue() as queue,
        Nv12FrameRing(device, capacity=2, frame_height=2, frame_width=2, output_height=2, output_width=2) as ring,
    ):
        first = ring.acquire()
        second = ring.acquire()
        first.y_plane.numpy()[:] = 100
        first.uv_plane.numpy()[:] = 128
        second.y_plane.numpy()[:] = 200
        second.uv_plane.numpy()[:] = 128
        first_event = ring.submit(first, queue=queue, color_range="full")
        ring.submit(second, queue=queue, color_range="full").wait()

        reused = ring.acquire()

        assert reused is first
        assert first_event.done
        np.testing.assert_allclose(reused.destination.numpy(), np.full((1, 3, 2, 2), 100 / 255.0), atol=1e-6)


def test_headless_nv12_preprocessing_reports_latency_and_retains_each_frame_output() -> None:
    frames = (
        Nv12Frame(np.full((2, 2), 64, dtype=np.uint8), np.full((1, 2), 128, dtype=np.uint8)),
        Nv12Frame(np.full((2, 2), 192, dtype=np.uint8), np.full((1, 2), 128, dtype=np.uint8)),
    )
    with Device.fake() as device:
        report = run_headless_nv12_preprocessing(device, frames, output_height=2, output_width=2, color_range="full")

    assert report.frame_count == 2
    assert len(report.latency_seconds) == 2
    assert report.p50_seconds > 0
    assert report.p95_seconds >= report.p50_seconds
    np.testing.assert_allclose(report.outputs[0], np.full((1, 3, 2, 2), 64 / 255.0), atol=1e-6)
    np.testing.assert_allclose(report.outputs[1], np.full((1, 3, 2, 2), 192 / 255.0), atol=1e-6)
