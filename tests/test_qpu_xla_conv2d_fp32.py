from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import conv2d_fp32


def _reference(source: np.ndarray, weight: np.ndarray, *, padding: int) -> np.ndarray:
    batch, _, height, width = source.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    output = np.zeros((batch, out_channels, height, width), dtype=np.float32)
    padded = np.pad(source, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    for n in range(batch):
        for out_channel in range(out_channels):
            for y in range(height):
                for x in range(width):
                    window = padded[n, :, y : y + kernel_height, x : x + kernel_width]
                    output[n, out_channel, y, x] = np.sum(window * weight[out_channel], dtype=np.float32)
    return output


def _run(device: Device) -> None:
    rng = np.random.default_rng(19)
    source_value = rng.standard_normal((1, 2, 6, 5), dtype=np.float32)
    weight_value = rng.standard_normal((4, 2, 3, 3), dtype=np.float32)
    expected = _reference(source_value, weight_value, padding=1)
    with device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        weight = device.tensor(weight_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_fp32(destination, source, weight, padding=1, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)


def test_conv2d_fp32_cpu_reference_matches_explicit_convolution() -> None:
    with Device.fake() as device:
        _run(device)


def test_conv2d_fp32_pointwise_1x1_path_matches_explicit_convolution() -> None:
    source_value = np.array(
        [[[[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]], [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.5]]]], dtype=np.float32
    )
    weight_value = np.array([[[[0.5]], [[-2.0]]], [[[-1.5]], [[0.25]]]], dtype=np.float32)
    expected = _reference(source_value, weight_value, padding=0)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        weight = device.tensor(weight_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_fp32(destination, source, weight, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_conv2d_fp32_qpu_gemm_path_matches_explicit_convolution() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        _run(device)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_conv2d_fp32_qpu_pointwise_1x1_path_matches_explicit_convolution() -> None:
    source_value = np.array(
        [[[[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]], [[-1.0, 0.25, 2.0], [3.0, -0.5, 1.5]]]], dtype=np.float32
    )
    weight_value = np.array([[[[0.5]], [[-2.0]]], [[[-1.5]], [[0.25]]]], dtype=np.float32)
    expected = _reference(source_value, weight_value, padding=0)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        weight = device.tensor(weight_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_fp32(destination, source, weight, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)
