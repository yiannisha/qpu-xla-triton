from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import conv2d_int32


def _reference_conv2d(
    source: np.ndarray,
    weight: np.ndarray,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> np.ndarray:
    """Compute an explicit INT32 NCHW reference for differential tests."""
    batch, channels, height, width = source.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    output_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    padded = np.pad(source, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    output = np.zeros((batch, out_channels, output_height, output_width), dtype=np.int32)
    for n in range(batch):
        for out_channel in range(out_channels):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    total = 0
                    for channel in range(channels):
                        for kernel_y in range(kernel_height):
                            for kernel_x in range(kernel_width):
                                total += int(
                                    padded[
                                        n,
                                        channel,
                                        output_y * stride[0] + kernel_y * dilation[0],
                                        output_x * stride[1] + kernel_x * dilation[1],
                                    ]
                                ) * int(weight[out_channel, channel, kernel_y, kernel_x])
                    output[n, out_channel, output_y, output_x] = total
    return output


def test_conv2d_int32_cpu_reference_handles_padding_stride_and_dilation() -> None:
    rng = np.random.default_rng(6)
    source_value = rng.integers(-4, 4, size=(1, 2, 6, 7), dtype=np.int32)
    weight_value = rng.integers(-4, 4, size=(3, 2, 3, 2), dtype=np.int32)
    expected = _reference_conv2d(source_value, weight_value, stride=(2, 1), padding=(1, 0), dilation=(1, 2))
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        weight = device.tensor(weight_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_int32(destination, source, weight, stride=(2, 1), padding=(1, 0), dilation=(1, 2), queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)


def test_conv2d_int32_pointwise_1x1_avoids_general_im2col_and_matches_reference() -> None:
    source_value = np.arange(2 * 2 * 3 * 5, dtype=np.int32).reshape(2, 2, 3, 5) - 20
    weight_value = np.array([[[[2]], [[-1]]], [[[3]], [[4]]], [[[-2]], [[5]]]], dtype=np.int32)
    expected = _reference_conv2d(source_value, weight_value, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        weight = device.tensor(weight_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_int32(destination, source, weight, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_conv2d_int32_qpu_gemm_path_matches_explicit_reference() -> None:
    rng = np.random.default_rng(7)
    source_value = rng.integers(-8, 8, size=(1, 2, 6, 5), dtype=np.int32)
    weight_value = rng.integers(-8, 8, size=(4, 2, 3, 3), dtype=np.int32)
    expected = _reference_conv2d(source_value, weight_value, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        weight = device.tensor(weight_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_int32(destination, source, weight, stride=1, padding=1, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_conv2d_int32_qpu_pointwise_1x1_path_matches_explicit_reference() -> None:
    source_value = np.arange(2 * 2 * 3 * 5, dtype=np.int32).reshape(2, 2, 3, 5) - 20
    weight_value = np.array([[[[2]], [[-1]]], [[[3]], [[4]]], [[[-2]], [[5]]]], dtype=np.int32)
    expected = _reference_conv2d(source_value, weight_value, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        weight = device.tensor(weight_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        weight.numpy()[:] = weight_value

        conv2d_int32(destination, source, weight, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)
