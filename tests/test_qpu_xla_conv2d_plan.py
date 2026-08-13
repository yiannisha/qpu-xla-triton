from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import Conv2dInt32Plan


def _reference(source: np.ndarray, weight: np.ndarray) -> np.ndarray:
    batch, _, height, width = source.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    output = np.empty((batch, out_channels, height - kernel_height + 1, width - kernel_width + 1), dtype=np.int32)
    for n in range(batch):
        for out_channel in range(out_channels):
            for y in range(output.shape[2]):
                for x in range(output.shape[3]):
                    output[n, out_channel, y, x] = np.sum(
                        source[n, :, y : y + kernel_height, x : x + kernel_width].astype(np.int64)
                        * weight[out_channel].astype(np.int64),
                        dtype=np.int64,
                    )
    return output


def _tensor(device: Device, values: np.ndarray):
    tensor = device.tensor(values.shape, np.int32)
    tensor.numpy()[:] = values
    return tensor


def _run_plan(device: Device, *, repeat: int) -> None:
    rng = np.random.default_rng(15)
    weight_value = rng.integers(-3, 4, size=(4, 2, 3, 3), dtype=np.int32)
    with (
        device.queue() as queue,
        Conv2dInt32Plan(device, source_shape=(1, 2, 6, 5), weight_shape=weight_value.shape) as plan,
    ):
        weight = _tensor(device, weight_value)
        plan.load_weight(weight, queue=queue).wait()
        for seed in range(repeat):
            source_value = np.random.default_rng(seed).integers(-3, 4, size=(1, 2, 6, 5), dtype=np.int32)
            source = _tensor(device, source_value)
            expected = _reference(source_value, weight_value)
            destination = device.tensor(expected.shape, np.int32)
            plan.execute(destination, source, queue=queue).wait()
            np.testing.assert_array_equal(destination.numpy(), expected)


def test_conv2d_int32_plan_reuses_weight_matrix_on_fake_backend() -> None:
    with Device.fake() as device:
        _run_plan(device, repeat=3)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_conv2d_int32_plan_reuses_qpu_weight_matrix() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        _run_plan(device, repeat=2)
