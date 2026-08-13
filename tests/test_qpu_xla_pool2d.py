from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops.pool2d import pool2d_int32


def _reference(source: np.ndarray, mode: str) -> np.ndarray:
    x00, x01 = source[:, :, 0::2, 0::2], source[:, :, 0::2, 1::2]
    x10, x11 = source[:, :, 1::2, 0::2], source[:, :, 1::2, 1::2]
    if mode == "max":
        return np.maximum(np.maximum(x00, x01), np.maximum(x10, x11))
    total = x00.astype(np.int64) + x01.astype(np.int64) + x10.astype(np.int64) + x11.astype(np.int64)
    return np.where(total < 0, -((-total) // 4), total // 4).astype(np.int32)


@pytest.mark.parametrize("mode", ("max", "avg"))
def test_pool2d_int32_cpu_reference_handles_negative_truncation(mode: str) -> None:
    source_value = np.array([[[[-7, -6, 1, 2], [-5, -4, 3, 4], [5, 6, -1, -2], [7, 8, -3, -4]]]], dtype=np.int32)
    expected = _reference(source_value, mode)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        pool2d_int32(destination, source, mode=mode, queue=queue).wait()
        np.testing.assert_array_equal(destination.numpy(), expected)


@pytest.mark.parametrize("mode", ("max", "avg"))
@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_pool2d_int32_qpu_matches_reference(mode: str) -> None:
    rng = np.random.default_rng(12)
    source_value = rng.integers(-100, 100, size=(1, 1, 8, 8), dtype=np.int32)
    expected = _reference(source_value, mode)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        pool2d_int32(destination, source, mode=mode, queue=queue).wait()
        np.testing.assert_array_equal(destination.numpy(), expected)
