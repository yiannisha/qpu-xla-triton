from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.w8a8_epilogue import W8A8_DEQUANTIZE_KERNEL


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_w8a8_dequantize_matches_fp32_reference() -> None:
    rng = np.random.default_rng(819)
    source_value = rng.integers(-100_000, 100_001, size=(32, 48), dtype=np.int32)
    row_value = rng.random(32, dtype=np.float32) / 127
    column_value = rng.random(48, dtype=np.float32) / 127
    expected = source_value.astype(np.float32) * row_value[:, None] * column_value[None, :]
    with Device.open(data_area_size=2 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        row_scales = device.tensor(row_value.shape, np.float32)
        column_scales = device.tensor(column_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        row_scales.numpy()[:] = row_value
        column_scales.numpy()[:] = column_value

        queue.submit(
            W8A8_DEQUANTIZE_KERNEL,
            (source, row_scales, column_scales, destination),
            grid=(3, 2, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
