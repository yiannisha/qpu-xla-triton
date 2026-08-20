from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import PreparedPool2DFP32, pool2d_fp32


def _reference(source: np.ndarray, mode: str) -> np.ndarray:
    x00, x01 = source[:, :, 0::2, 0::2], source[:, :, 0::2, 1::2]
    x10, x11 = source[:, :, 1::2, 0::2], source[:, :, 1::2, 1::2]
    if mode == "max":
        return np.maximum(np.maximum(x00, x01), np.maximum(x10, x11))
    return ((x00 + x01) + (x10 + x11)) * np.float32(0.25)


@pytest.mark.parametrize("mode", ("max", "avg"))
def test_pool2d_fp32_cpu_reference_matches_vectorized_result(mode: str) -> None:
    source_value = np.array(
        [[[[-7.5, -6.5, 1.0, 2.0], [-5.5, -4.5, 3.0, 4.0], [5.0, 6.0, -1.0, -2.0], [7.0, 8.0, -3.0, -4.0]]]],
        dtype=np.float32,
    )
    expected = _reference(source_value, mode)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        pool2d_fp32(destination, source, mode=mode, queue=queue).wait()
        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ("max", "avg"))
@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_pool2d_fp32_qpu_matches_reference(mode: str) -> None:
    source_value = np.random.default_rng(21).standard_normal((1, 1, 8, 8), dtype=np.float32)
    expected = _reference(source_value, mode)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        pool2d_fp32(destination, source, mode=mode, queue=queue).wait()
        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ("max", "avg"))
@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", ((1, 2, 16, 16), (1, 3, 16, 16), (1, 11, 8, 8)))
def test_prepared_pool2d_fp32_uses_multi_qpu_path_exactly(
    mode: str, shape: tuple[int, int, int, int]
) -> None:
    source_value = np.random.default_rng(22).standard_normal(shape, dtype=np.float32)
    expected = _reference(source_value, mode)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with PreparedPool2DFP32(source, destination) as plan:
            event = plan.execute(mode=mode, queue=queue)
            event.wait()
            assert event.name == f"vc7.{mode}pool2d_fp32"
            np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)
