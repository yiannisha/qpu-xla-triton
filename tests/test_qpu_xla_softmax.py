from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import softmax_fp32
from qpu_xla.scheduler import Placement


def _reference(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def test_softmax_fp32_cpu_reference_supports_unaligned_width() -> None:
    rng = np.random.default_rng(81)
    values = rng.standard_normal((3, 19), dtype=np.float32)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(values.shape, np.float32)
        destination = device.tensor(values.shape, np.float32)
        source.numpy()[:] = values
        softmax_fp32(destination, source, queue=queue).wait()
        np.testing.assert_allclose(destination.numpy(), _reference(values), atol=1e-7, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", ((1, 16), (16, 64), (17, 512)))
def test_softmax_fp32_qpu_matches_numpy(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(900 + shape[0])
    values = rng.standard_normal(shape, dtype=np.float32) * np.float32(3.0)
    expected = _reference(values)
    with Device.open(data_area_size=2 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(shape, np.float32)
        destination = device.tensor(shape, np.float32)
        source.numpy()[:] = values
        event = softmax_fp32(destination, source, queue=queue, placement=Placement.QPU)
        event.wait()
        actual = np.array(destination.numpy(), copy=True)
        assert event.name == "vc7.softmax_fp32"
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_softmax_fp32_hybrid_row_split_matches_numpy() -> None:
    rng = np.random.default_rng(933)
    values = rng.standard_normal((17, 64), dtype=np.float32)
    expected = _reference(values)
    with (
        Device.open(data_area_size=2 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(values.shape, np.float32)
        destination = device.tensor(values.shape, np.float32)
        source.numpy()[:] = values
        event = softmax_fp32(
            destination,
            source,
            queue=qpu_queue,
            cpu_queue=cpu_queue,
            placement=Placement.HYBRID,
            qpu_rows=8,
        )
        event.wait()
        actual = np.array(destination.numpy(), copy=True)
        assert event.name == "softmax_fp32.hybrid_join"
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
