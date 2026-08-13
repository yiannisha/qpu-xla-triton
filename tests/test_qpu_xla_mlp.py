from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import mlp_int32


def _reference_mlp(
    source: np.ndarray,
    weight1: np.ndarray,
    bias1: np.ndarray,
    weight2: np.ndarray,
    bias2: np.ndarray,
) -> np.ndarray:
    """Compute the strict INT32 MLP reference used by QPU differential tests."""
    hidden = source.astype(np.int64) @ weight1.astype(np.int64) + bias1.astype(np.int64)
    hidden = np.maximum(hidden, 0)
    result = hidden @ weight2.astype(np.int64) + bias2.astype(np.int64)
    return result.astype(np.int32)


def test_mlp_int32_cpu_reference_pads_all_tiled_dimensions() -> None:
    rng = np.random.default_rng(8)
    source_value = rng.integers(-4, 4, size=(3, 5), dtype=np.int32)
    weight1_value = rng.integers(-4, 4, size=(5, 7), dtype=np.int32)
    bias1_value = rng.integers(-8, 8, size=(7,), dtype=np.int32)
    weight2_value = rng.integers(-4, 4, size=(7, 6), dtype=np.int32)
    bias2_value = rng.integers(-8, 8, size=(6,), dtype=np.int32)
    expected = _reference_mlp(source_value, weight1_value, bias1_value, weight2_value, bias2_value)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        weight1 = device.tensor(weight1_value.shape, np.int32)
        bias1 = device.tensor(bias1_value.shape, np.int32)
        weight2 = device.tensor(weight2_value.shape, np.int32)
        bias2 = device.tensor(bias2_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = source_value
        weight1.numpy()[:] = weight1_value
        bias1.numpy()[:] = bias1_value
        weight2.numpy()[:] = weight2_value
        bias2.numpy()[:] = bias2_value

        mlp_int32(destination, source, weight1, bias1, weight2, bias2, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_mlp_int32_qpu_dense_stages_match_reference() -> None:
    rng = np.random.default_rng(9)
    source_value = rng.integers(-4, 4, size=(17, 8), dtype=np.int32)
    weight1_value = rng.integers(-4, 4, size=(8, 20), dtype=np.int32)
    bias1_value = rng.integers(-8, 8, size=(20,), dtype=np.int32)
    weight2_value = rng.integers(-4, 4, size=(20, 18), dtype=np.int32)
    bias2_value = rng.integers(-8, 8, size=(18,), dtype=np.int32)
    expected = _reference_mlp(source_value, weight1_value, bias1_value, weight2_value, bias2_value)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        tensors = [
            device.tensor(value.shape, np.int32)
            for value in (source_value, weight1_value, bias1_value, weight2_value, bias2_value, expected)
        ]
        source, weight1, bias1, weight2, bias2, destination = tensors
        values = (source_value, weight1_value, bias1_value, weight2_value, bias2_value)
        for tensor, value in zip(tensors[:-1], values, strict=True):
            tensor.numpy()[:] = value

        mlp_int32(destination, source, weight1, bias1, weight2, bias2, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)
