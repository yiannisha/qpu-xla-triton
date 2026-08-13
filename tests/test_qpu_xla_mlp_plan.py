from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import MlpInt32Plan


def _reference(
    source: np.ndarray,
    weight1: np.ndarray,
    bias1: np.ndarray,
    weight2: np.ndarray,
    bias2: np.ndarray,
) -> np.ndarray:
    hidden = source.astype(np.int64) @ weight1.astype(np.int64) + bias1.astype(np.int64)
    return (np.maximum(hidden, 0) @ weight2.astype(np.int64) + bias2.astype(np.int64)).astype(np.int32)


def _tensor(device: Device, values: np.ndarray):
    tensor = device.tensor(values.shape, np.int32)
    tensor.numpy()[:] = values
    return tensor


def _run_plan(device: Device, *, repeat: int) -> None:
    rng = np.random.default_rng(14)
    weight1_value = rng.integers(-3, 4, size=(5, 7), dtype=np.int32)
    bias1_value = rng.integers(-5, 6, size=(7,), dtype=np.int32)
    weight2_value = rng.integers(-3, 4, size=(7, 6), dtype=np.int32)
    bias2_value = rng.integers(-5, 6, size=(6,), dtype=np.int32)
    with (
        device.queue() as queue,
        MlpInt32Plan(device, batch=3, in_features=5, hidden_features=7, out_features=6) as plan,
    ):
        weights = (_tensor(device, value) for value in (weight1_value, bias1_value, weight2_value, bias2_value))
        weight1, bias1, weight2, bias2 = weights
        plan.load_weights(weight1, bias1, weight2, bias2, queue=queue).wait()
        for seed in range(repeat):
            source_value = np.random.default_rng(seed).integers(-3, 4, size=(3, 5), dtype=np.int32)
            source = _tensor(device, source_value)
            destination = device.tensor((3, 6), np.int32)
            plan.execute(destination, source, queue=queue).wait()
            np.testing.assert_array_equal(
                destination.numpy(),
                _reference(source_value, weight1_value, bias1_value, weight2_value, bias2_value),
            )


def test_mlp_int32_plan_reuses_loaded_weights_on_fake_backend() -> None:
    with Device.fake() as device:
        _run_plan(device, repeat=3)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_mlp_int32_plan_reuses_qpu_buffers_and_weights() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        _run_plan(device, repeat=2)
