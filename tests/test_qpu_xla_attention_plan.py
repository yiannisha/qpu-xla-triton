from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import AttentionInt32Plan


def _reference(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    return ((query.astype(np.int64) @ key.astype(np.int64).T) @ value.astype(np.int64)).astype(np.int32)


def _tensor(device: Device, values: np.ndarray):
    tensor = device.tensor(values.shape, np.int32)
    tensor.numpy()[:] = values
    return tensor


def _run_plan(device: Device, *, repeat: int) -> None:
    rng = np.random.default_rng(16)
    key_value = rng.integers(-3, 4, size=(20, 8), dtype=np.int32)
    value_value = rng.integers(-3, 4, size=(20, 18), dtype=np.int32)
    with (
        device.queue() as queue,
        AttentionInt32Plan(
            device, query_shape=(17, 8), key_shape=key_value.shape, value_shape=value_value.shape
        ) as plan,
    ):
        key, value = (_tensor(device, values) for values in (key_value, value_value))
        plan.load_key_value(key, value, queue=queue).wait()
        for seed in range(repeat):
            query_value = np.random.default_rng(seed).integers(-3, 4, size=(17, 8), dtype=np.int32)
            query = _tensor(device, query_value)
            destination = device.tensor((17, 18), np.int32)
            plan.execute(destination, query, queue=queue).wait()
            np.testing.assert_array_equal(destination.numpy(), _reference(query_value, key_value, value_value))


def test_attention_int32_plan_reuses_key_value_preparation_on_fake_backend() -> None:
    with Device.fake() as device:
        _run_plan(device, repeat=3)


def test_attention_int32_plan_retains_the_smul24_range_contract() -> None:
    with (
        Device.fake() as device,
        device.queue() as queue,
        AttentionInt32Plan(device, query_shape=(1, 1), key_shape=(1, 1), value_shape=(1, 1)) as plan,
    ):
        key = _tensor(device, np.array([[4096]], dtype=np.int32))
        value = _tensor(device, np.array([[1]], dtype=np.int32))
        query = _tensor(device, np.array([[4096]], dtype=np.int32))
        destination = device.tensor((1, 1), np.int32)
        plan.load_key_value(key, value, queue=queue).wait()

        with pytest.raises(ValueError, match="score values"):
            plan.execute(destination, query, queue=queue).wait()


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_attention_int32_plan_reuses_qpu_key_value_buffers() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        _run_plan(device, repeat=2)
