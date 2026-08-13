from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.models.tinyllama import KvCacheFp32


def _reference(query: np.ndarray, key: np.ndarray, value: np.ndarray, *, offset: int) -> np.ndarray:
    scores = query @ key.T / np.sqrt(np.float32(query.shape[1]))
    columns = np.arange(key.shape[0])[None, :]
    rows = np.arange(query.shape[0])[:, None]
    scores[columns > rows + offset] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ value


def _tensor(device: Device, values: np.ndarray):
    tensor = device.tensor(values.shape, np.float32)
    tensor.numpy()[:] = values
    return tensor


def _run_cache(device: Device) -> None:
    rng = np.random.default_rng(24)
    key_value = rng.standard_normal((5, 4), dtype=np.float32)
    value_value = rng.standard_normal((5, 6), dtype=np.float32)
    query_value = rng.standard_normal((2, 4), dtype=np.float32)
    expected = _reference(query_value, key_value, value_value, offset=3)
    with device.queue() as queue, KvCacheFp32(device, capacity=8, depth=4, value_dim=6) as cache:
        cache.append(_tensor(device, key_value[:3]), _tensor(device, value_value[:3]), queue=queue).wait()
        cache.append(_tensor(device, key_value[3:]), _tensor(device, value_value[3:]), queue=queue).wait()
        destination = device.tensor((2, 6), np.float32)
        cache.attend(destination, _tensor(device, query_value), queue=queue).wait()
        assert cache.length == 5
        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)


def test_kv_cache_fp32_appends_and_uses_position_aware_masking() -> None:
    with Device.fake() as device:
        _run_cache(device)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_kv_cache_fp32_qpu_attention_stages_match_reference() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        _run_cache(device)
