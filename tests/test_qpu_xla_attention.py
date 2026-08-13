from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import attention_int32


def _reference_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Compute the explicit widened reference for the unnormalized operator."""
    scores = query.astype(np.int64) @ key.astype(np.int64).T
    return (scores @ value.astype(np.int64)).astype(np.int32)


def _tensor(device: Device, values: np.ndarray):
    tensor = device.tensor(values.shape, np.int32)
    tensor.numpy()[:] = values
    return tensor


def test_attention_int32_cpu_reference_pads_both_gemm_stages() -> None:
    rng = np.random.default_rng(10)
    query_value = rng.integers(-4, 4, size=(3, 5), dtype=np.int32)
    key_value = rng.integers(-4, 4, size=(7, 5), dtype=np.int32)
    value_value = rng.integers(-4, 4, size=(7, 6), dtype=np.int32)
    expected = _reference_attention(query_value, key_value, value_value)
    with Device.fake() as device, device.queue() as queue:
        query, key, value = (_tensor(device, values) for values in (query_value, key_value, value_value))
        destination = device.tensor(expected.shape, np.int32)

        attention_int32(destination, query, key, value, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)


def test_attention_int32_rejects_score_values_too_wide_for_second_smul24_stage() -> None:
    with Device.fake() as device:
        query = _tensor(device, np.full((1, 1), 4096, dtype=np.int32))
        key = _tensor(device, np.full((1, 1), 4096, dtype=np.int32))
        value = _tensor(device, np.ones((1, 1), dtype=np.int32))
        destination = device.tensor((1, 1), np.int32)

        with pytest.raises(ValueError, match="score values"):
            attention_int32(destination, query, key, value)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_attention_int32_qpu_two_tiled_stages_match_reference() -> None:
    rng = np.random.default_rng(11)
    query_value = rng.integers(-3, 4, size=(17, 8), dtype=np.int32)
    key_value = rng.integers(-3, 4, size=(20, 8), dtype=np.int32)
    value_value = rng.integers(-3, 4, size=(20, 18), dtype=np.int32)
    expected = _reference_attention(query_value, key_value, value_value)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        query, key, value = (_tensor(device, values) for values in (query_value, key_value, value_value))
        destination = device.tensor(expected.shape, np.int32)

        attention_int32(destination, query, key, value, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)
