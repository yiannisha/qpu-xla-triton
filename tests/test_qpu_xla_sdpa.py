from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import scaled_dot_product_attention_fp32
from qpu_xla.scheduler import Placement


def _reference(query: np.ndarray, key: np.ndarray, value: np.ndarray, *, causal: bool) -> np.ndarray:
    scores = query @ key.T / np.sqrt(np.float32(query.shape[1]))
    if causal:
        rows, columns = np.triu_indices(query.shape[0], k=1, m=key.shape[0])
        scores[rows, columns] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ value


@pytest.mark.parametrize("causal", (False, True))
def test_sdpa_fp32_cpu_matches_stable_reference(causal: bool) -> None:
    rng = np.random.default_rng(22)
    query_value = rng.standard_normal((3, 5), dtype=np.float32)
    key_value = rng.standard_normal((4, 5), dtype=np.float32)
    value_value = rng.standard_normal((4, 6), dtype=np.float32)
    expected = _reference(query_value, key_value, value_value, causal=causal)
    with Device.fake() as device, device.queue() as queue:
        query = device.tensor(query_value.shape, np.float32)
        key = device.tensor(key_value.shape, np.float32)
        value = device.tensor(value_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        query.numpy()[:] = query_value
        key.numpy()[:] = key_value
        value.numpy()[:] = value_value

        scaled_dot_product_attention_fp32(destination, query, key, value, causal=causal, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_sdpa_fp32_qpu_gemm_stages_match_reference() -> None:
    rng = np.random.default_rng(23)
    query_value = rng.standard_normal((17, 8), dtype=np.float32)
    key_value = rng.standard_normal((20, 8), dtype=np.float32)
    value_value = rng.standard_normal((20, 18), dtype=np.float32)
    expected = _reference(query_value, key_value, value_value, causal=True)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        query = device.tensor(query_value.shape, np.float32)
        key = device.tensor(key_value.shape, np.float32)
        value = device.tensor(value_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        query.numpy()[:] = query_value
        key.numpy()[:] = key_value
        value.numpy()[:] = value_value

        scaled_dot_product_attention_fp32(
            destination,
            query,
            key,
            value,
            causal=True,
            queue=queue,
            softmax_placement=Placement.QPU,
        ).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)
