from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import scaled_dot_product_attention_fp32
from qpu_xla.scheduler import Placement


def _reference(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    scores = query @ key.T / np.sqrt(np.float32(query.shape[1]))
    probabilities = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ value


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("hybrid", [False, True])
def test_decode_sdpa_qpu_and_hybrid_gemv_stages_match_reference(hybrid: bool) -> None:
    rng = np.random.default_rng(995)
    query_value = rng.standard_normal((1, 64), dtype=np.float32)
    key_value = rng.standard_normal((32, 64), dtype=np.float32)
    value_value = rng.standard_normal((32, 64), dtype=np.float32)
    expected = _reference(query_value, key_value, value_value)
    with (
        Device.open(data_area_size=4 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        query = device.tensor(query_value.shape, np.float32)
        key = device.tensor(key_value.shape, np.float32)
        value = device.tensor(value_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        query.numpy()[:] = query_value
        key.numpy()[:] = key_value
        value.numpy()[:] = value_value
        placement = Placement.HYBRID if hybrid else Placement.QPU
        scaled_dot_product_attention_fp32(
            destination,
            query,
            key,
            value,
            queue=qpu_queue,
            matmul_cpu_queue=cpu_queue if hybrid else None,
            score_placement=placement,
            value_placement=placement,
            score_qpu_units=16 if hybrid else None,
            value_qpu_units=32 if hybrid else None,
            softmax_placement=Placement.QPU,
        ).wait()
        np.testing.assert_allclose(destination.numpy(), expected, atol=8e-4, rtol=8e-4)
