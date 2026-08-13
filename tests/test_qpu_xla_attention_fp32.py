from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import attention_fp32


def _run(device: Device) -> None:
    rng = np.random.default_rng(20)
    query_value = rng.standard_normal((17, 8), dtype=np.float32)
    key_value = rng.standard_normal((20, 8), dtype=np.float32)
    value_value = rng.standard_normal((20, 18), dtype=np.float32)
    expected = query_value @ key_value.T @ value_value
    with device.queue() as queue:
        query = device.tensor(query_value.shape, np.float32)
        key = device.tensor(key_value.shape, np.float32)
        value = device.tensor(value_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        query.numpy()[:] = query_value
        key.numpy()[:] = key_value
        value.numpy()[:] = value_value

        attention_fp32(destination, query, key, value, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)


def test_attention_fp32_cpu_reference_matches_two_matmul_stages() -> None:
    with Device.fake() as device:
        _run(device)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_attention_fp32_qpu_two_tiled_stages_match_reference() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        _run(device)
