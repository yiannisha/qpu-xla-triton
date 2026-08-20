from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import mlp_fp32


def _make_case(device: Device, *, batch: int, inputs: int, hidden: int, outputs: int) -> tuple:
    rng = np.random.default_rng(31)
    values = (
        rng.standard_normal((batch, inputs), dtype=np.float32),
        rng.standard_normal((inputs, hidden), dtype=np.float32),
        rng.standard_normal((hidden,), dtype=np.float32),
        rng.standard_normal((hidden, outputs), dtype=np.float32),
        rng.standard_normal((outputs,), dtype=np.float32),
    )
    tensors = tuple(device.tensor(value.shape, np.float32) for value in values)
    for tensor, value in zip(tensors, values, strict=True):
        tensor.numpy()[:] = value
    return (*tensors, values)


def test_mlp_fp32_cpu_reference_matches_numpy() -> None:
    with Device.fake() as device, device.queue() as queue:
        source, weight1, bias1, weight2, bias2, values = _make_case(
            device, batch=3, inputs=5, hidden=7, outputs=6
        )
        destination = device.tensor((3, 6), np.float32)
        source_value, weight1_value, bias1_value, weight2_value, bias2_value = values
        expected = np.maximum(source_value @ weight1_value + bias1_value, 0.0) @ weight2_value + bias2_value

        mlp_fp32(destination, source, weight1, bias1, weight2, bias2, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_mlp_fp32_qpu_stages_match_numpy() -> None:
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        source, weight1, bias1, weight2, bias2, values = _make_case(
            device, batch=32, inputs=32, hidden=32, outputs=32
        )
        destination = device.tensor((32, 32), np.float32)
        source_value, weight1_value, bias1_value, weight2_value, bias2_value = values
        expected = np.maximum(source_value @ weight1_value + bias1_value, 0.0) @ weight2_value + bias2_value

        mlp_fp32(destination, source, weight1, bias1, weight2, bias2, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-4, rtol=1e-4)
