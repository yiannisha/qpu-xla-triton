from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import PreparedFP32Linear
from qpu_xla.scheduler import Placement


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("batch,inputs,outputs", [(1, 127, 93), (17, 68, 47), (32, 64, 64)])
def test_prepared_fp32_linear_qpu_matches_numpy(batch: int, inputs: int, outputs: int) -> None:
    rng = np.random.default_rng(722)
    weight_value = rng.standard_normal((outputs, inputs), dtype=np.float32)
    source_value = rng.standard_normal((batch, inputs), dtype=np.float32)
    with Device.open(data_area_size=32 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor((batch, outputs), np.float32)
        source.numpy()[:] = source_value
        with PreparedFP32Linear(device, weight_value, max_batch=batch) as linear:
            linear.execute(destination, source, queue=queue, placement=Placement.QPU).wait()
            np.testing.assert_allclose(destination.numpy(), source_value @ weight_value.T, atol=5e-4, rtol=5e-4)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_prepared_fp32_linear_decode_hybrid_matches_numpy() -> None:
    rng = np.random.default_rng(723)
    weight_value = rng.standard_normal((80, 128), dtype=np.float32)
    source_value = rng.standard_normal((1, 128), dtype=np.float32)
    with (
        Device.open(data_area_size=4 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor((1, 80), np.float32)
        source.numpy()[:] = source_value
        with PreparedFP32Linear(device, weight_value, max_batch=1) as linear:
            linear.execute(
                destination,
                source,
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                placement=Placement.HYBRID,
                qpu_units=32,
            ).wait()
            np.testing.assert_allclose(destination.numpy(), source_value @ weight_value.T, atol=3e-4, rtol=3e-4)
