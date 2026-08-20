from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import residual_add_fp32
from qpu_xla.scheduler import Placement


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", [(1, 2048), (1, 2080), (32, 512)])
def test_qpu_residual_add_matches_numpy(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(810)
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        left = device.tensor(shape, np.float32)
        right = device.tensor(shape, np.float32)
        destination = device.tensor(shape, np.float32)
        left.numpy()[:] = rng.standard_normal(shape, dtype=np.float32)
        right.numpy()[:] = rng.standard_normal(shape, dtype=np.float32)
        residual_add_fp32(destination, left, right, queue=queue, placement=Placement.QPU).wait()
        np.testing.assert_array_equal(destination.numpy(), left.numpy() + right.numpy())


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_hybrid_residual_add_matches_numpy() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as qpu_queue, device.queue() as cpu_queue:
        left = device.tensor((17, 64), np.float32)
        right = device.tensor((17, 64), np.float32)
        destination = device.tensor((17, 64), np.float32)
        left.numpy()[:] = 2.0
        right.numpy()[:] = 3.0
        residual_add_fp32(
            destination, left, right, queue=qpu_queue, cpu_queue=cpu_queue, placement=Placement.HYBRID, qpu_rows=8
        ).wait()
        np.testing.assert_array_equal(destination.numpy(), np.full((17, 64), 5.0, np.float32))
