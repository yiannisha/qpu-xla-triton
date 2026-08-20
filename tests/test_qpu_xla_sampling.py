from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import greedy_sample_fp32
from qpu_xla.scheduler import Placement


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", [(1, 32000), (17, 128)])
def test_qpu_argmax_matches_numpy(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(991)
    values = rng.standard_normal(shape, dtype=np.float32)
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        logits = device.tensor(shape, np.float32)
        destination = device.tensor((shape[0],), np.int32)
        logits.numpy()[:] = values
        greedy_sample_fp32(destination, logits, queue=queue, placement=Placement.QPU).wait()
        np.testing.assert_array_equal(destination.numpy(), np.argmax(values, axis=1))


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_qpu_argmax_uses_first_index_on_ties() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        logits = device.tensor((1, 32), np.float32)
        destination = device.tensor((1,), np.int32)
        logits.numpy().fill(-1.0)
        logits.numpy()[0, 3] = logits.numpy()[0, 19] = 5.0
        greedy_sample_fp32(destination, logits, queue=queue, placement=Placement.QPU).wait()
        assert destination.numpy()[0] == 3
