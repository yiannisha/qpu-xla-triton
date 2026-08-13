from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import DependencyError, Device
from qpu_xla.ops import hybrid_matmul


def test_hybrid_matmul_requires_distinct_queues() -> None:
    with Device.fake() as device, device.queue() as queue:
        left = device.tensor((16, 4), np.int32)
        right = device.tensor((4, 16), np.int32)
        destination = device.tensor((16, 16), np.int32)
        with pytest.raises(DependencyError, match="separate"):
            hybrid_matmul(destination, left, right, qpu_queue=queue, cpu_queue=queue)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_hybrid_matmul_qpu_prefix_and_cpu_tail_match_reference() -> None:
    rng = np.random.default_rng(17)
    left_value = rng.integers(-4, 4, size=(17, 4), dtype=np.int32)
    right_value = rng.integers(-4, 4, size=(4, 16), dtype=np.int32)
    expected = (left_value.astype(np.int64) @ right_value.astype(np.int64)).astype(np.int32)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as qpu_queue, device.queue() as cpu_queue:
        left = device.tensor(left_value.shape, np.int32)
        right = device.tensor(right_value.shape, np.int32)
        destination = device.tensor(expected.shape, np.int32)
        left.numpy()[:] = left_value
        right.numpy()[:] = right_value

        hybrid_matmul(destination, left, right, qpu_queue=qpu_queue, cpu_queue=cpu_queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected)
