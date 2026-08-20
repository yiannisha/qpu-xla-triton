from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import embedding_lookup_fp32
from qpu_xla.scheduler import Placement


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("tokens,width", [(1, 64), (17, 128)])
def test_qpu_embedding_lookup_matches_numpy(tokens: int, width: int) -> None:
    rng = np.random.default_rng(818)
    table_value = rng.standard_normal((97, width), dtype=np.float32)
    ids_value = rng.integers(0, 97, size=(tokens,), dtype=np.int32)
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        table = device.tensor(table_value.shape, np.float32)
        ids = device.tensor(ids_value.shape, np.int32)
        destination = device.tensor((tokens, width), np.float32)
        table.numpy()[:] = table_value
        ids.numpy()[:] = ids_value
        embedding_lookup_fp32(destination, ids, table, queue=queue, placement=Placement.QPU).wait()
        np.testing.assert_array_equal(destination.numpy(), table_value[ids_value])


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_hybrid_embedding_lookup_matches_numpy() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as qpu_queue, device.queue() as cpu_queue:
        table = device.tensor((32, 64), np.float32)
        ids = device.tensor((5,), np.int32)
        destination = device.tensor((5, 64), np.float32)
        table.numpy()[:] = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
        ids.numpy()[:] = [7, 2, 9, 1, 4]
        embedding_lookup_fp32(
            destination, ids, table, queue=qpu_queue, cpu_queue=cpu_queue, placement=Placement.HYBRID, qpu_tokens=2
        ).wait()
        np.testing.assert_array_equal(destination.numpy(), table.numpy()[ids.numpy()])
