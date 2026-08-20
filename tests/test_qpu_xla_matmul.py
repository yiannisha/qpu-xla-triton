from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import hybrid_matmul, matmul
from qpu_xla.scheduler import Placement


def test_matmul_cpu_reference_handles_non_tile_shapes() -> None:
    with Device.fake() as device, device.queue() as queue:
        left = device.tensor((3, 5), np.int32)
        right = device.tensor((5, 2), np.int32)
        destination = device.tensor((3, 2), np.int32)
        left.numpy()[:] = np.arange(15, dtype=np.int32).reshape(3, 5) - 7
        right.numpy()[:] = np.arange(10, dtype=np.int32).reshape(5, 2) - 3

        matmul(destination, left, right, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), left.numpy() @ right.numpy())


def test_matmul_rejects_shape_and_dtype_mismatches() -> None:
    with Device.fake() as device:
        left = device.tensor((2, 3), np.int32)
        right = device.tensor((4, 2), np.int32)
        destination = device.tensor((2, 2), np.int32)
        with pytest.raises(ValueError, match="align"):
            matmul(destination, left, right)

        right_float = device.tensor((3, 2), np.float32)
        with pytest.raises(ValueError, match="dtypes"):
            matmul(destination, left, right_float)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_tiled_int32_matmul_matches_numpy_on_multiple_output_tiles() -> None:
    rng = np.random.default_rng(5)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        left = device.tensor((32, 8), np.int32)
        right = device.tensor((8, 32), np.int32)
        destination = device.tensor((32, 32), np.int32)
        left.numpy()[:] = rng.integers(-16, 16, size=left.shape, dtype=np.int32)
        right.numpy()[:] = rng.integers(-16, 16, size=right.shape, dtype=np.int32)
        expected = left.numpy().astype(np.int64) @ right.numpy().astype(np.int64)

        matmul(destination, left, right, queue=queue).wait()

        np.testing.assert_array_equal(destination.numpy(), expected.astype(np.int32))


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_tiled_fp32_matmul_matches_numpy_on_multiple_output_tiles() -> None:
    rng = np.random.default_rng(18)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        left = device.tensor((32, 8), np.float32)
        right = device.tensor((8, 32), np.float32)
        destination = device.tensor((32, 32), np.float32)
        left.numpy()[:] = rng.standard_normal(left.shape, dtype=np.float32)
        right.numpy()[:] = rng.standard_normal(right.shape, dtype=np.float32)
        expected = left.numpy() @ right.numpy()

        matmul(destination, left, right, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("input_width,output_width", [(64, 32), (512, 512), (2048, 256)])
def test_fp32_gemv_matches_numpy(input_width: int, output_width: int) -> None:
    rng = np.random.default_rng(918)
    with Device.open(data_area_size=32 * 1024 * 1024) as device, device.queue() as queue:
        left = device.tensor((1, input_width), np.float32)
        right = device.tensor((input_width, output_width), np.float32)
        destination = device.tensor((1, output_width), np.float32)
        left.numpy()[:] = rng.standard_normal(left.shape, dtype=np.float32)
        right.numpy()[:] = rng.standard_normal(right.shape, dtype=np.float32)
        expected = left.numpy() @ right.numpy()

        matmul(destination, left, right, queue=queue, placement=Placement.QPU).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=3e-4, rtol=3e-4)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_fp32_decode_hybrid_splits_output_columns() -> None:
    rng = np.random.default_rng(919)
    with (
        Device.open(data_area_size=4 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        left = device.tensor((1, 128), np.float32)
        right = device.tensor((128, 64), np.float32)
        destination = device.tensor((1, 64), np.float32)
        left.numpy()[:] = rng.standard_normal(left.shape, dtype=np.float32)
        right.numpy()[:] = rng.standard_normal(right.shape, dtype=np.float32)
        expected = left.numpy() @ right.numpy()

        hybrid_matmul(
            destination,
            left,
            right,
            qpu_queue=qpu_queue,
            cpu_queue=cpu_queue,
            qpu_columns=32,
        ).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=2e-4, rtol=2e-4)
