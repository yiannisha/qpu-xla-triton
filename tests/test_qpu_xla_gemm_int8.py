from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.gemm_int8 import (
    TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
    TILED_W8A8_GEMM_KERNEL,
    pack_int8_gemm_operands,
    pack_int8_quads,
)


def test_pack_int8_quads_preserves_signed_bytes_and_operand_layout() -> None:
    left = np.array([[-128, -1, 0, 127, 3, -4, 5, -6]], dtype=np.int8)
    right = np.arange(-16, 16, dtype=np.int8).reshape(8, 4)
    packed_left, packed_right = pack_int8_gemm_operands(left, right)

    assert packed_left.shape == (1, 2)
    assert packed_right.shape == (2, 4)
    np.testing.assert_array_equal(packed_left.view(np.uint8).reshape(1, 8).view(np.int8), left)
    np.testing.assert_array_equal(pack_int8_quads(right.T).view(np.uint8).reshape(4, 8).view(np.int8), right.T)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", ((16, 16, 16), (32, 64, 32)))
def test_tiled_w8a8_gemm_matches_exact_int32_reference(shape: tuple[int, int, int]) -> None:
    p, q, r = shape
    rng = np.random.default_rng(91 + q)
    left_value = rng.integers(-127, 128, size=(p, q), dtype=np.int8)
    right_value = rng.integers(-127, 128, size=(q, r), dtype=np.int8)
    expected = (left_value.astype(np.int64) @ right_value.astype(np.int64)).astype(np.int32)
    packed_left, packed_right = pack_int8_gemm_operands(left_value, right_value)
    data_area_size = packed_left.nbytes + packed_right.nbytes + expected.nbytes + 1024 * 1024

    with Device.open(data_area_size=data_area_size) as device, device.queue() as queue:
        left = device.tensor(packed_left.shape, np.uint32)
        right = device.tensor(packed_right.shape, np.uint32)
        destination = device.tensor(expected.shape, np.int32)
        left.numpy()[:] = packed_left
        right.numpy()[:] = packed_right

        queue.submit(
            TILED_W8A8_GEMM_KERNEL,
            (left, right, destination),
            grid=(r // 16, p // 16, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_tiled_w8a8_gemm_fused_dequantize_matches_fp32_epilogue() -> None:
    p, q, r = 32, 64, 48
    rng = np.random.default_rng(1765)
    left_value = rng.integers(-127, 128, size=(p, q), dtype=np.int8)
    right_value = rng.integers(-127, 128, size=(q, r), dtype=np.int8)
    row_scale_value = rng.uniform(0.001, 0.1, size=p).astype(np.float32)
    column_scale_value = rng.uniform(0.001, 0.1, size=r).astype(np.float32)
    accumulation = (left_value.astype(np.int64) @ right_value.astype(np.int64)).astype(np.int32)
    expected = accumulation.astype(np.float32) * row_scale_value[:, None] * column_scale_value[None, :]
    packed_left, packed_right = pack_int8_gemm_operands(left_value, right_value)
    data_area_size = sum(
        value.nbytes
        for value in (packed_left, packed_right, row_scale_value, column_scale_value, expected)
    ) + 1024 * 1024

    with Device.open(data_area_size=data_area_size) as device, device.queue() as queue:
        left = device.tensor(packed_left.shape, np.uint32)
        right = device.tensor(packed_right.shape, np.uint32)
        row_scales = device.tensor(row_scale_value.shape, np.float32)
        column_scales = device.tensor(column_scale_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        left.numpy()[:] = packed_left
        right.numpy()[:] = packed_right
        row_scales.numpy()[:] = row_scale_value
        column_scales.numpy()[:] = column_scale_value

        queue.submit(
            TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
            (left, right, row_scales, column_scales, destination),
            grid=(r // 16, p // 16, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_array_equal(actual, expected)
