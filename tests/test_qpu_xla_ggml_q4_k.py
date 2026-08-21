from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.ggml_q4_k import (
    GGML_Q4_K_Q8_K_LINEAR_M4_KERNEL,
    ggml_q4_k_q8_k_reference,
    pack_ggml_q4_k_blocks,
    pack_ggml_q8_k_blocks,
    unpack_ggml_q4_k_blocks,
    unpack_ggml_q8_k_blocks,
)


def test_native_q4_k_and_q8_k_round_trip_extreme_fields() -> None:
    super_scales = np.asarray([0.0, -0.5], dtype=np.float16)
    super_mins = np.asarray([np.float16(2**-14), 0.25], dtype=np.float16)
    sub_scales = np.asarray([[0, 1, 15, 16, 31, 32, 62, 63]] * 2, dtype=np.uint8)
    sub_mins = np.asarray([[63, 62, 32, 31, 16, 15, 1, 0]] * 2, dtype=np.uint8)
    codes = np.stack((np.arange(256) % 16, 15 - np.arange(256) % 16)).astype(np.uint8)
    blocks = pack_ggml_q4_k_blocks(
        super_scales,
        super_mins,
        sub_scales,
        sub_mins,
        codes,
    )
    actual_scale, actual_min, actual_sub_scale, actual_sub_min, actual_codes = (
        unpack_ggml_q4_k_blocks(blocks)
    )
    np.testing.assert_array_equal(actual_scale, super_scales.astype(np.float32))
    np.testing.assert_array_equal(actual_min, super_mins.astype(np.float32))
    np.testing.assert_array_equal(actual_sub_scale, sub_scales)
    np.testing.assert_array_equal(actual_sub_min, sub_mins)
    np.testing.assert_array_equal(actual_codes, codes)

    q8_scales = np.asarray([0.0, -0.125], dtype=np.float32)
    q8_values = np.stack(
        (
            np.resize(np.asarray([-128, 127], dtype=np.int16), 256),
            np.arange(256, dtype=np.int16) % 255 - 127,
        )
    )
    q8_blocks = pack_ggml_q8_k_blocks(q8_scales, q8_values)
    decoded_scales, decoded_values, decoded_sums = unpack_ggml_q8_k_blocks(q8_blocks)
    np.testing.assert_array_equal(decoded_scales, q8_scales)
    np.testing.assert_array_equal(decoded_values, q8_values.astype(np.int8))
    np.testing.assert_array_equal(
        decoded_sums,
        q8_values.astype(np.int16).reshape(2, 16, 16).sum(axis=-1),
    )


def test_q4_k_q8_k_reference_matches_explicit_dequantized_dot() -> None:
    rng = np.random.default_rng(4540)
    rows, outputs, blocks = 4, 3, 2
    activation_scales = rng.uniform(-0.1, 0.1, size=(rows, blocks)).astype(np.float32)
    activation_values = rng.integers(-127, 128, size=(rows, blocks, 256), dtype=np.int16)
    weight_scales = rng.uniform(-0.2, 0.2, size=(outputs, blocks)).astype(np.float16)
    weight_mins = rng.uniform(-0.2, 0.2, size=(outputs, blocks)).astype(np.float16)
    sub_scales = rng.integers(0, 64, size=(outputs, blocks, 8), dtype=np.uint8)
    sub_mins = rng.integers(0, 64, size=(outputs, blocks, 8), dtype=np.uint8)
    codes = rng.integers(0, 16, size=(outputs, blocks, 256), dtype=np.uint8)
    activation_blocks = pack_ggml_q8_k_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q4_k_blocks(
        weight_scales,
        weight_mins,
        sub_scales,
        sub_mins,
        codes,
    )
    actual = ggml_q4_k_q8_k_reference(activation_blocks, weight_blocks)
    q8 = activation_scales[..., None] * activation_values.astype(np.float32)
    weight_scales_f32 = weight_scales.astype(np.float32)
    weight_mins_f32 = weight_mins.astype(np.float32)
    q4 = np.empty((outputs, blocks, 256), dtype=np.float32)
    for subblock in range(8):
        span = slice(subblock * 32, (subblock + 1) * 32)
        q4[..., span] = (
            weight_scales_f32[..., None]
            * sub_scales[..., subblock, None]
            * codes[..., span]
            - weight_mins_f32[..., None] * sub_mins[..., subblock, None]
        )
    expected = np.einsum("rbk,obk->ro", q8, q4, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-3)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII unavailable")
def test_native_q4_k_q8_k_m4_matches_oracle_and_preserves_partition() -> None:
    rng = np.random.default_rng(4544)
    rows, outputs, blocks = 4, 32, 2
    activation_scales = rng.uniform(-0.1, 0.1, size=(rows, blocks)).astype(np.float32)
    activation_values = rng.integers(-127, 128, size=(rows, blocks, 256), dtype=np.int16)
    weight_scales = rng.uniform(-0.1, 0.1, size=(outputs, blocks)).astype(np.float16)
    weight_mins = rng.uniform(-0.1, 0.1, size=(outputs, blocks)).astype(np.float16)
    sub_scales = rng.integers(0, 64, size=(outputs, blocks, 8), dtype=np.uint8)
    sub_mins = rng.integers(0, 64, size=(outputs, blocks, 8), dtype=np.uint8)
    codes = rng.integers(0, 16, size=(outputs, blocks, 256), dtype=np.uint8)
    activation_blocks = pack_ggml_q8_k_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q4_k_blocks(
        weight_scales,
        weight_mins,
        sub_scales,
        sub_mins,
        codes,
    )
    expected = ggml_q4_k_q8_k_reference(activation_blocks, weight_blocks)
    resident_weight = np.ascontiguousarray(weight_blocks[16:])
    sentinel = np.float32(123456.0)
    data_area_size = activation_blocks.nbytes + resident_weight.nbytes + expected.nbytes + (1 << 20)
    with Device.open(data_area_size=data_area_size) as device, device.queue() as queue:
        activation = device.tensor(activation_blocks.shape, np.uint8)
        weight = device.tensor(resident_weight.shape, np.uint8)
        destination = device.tensor(expected.shape, np.float32)
        activation.numpy()[:] = activation_blocks
        weight.numpy()[:] = resident_weight
        destination.numpy()[:] = sentinel
        queue.submit(
            GGML_Q4_K_Q8_K_LINEAR_M4_KERNEL,
            (activation, weight, destination, 0, 16, 16),
            grid=(1, 1, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)
    np.testing.assert_array_equal(actual[:, :16], sentinel)
    np.testing.assert_allclose(actual[:, 16:], expected[:, 16:], rtol=3e-6, atol=3e-4)
