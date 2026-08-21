from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.ggml_q4_k import pack_ggml_q8_k_blocks
from qpu_xla.kernels.ggml_q6_k import (
    GGML_Q6_K_Q8_K_LINEAR_M4_KERNEL,
    ggml_q6_k_q8_k_reference,
    pack_ggml_q6_k_blocks,
    unpack_ggml_q6_k_blocks,
)


def test_native_q6_k_round_trip_extreme_fields() -> None:
    super_scales = np.asarray([0.0, -0.5], dtype=np.float16)
    sub_scales = np.stack(
        (
            np.resize(np.asarray([-128, 127], dtype=np.int16), 16),
            np.arange(16, dtype=np.int16) - 8,
        )
    )
    codes = np.stack(
        (
            np.arange(256, dtype=np.int16) % 64 - 32,
            31 - np.arange(256, dtype=np.int16) % 64,
        )
    )
    blocks = pack_ggml_q6_k_blocks(super_scales, sub_scales, codes)
    actual_scale, actual_sub_scales, actual_codes = unpack_ggml_q6_k_blocks(blocks)
    np.testing.assert_array_equal(actual_scale, super_scales.astype(np.float32))
    np.testing.assert_array_equal(actual_sub_scales, sub_scales.astype(np.int8))
    np.testing.assert_array_equal(actual_codes, codes.astype(np.int8))


def test_q6_k_q8_k_reference_matches_explicit_dequantized_dot() -> None:
    rng = np.random.default_rng(4650)
    rows, outputs, blocks = 4, 3, 2
    activation_scales = rng.uniform(-0.1, 0.1, size=(rows, blocks)).astype(np.float32)
    activation_values = rng.integers(-127, 128, size=(rows, blocks, 256), dtype=np.int16)
    weight_scales = rng.uniform(-0.2, 0.2, size=(outputs, blocks)).astype(np.float16)
    sub_scales = rng.integers(-128, 128, size=(outputs, blocks, 16), dtype=np.int16)
    codes = rng.integers(-32, 32, size=(outputs, blocks, 256), dtype=np.int16)
    activation_blocks = pack_ggml_q8_k_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q6_k_blocks(weight_scales, sub_scales, codes)
    actual = ggml_q6_k_q8_k_reference(activation_blocks, weight_blocks)
    q8 = activation_scales[..., None] * activation_values.astype(np.float32)
    q6 = np.empty((outputs, blocks, 256), dtype=np.float32)
    for subblock in range(16):
        span = slice(subblock * 16, (subblock + 1) * 16)
        q6[..., span] = (
            weight_scales.astype(np.float32)[..., None]
            * sub_scales[..., subblock, None]
            * codes[..., span]
        )
    expected = np.einsum("rbk,obk->ro", q8, q6, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-3)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII unavailable")
def test_native_q6_k_q8_k_m4_matches_oracle_and_preserves_partition() -> None:
    rng = np.random.default_rng(4654)
    rows, outputs, blocks = 4, 32, 2
    activation_scales = rng.uniform(-0.1, 0.1, size=(rows, blocks)).astype(np.float32)
    activation_values = rng.integers(-127, 128, size=(rows, blocks, 256), dtype=np.int16)
    weight_scales = rng.uniform(-0.1, 0.1, size=(outputs, blocks)).astype(np.float16)
    sub_scales = rng.integers(-128, 128, size=(outputs, blocks, 16), dtype=np.int16)
    codes = rng.integers(-32, 32, size=(outputs, blocks, 256), dtype=np.int16)
    activation_blocks = pack_ggml_q8_k_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q6_k_blocks(weight_scales, sub_scales, codes)
    expected = ggml_q6_k_q8_k_reference(activation_blocks, weight_blocks)
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
            GGML_Q6_K_Q8_K_LINEAR_M4_KERNEL,
            (activation, weight, destination, 0, 16, 16),
            grid=(1, 1, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)
    np.testing.assert_array_equal(actual[:, :16], sentinel)
    np.testing.assert_allclose(actual[:, 16:], expected[:, 16:], rtol=4e-6, atol=8e-4)
