from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.ggml_q4_0 import pack_ggml_q8_0_blocks
from qpu_xla.kernels.ggml_q8_0 import (
    GGML_Q8_0_Q8_0_LINEAR_M4_KERNEL,
    ggml_q8_0_q8_0_reference,
)


def test_q8_0_q8_0_reference_matches_explicit_dequantized_dot() -> None:
    rng = np.random.default_rng(4850)
    rows, outputs, blocks = 4, 3, 3
    activation_scales = rng.uniform(-0.1, 0.1, size=(rows, blocks)).astype(np.float16)
    activation_values = rng.integers(-128, 128, size=(rows, blocks, 32), dtype=np.int16)
    weight_scales = rng.uniform(-0.2, 0.2, size=(outputs, blocks)).astype(np.float16)
    weight_values = rng.integers(-128, 128, size=(outputs, blocks, 32), dtype=np.int16)
    activation_blocks = pack_ggml_q8_0_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q8_0_blocks(weight_scales, weight_values)
    actual = ggml_q8_0_q8_0_reference(activation_blocks, weight_blocks)
    activation = activation_scales.astype(np.float32)[..., None] * activation_values.astype(
        np.float32
    )
    weights = weight_scales.astype(np.float32)[..., None] * weight_values.astype(np.float32)
    expected = np.einsum("rbk,obk->ro", activation, weights, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-3)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII unavailable")
def test_native_q8_0_q8_0_m4_matches_oracle_and_preserves_partition() -> None:
    rng = np.random.default_rng(4854)
    rows, outputs, blocks = 4, 32, 3
    activation_scales = rng.uniform(-0.1, 0.1, size=(rows, blocks)).astype(np.float16)
    activation_values = rng.integers(-128, 128, size=(rows, blocks, 32), dtype=np.int16)
    weight_scales = rng.uniform(-0.1, 0.1, size=(outputs, blocks)).astype(np.float16)
    weight_values = rng.integers(-128, 128, size=(outputs, blocks, 32), dtype=np.int16)
    activation_blocks = pack_ggml_q8_0_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q8_0_blocks(weight_scales, weight_values)
    expected = ggml_q8_0_q8_0_reference(activation_blocks, weight_blocks)
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
            GGML_Q8_0_Q8_0_LINEAR_M4_KERNEL,
            (activation, weight, destination, 0, 16, 16),
            grid=(1, 1, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)
    np.testing.assert_array_equal(actual[:, :16], sentinel)
    np.testing.assert_allclose(actual[:, 16:], expected[:, 16:], rtol=4e-6, atol=8e-4)
