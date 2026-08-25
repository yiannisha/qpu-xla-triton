from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.ggml_geglu_q8 import (
    GGML_GEGLU_Q8_0_KERNEL,
    GGML_GEGLU_Q8_0_SPLIT_KERNEL,
    ggml_geglu_q8_0_reference,
    ggml_gelu_fp16_table,
)
from qpu_xla.kernels.ggml_q4_0 import (
    GGML_Q4_0_Q8_0_LINEAR_KERNEL,
    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
    ggml_q4_0_q8_0_reference,
    pack_ggml_q4_0_blocks,
    pack_ggml_q4_0_tiled_weights,
)


def test_geglu_q8_reference_has_native_block_layout() -> None:
    gate = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(1, 64)
    up = np.linspace(0.5, 1.5, 64, dtype=np.float32).reshape(1, 64)
    packed = ggml_geglu_q8_0_reference(gate, up)
    assert packed.shape == (1, 2, 34)
    assert packed.dtype == np.uint8
    assert np.any(packed[..., 2:].view(np.int8) != 0)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII unavailable")
def test_fused_geglu_q8_matches_reference_and_chains_into_q4_down() -> None:
    rng = np.random.default_rng(4964)
    rows, columns, outputs = 16, 64, 32
    gate_host = rng.normal(0.0, 0.8, size=(rows, columns)).astype(np.float32)
    up_host = rng.normal(0.0, 0.8, size=(rows, columns)).astype(np.float32)
    expected_q8 = ggml_geglu_q8_0_reference(gate_host, up_host)
    weight_scales = rng.uniform(-0.15, 0.15, size=(outputs, columns // 32)).astype(
        np.float16
    )
    weight_codes = rng.integers(
        0, 16, size=(outputs, columns // 32, 32), dtype=np.uint8
    )
    weight_host = pack_ggml_q4_0_blocks(weight_scales, weight_codes)
    expected_down = ggml_q4_0_q8_0_reference(expected_q8, weight_host)
    table_host = ggml_gelu_fp16_table()

    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        gate = device.tensor(gate_host.shape, np.float32)
        up = device.tensor(up_host.shape, np.float32)
        q8 = device.tensor(expected_q8.shape, np.uint8)
        table = device.tensor(table_host.shape, np.uint16)
        weight = device.tensor(weight_host.shape, np.uint8)
        destination = device.tensor(expected_down.shape, np.float32)
        gate.numpy()[:] = gate_host
        up.numpy()[:] = up_host
        table.numpy()[:] = table_host
        weight.numpy()[:] = weight_host
        q8.numpy()[:] = 0xCD
        destination.numpy()[:] = np.nan
        queue.submit(
            GGML_GEGLU_Q8_0_KERNEL,
            (gate, up, q8, table),
            grid=(columns // 32, rows, 1),
        ).wait()
        actual_q8 = np.array(q8.numpy(), copy=True)
        queue.submit(
            GGML_Q4_0_Q8_0_LINEAR_KERNEL,
            (q8, weight, destination, 0, 0, outputs),
            grid=(outputs // 16, rows // 4, 1),
        ).wait()
        actual_down = np.array(destination.numpy(), copy=True)

    np.testing.assert_array_equal(actual_q8, expected_q8)
    np.testing.assert_allclose(actual_down, expected_down, rtol=2e-6, atol=2e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII unavailable")
def test_split_geglu_q8_chains_into_optimized_tiled_q4_down() -> None:
    rng = np.random.default_rng(4965)
    rows, columns, outputs = 16, 64, 32
    gate_host = rng.normal(0.0, 0.8, size=(rows, columns)).astype(np.float32)
    up_host = rng.normal(0.0, 0.8, size=(rows, columns)).astype(np.float32)
    expected_q8 = ggml_geglu_q8_0_reference(gate_host, up_host)
    weight_host = pack_ggml_q4_0_blocks(
        rng.uniform(-0.15, 0.15, size=(outputs, columns // 32)).astype(np.float16),
        rng.integers(0, 16, size=(outputs, columns // 32, 32), dtype=np.uint8),
    )
    weight_scale_host, weight_q_host = pack_ggml_q4_0_tiled_weights(weight_host)
    expected_down = ggml_q4_0_q8_0_reference(expected_q8, weight_host)
    table_host = ggml_gelu_fp16_table()
    blocks = columns // 32

    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        gate = device.tensor(gate_host.shape, np.float32)
        up = device.tensor(up_host.shape, np.float32)
        activation_scales = device.tensor((rows, blocks), np.uint32)
        activation_q = device.tensor((rows, columns), np.uint8)
        table = device.tensor(table_host.shape, np.uint16)
        weight_scales = device.tensor(weight_scale_host.shape, np.uint16)
        weight_q = device.tensor(weight_q_host.shape, np.uint32)
        destination = device.tensor(expected_down.shape, np.float32)
        gate.numpy()[:] = gate_host
        up.numpy()[:] = up_host
        table.numpy()[:] = table_host
        weight_scales.numpy()[:] = weight_scale_host
        weight_q.numpy()[:] = weight_q_host
        queue.submit(
            GGML_GEGLU_Q8_0_SPLIT_KERNEL,
            (gate, up, activation_scales, activation_q, table),
            grid=(blocks, rows, 1),
        ).wait()
        queue.submit(
            GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
            (activation_q, activation_scales, weight_q, weight_scales, destination),
            grid=(outputs // 16, rows // 16, 1),
        ).wait()
        actual_scales = np.array(activation_scales.numpy(), copy=True)
        actual_q = np.array(activation_q.numpy(), copy=True)
        actual_down = np.array(destination.numpy(), copy=True)

    np.testing.assert_array_equal(
        actual_scales.astype(np.uint16),
        expected_q8[..., :2].copy().view(np.uint16).reshape(rows, blocks),
    )
    np.testing.assert_array_equal(
        actual_q.reshape(rows, blocks, 32),
        expected_q8[..., 2:],
    )
    np.testing.assert_allclose(actual_down, expected_down, rtol=2e-6, atol=2e-5)
