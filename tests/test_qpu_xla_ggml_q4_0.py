from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.ggml_q4_0 import (
    GGML_Q4_0_Q8_0_LINEAR_KERNEL,
    ggml_q4_0_q8_0_reference,
    pack_ggml_q4_0_blocks,
    pack_ggml_q4_0_tiled_weights,
    pack_ggml_q8_0_blocks,
    unpack_ggml_q4_0_blocks,
    unpack_ggml_q8_0_blocks,
)
from qpu_xla.kernels.ggml_q4_0_persistent import (
    qpu_ggml_q4_0_q8_0_tiled_gemm_persistent,
)
from videocore7.driver import Driver


def test_native_block_decoders_cover_extreme_codes_and_scales() -> None:
    q4_scales = np.array([0.0, -0.5, np.float16(2**-14)], dtype=np.float16)
    q4_codes = np.stack(
        (
            np.zeros(32, dtype=np.uint8),
            np.full(32, 15, dtype=np.uint8),
            np.arange(32, dtype=np.uint8) % 16,
        )
    )
    q4_blocks = pack_ggml_q4_0_blocks(q4_scales, q4_codes)
    decoded_scales, decoded_values = unpack_ggml_q4_0_blocks(q4_blocks)
    np.testing.assert_array_equal(decoded_scales, q4_scales.astype(np.float32))
    np.testing.assert_array_equal(decoded_values, q4_codes.astype(np.int8) - 8)

    q8_scales = np.array([0.0, 0.125], dtype=np.float16)
    q8_values = np.stack((np.full(32, -128, np.int16), np.full(32, 127, np.int16)))
    q8_blocks = pack_ggml_q8_0_blocks(q8_scales, q8_values)
    actual_scales, actual_values = unpack_ggml_q8_0_blocks(q8_blocks)
    np.testing.assert_array_equal(actual_scales, q8_scales.astype(np.float32))
    np.testing.assert_array_equal(actual_values, q8_values.astype(np.int8))


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("rows", (4,))
def test_native_q4_0_q8_0_linear_matches_block_oracle_and_preserves_partition(rows: int) -> None:
    rng = np.random.default_rng(4400 + rows)
    blocks = 5
    outputs = 32
    activation_scales = rng.uniform(0.0001, 0.2, size=(rows, blocks)).astype(np.float32)
    activation_values = rng.integers(-128, 128, size=(rows, blocks, 32), dtype=np.int16)
    weight_scales = rng.uniform(-0.2, 0.2, size=(outputs, blocks)).astype(np.float16)
    weight_codes = rng.integers(0, 16, size=(outputs, blocks, 32), dtype=np.uint8)
    activation_blocks = pack_ggml_q8_0_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q4_0_blocks(weight_scales, weight_codes)
    expected = ggml_q4_0_q8_0_reference(activation_blocks, weight_blocks)
    resident_weight_blocks = np.ascontiguousarray(weight_blocks[16:])
    sentinel = np.float32(123456.0)
    data_area_size = activation_blocks.nbytes + resident_weight_blocks.nbytes + expected.nbytes + (1 << 20)

    with Device.open(data_area_size=data_area_size) as device, device.queue() as queue:
        activation = device.tensor(activation_blocks.shape, np.uint8)
        weight = device.tensor(resident_weight_blocks.shape, np.uint8)
        destination = device.tensor(expected.shape, np.float32)
        activation.numpy()[:] = activation_blocks
        weight.numpy()[:] = resident_weight_blocks
        destination.numpy()[:] = sentinel
        queue.submit(
            GGML_Q4_0_Q8_0_LINEAR_KERNEL,
            (activation, weight, destination, 0, 16, 16),
            grid=(1, 1, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_array_equal(actual[:, :16], sentinel)
    np.testing.assert_allclose(actual[:, 16:], expected[:, 16:], rtol=2e-6, atol=2e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_persistent_q4_0_streams_cover_24_threads_and_cross_phase_barrier() -> None:
    rng = np.random.default_rng(4488)
    rows = 16
    blocks = 2
    outputs = 48 * 16
    activation_scales = rng.uniform(0.0001, 0.2, size=(rows, blocks)).astype(np.float16)
    activation_values = rng.integers(-127, 128, size=(rows, blocks, 32), dtype=np.int16)
    weight_scales = rng.uniform(-0.2, 0.2, size=(outputs, blocks)).astype(np.float16)
    weight_codes = rng.integers(0, 16, size=(outputs, blocks, 32), dtype=np.uint8)
    activation_blocks = pack_ggml_q8_0_blocks(activation_scales, activation_values)
    weight_blocks = pack_ggml_q4_0_blocks(weight_scales, weight_codes)
    expected = ggml_q4_0_q8_0_reference(activation_blocks, weight_blocks)
    split_activation_scales = np.ascontiguousarray(activation_blocks[..., :2].view(np.uint16).reshape(rows, blocks))
    split_activation_q = np.ascontiguousarray(activation_blocks[..., 2:].reshape(rows, blocks * 32))
    tiled_weight_scales, tiled_weight_q = pack_ggml_q4_0_tiled_weights(weight_blocks)
    data_area_size = int(
        split_activation_scales.nbytes
        + split_activation_q.nbytes
        + tiled_weight_scales.nbytes
        + tiled_weight_q.nbytes
        + 2 * expected.nbytes
        + (1 << 20)
    )

    with Driver(data_area_size=data_area_size) as drv:
        code = drv.program(qpu_ggml_q4_0_q8_0_tiled_gemm_persistent)
        activation_scale_buffer = drv.alloc(split_activation_scales.shape, dtype=np.uint16)
        activation_q_buffer = drv.alloc(split_activation_q.shape, dtype=np.uint8)
        weight_scale_buffer = drv.alloc(tiled_weight_scales.shape, dtype=np.uint16)
        weight_q_buffer = drv.alloc(tiled_weight_q.shape, dtype=np.uint32)
        destination = drv.alloc(expected.shape, dtype=np.float32)
        destination_second = drv.alloc(expected.shape, dtype=np.float32)
        streams = drv.alloc((24, 51), dtype=np.uint32)
        uniforms = drv.alloc(2, dtype=np.uint32)

        activation_scale_buffer[:] = split_activation_scales
        activation_q_buffer[:] = split_activation_q
        weight_scale_buffer[:] = tiled_weight_scales
        weight_q_buffer[:] = tiled_weight_q
        destination[:] = np.nan
        destination_second[:] = np.nan
        streams[:] = 0
        for thread in range(24):
            streams[thread, 0] = 2
            for phase, phase_destination in enumerate((destination, destination_second)):
                phase_start = 1 + phase * 25
                streams[thread, phase_start] = 2
                for slot in range(2):
                    record_start = phase_start + 1 + slot * 12
                    streams[thread, record_start : record_start + 12] = (
                        thread + slot * 24,
                        activation_q_buffer.strides[0],
                        activation_q_buffer.addresses()[0, 0],
                        weight_q_buffer.strides[0],
                        weight_q_buffer.addresses()[0, 0],
                        phase_destination.strides[0],
                        phase_destination.addresses()[0, 0],
                        blocks,
                        activation_scale_buffer.strides[0],
                        activation_scale_buffer.addresses()[0, 0],
                        weight_scale_buffer.strides[0],
                        weight_scale_buffer.addresses()[0, 0],
                    )
        uniforms[:] = (streams.addresses()[0, 0], streams.strides[0])

        drv.execute(
            code,
            local_invocation=(16, 1, 1),
            uniforms=uniforms.addresses()[0],
            wgs_per_sg=24,
            thread=24,
        )
        actual = np.array(destination, copy=True)
        actual_second = np.array(destination_second, copy=True)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(actual_second, expected, rtol=2e-6, atol=2e-5)
