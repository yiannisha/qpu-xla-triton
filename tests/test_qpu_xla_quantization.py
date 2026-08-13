from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.models.tinyllama import (
    QuantizedMatrixInt8,
    quantize_per_output_channel_int8,
    quantized_linear_fp32,
    quantized_linear_int8_gemm_fp32,
)


def test_per_output_channel_int8_quantization_has_bounded_reconstruction_error() -> None:
    weight = np.array([[0.0, -1.0, 0.25], [2.0, -0.5, 1.0]], dtype=np.float32)
    quantized = quantize_per_output_channel_int8(weight)

    assert quantized.shape == weight.shape
    assert quantized.values.dtype == np.int8
    np.testing.assert_allclose(quantized.dequantize(), weight, atol=0.01, rtol=0.01)


def test_quantized_linear_fp32_matches_its_dequantized_reference() -> None:
    weight = np.array([[0.0, -1.0, 0.25], [2.0, -0.5, 1.0]], dtype=np.float32)
    source_value = np.array([[1.0, 2.0, -3.0], [-2.0, 0.5, 1.5]], dtype=np.float32)
    quantized = quantize_per_output_channel_int8(weight)
    expected = source_value @ quantized.dequantize().T
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        quantized_linear_fp32(destination, source, quantized, queue=queue).wait()
        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


def _dynamic_int8_reference(source: np.ndarray, weight: QuantizedMatrixInt8) -> np.ndarray:
    scales = np.maximum(np.max(np.abs(source), axis=1) / np.float32(127.0), np.float32(1.0 / 127.0))
    quantized_source = np.rint(source / scales[:, None]).clip(-127, 127).astype(np.int32)
    accumulation = quantized_source @ weight.values.astype(np.int32).T
    return accumulation.astype(np.float32) * scales[:, None] * weight.scales[None, :]


def test_quantized_linear_int8_gemm_fp32_matches_dynamic_int8_reference_on_fake_device() -> None:
    source_value = np.array([[0.2, -1.4, 0.5, 0.0, 1.0], [-0.5, 0.25, 2.0, -1.0, 0.75]], dtype=np.float32)
    weight_value = np.array(
        [[0.25, -0.75, 0.5, 0.0, 1.0], [-1.0, 0.5, 0.25, -0.5, 0.75], [0.0, 1.0, -0.25, 0.5, -0.75]],
        dtype=np.float32,
    )
    quantized = quantize_per_output_channel_int8(weight_value)
    expected = _dynamic_int8_reference(source_value, quantized)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value

        quantized_linear_int8_gemm_fp32(destination, source, quantized, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_quantized_linear_int8_gemm_fp32_matches_dynamic_int8_reference_on_qpu() -> None:
    rng = np.random.default_rng(73)
    source_value = rng.standard_normal((2, 5), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((3, 5), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value

        quantized_linear_int8_gemm_fp32(destination, source, quantized, queue=queue).wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)
