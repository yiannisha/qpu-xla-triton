from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.benchmark import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PartitionEvidence,
    PerformanceEvidence,
)
from qpu_xla.models.tinyllama import (
    CalibratedW8A8Linear,
    PreparedW8A8Linear,
    QuantizedMatrixInt8,
    quantize_per_output_channel_int8,
    quantized_linear_fp32,
    quantized_linear_int8_gemm_fp32,
)
from qpu_xla.scheduler import Placement


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


def test_prepared_w8a8_linear_rejects_a_cpu_only_backend() -> None:
    quantized = quantize_per_output_channel_int8(np.ones((16, 16), dtype=np.float32))
    with Device.fake() as device, pytest.raises(ValueError, match="VideoCore VII"):
        PreparedW8A8Linear(device, quantized, max_batch=16)


def test_calibrated_w8a8_auto_uses_cpu_when_qpu_backend_is_unavailable() -> None:
    source_value = np.arange(32, dtype=np.float32).reshape(2, 16) / 7
    quantized = quantize_per_output_channel_int8(np.arange(256, dtype=np.float32).reshape(16, 16) / 31)
    expected = _dynamic_int8_reference(source_value, quantized)
    winning = CandidateRecord(
        "measured-2x16x16",
        "linear",
        "w8a8-i32-fp32",
        "row-major-packed-k4",
        "2x16x16",
        "abc",
        CandidateStatus.SUPPORTED_WIN,
        CorrectnessEvidence("numpy", 1, True, 0.0, 0.0, 0.0, 0.0),
        PerformanceEvidence("torch", (0.002,), (0.001,)),
    )
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with CalibratedW8A8Linear(
            device,
            quantized,
            max_batch=2,
            candidates=CandidateRegistry((winning,)),
        ) as plan:
            event = plan.execute(destination, source, queue=queue)
            event.wait()
            assert event.name == "w8a8_linear.cpu_reference"
            np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)
            with pytest.raises(ValueError, match="supported-win"):
                plan.execute(destination, source, queue=queue, placement=Placement.QPU)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_calibrated_w8a8_auto_dispatches_an_exact_supported_shape_to_qpu() -> None:
    rng = np.random.default_rng(1921)
    source_value = rng.standard_normal((16, 16), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((16, 16), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    winning = CandidateRecord(
        "measured-16x16x16",
        "linear",
        "w8a8-i32-fp32",
        "row-major-packed-k4",
        "16x16x16",
        "abc",
        CandidateStatus.SUPPORTED_WIN,
        CorrectnessEvidence("numpy", 1, True, 0.0, 0.0, 0.0, 0.0),
        PerformanceEvidence("torch", (0.002,), (0.001,)),
    )
    with Device.open(data_area_size=2 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with CalibratedW8A8Linear(
            device,
            quantized,
            max_batch=16,
            candidates=CandidateRegistry((winning,)),
        ) as plan:
            event = plan.execute(destination, source, queue=queue)
            event.wait()
            actual = np.array(destination.numpy(), copy=True)
            assert event.name == "vc7.tiled_w8a8_gemm_dequantize"

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_prepared_w8a8_linear_matches_dynamic_int8_reference_on_qpu() -> None:
    rng = np.random.default_rng(731)
    source_value = rng.standard_normal((17, 20), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((19, 20), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    with Device.open(data_area_size=2 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value

        with PreparedW8A8Linear(device, quantized, max_batch=32) as plan:
            plan.execute(destination, source, queue=queue).wait()
            actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("batch", (1, 17))
def test_prepared_w8a8_hybrid_output_split_matches_dynamic_reference_on_qpu(batch: int) -> None:
    rng = np.random.default_rng(801 + batch)
    source_value = rng.standard_normal((batch, 32), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((48, 32), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    with (
        Device.open(data_area_size=2 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value

        with PreparedW8A8Linear(device, quantized, max_batch=max(16, batch)) as plan:
            event = plan.execute_hybrid(
                destination,
                source,
                qpu_queue=qpu_queue,
                cpu_queue=cpu_queue,
                qpu_outputs=16,
            )
            event.wait()
            actual = np.array(destination.numpy(), copy=True)
            assert event.name == "w8a8_linear.hybrid_join"

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_prepared_w8a8_hybrid_row_split_matches_dynamic_reference_on_qpu() -> None:
    rng = np.random.default_rng(1907)
    source_value = rng.standard_normal((31, 20), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((19, 20), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    with (
        Device.open(data_area_size=2 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value

        with PreparedW8A8Linear(device, quantized, max_batch=32) as plan:
            event = plan.execute_hybrid_rows(
                destination,
                source,
                qpu_queue=qpu_queue,
                cpu_queue=cpu_queue,
                qpu_rows=16,
            )
            event.wait()
            actual = np.array(destination.numpy(), copy=True)
            assert event.name == "w8a8_linear.hybrid_join"

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_calibrated_w8a8_rejects_ambiguous_hybrid_partition_axes() -> None:
    quantized = quantize_per_output_channel_int8(np.ones((32, 16), dtype=np.float32))
    with (
        Device.fake() as device,
        device.queue() as queue,
        CalibratedW8A8Linear(
            device,
            quantized,
            max_batch=32,
            candidates=CandidateRegistry(),
        ) as plan,
    ):
        source = device.tensor((32, 16), np.float32)
        destination = device.tensor((32, 32), np.float32)
        with pytest.raises(ValueError, match="mutually exclusive"):
            plan.execute(
                destination,
                source,
                queue=queue,
                placement=Placement.AUTO,
                qpu_rows=16,
                qpu_outputs=16,
            )


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_calibrated_w8a8_auto_selects_the_fastest_exact_partition_axis() -> None:
    rng = np.random.default_rng(2921)
    source_value = rng.standard_normal((32, 16), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((32, 16), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    correctness = CorrectnessEvidence("numpy", 1, True, 0.0, 0.0, 0.0, 0.0)
    records = (
        CandidateRecord(
            "qpu",
            "linear",
            "w8a8-i32-fp32",
            "row-major-packed-k4",
            "32x16x32",
            "hash",
            CandidateStatus.SUPPORTED_WIN,
            correctness,
            PerformanceEvidence("torch-fp32", (0.010,), (0.004,)),
        ),
        CandidateRecord(
            "hybrid-rows",
            "linear",
            "w8a8-i32-fp32",
            "row-major-packed-k4",
            "32x16x32",
            "hash",
            CandidateStatus.SUPPORTED_WIN,
            correctness,
            PerformanceEvidence("torch-fp32", (0.010,), (0.001,)),
            placement="hybrid",
            partition=PartitionEvidence("rows", 16, 32, 16),
        ),
        CandidateRecord(
            "hybrid-outputs",
            "linear",
            "w8a8-i32-fp32",
            "row-major-packed-k4",
            "32x16x32",
            "hash",
            CandidateStatus.SUPPORTED_WIN,
            correctness,
            PerformanceEvidence("torch-fp32", (0.010,), (0.002,)),
            placement="hybrid",
            partition=PartitionEvidence("outputs", 16, 32, 16),
        ),
    )
    with (
        Device.open(data_area_size=2 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with CalibratedW8A8Linear(
            device,
            quantized,
            max_batch=32,
            candidates=CandidateRegistry(records),
        ) as plan:
            plan.execute(destination, source, queue=qpu_queue, cpu_queue=cpu_queue).wait()
            np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)
            assert plan._qpu is not None and plan._qpu._hybrid_weight_full is not None
            assert plan._qpu._hybrid_weight_tail is None

            plan.execute(
                destination,
                source,
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                qpu_outputs=16,
            ).wait()
            np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)
            assert plan._qpu._hybrid_weight_tail is not None


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_prepared_w8a8_linear_decode_uses_exact_gemv_path() -> None:
    rng = np.random.default_rng(1731)
    source_value = rng.standard_normal((1, 512), dtype=np.float32)
    quantized = quantize_per_output_channel_int8(rng.standard_normal((512, 512), dtype=np.float32))
    expected = _dynamic_int8_reference(source_value, quantized)
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value

        with PreparedW8A8Linear(device, quantized, max_batch=1) as plan:
            plan.execute(destination, source, queue=queue).wait()
            actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
