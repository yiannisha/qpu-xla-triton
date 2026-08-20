from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from qpu_xla import Device
from qpu_xla.benchmark import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PartitionEvidence,
    PerformanceEvidence,
    QualityEvidence,
    collect_metadata,
    numpy_backend_label,
)
from qpu_xla.kernels import (
    TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
    TILED_W8A8_GEMM_KERNEL,
    W8A8_DEQUANTIZE_KERNEL,
    W8A8_GEMV_KERNEL,
)
from qpu_xla.memory import AccessMode
from qpu_xla.models.tinyllama import (
    PreparedW8A8Linear,
    QuantizedMatrixInt8,
    quantize_per_output_channel_int8,
)
from qpu_xla.workloads import LLAMA_DENSE_V1, LlamaWorkload

SQUARE_CASES = (
    LlamaWorkload("square-64x64x64", "prefill", 64, 64, 64, 1, 1, 64, 64, 64),
    LlamaWorkload("square-512x512x512", "prefill", 512, 512, 512, 8, 8, 64, 512, 512),
)
ALL_CASES = (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout, *SQUARE_CASES)


def _samples(fn: Any, *, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    durations: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        durations.append(perf_counter() - start)
    return tuple(durations)


def _projection_shapes(
    case: LlamaWorkload,
    selection: str,
) -> tuple[tuple[str, int, int, tuple[str, ...]], ...]:
    shapes = {
        "hidden": ("hidden", case.hidden_size, case.hidden_size, ("q", "o")),
        "kv": ("kv", case.hidden_size, case.kv_heads * case.head_dim, ("k", "v")),
        "up": ("up", case.hidden_size, case.intermediate_size, ("gate", "up")),
        "down": ("down", case.intermediate_size, case.hidden_size, ("down",)),
        "lm_head": ("lm_head", case.hidden_size, case.vocabulary_size, ("lm_head",)),
    }
    if selection == "all":
        return tuple(shapes.values())
    return (shapes[selection],)


def _hybrid_output_partitions(outputs: int) -> tuple[int, ...]:
    """Return aligned output-prefix splits that retain non-empty CPU work."""
    return tuple(
        sorted(
            {
                qpu_outputs
                for fraction in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75, 0.875)
                if 0 < (qpu_outputs := int(outputs * fraction) // 16 * 16) < outputs
            }
        )
    )


def _hybrid_row_partitions(rows: int) -> tuple[int, ...]:
    """Return aligned row-prefix splits that retain non-empty CPU work."""
    return tuple(
        sorted(
            {
                qpu_rows
                for fraction in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
                if 0 < (qpu_rows := int(rows * fraction) // 16 * 16) < rows
            }
        )
    )


def _error_metrics(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]) -> CorrectnessEvidence:
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = absolute / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    return CorrectnessEvidence(
        reference="dynamic-per-row-int8 + numpy-int32-matmul + fp32-scales",
        cases=actual.size,
        exact=bool(np.array_equal(actual, expected)),
        max_abs_error=float(np.max(absolute, initial=0.0)),
        mean_abs_error=float(np.mean(absolute)),
        p99_abs_error=float(np.percentile(absolute, 99)),
        max_relative_error=float(np.max(relative, initial=0.0)),
        nan_count=int(np.count_nonzero(np.isnan(actual))),
        inf_count=int(np.count_nonzero(np.isinf(actual))),
    )


def _quality_metrics(
    actual: npt.NDArray[np.float32],
    expected: npt.NDArray[np.float32],
    *,
    logits: bool,
) -> QualityEvidence:
    """Measure quantized output quality against the deployable FP32 result."""
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    finite_actual = np.nan_to_num(actual64)
    finite_expected = np.nan_to_num(expected64)
    absolute = np.abs(actual64 - expected64)
    rmse = float(np.sqrt(np.mean(np.square(finite_actual - finite_expected))))
    reference_rms = float(np.sqrt(np.mean(np.square(finite_expected))))
    denominator = float(np.linalg.norm(finite_actual.ravel()) * np.linalg.norm(finite_expected.ravel()))
    cosine = 1.0 if denominator == 0.0 else float(np.dot(finite_actual.ravel(), finite_expected.ravel()) / denominator)
    top1 = float(np.mean(np.argmax(actual, axis=1) == np.argmax(expected, axis=1))) if logits else None
    return QualityEvidence(
        reference="numpy-openblas-fp32",
        normalized_rmse=rmse / max(reference_rms, 1e-12),
        cosine_similarity=max(-1.0, min(1.0, cosine)),
        max_abs_error=float(np.nanmax(absolute, initial=0.0)),
        mean_abs_error=float(np.nanmean(absolute)),
        p99_abs_error=float(np.nanpercentile(absolute, 99)),
        nan_count=int(np.count_nonzero(np.isnan(actual))),
        inf_count=int(np.count_nonzero(np.isinf(actual))),
        top1_agreement=top1,
    )


def _candidate_status(
    correctness: CorrectnessEvidence,
    quality: QualityEvidence,
    performance: PerformanceEvidence,
) -> tuple[CandidateStatus, str]:
    correct = correctness.nan_count == 0 and correctness.inf_count == 0 and correctness.max_abs_error <= 1e-4
    if not correct:
        return CandidateStatus.QUARANTINED_INCORRECT, "failed the dynamic-W8A8 differential tolerance"
    if quality.passes_default_gate and performance.measured_win:
        return CandidateStatus.SUPPORTED_WIN, ""
    if performance.same_contract_win:
        return CandidateStatus.SAME_CONTRACT_WIN, "beats dynamic W8A8 CPU but not deployable FP32 CPU"
    if not quality.passes_default_gate:
        return CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER, "correct W8A8 execution, but FP32 quality gate failed"
    return CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER, "correct, but below both 1.05x whole-operation gates"


def _source_hash() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in (
        "src/qpu_xla/kernels/gemm_int8.py",
        "src/qpu_xla/kernels/gemv_int8.py",
        "src/qpu_xla/models/tinyllama/quantization.py",
    ):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _automatic_arena_mib(
    case: LlamaWorkload,
    projections: tuple[tuple[str, int, int, tuple[str, ...]], ...],
) -> int:
    """Size the device arena for the largest isolated projection in this job."""
    largest = 0
    for _, inputs, outputs, _ in projections:
        padded_rows = (case.tokens + 15) // 16 * 16
        padded_inputs = (inputs + 15) // 16 * 16
        padded_outputs = (outputs + 15) // 16 * 16
        plan_bytes = (
            padded_rows * padded_inputs
            + padded_inputs * padded_outputs
            + padded_rows * padded_outputs * 4
            + (padded_rows + padded_outputs) * 4
        )
        io_bytes = (case.tokens * inputs + case.tokens * outputs) * 4
        largest = max(largest, plan_bytes + io_bytes)
    return max(16, (largest * 3 // 2 + (1 << 20) - 1) // (1 << 20))


def _benchmark_cpu_int32(
    source_value: npt.NDArray[np.float32],
    weight: QuantizedMatrixInt8,
    *,
    warmup: int,
    repeat: int,
) -> tuple[
    npt.NDArray[np.float32],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Measure prepared INT32 CPU references and release their widened weight."""
    weight_int32 = weight.values.astype(np.int32)

    def quantize_source_numpy() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        source_scales = np.maximum(
            np.max(np.abs(source_value), axis=1) / np.float32(127.0),
            np.float32(1.0 / 127.0),
        )
        quantized_source = np.rint(source_value / source_scales[:, None]).clip(-127, 127).astype(np.int32)
        return source_scales.astype(np.float32, copy=False), quantized_source

    def numpy_reference() -> npt.NDArray[np.float32]:
        source_scales, quantized_source = quantize_source_numpy()
        accumulation = quantized_source @ weight_int32.T
        result = accumulation.astype(np.float32) * source_scales[:, None] * weight.scales[None, :]
        return np.asarray(result, dtype=np.float32)

    torch_source = torch.from_numpy(source_value)
    torch_weight = torch.from_numpy(weight_int32)
    torch_weight_scales = torch.from_numpy(weight.scales)

    def torch_reference() -> torch.Tensor:
        source_scales = torch.maximum(
            torch.amax(torch.abs(torch_source), dim=1) / 127,
            torch.tensor(1.0 / 127.0),
        )
        quantized_source = torch.round(torch_source / source_scales[:, None]).clamp(-127, 127).to(torch.int32)
        accumulation = quantized_source @ torch_weight.T
        return accumulation.to(torch.float32) * source_scales[:, None] * torch_weight_scales[None, :]

    expected = numpy_reference()
    numpy_seconds = _samples(numpy_reference, warmup=warmup, repeat=repeat)
    torch_seconds = _samples(torch_reference, warmup=warmup, repeat=repeat)
    return expected, numpy_seconds, torch_seconds


def _benchmark_cpu_fp32(
    source_value: npt.NDArray[np.float32],
    weight_value: npt.NDArray[np.float32],
    *,
    warmup: int,
    repeat: int,
) -> tuple[npt.NDArray[np.float32], tuple[float, ...], tuple[float, ...]]:
    """Measure NumPy and Torch native FP32 projection references."""
    torch_fp32_source = torch.from_numpy(source_value)
    torch_fp32_weight = torch.from_numpy(weight_value)

    def numpy_fp32_reference() -> npt.NDArray[np.float32]:
        return np.asarray(source_value @ weight_value.T, dtype=np.float32)

    def torch_fp32_reference() -> torch.Tensor:
        return torch_fp32_source @ torch_fp32_weight.T

    numpy_fp32_seconds = _samples(numpy_fp32_reference, warmup=warmup, repeat=repeat)
    torch_fp32_seconds = _samples(torch_fp32_reference, warmup=warmup, repeat=repeat)
    return numpy_fp32_reference(), numpy_fp32_seconds, torch_fp32_seconds


def _run_projection(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    projection: tuple[str, int, int, tuple[str, ...]],
    *,
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
    epilogue: str,
) -> tuple[list[dict[str, object]], list[CandidateRecord]]:
    projection_name, input_features, output_features, aliases = projection
    source_value = rng.standard_normal((case.tokens, input_features), dtype=np.float32)
    weight_value = rng.standard_normal((output_features, input_features), dtype=np.float32)
    weight = quantize_per_output_channel_int8(weight_value)
    fp32_reference, numpy_fp32_seconds, torch_fp32_seconds = _benchmark_cpu_fp32(
        source_value,
        weight_value,
        warmup=warmup,
        repeat=repeat,
    )
    del weight_value
    gc.collect()
    expected, numpy_seconds, torch_seconds = _benchmark_cpu_int32(
        source_value,
        weight,
        warmup=warmup,
        repeat=repeat,
    )
    gc.collect()
    same_contract_name, same_contract_seconds = min(
        (
            (f"{numpy_backend_label()}-int32", numpy_seconds),
            ("torch-int32", torch_seconds),
        ),
        key=lambda item: median(item[1]),
    )
    cpu_name, cpu_seconds = min(
        (
            (f"{numpy_backend_label()}-fp32", numpy_fp32_seconds),
            ("torch-fp32", torch_fp32_seconds),
        ),
        key=lambda item: median(item[1]),
    )

    source = device.tensor(source_value.shape, np.float32)
    destination = device.tensor(expected.shape, np.float32)
    source.numpy()[:] = source_value
    with PreparedW8A8Linear(
        device,
        weight,
        max_batch=case.tokens,
        qpu_dequantize=epilogue != "cpu",
        fuse_dequantize=epilogue == "fused-qpu",
    ) as plan:
        host_prep_seconds = _samples(
            lambda: plan._prepare_values(source_value),
            warmup=warmup,
            repeat=repeat,
        )

        def qpu_total() -> None:
            plan.execute(destination, source, queue=qpu_queue).wait()

        qpu_seconds = _samples(qpu_total, warmup=warmup, repeat=repeat)
        actual = np.array(destination.numpy(), copy=True)

        fused_kernel = (
            epilogue == "fused-qpu"
            and case.tokens > 1
            and case.tokens == plan.padded_batch
            and output_features == plan.padded_outputs
        )
        raw_kernel = (
            TILED_W8A8_GEMM_DEQUANTIZE_KERNEL
            if fused_kernel
            else W8A8_GEMV_KERNEL
            if case.tokens == 1
            else TILED_W8A8_GEMM_KERNEL
        )
        raw_source = plan._gemv_source if case.tokens == 1 else plan._packed_source
        raw_accumulator = plan._gemv_accumulator if case.tokens == 1 else plan._accumulator
        raw_args = (
            (raw_source, plan._packed_weight, plan._row_scales, plan._column_scales, destination)
            if fused_kernel
            else (raw_source, plan._packed_weight, raw_accumulator)
        )

        def qpu_kernel_only() -> None:
            qpu_queue.submit(
                raw_kernel,
                raw_args,
                grid=(plan.padded_outputs // 16, 1 if case.tokens == 1 else plan.padded_batch // 16, 1),
                buffers=(
                    raw_source.access(AccessMode.READ),
                    plan._packed_weight.access(AccessMode.READ),
                    *(
                        (
                            plan._row_scales.access(AccessMode.READ),
                            plan._column_scales.access(AccessMode.READ),
                            destination.access(AccessMode.WRITE),
                        )
                        if fused_kernel
                        else (raw_accumulator.access(AccessMode.WRITE),)
                    ),
                ),
            ).wait()

        kernel_seconds = _samples(qpu_kernel_only, warmup=warmup, repeat=repeat)

        if fused_kernel:
            dequantization_seconds: tuple[float, ...] = ()
        elif epilogue == "standalone-qpu" and case.tokens % 16 == 0 and output_features % 16 == 0:
            active_accumulator = plan._accumulator.slice((slice(0, case.tokens), slice(0, output_features)))
            active_row_scales = plan._row_scales.slice((slice(0, case.tokens),))
            active_column_scales = plan._column_scales.slice((slice(0, output_features),))

            def standalone_dequantize() -> None:
                qpu_queue.submit(
                    W8A8_DEQUANTIZE_KERNEL,
                    (active_accumulator, active_row_scales, active_column_scales, destination),
                    grid=(output_features // 16, case.tokens // 16, 1),
                    buffers=(
                        active_accumulator.access(AccessMode.READ),
                        active_row_scales.access(AccessMode.READ),
                        active_column_scales.access(AccessMode.READ),
                        destination.access(AccessMode.WRITE),
                    ),
                ).wait()

            dequantization_seconds = _samples(standalone_dequantize, warmup=warmup, repeat=repeat)
        else:

            def cpu_dequantize() -> None:
                active = plan._accumulator.numpy()[: case.tokens, :output_features].astype(np.float32)
                destination.numpy()[:] = active * plan._source_scales[: case.tokens, None] * weight.scales[None, :]

            dequantization_seconds = _samples(cpu_dequantize, warmup=warmup, repeat=repeat)

    correctness = _error_metrics(actual, expected)
    quality = _quality_metrics(actual, fp32_reference, logits=projection_name == "lm_head")
    performance = PerformanceEvidence(
        cpu_name,
        cpu_seconds,
        qpu_seconds,
        host_prep_seconds=host_prep_seconds,
        kernel_seconds=kernel_seconds,
        dequantization_seconds=dequantization_seconds,
        same_contract_reference=same_contract_name,
        same_contract_seconds=same_contract_seconds,
    )
    status, reason = _candidate_status(correctness, quality, performance)
    execution_form = "fused-qpu-dequant" if fused_kernel else f"{epilogue}-dequant"
    name = f"vc7.w8a8.{case.name}.{projection_name}.{execution_form}"
    candidate = CandidateRecord(
        name=name,
        operation="linear",
        dtype="w8a8-i32-fp32",
        layout="row-major-packed-k4",
        shape_class=f"{case.tokens}x{input_features}x{output_features}",
        source_hash=_source_hash(),
        status=status,
        correctness=correctness,
        performance=performance,
        reason=reason,
        placement="qpu",
        kernels=(raw_kernel.name,),
        quality=quality,
    )
    result: dict[str, object] = {
        "case": asdict(case),
        "projection": projection_name,
        "aliases": list(aliases),
        "shape": [case.tokens, input_features, output_features],
        "deployable_cpu_reference": cpu_name,
        "same_contract_cpu_reference": same_contract_name,
        "numpy_seconds": list(numpy_seconds),
        "numpy_openblas_int32_seconds": list(numpy_seconds),
        "torch_seconds": list(torch_seconds),
        "numpy_fp32_seconds": list(numpy_fp32_seconds),
        "numpy_openblas_fp32_seconds": list(numpy_fp32_seconds),
        "torch_fp32_seconds": list(torch_fp32_seconds),
        "qpu_total_seconds": list(qpu_seconds),
        "host_quantize_pack_seconds": list(host_prep_seconds),
        "qpu_kernel_only_seconds": list(kernel_seconds),
        "dequantization_seconds": list(dequantization_seconds),
        "speedup_over_deployable_fp32_cpu": performance.speedup,
        "speedup_over_dynamic_w8a8_cpu": performance.same_contract_speedup,
        "correctness": asdict(correctness),
        "quality": asdict(quality),
        "execution_form": execution_form,
        "status": status.value,
        "placement": "qpu",
        "partition": None,
    }
    print(
        f"{name}: CPU {median(cpu_seconds) * 1e3:.3f} ms ({cpu_name}), "
        f"QPU total {median(qpu_seconds) * 1e3:.3f} ms, "
        f"kernel {median(kernel_seconds) * 1e3:.3f} ms, "
        f"FP32 speedup {performance.speedup:.3f}x, W8A8 speedup "
        f"{performance.same_contract_speedup:.3f}x, max error {correctness.max_abs_error:g}, {status.value}"
    )
    results = [result]
    candidates = [candidate]
    with PreparedW8A8Linear(
        device,
        weight,
        max_batch=case.tokens,
        qpu_dequantize=epilogue != "cpu",
        fuse_dequantize=epilogue == "fused-qpu",
    ) as hybrid_plan:

        def record_hybrid(axis: str, qpu_units: int, total_units: int, fn: Any) -> None:
            hybrid_seconds = _samples(fn, warmup=warmup, repeat=repeat)
            hybrid_actual = np.array(destination.numpy(), copy=True)
            hybrid_correctness = _error_metrics(hybrid_actual, expected)
            hybrid_quality = _quality_metrics(
                hybrid_actual,
                fp32_reference,
                logits=projection_name == "lm_head",
            )
            hybrid_performance = PerformanceEvidence(
                cpu_name,
                cpu_seconds,
                hybrid_seconds,
                same_contract_reference=same_contract_name,
                same_contract_seconds=same_contract_seconds,
            )
            hybrid_status, hybrid_reason = _candidate_status(
                hybrid_correctness,
                hybrid_quality,
                hybrid_performance,
            )
            axis_suffix = "r" if axis == "rows" else "o"
            hybrid_name = f"vc7.w8a8.{case.name}.{projection_name}.{execution_form}.hybrid-{axis_suffix}{qpu_units}"
            hybrid_candidate = CandidateRecord(
                name=hybrid_name,
                operation="linear",
                dtype="w8a8-i32-fp32",
                layout="row-major-packed-k4",
                shape_class=f"{case.tokens}x{input_features}x{output_features}",
                source_hash=_source_hash(),
                status=hybrid_status,
                correctness=hybrid_correctness,
                performance=hybrid_performance,
                reason=hybrid_reason,
                placement="hybrid",
                kernels=(raw_kernel.name, "numpy.int32_matmul"),
                partition=PartitionEvidence(axis, qpu_units, total_units, 16),
                quality=hybrid_quality,
            )
            results.append(
                {
                    "case": asdict(case),
                    "projection": projection_name,
                    "aliases": list(aliases),
                    "execution_form": execution_form,
                    "shape": [case.tokens, input_features, output_features],
                    "deployable_cpu_reference": cpu_name,
                    "same_contract_cpu_reference": same_contract_name,
                    "numpy_seconds": list(numpy_seconds),
                    "numpy_openblas_int32_seconds": list(numpy_seconds),
                    "torch_seconds": list(torch_seconds),
                    "numpy_fp32_seconds": list(numpy_fp32_seconds),
                    "numpy_openblas_fp32_seconds": list(numpy_fp32_seconds),
                    "torch_fp32_seconds": list(torch_fp32_seconds),
                    "hybrid_total_seconds": list(hybrid_seconds),
                    "speedup_over_deployable_fp32_cpu": hybrid_performance.speedup,
                    "speedup_over_dynamic_w8a8_cpu": hybrid_performance.same_contract_speedup,
                    "correctness": asdict(hybrid_correctness),
                    "quality": asdict(hybrid_quality),
                    "status": hybrid_status.value,
                    "placement": "hybrid",
                    "partition": {
                        "axis": axis,
                        "qpu_units": qpu_units,
                        "cpu_units": total_units - qpu_units,
                        "total_units": total_units,
                        "alignment": 16,
                    },
                }
            )
            candidates.append(hybrid_candidate)
            print(
                f"{hybrid_name}: CPU {median(cpu_seconds) * 1e3:.3f} ms ({cpu_name}), "
                f"hybrid total {median(hybrid_seconds) * 1e3:.3f} ms, "
                f"FP32 speedup {hybrid_performance.speedup:.3f}x, "
                f"max error {hybrid_correctness.max_abs_error:g}, {hybrid_status.value}"
            )

        for qpu_outputs in _hybrid_output_partitions(output_features):

            def hybrid_outputs(qpu_outputs: int = qpu_outputs) -> None:
                hybrid_plan.execute_hybrid(
                    destination,
                    source,
                    qpu_queue=qpu_queue,
                    cpu_queue=cpu_queue,
                    qpu_outputs=qpu_outputs,
                ).wait()

            record_hybrid("outputs", qpu_outputs, output_features, hybrid_outputs)

        for qpu_rows in _hybrid_row_partitions(case.tokens):

            def hybrid_rows(qpu_rows: int = qpu_rows) -> None:
                hybrid_plan.execute_hybrid_rows(
                    destination,
                    source,
                    qpu_queue=qpu_queue,
                    cpu_queue=cpu_queue,
                    qpu_rows=qpu_rows,
                ).wait()

            record_hybrid("rows", qpu_rows, case.tokens, hybrid_rows)
    source.buffer.close()
    destination.buffer.close()
    return results, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark persistent packed W8A8 Llama projections")
    parser.add_argument(
        "--case",
        default="prefill-h512-t16",
        choices=[case.name for case in ALL_CASES],
    )
    parser.add_argument(
        "--projection",
        choices=("hidden", "kv", "up", "down", "lm_head", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=0, help="0 selects an automatic per-shape arena")
    parser.add_argument(
        "--epilogue",
        choices=("cpu", "standalone-qpu", "fused-qpu"),
        default="cpu",
    )
    parser.add_argument("--qpu-dequantize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("w8a8-benchmark.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.data_area_mib < 0:
        parser.error("warmup and data-area-mib must be non-negative; repeat must be positive")
    case = cast(
        LlamaWorkload,
        next(case for case in ALL_CASES if case.name == args.case),
    )
    if args.qpu_dequantize:
        if args.epilogue != "cpu":
            parser.error("legacy --qpu-dequantize cannot be combined with --epilogue")
        args.epilogue = "fused-qpu"
    projections = (
        (("square", case.hidden_size, case.hidden_size, ("square",)),)
        if case in SQUARE_CASES
        else _projection_shapes(case, args.projection)
    )
    data_area_mib = args.data_area_mib or _automatic_arena_mib(case, projections)
    rng = np.random.default_rng(args.seed)
    registry = CandidateRegistry()
    results: list[dict[str, object]] = []
    with (
        Device.open(data_area_size=data_area_mib * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        metadata = collect_metadata(
            device,
            extra={
                "manifest": f"{LLAMA_DENSE_V1.name}-v{LLAMA_DENSE_V1.version}",
                "semantics": "steady-state",
                "cpu_reference": "fastest-of-torch-and-numpy-blas",
                "torch": torch.__version__,
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
                "data_area_mib": str(data_area_mib),
                "epilogue": args.epilogue,
            },
        )
        for projection in projections:
            projection_results, projection_candidates = _run_projection(
                device,
                qpu_queue,
                cpu_queue,
                case,
                projection,
                warmup=args.warmup,
                repeat=args.repeat,
                rng=rng,
                epilogue=args.epilogue,
            )
            results.extend(projection_results)
            for candidate in projection_candidates:
                registry.register(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump({"metadata": metadata, "results": results}, output, indent=2, sort_keys=True)
    registry_path = args.output.with_suffix(".candidates.json")
    registry.save(registry_path)
    print(f"saved {args.output} and {registry_path}")


if __name__ == "__main__":
    main()
