from __future__ import annotations

import argparse
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
import torch.nn.functional as torch_functional

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
from qpu_xla.models.tinyllama import quantize_per_output_channel_int8
from qpu_xla.ops import Conv2dW8A8Plan
from qpu_xla.ops.conv2d import _im2col_nchw, _output_hw
from qpu_xla.workloads import YOLO_DETECTION_V1, YoloConvWorkload


def _samples(fn: Any, *, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    durations: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        durations.append(perf_counter() - start)
    return tuple(durations)


def _dynamic_reference(
    source: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    case: YoloConvWorkload,
) -> npt.NDArray[np.float32]:
    padding = case.kernel // 2
    output_height, output_width = _output_hw(
        case.height,
        case.width,
        case.kernel,
        case.kernel,
        (case.stride, case.stride),
        (padding, padding),
        (1, 1),
    )
    rows = output_height * output_width
    input_group = case.in_channels // case.groups
    output_group = case.out_channels // case.groups
    lowered = _im2col_nchw(
        source,
        case.kernel,
        case.kernel,
        stride=(case.stride, case.stride),
        padding=(padding, padding),
        dilation=(1, 1),
    ).reshape(rows, case.in_channels, case.kernel, case.kernel)
    results: list[npt.NDArray[np.float32]] = []
    for group in range(case.groups):
        source_matrix = lowered[:, group * input_group : (group + 1) * input_group].reshape(rows, -1)
        weight_matrix = np.ascontiguousarray(
            weight[group * output_group : (group + 1) * output_group].reshape(output_group, -1)
        )
        quantized_weight = quantize_per_output_channel_int8(weight_matrix)
        source_scales = np.maximum(
            np.max(np.abs(source_matrix), axis=1) / np.float32(127),
            np.float32(1 / 127),
        )
        quantized_source = np.rint(source_matrix / source_scales[:, None]).clip(-127, 127).astype(np.int32)
        accumulation = quantized_source @ quantized_weight.values.astype(np.int32).T
        result = accumulation.astype(np.float32) * source_scales[:, None] * quantized_weight.scales[None, :]
        results.append(np.asarray(result, dtype=np.float32))
    matrix = np.concatenate(results, axis=1)
    return matrix.reshape(1, output_height, output_width, case.out_channels).transpose(0, 3, 1, 2)


def _numpy_fp32_reference(
    source: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    case: YoloConvWorkload,
) -> npt.NDArray[np.float32]:
    """Run the NumPy lowered-FP32 convolution baseline."""
    padding = case.kernel // 2
    output_height, output_width = _output_hw(
        case.height,
        case.width,
        case.kernel,
        case.kernel,
        (case.stride, case.stride),
        (padding, padding),
        (1, 1),
    )
    rows = output_height * output_width
    input_group = case.in_channels // case.groups
    output_group = case.out_channels // case.groups
    lowered = _im2col_nchw(
        source,
        case.kernel,
        case.kernel,
        stride=(case.stride, case.stride),
        padding=(padding, padding),
        dilation=(1, 1),
    ).reshape(rows, case.in_channels, case.kernel, case.kernel)
    results = []
    for group in range(case.groups):
        source_matrix = lowered[:, group * input_group : (group + 1) * input_group].reshape(rows, -1)
        weight_matrix = weight[group * output_group : (group + 1) * output_group].reshape(output_group, -1)
        results.append(source_matrix @ weight_matrix.T)
    matrix = np.concatenate(results, axis=1)
    return matrix.reshape(1, output_height, output_width, case.out_channels).transpose(0, 3, 1, 2)


def _torch_dynamic_reference(
    source: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    case: YoloConvWorkload,
) -> torch.Tensor:
    """Run the Torch INT32 dynamic-W8A8 lowered convolution baseline."""
    padding = case.kernel // 2
    output_height, output_width = _output_hw(
        case.height,
        case.width,
        case.kernel,
        case.kernel,
        (case.stride, case.stride),
        (padding, padding),
        (1, 1),
    )
    rows = output_height * output_width
    input_group = case.in_channels // case.groups
    output_group = case.out_channels // case.groups
    lowered = _im2col_nchw(
        source,
        case.kernel,
        case.kernel,
        stride=(case.stride, case.stride),
        padding=(padding, padding),
        dilation=(1, 1),
    ).reshape(rows, case.in_channels, case.kernel, case.kernel)
    results = []
    for group in range(case.groups):
        source_matrix = torch.from_numpy(
            np.ascontiguousarray(lowered[:, group * input_group : (group + 1) * input_group].reshape(rows, -1))
        )
        weight_matrix = np.ascontiguousarray(
            weight[group * output_group : (group + 1) * output_group].reshape(output_group, -1)
        )
        quantized_weight = quantize_per_output_channel_int8(weight_matrix)
        source_scales = torch.maximum(
            torch.amax(torch.abs(source_matrix), dim=1) / 127,
            torch.tensor(1 / 127, dtype=torch.float32),
        )
        quantized_source = torch.round(source_matrix / source_scales[:, None]).clamp(-127, 127).to(torch.int32)
        accumulation = quantized_source @ torch.from_numpy(quantized_weight.values.astype(np.int32)).T
        results.append(
            accumulation.to(torch.float32)
            * source_scales[:, None]
            * torch.from_numpy(quantized_weight.scales)[None, :]
        )
    matrix = torch.cat(results, dim=1)
    return matrix.reshape(1, output_height, output_width, case.out_channels).permute(0, 3, 1, 2)


def _correctness(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]) -> CorrectnessEvidence:
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = absolute / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    return CorrectnessEvidence(
        "dynamic-per-row-int8 im2col + numpy-int32-matmul + fp32-scales",
        actual.size,
        bool(np.array_equal(actual, expected)),
        float(np.max(absolute, initial=0.0)),
        float(np.mean(absolute)),
        float(np.percentile(absolute, 99)),
        float(np.max(relative, initial=0.0)),
        nan_count=int(np.count_nonzero(np.isnan(actual))),
        inf_count=int(np.count_nonzero(np.isinf(actual))),
    )


def _quality(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]) -> QualityEvidence:
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    absolute = np.abs(actual64 - expected64)
    rmse = float(np.sqrt(np.mean(np.square(actual64 - expected64))))
    reference_rms = float(np.sqrt(np.mean(np.square(expected64))))
    denominator = float(np.linalg.norm(actual64.ravel()) * np.linalg.norm(expected64.ravel()))
    cosine = 1.0 if denominator == 0.0 else float(np.dot(actual64.ravel(), expected64.ravel()) / denominator)
    return QualityEvidence(
        "torch-native-conv2d-fp32",
        rmse / max(reference_rms, 1e-12),
        max(-1.0, min(1.0, cosine)),
        max_abs_error=float(np.max(absolute, initial=0.0)),
        mean_abs_error=float(np.mean(absolute)),
        p99_abs_error=float(np.percentile(absolute, 99)),
        nan_count=int(np.count_nonzero(np.isnan(actual))),
        inf_count=int(np.count_nonzero(np.isinf(actual))),
    )


def _status(
    correctness: CorrectnessEvidence,
    quality: QualityEvidence,
    performance: PerformanceEvidence,
) -> tuple[CandidateStatus, str]:
    if correctness.max_abs_error > 1e-4 or correctness.nan_count or correctness.inf_count:
        return CandidateStatus.QUARANTINED_INCORRECT, "failed the dynamic-W8A8 differential tolerance"
    if quality.passes_default_gate and performance.measured_win:
        return CandidateStatus.SUPPORTED_WIN, ""
    if performance.same_contract_win:
        return CandidateStatus.SAME_CONTRACT_WIN, "beats dynamic W8A8 CPU but not deployable FP32 CPU"
    return CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER, "correct, but below both 1.05x whole-operation gates"


def _aligned_partitions(total: int, alignment: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                units
                for fraction in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
                if 0 < (units := int(total * fraction) // alignment * alignment) < total
            }
        )
    )


def _source_hash() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in ("src/qpu_xla/kernels/gemm_int8.py", "src/qpu_xla/ops/conv2d_w8a8.py"):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def main() -> None:
    cases = (*YOLO_DETECTION_V1.tuning, *YOLO_DETECTION_V1.holdout)
    parser = argparse.ArgumentParser(description="Benchmark grouped W8A8 YOLO convolution candidates")
    parser.add_argument("--case", default="p3-1x1", choices=[case.name for case in cases])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("yolo-w8a8-benchmark.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.data_area_mib <= 0:
        parser.error("warmup must be non-negative; repeat and data-area-mib must be positive")
    case = next(case for case in cases if case.name == args.case)
    rng = np.random.default_rng(args.seed)
    source_value = rng.standard_normal((1, case.in_channels, case.height, case.width), dtype=np.float32)
    weight_value = np.ascontiguousarray(
        rng.standard_normal(
            (case.out_channels, case.in_channels // case.groups, case.kernel, case.kernel),
            dtype=np.float32,
        )
    )
    padding = case.kernel // 2
    expected = _dynamic_reference(source_value, weight_value, case)
    numpy_fp32_expected = _numpy_fp32_reference(source_value, weight_value, case)
    torch_source = torch.from_numpy(source_value)
    torch_weight = torch.from_numpy(weight_value)

    def torch_native() -> torch.Tensor:
        return torch_functional.conv2d(
            torch_source,
            torch_weight,
            stride=case.stride,
            padding=padding,
            groups=case.groups,
        )

    torch_fp32_seconds = _samples(torch_native, warmup=args.warmup, repeat=args.repeat)
    numpy_fp32_seconds = _samples(
        lambda: _numpy_fp32_reference(source_value, weight_value, case),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    numpy_w8a8_seconds = _samples(
        lambda: _dynamic_reference(source_value, weight_value, case),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    torch_w8a8_seconds = _samples(
        lambda: _torch_dynamic_reference(source_value, weight_value, case),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    same_contract_name, same_contract_seconds = min(
        (
            (f"{numpy_backend_label()}-dynamic-w8a8", numpy_w8a8_seconds),
            ("torch-int32-dynamic-w8a8", torch_w8a8_seconds),
        ),
        key=lambda item: median(item[1]),
    )
    cpu_name, cpu_seconds = min(
        (
            (f"{numpy_backend_label()}-lowered-fp32", numpy_fp32_seconds),
            ("torch-native-conv2d-fp32", torch_fp32_seconds),
        ),
        key=lambda item: median(item[1]),
    )
    torch_output = torch_native().numpy()
    np.testing.assert_allclose(numpy_fp32_expected, torch_output, atol=1e-4, rtol=1e-5)
    shape_class = (
        f"1x{case.in_channels}x{case.height}x{case.width}-"
        f"{case.out_channels}x{case.kernel}x{case.kernel}-s{case.stride}-g{case.groups}"
    )
    common_result: dict[str, object] = {
        "case": asdict(case),
        "shape_class": shape_class,
        "deployable_cpu_reference": cpu_name,
        "same_contract_cpu_reference": same_contract_name,
        "numpy_openblas_dynamic_w8a8_seconds": list(numpy_w8a8_seconds),
        "torch_dynamic_w8a8_seconds": list(torch_w8a8_seconds),
        "numpy_openblas_lowered_fp32_seconds": list(numpy_fp32_seconds),
        "torch_native_fp32_seconds": list(torch_fp32_seconds),
    }
    results: list[dict[str, object]] = []
    records: list[CandidateRecord] = []

    with (
        Device.open(data_area_size=args.data_area_mib * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with Conv2dW8A8Plan(
            device,
            source_shape=cast(tuple[int, int, int, int], source_value.shape),
            weight=weight_value,
            stride=case.stride,
            padding=padding,
            groups=case.groups,
        ) as plan:

            def retain_configuration(
                configuration: str,
                placement: str,
                partition: PartitionEvidence | None,
                run: Any,
            ) -> None:
                qpu_trace_start = len(qpu_queue.chrome_trace()["traceEvents"])
                cpu_trace_start = len(cpu_queue.chrome_trace()["traceEvents"])
                total_seconds = _samples(run, warmup=args.warmup, repeat=args.repeat)
                actual = np.array(destination.numpy(), dtype=np.float32, copy=True)
                correctness = _correctness(actual, expected)
                quality = _quality(actual, torch_output)
                qpu_trace = qpu_queue.chrome_trace()["traceEvents"][qpu_trace_start:]
                cpu_trace = cpu_queue.chrome_trace()["traceEvents"][cpu_trace_start:]
                qpu_events = [event for event in qpu_trace if event.get("ph") == "X" and event.get("cat") == "qpu"]
                launches = 1 if plan._direct_depthwise else case.groups
                measured_qpu_events = qpu_events[-args.repeat * launches :]
                kernel_seconds = tuple(
                    sum(cast(float, event["dur"]) for event in measured_qpu_events[index : index + launches]) / 1e6
                    for index in range(0, len(measured_qpu_events), launches)
                )
                host_events = [
                    event
                    for event in (*qpu_trace, *cpu_trace)
                    if event.get("ph") == "X"
                    and event.get("cat") == "host"
                    and event.get("name") == "conv2d_w8a8.prepare_columns"
                ]
                host_prep_seconds = tuple(cast(float, event["dur"]) / 1e6 for event in host_events[-args.repeat :])
                performance = PerformanceEvidence(
                    cpu_name,
                    cpu_seconds,
                    total_seconds,
                    host_prep_seconds=host_prep_seconds,
                    kernel_seconds=kernel_seconds,
                    same_contract_reference=same_contract_name,
                    same_contract_seconds=same_contract_seconds,
                )
                status, reason = _status(correctness, quality, performance)
                name = f"vc7.w8a8.conv2d.{case.name}.{configuration}"
                kernels = tuple(dict.fromkeys(str(event["name"]) for event in qpu_events))
                records.append(
                    CandidateRecord(
                        name,
                        "conv2d",
                        "w8a8-i32-fp32",
                        "nchw-oihw-packed-k4",
                        shape_class,
                        _source_hash(),
                        status,
                        correctness,
                        performance,
                        reason,
                        placement=placement,
                        kernels=kernels,
                        partition=partition,
                        quality=quality,
                    )
                )
                results.append(
                    {
                        **common_result,
                        "configuration": configuration,
                        "placement": placement,
                        "partition": asdict(partition) if partition is not None else None,
                        "whole_operation_seconds": list(total_seconds),
                        "hybrid_total_seconds": list(total_seconds) if placement == "hybrid" else [],
                        "host_quantize_pack_seconds": list(host_prep_seconds),
                        "kernel_only_seconds": list(kernel_seconds),
                        "dequantization_seconds": [],
                        "speedup_over_deployable_fp32_cpu": performance.speedup,
                        "speedup_over_dynamic_w8a8_cpu": performance.same_contract_speedup,
                        "correctness": asdict(correctness),
                        "quality": asdict(quality),
                        "status": status.value,
                    }
                )
                print(
                    f"{name}: FP32 {median(cpu_seconds) * 1e3:.3f} ms ({cpu_name}), "
                    f"candidate {median(total_seconds) * 1e3:.3f} ms, "
                    f"FP32 speedup {performance.speedup:.3f}x, {status.value}"
                )

            retain_configuration(
                "qpu",
                "qpu",
                None,
                lambda: plan.execute(destination, source, queue=qpu_queue).wait(),
            )
            for qpu_rows in _aligned_partitions(plan.rows, 16):
                retain_configuration(
                    f"hybrid-r{qpu_rows}",
                    "hybrid",
                    PartitionEvidence("rows", qpu_rows, plan.rows, 16),
                    lambda qpu_rows=qpu_rows: plan.execute_hybrid(
                        destination,
                        source,
                        qpu_queue=qpu_queue,
                        cpu_queue=cpu_queue,
                        qpu_rows=qpu_rows,
                    ).wait(),
                )
            output_alignment = plan.output_split_alignment
            if output_alignment is None:
                results.append(
                    {
                        **common_result,
                        "configuration": "hybrid-outputs",
                        "placement": "hybrid",
                        "partition": {"axis": "outputs"},
                        "status": "unsupported",
                        "reason": "per-group output channels do not contain an aligned CPU/QPU split",
                    }
                )
            else:
                for qpu_outputs in _aligned_partitions(case.out_channels, output_alignment):
                    retain_configuration(
                        f"hybrid-o{qpu_outputs}",
                        "hybrid",
                        PartitionEvidence("outputs", qpu_outputs, case.out_channels, output_alignment),
                        lambda qpu_outputs=qpu_outputs: plan.execute_hybrid(
                            destination,
                            source,
                            qpu_queue=qpu_queue,
                            cpu_queue=cpu_queue,
                            qpu_outputs=qpu_outputs,
                        ).wait(),
                    )
        metadata = collect_metadata(
            device,
            extra={
                "manifest": f"{YOLO_DETECTION_V1.name}-v{YOLO_DETECTION_V1.version}",
                "semantics": "steady-state",
                "cpu_reference": "fastest-deployable-fp32-and-fastest-same-contract-w8a8",
                "torch": torch.__version__,
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
            },
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump({"metadata": metadata, "results": results}, output, indent=2, sort_keys=True)
    registry_path = args.output.with_suffix(".candidates.json")
    CandidateRegistry(tuple(records)).save(registry_path)
    print(f"saved {args.output} and {registry_path}")


if __name__ == "__main__":
    main()
