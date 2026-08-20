#!/usr/bin/env python3
"""Validate W8A8 archives and render evaluation, backend, scoreboard, and registry outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any, cast

from qpu_xla.benchmark import CandidateRecord, CandidateRegistry, CandidateStatus
from qpu_xla.workloads import LLAMA_DENSE_V1, YOLO_DETECTION_V1

DENSE_PROJECTIONS = ("hidden", "kv", "up", "down", "lm_head")
SQUARE_CASES = ("square-64x64x64", "square-512x512x512")
FP32_MATRIX = "../20260819-qpu-xla-kernel-suite/FP32_EVALUATION_MATRIX.md"


def _load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"{path} is missing benchmark metadata")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path} is missing retained configuration results")
    if not all(isinstance(result, dict) for result in results):
        raise ValueError(f"{path} contains a non-object result")
    return cast(dict[str, Any], payload["metadata"]), cast(list[dict[str, Any]], results)


def _samples(record: dict[str, Any], key: str, *, required: bool = True) -> list[float]:
    raw = record.get(key)
    if not isinstance(raw, list) or (required and not raw):
        if required:
            raise ValueError(f"result is missing raw {key} samples")
        return []
    values = [float(value) for value in raw]
    if any(value <= 0 for value in values):
        raise ValueError(f"result has invalid {key} samples")
    return values


def _validate_result(record: dict[str, Any]) -> None:
    if record.get("status") == "unsupported":
        if not record.get("reason") or not isinstance(record.get("partition"), dict):
            raise ValueError("unsupported configurations require a partition and reason")
        return
    for key in ("correctness", "quality", "placement"):
        if key not in record:
            raise ValueError(f"result is missing {key}")
    correctness = record["correctness"]
    quality = record["quality"]
    if not isinstance(correctness, dict) or not isinstance(quality, dict):
        raise ValueError("correctness and quality evidence must be objects")
    for key in ("max_abs_error", "mean_abs_error", "p99_abs_error", "nan_count", "inf_count"):
        if key not in quality:
            raise ValueError(f"quality evidence is missing {key}")
    for key in ("normalized_rmse", "cosine_similarity"):
        if key not in quality:
            raise ValueError(f"quality evidence is missing {key}")
    if record.get("projection") == "lm_head" and quality.get("top1_agreement") is None:
        raise ValueError("logit quality evidence is missing top1_agreement")
    if "projection" in record:
        for key in (
            "numpy_openblas_int32_seconds",
            "torch_seconds",
            "numpy_openblas_fp32_seconds",
            "torch_fp32_seconds",
        ):
            _samples(record, key)
        if record.get("placement") == "qpu":
            _samples(record, "qpu_total_seconds")
            _samples(record, "host_quantize_pack_seconds")
            _samples(record, "qpu_kernel_only_seconds")
            _samples(record, "dequantization_seconds", required=False)
        else:
            _samples(record, "hybrid_total_seconds")
    else:
        for key in (
            "numpy_openblas_dynamic_w8a8_seconds",
            "torch_dynamic_w8a8_seconds",
            "numpy_openblas_lowered_fp32_seconds",
            "torch_native_fp32_seconds",
        ):
            _samples(record, key)
        _samples(record, "whole_operation_seconds")
        _samples(record, "host_quantize_pack_seconds")
        _samples(record, "kernel_only_seconds")
        _samples(record, "dequantization_seconds", required=False)


def _case_name(record: dict[str, Any]) -> str:
    case = record.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("name"), str):
        raise ValueError("result is missing its workload case")
    return str(case["name"])


def _operation(record: dict[str, Any]) -> str:
    projection = record.get("projection")
    return str(projection) if isinstance(projection, str) else "conv2d"


def _whole_samples(record: dict[str, Any]) -> list[float]:
    for key in ("qpu_total_seconds", "hybrid_total_seconds", "whole_operation_seconds"):
        values = record.get(key)
        if isinstance(values, list) and values:
            return [float(value) for value in values]
    return []


def _fmt_ms(samples: list[float]) -> str:
    return "-" if not samples else f"{median(samples) * 1_000:.3f}"


def _fmt(value: object, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _partition(record: dict[str, Any]) -> str:
    partition = record.get("partition")
    if not isinstance(partition, dict) or not partition.get("qpu_units"):
        return "-" if record.get("status") != "unsupported" else f"{partition.get('axis')} unsupported"
    return f"{partition['axis']} {partition['qpu_units']}/{partition.get('total_units', '?')}"


def _validate_coverage(rows: list[dict[str, Any]]) -> None:
    dense_cases = [case.name for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)]
    yolo_cases = [case.name for case in (*YOLO_DETECTION_V1.tuning, *YOLO_DETECTION_V1.holdout)]
    available = {(_case_name(row), _operation(row)) for row in rows}
    missing = {
        *(f"{case}/{projection}" for case in dense_cases for projection in DENSE_PROJECTIONS),
        *(f"{case}/square" for case in SQUARE_CASES),
        *(f"{case}/conv2d" for case in yolo_cases),
    } - {f"{case}/{operation}" for case, operation in available}
    if missing:
        raise ValueError(f"evaluation matrix is missing manifest shapes: {', '.join(sorted(missing))}")
    dense_shapes = (
        *((case, projection) for case in dense_cases for projection in DENSE_PROJECTIONS),
        *((case, "square") for case in SQUARE_CASES),
    )
    for case, projection in dense_shapes:
        group = [row for row in rows if _case_name(row) == case and _operation(row) == projection]
        epilogues = {str(row.get("execution_form")) for row in group if row.get("placement") == "qpu"}
        required = {"cpu-dequant", "standalone-qpu-dequant", "fused-qpu-dequant"}
        if epilogues != required:
            raise ValueError(f"{case}/{projection} does not contain all dequantization forms")
        axes = ("rows", "outputs") if projection == "square" else ("outputs",)
        for axis in axes:
            if not any(
                isinstance(row.get("partition"), dict) and row["partition"].get("axis") == axis for row in group
            ):
                raise ValueError(f"{case}/{projection} has no {axis} partition")
    for case in yolo_cases:
        group = [row for row in rows if _case_name(row) == case]
        axes = {row["partition"].get("axis") for row in group if isinstance(row.get("partition"), dict)}
        if not {"rows", "outputs"} <= axes:
            raise ValueError(f"{case} is missing row or explicit output partition evidence")


def _runtime_outcomes(logs: Path) -> dict[str, tuple[float, bool]]:
    outcomes: dict[str, tuple[float, bool]] = {}
    for path in sorted(logs.glob("runtime-*-l1.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = str(payload["case"]["name"])
        timings = payload["timings"]
        speedup = float(timings.get("speedup_over_fp32_cpu", timings.get("speedup_over_fp32_numpy", 0.0)))
        quality = payload["correctness"]["qpu_vs_fp32"]
        safe = (
            float(quality.get("normalized_rmse", float("inf"))) <= 0.10
            and float(quality.get("cosine_similarity", -1.0)) >= 0.99
            and int(quality.get("nan_count", 1)) == 0
            and int(quality.get("inf_count", 1)) == 0
        )
        outcomes[case] = (speedup, safe)
    return outcomes


def _combine_registry(logs: Path) -> CandidateRegistry:
    records: list[CandidateRecord] = []
    names: set[str] = set()
    for path in sorted(logs.glob("*.candidates.json")):
        if path.name == "w8a8-calibrated.candidates.json":
            continue
        for record in CandidateRegistry.load(path).records:
            if record.name in names:
                raise ValueError(f"duplicate candidate name {record.name!r} in {path}")
            names.add(record.name)
            records.append(record)
    outcomes = _runtime_outcomes(logs)
    revised: list[CandidateRecord] = []
    dense_cases = [case.name for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)]
    for record in records:
        matched_case = next((case for case in dense_cases if case in record.name), None)
        outcome = outcomes.get(matched_case) if matched_case is not None else None
        if record.status is CandidateStatus.SUPPORTED_WIN and outcome is not None:
            speedup, safe = outcome
            if speedup < 1.05 or not safe:
                record = replace(
                    record,
                    status=CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER,
                    reason=(
                        f"standalone winner demoted by one-layer regression: "
                        f"{speedup:.3f}x FP32 speedup, quality_safe={safe}"
                    ),
                )
        revised.append(record)
    return CandidateRegistry(tuple(revised))


def _render_evaluation(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# W8A8 Evaluation Matrix",
        "",
        "Steady-state whole-operation evidence. `same-contract-win` compares against the fastest dynamic-W8A8 "
        "CPU backend; `supported-win` compares against the fastest deployable FP32 backend and is the only AUTO gate.",
        "",
        f"FP32-only RMSNorm, RoPE, softmax, SwiGLU, residual, embedding, sampling, and attention evidence is in "
        f"the [FP32 evaluation matrix]({FP32_MATRIX}).",
        "",
        "| Case | Operation | Aliases/config | Placement | Partition | Candidate ms | FP32 speedup | W8A8 speedup | "
        "NRMSE | Cosine | Max/P99 error | Status | Log |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            _case_name(item),
            _operation(item),
            str(item.get("placement")),
            _partition(item),
            str(item.get("execution_form", item.get("configuration", ""))),
        ),
    ):
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        alias = row.get("aliases", row.get("configuration", row.get("execution_form", "-")))
        alias_text = "/".join(alias) if isinstance(alias, list) else str(alias)
        if isinstance(alias, list) and row.get("execution_form"):
            alias_text = f"{alias_text}; {row['execution_form']}"
        max_p99 = (
            f"{float(quality.get('max_abs_error', 0.0)):.3g}/{float(quality.get('p99_abs_error', 0.0)):.3g}"
            if quality
            else "-"
        )
        lines.append(
            f"| {_case_name(row)} | {_operation(row)} | {alias_text} | {row.get('placement')} | "
            f"{_partition(row)} | {_fmt_ms(_whole_samples(row))} | "
            f"{_fmt(row.get('speedup_over_deployable_fp32_cpu'))}x | "
            f"{_fmt(row.get('speedup_over_dynamic_w8a8_cpu'))}x | "
            f"{_fmt(quality.get('normalized_rmse'), 5)} | {_fmt(quality.get('cosine_similarity'), 5)} | "
            f"{max_p99} | {row.get('status')} | [{row['_path'].name}]({row['_path'].name}) |"
        )
    return "\n".join(lines) + "\n"


def _render_backend(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# W8A8 Backend Latency Matrix",
        "",
        "Best measured QPU and hybrid whole-operation configuration per exact shape.",
        "",
        "| Case | Operation | Best QPU | QPU ms | Best hybrid | Hybrid ms | Fastest | FP32 speedup | Gate |",
        "|---|---|---|---:|---|---:|---|---:|---|",
    ]
    keys = sorted({(_case_name(row), _operation(row)) for row in rows})
    for case, operation in keys:
        group = [
            row for row in rows if _case_name(row) == case and _operation(row) == operation and _whole_samples(row)
        ]
        qpu = min(
            (row for row in group if row.get("placement") == "qpu"),
            key=lambda row: median(_whole_samples(row)),
            default=None,
        )
        hybrid = min(
            (row for row in group if row.get("placement") == "hybrid"),
            key=lambda row: median(_whole_samples(row)),
            default=None,
        )
        fastest = min(
            (row for row in (qpu, hybrid) if row is not None),
            key=lambda row: median(_whole_samples(row)),
            default=None,
        )
        qpu_name = str(qpu.get("execution_form", qpu.get("configuration", "qpu"))) if qpu else "-"
        hybrid_name = _partition(hybrid) if hybrid else "-"
        lines.append(
            f"| {case} | {operation} | {qpu_name} | {_fmt_ms(_whole_samples(qpu)) if qpu else '-'} | "
            f"{hybrid_name} | {_fmt_ms(_whole_samples(hybrid)) if hybrid else '-'} | "
            f"{fastest.get('placement') if fastest else '-'} | "
            f"{_fmt(fastest.get('speedup_over_deployable_fp32_cpu') if fastest else None)}x | "
            f"{fastest.get('status') if fastest else '-'} |"
        )
    return "\n".join(lines) + "\n"


def _render_scoreboard(registry: CandidateRegistry) -> str:
    lines = [
        "# W8A8 Kernel Scoreboard",
        "",
        "Only exact-shape `supported-win` records are visible to AUTO placement.",
        "",
        "| Candidate | Shape | Placement | Partition | FP32 speedup | W8A8 speedup | Quality | Status |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for record in sorted(registry.records, key=lambda candidate: candidate.name):
        performance = record.performance
        quality = record.quality
        partition = (
            "-"
            if record.partition is None
            else f"{record.partition.axis} {record.partition.qpu_units}/{record.partition.total_units}"
        )
        quality_text = (
            "-" if quality is None else f"NRMSE {quality.normalized_rmse:.4f}, cos {quality.cosine_similarity:.4f}"
        )
        lines.append(
            f"| `{record.name}` | `{record.shape_class}` | {record.placement} | {partition} | "
            f"{performance.speedup:.3f}x | "
            f"{_fmt(performance.same_contract_speedup)}x | {quality_text} | {record.status.value} |"
            if performance is not None
            else f"| `{record.name}` | `{record.shape_class}` | {record.placement} | {partition} | - | - | "
            f"{quality_text} | {record.status.value} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Validate one log root and regenerate all derived W8A8 artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    logs = args.logs.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(logs.glob("*.json")):
        if path.name.endswith(".candidates.json") or path.name.startswith("runtime-"):
            continue
        _, results = _load_results(path)
        for result in results:
            _validate_result(result)
            rows.append({**result, "_path": path})
    if not rows:
        raise ValueError(f"{logs} contains no W8A8 benchmark results")
    if args.require_complete:
        _validate_coverage(rows)
    registry = _combine_registry(logs)
    (logs / "W8A8_EVALUATION_MATRIX.md").write_text(_render_evaluation(rows), encoding="utf-8")
    (logs / "W8A8_BACKEND_LATENCY_MATRIX.md").write_text(_render_backend(rows), encoding="utf-8")
    (logs / "W8A8_KERNEL_SCOREBOARD.md").write_text(_render_scoreboard(registry), encoding="utf-8")
    registry.save(logs / "w8a8-calibrated.candidates.json")
    print(f"rendered {len(rows)} configurations and {len(registry.records)} candidates in {logs}")


if __name__ == "__main__":
    main()
