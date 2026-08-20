#!/usr/bin/env python3
"""Build a compact all-kernel latency matrix from retained benchmark JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class Row:
    """One exact FP32 operation shape summarized across execution backends."""

    case: str
    operation: str
    kernel: str
    shape: str
    numpy_ms: float | None
    torch_ms: float | None
    runtime_cpu_ms: float | None
    qpu_ms: float | None
    hybrid_ms: float | None
    hybrid_configuration: str
    max_error: float


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _median_ms(values: object) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    return float(median(float(value) for value in values)) * 1_000


def _minimum(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _shape(values: object) -> str:
    return "×".join(str(value) for value in values) if isinstance(values, list) else "-"


def _partition(record: dict[str, Any]) -> str:
    partition = record.get("partition")
    if not isinstance(partition, dict):
        return "-"
    axis = partition.get("axis", "units")
    qpu = partition.get("qpu_units", "?")
    total = partition.get("total_units")
    if total is None and isinstance(partition.get("cpu_units"), int) and isinstance(qpu, int):
        total = qpu + int(partition["cpu_units"])
    return f"{axis} {qpu}/{total if total is not None else '?'} QPU"


def _torch_ms(record: dict[str, Any]) -> float | None:
    values = [
        _median_ms(record.get("torch_cpu_seconds")),
        _median_ms(record.get("torch_seconds")),
        _median_ms(record.get("torch_complex_seconds")),
        _median_ms(record.get("torch_pairwise_seconds")),
    ]
    backend = record.get("cpu_backend_seconds")
    if isinstance(backend, dict):
        values.extend(_median_ms(samples) for name, samples in backend.items() if str(name).startswith("torch"))
    return _minimum(values)


def _numpy_ms(record: dict[str, Any]) -> float | None:
    values = [
        _median_ms(record.get("numpy_openblas_seconds")),
        _median_ms(record.get("numpy_cpu_seconds")),
    ]
    backend = record.get("cpu_backend_seconds")
    if isinstance(backend, dict):
        values.extend(_median_ms(samples) for name, samples in backend.items() if str(name).startswith("numpy"))
    return _minimum(values)


def _error(record: dict[str, Any]) -> float:
    correctness = record.get("correctness")
    nested = correctness.get("max_abs_error", 0.0) if isinstance(correctness, dict) else 0.0
    return float(record.get("max_abs_error", nested))


def _candidate_ms(record: dict[str, Any]) -> float | None:
    return _minimum(
        [
            _median_ms(record.get("candidate_seconds")),
            _median_ms(record.get("qpu_seconds")),
        ]
    )


def _projection_rows(logs: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(logs.glob("*-fp32-linear.json")):
        records = _read(path)["results"]
        keys = sorted({(record["case"]["name"], record["projection"]) for record in records})
        for case, operation in keys:
            group = [
                record for record in records if record["case"]["name"] == case and record["projection"] == operation
            ]
            qpu = next(record for record in group if record["placement"] == "qpu")
            hybrids = [record for record in group if record["placement"] == "hybrid"]
            best_hybrid = min(hybrids, key=lambda record: _candidate_ms(record) or float("inf"), default=None)
            shape = qpu["shape"]
            kernel = "vc7.fp32_gemv" if shape[0] == 1 else "vc7.tiled_fp32_gemm"
            rows.append(
                Row(
                    case,
                    operation,
                    kernel,
                    _shape(shape),
                    _numpy_ms(qpu),
                    _torch_ms(qpu),
                    None,
                    _candidate_ms(qpu),
                    _candidate_ms(best_hybrid) if best_hybrid is not None else None,
                    _partition(best_hybrid) if best_hybrid is not None else "-",
                    max(_error(record) for record in group),
                )
            )
    return rows


def _stage_rows(logs: Path) -> list[Row]:
    kernels = {
        "rms-norm": "vc7.rms_norm_fp32",
        "rope": "vc7.rope_fp32",
        "softmax": "vc7.softmax_fp32",
        "swiglu": "vc7.swiglu_fp32",
    }
    rows: list[Row] = []
    for path in sorted(logs.glob("*-stages.json")):
        records = _read(path)["results"]
        for operation in sorted({record["operation"] for record in records}):
            group = [record for record in records if record["operation"] == operation]
            qpu = next(record for record in group if record["placement"] == "qpu")
            hybrids = [record for record in group if record["placement"] == "hybrid"]
            best_hybrid = min(hybrids, key=lambda record: _candidate_ms(record) or float("inf"), default=None)
            rows.append(
                Row(
                    qpu["case"]["name"],
                    operation,
                    kernels[operation],
                    _shape(qpu["shape"]),
                    _numpy_ms(qpu),
                    _torch_ms(qpu),
                    None,
                    _candidate_ms(qpu),
                    _candidate_ms(best_hybrid) if best_hybrid is not None else None,
                    _partition(best_hybrid) if best_hybrid is not None else "-",
                    max(_error(record) for record in group),
                )
            )
    return rows


def _post_rows(logs: Path) -> list[Row]:
    path = logs / "prefill-h512-t16-post-ops.json"
    payload = _read(path)
    case = payload["case"]
    shapes = {
        "argmax": [case["tokens"], case["vocabulary_size"]],
        "embedding": [case["tokens"], case["hidden_size"]],
        "kv_append": [case["tokens"], case["kv_heads"] * case["head_dim"]],
        "residual_add": [case["tokens"], case["hidden_size"]],
    }
    kernels = {
        "argmax": "vc7.argmax_fp32",
        "embedding": "vc7.embedding_lookup_fp32",
        "kv_append": "vc7.copy_words",
        "residual_add": "vc7.residual_add_fp32",
    }
    rows: list[Row] = []
    records = payload["results"]
    for operation in sorted(shapes):
        group = [record for record in records if record["operation"] == operation]
        qpu = next(record for record in group if record["placement"] == "qpu")
        hybrids = [record for record in group if record["placement"] == "hybrid"]
        best_hybrid = min(hybrids, key=lambda record: _candidate_ms(record) or float("inf"), default=None)
        rows.append(
            Row(
                case["name"],
                operation,
                kernels[operation],
                _shape(shapes[operation]),
                _numpy_ms(qpu),
                _torch_ms(qpu),
                None,
                _candidate_ms(qpu),
                _candidate_ms(best_hybrid) if best_hybrid is not None else None,
                _partition(best_hybrid) if best_hybrid is not None else "-",
                max(_error(record) for record in group),
            )
        )
    return rows


def _matrix_rows(logs: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(logs.glob("fp32-matmul-attention-s*.json")):
        records = _read(path)["results"]
        for operation in sorted({record["operation"] for record in records}):
            group = [record for record in records if record["operation"] == operation]

            def timing(prefix: str) -> float | None:
                match = next((record for record in group if str(record["configuration"]).startswith(prefix)), None)
                return None if match is None else float(match["median_seconds"]) * 1_000

            hybrids = [record for record in group if str(record["configuration"]).startswith("CPU+QPU")]
            best_hybrid = min(hybrids, key=lambda record: float(record["median_seconds"]), default=None)
            qpu_prefix = "QPU only" if operation == "matmul" else "QPU-only staged"
            shape = group[0]["shape"]
            rows.append(
                Row(
                    path.stem.removeprefix("fp32-matmul-attention-"),
                    operation,
                    ("vc7.tiled_fp32_gemm" if operation == "matmul" else "vc7.tiled_fp32_gemm + vc7.softmax_fp32"),
                    _shape(shape),
                    timing("numpy-"),
                    timing("torch.") if operation == "matmul" else timing("Torch"),
                    timing("qpu_xla CPU"),
                    timing(qpu_prefix),
                    None if best_hybrid is None else float(best_hybrid["median_seconds"]) * 1_000,
                    "-" if best_hybrid is None else str(best_hybrid["configuration"]),
                    max(float(record["max_abs_error"]) for record in group),
                )
            )
    return rows


def _pool_rows(logs: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(logs.glob("fp32-pool2d*.json")):
        payload = _read(path)
        case = payload["case"]
        for record in payload["results"]:
            mode = str(record["mode"])
            shape = f"{_shape(record['input_shape'])} → {_shape(record['output_shape'])}"
            rows.append(
                Row(
                    str(case["name"]),
                    f"{mode}pool2d",
                    f"vc7.{mode}pool2d_fp32",
                    shape,
                    _median_ms(record.get("numpy_cpu_seconds")),
                    _median_ms(record.get("torch_cpu_seconds")),
                    None,
                    _median_ms(record.get("qpu_seconds")),
                    None,
                    "-",
                    _error(record),
                )
            )
    return rows


def _epilogue_rows(logs: Path) -> list[Row]:
    kernels = {
        "bias": "vc7.bias_fp32",
        "relu": "vc7.relu_fp32",
        "bias_relu": "vc7.bias_relu_fp32",
    }
    rows: list[Row] = []
    for path in sorted(logs.glob("fp32-epilogues*.json")):
        payload = _read(path)
        for record in payload["results"]:
            operation = str(record["operation"])
            rows.append(
                Row(
                    str(payload["case"]["name"]),
                    operation,
                    kernels[operation],
                    _shape(record["shape"]),
                    _median_ms(record.get("numpy_cpu_seconds")),
                    _median_ms(record.get("torch_cpu_seconds")),
                    None,
                    _median_ms(record.get("qpu_seconds")),
                    None,
                    "-",
                    _error(record),
                )
            )
    return rows


def _minmax_rows(logs: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(logs.glob("fp32-minmax*.json")):
        payload = _read(path)
        for record in payload["results"]:
            operation = str(record["operation"])
            rows.append(
                Row(
                    str(payload["case"]["name"]),
                    operation,
                    f"vc7.{operation}_words",
                    _shape(record["shape"]),
                    _median_ms(record.get("numpy_cpu_seconds")),
                    _median_ms(record.get("torch_cpu_seconds")),
                    None,
                    _median_ms(record.get("qpu_seconds")),
                    None,
                    "-",
                    _error(record),
                )
            )
    return rows


def _copy_rows(logs: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(logs.glob("fp32-copy*.json")):
        payload = _read(path)
        for record in payload["results"]:
            rows.append(
                Row(
                    str(payload["case"]["name"]),
                    "copy",
                    "vc7.copy_words",
                    _shape(record["shape"]),
                    _median_ms(record.get("numpy_cpu_seconds")),
                    _median_ms(record.get("torch_cpu_seconds")),
                    None,
                    _median_ms(record.get("qpu_seconds")),
                    None,
                    "-",
                    _error(record),
                )
            )
    return rows


def _residual_rows(logs: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(logs.glob("fp32-residual*.json")):
        payload = _read(path)
        for record in payload["results"]:
            rows.append(
                Row(
                    str(payload["case"]["name"]),
                    "residual_add",
                    "vc7.residual_add_fp32",
                    _shape(record["shape"]),
                    _median_ms(record.get("numpy_cpu_seconds")),
                    _median_ms(record.get("torch_cpu_seconds")),
                    None,
                    _median_ms(record.get("qpu_seconds")),
                    None,
                    "-",
                    _error(record),
                )
            )
    return rows


def _winner(row: Row) -> tuple[str, float | None]:
    candidates = {
        "NumPy/OpenBLAS": row.numpy_ms,
        "Torch": row.torch_ms,
        "qpu_xla CPU": row.runtime_cpu_ms,
        "QPU": row.qpu_ms,
        "CPU/QPU": row.hybrid_ms,
    }
    present = {name: value for name, value in candidates.items() if value is not None}
    return min(present.items(), key=lambda item: item[1]) if present else ("-", None)


def _render_fp32(rows: list[Row]) -> list[str]:
    lines = [
        "## FP32 backend matrix",
        "",
        "| Case | Operation | QPU kernel | Shape | NumPy/OpenBLAS ms | Torch ms | qpu_xla CPU ms | "
        "QPU-only ms | Best CPU/QPU ms | Best split | Fastest | Best acceleration | Promotion | Max error |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.case, item.operation, item.shape)):
        winner, latency = _winner(row)
        cpu_ms = _minimum([row.numpy_ms, row.torch_ms])
        accelerated_ms = _minimum([row.qpu_ms, row.hybrid_ms])
        speedup = None if cpu_ms is None or accelerated_ms is None else cpu_ms / accelerated_ms
        promotion = "supported-win" if speedup is not None and speedup >= 1.05 else "not promoted"
        lines.append(
            f"| {row.case} | {row.operation} | `{row.kernel}` | `{row.shape}` | {_fmt(row.numpy_ms)} | "
            f"{_fmt(row.torch_ms)} | {_fmt(row.runtime_cpu_ms)} | {_fmt(row.qpu_ms)} | "
            f"{_fmt(row.hybrid_ms)} | {row.hybrid_configuration} | {winner} ({_fmt(latency)} ms) | "
            f"{'-' if speedup is None else f'{speedup:.3f}×'} | {promotion} | {row.max_error:.3g} |"
        )
    return lines


def _render_w8a8(logs: Path, yolo_logs: Path) -> list[str]:
    dense = _read(logs / "prefill-h512-t16-w8a8-hidden.json")["results"]
    qpu = next(record for record in dense if record["placement"] == "qpu")
    hybrid = min(
        (record for record in dense if record["placement"] == "hybrid"),
        key=lambda record: _median_ms(record["candidate_total_seconds"]) or float("inf"),
    )
    lines = [
        "## W8A8 backend matrix",
        "",
        "Dense W8A8 reports both the same-quantized-contract CPU path and the deployable FP32 CPU alternative.",
        "",
        "| Case | Operation/kernel | Shape | NumPy W8A8 ms | Torch W8A8 ms | NumPy FP32 ms | Torch FP32 ms | "
        "QPU-only ms | QPU kernel-only ms | Best CPU/QPU ms | Best split | Fastest deployable backend |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        f"| prefill-h512-t16 | hidden linear / `vc7.tiled_w8a8_gemm_dequantize` | `16×512×512` | "
        f"{_fmt(_median_ms(qpu['numpy_openblas_int32_seconds']))} | {_fmt(_median_ms(qpu['torch_seconds']))} | "
        f"{_fmt(_median_ms(qpu['numpy_openblas_fp32_seconds']))} | {_fmt(_median_ms(qpu['torch_fp32_seconds']))} | "
        f"{_fmt(_median_ms(qpu['qpu_total_seconds']))} | {_fmt(_median_ms(qpu['qpu_kernel_only_seconds']))} | "
        f"{_fmt(_median_ms(hybrid['candidate_total_seconds']))} | {_partition(hybrid)} | NumPy/OpenBLAS FP32 |",
        "",
        "| Case | Operation/kernel | Shape | NumPy dynamic W8A8 ms | Torch native FP32 ms | QPU prepared total ms | "
        "QPU execute-event ms | QPU host-prep-event ms | Fastest backend | Max error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for path in sorted(yolo_logs.glob("yolo-*.json")):
        if path.name.endswith(".candidates.json"):
            continue
        payload = _read(path)
        case = payload["case"]
        shape = (
            f"1×{case['in_channels']}×{case['height']}×{case['width']} → {case['out_channels']}×"
            f"{case['kernel']}×{case['kernel']}, s{case['stride']}, g{case['groups']}"
        )
        numpy_ms = _median_ms(payload["numpy_openblas_dynamic_w8a8_seconds"])
        torch_ms = _median_ms(payload["torch_native_seconds"])
        qpu_ms = _median_ms(payload["qpu_prep_cached_total_seconds"])
        fastest = min(
            (("NumPy W8A8", numpy_ms), ("Torch FP32", torch_ms), ("QPU", qpu_ms)),
            key=lambda item: float("inf") if item[1] is None else item[1],
        )[0]
        lines.append(
            f"| {case['name']} | conv2d / `vc7.tiled_w8a8_gemm` | `{shape}` | {_fmt(numpy_ms)} | "
            f"{_fmt(torch_ms)} | {_fmt(qpu_ms)} | {_fmt(_median_ms(payload['qpu_execute_event_seconds']))} | "
            f"{_fmt(_median_ms(payload['qpu_host_prep_event_seconds']))} | {fastest} | "
            f"{float(payload['correctness']['max_abs_error']):.3g} |"
        )
    return lines


def _render_unmeasured() -> list[str]:
    kernels = (
        ("vc7.avgpool2d_int32", "INT32", "pooling benchmark not included in the current rerun"),
        ("vc7.maxpool2d_int32", "INT32", "pooling benchmark not included in the current rerun"),
        ("vc7.bias_int32", "INT32", "standalone epilogue not included"),
        ("vc7.bias_relu_int32", "INT32", "standalone epilogue not included"),
        ("vc7.relu_int32", "INT32", "standalone epilogue not included"),
        ("vc7.tiled_int32_gemm", "INT32", "legacy result was not rerun in this backend matrix"),
        ("vc7.w8a8_dequantize", "W8A8→FP32", "only measured fused into dense GEMM"),
        ("vc7.w8a8_gemv", "W8A8→INT32", "decode W8A8 was not rerun in the current matrix"),
    )
    lines = [
        "## Packaged kernels without current comparable latency",
        "",
        "| Kernel | Dtype | Current gap |",
        "|---|---|---|",
    ]
    lines.extend(f"| `{kernel}` | {dtype} | {reason} |" for kernel, dtype, reason in kernels)
    return lines


def main() -> None:
    """Render the consolidated Markdown latency matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--yolo-logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fp32_rows = [
        *_projection_rows(args.logs),
        *_stage_rows(args.logs),
        *_post_rows(args.logs),
        *_matrix_rows(args.logs),
        *_pool_rows(args.logs),
        *_epilogue_rows(args.logs),
        *_minmax_rows(args.logs),
        *_copy_rows(args.logs),
        *_residual_rows(args.logs),
    ]
    lines = [
        "# Kernel Backend Latency Matrix",
        "",
        "Current packaged qpu_xla kernel-suite results. Median steady-state whole-operation latency in milliseconds. "
        "CPU measurements use four threads. "
        "NumPy is linked to OpenBLAS 0.3.28; non-matrix NumPy operations do not necessarily invoke BLAS. "
        "QPU-only and CPU/QPU columns include submission and synchronization unless explicitly labeled kernel-only.",
        "",
        "The FP32 table shows the fastest measured hybrid partition while the exhaustive per-partition records "
        "remain in "
        "[FP32_EVALUATION_MATRIX.md](FP32_EVALUATION_MATRIX.md).",
        "",
        *_render_fp32(fp32_rows),
        "",
        *_render_w8a8(args.logs, args.yolo_logs),
        "",
        *_render_unmeasured(),
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"updated {args.output} ({len(fp32_rows)} FP32 operation/shape rows)")


if __name__ == "__main__":
    main()
