#!/usr/bin/env python3
"""Build a live Markdown scoreboard from structured QPU candidate records."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate record together with the archive that supplied it."""

    record: dict[str, Any]
    archive: Path
    modified_ns: int


@dataclass(frozen=True, slots=True)
class PackagedKernel:
    """One statically discovered public kernel descriptor."""

    symbol: str
    name: str
    source: Path
    line: int


IMPLEMENTATION_NOTES = {
    "vc7.copy_words": "Linear 32-bit word copy.",
    "vc7.minimum_words": "Elementwise minimum over 32-bit words.",
    "vc7.maximum_words": "Elementwise maximum over 32-bit words.",
    "vc7.tiled_int32_gemm": "Tiled int32 GEMM using signed 24-bit multiplies.",
    "vc7.tiled_fp32_gemm": "Tiled FP32 GEMM.",
    "vc7.fp32_gemv": "Single-row FP32 GEMV for autoregressive decode.",
    "vc7.tiled_w8a8_gemm": "16x16 output-tiled, packed-K4 W8A8 GEMM with int32 accumulation.",
    "vc7.tiled_w8a8_gemm_dequantize": "Native-dot packed-K4 W8A8 GEMM with fused FP32 scaling.",
    "vc7.w8a8_gemv": "Single-row packed-K4 W8A8 GEMV with int32 accumulation.",
    "vc7.w8a8_dequantize": "Int32-to-FP32 W8A8 scale epilogue.",
    "vc7.depthwise_w8a8_3x3": "Direct packed depthwise 3x3 W8A8 convolution with FP32 scaling.",
    "vc7.swiglu_fp32": "Fused FP32 SiLU-gate multiplication.",
    "vc7.rms_norm_fp32": "Row-parallel FP32 RMSNorm with a 16-lane reduction.",
    "vc7.rope_fp32": "FP32 rotary embedding using caller-cached cosine and signed-sine tables.",
    "vc7.softmax_fp32": "Numerically stable row-parallel FP32 softmax.",
    "vc7.residual_add_fp32": "Contiguous vectorized FP32 residual addition.",
    "vc7.embedding_lookup_fp32": "Row-parallel FP32 embedding gather.",
    "vc7.argmax_fp32": "Row-parallel FP32 argmax with first-index tie semantics.",
    "vc7.maxpool2d_int32": "NCHW int32 max-pooling specialization.",
    "vc7.avgpool2d_int32": "NCHW int32 average-pooling specialization.",
    "vc7.maxpool2d_fp32": "NCHW FP32 max-pooling specialization.",
    "vc7.avgpool2d_fp32": "NCHW FP32 average-pooling specialization.",
    "vc7.bias_relu_fp32": "Fused FP32 bias and ReLU epilogue.",
    "vc7.bias_fp32": "FP32 bias epilogue.",
    "vc7.relu_fp32": "FP32 ReLU epilogue.",
    "vc7.bias_relu_int32": "Fused int32 bias and ReLU epilogue.",
    "vc7.bias_int32": "Int32 bias epilogue.",
    "vc7.relu_int32": "Int32 ReLU epilogue.",
}


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root, help="repository root")
    parser.add_argument("--logs", type=Path, help="candidate archive root; defaults to ROOT/experiment_logs")
    parser.add_argument("--output", type=Path, help="Markdown output; defaults to LOGS/KERNEL_SCOREBOARD.md")
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="keep watching and regenerate after changes (minimum interval: 0.25 seconds)",
    )
    return parser.parse_args()


def _candidate_paths(logs: Path) -> tuple[Path, ...]:
    return tuple(sorted(logs.rglob("*.candidates.json"))) if logs.exists() else ()


def _load_candidates(paths: tuple[Path, ...]) -> tuple[list[Candidate], list[str]]:
    newest: dict[str, Candidate] = {}
    warnings: list[str] = []
    for path in paths:
        try:
            modified_ns = path.stat().st_mtime_ns
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped `{path}` while it was unreadable: {exc}")
            continue
        if payload.get("schema_version") not in (1, 2, 3) or not isinstance(payload.get("records"), list):
            warnings.append(f"Skipped `{path}` because it is not a supported candidate archive.")
            continue
        for record in payload["records"]:
            if not isinstance(record, dict) or not isinstance(record.get("name"), str):
                warnings.append(f"Skipped an invalid record in `{path}`.")
                continue
            candidate = Candidate(record, path, modified_ns)
            prior = newest.get(record["name"])
            if prior is None or (candidate.modified_ns, str(candidate.archive)) > (
                prior.modified_ns,
                str(prior.archive),
            ):
                newest[record["name"]] = candidate
    return sorted(newest.values(), key=lambda item: item.record["name"]), warnings


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _discover_packaged_kernels(root: Path) -> tuple[PackagedKernel, ...]:
    kernels: list[PackagedKernel] = []
    for source in sorted((root / "src/qpu_xla/kernels").glob("*.py")):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if (
                not isinstance(value, ast.Call)
                or _call_name(value.func) not in {"Kernel", "_kernel"}
                or not value.args
            ):
                continue
            name_arg = value.args[0]
            if not isinstance(name_arg, ast.Constant) or not isinstance(name_arg.value, str):
                continue
            for symbol in _assignment_names(node):
                if symbol.endswith("_KERNEL"):
                    kernels.append(PackagedKernel(symbol, name_arg.value, source, node.lineno))
    return tuple(sorted(kernels, key=lambda item: item.name))


def _source_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in sorted((root / "src").rglob("*.py")):
        try:
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            continue
    for group in (
        (
            "src/qpu_xla/kernels/gemm_int8.py",
            "src/qpu_xla/kernels/gemv_int8.py",
            "src/qpu_xla/models/tinyllama/quantization.py",
        ),
        (
            "src/qpu_xla/kernels/swiglu.py",
            "src/qpu_xla/ops/swiglu.py",
            "examples/benchmark_qpu_xla_llama_stages.py",
        ),
        (
            "src/qpu_xla/kernels/rms_norm.py",
            "src/qpu_xla/models/tinyllama/functional.py",
            "examples/benchmark_qpu_xla_llama_stages.py",
        ),
        (
            "src/qpu_xla/kernels/rope.py",
            "src/qpu_xla/ops/rope.py",
            "examples/benchmark_qpu_xla_llama_stages.py",
        ),
        (
            "src/qpu_xla/kernels/softmax.py",
            "src/qpu_xla/ops/softmax.py",
            "examples/benchmark_qpu_xla_llama_stages.py",
        ),
    ):
        digest = hashlib.sha256()
        try:
            for relative in group:
                digest.update((root / relative).read_bytes())
        except OSError:
            continue
        hashes.add(digest.hexdigest())
    return hashes


def _underlying_kernels(candidate: Candidate) -> tuple[str, ...]:
    record = candidate.record
    explicit = record.get("kernels")
    if isinstance(explicit, list) and all(isinstance(name, str) for name in explicit):
        return tuple(explicit)
    dtype = str(record.get("dtype", ""))
    shape = str(record.get("shape_class", ""))
    if dtype == "w8a8-i32-fp32":
        try:
            rows = int(shape.split("x", 1)[0])
        except ValueError:
            return ()
        identity = f"{record.get('name', '')} {candidate.archive.name}"
        if rows == 1:
            kernels = ["vc7.w8a8_gemv"]
        elif "fused-qpu-dequant" in identity:
            kernels = ["vc7.tiled_w8a8_gemm_dequantize"]
        else:
            kernels = ["vc7.tiled_w8a8_gemm"]
        if "qpu-epilogue" in identity:
            kernels.append("vc7.w8a8_dequantize")
        return tuple(kernels)
    if record.get("operation") == "swiglu":
        return ("vc7.swiglu_fp32",)
    name = str(record.get("name", ""))
    return (name,) if name.startswith("vc7.") else ()


def _timings(record: dict[str, Any]) -> tuple[float, float, float] | None:
    performance = record.get("performance")
    if not isinstance(performance, dict):
        return None
    cpu = performance.get("cpu_seconds")
    candidate = performance.get("candidate_seconds")
    if not isinstance(cpu, list) or not isinstance(candidate, list) or not cpu or not candidate:
        return None
    try:
        cpu_median = float(median(float(value) for value in cpu))
        candidate_median = float(median(float(value) for value in candidate))
    except (TypeError, ValueError):
        return None
    if cpu_median <= 0 or candidate_median <= 0:
        return None
    return cpu_median, candidate_median, cpu_median / candidate_median


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1_000:.3f}"


def _format_speedup(value: float) -> str:
    return f"{value:.3f}x"


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _correctness(record: dict[str, Any]) -> str:
    evidence = record.get("correctness")
    if not isinstance(evidence, dict):
        return "not recorded"
    maximum = evidence.get("max_abs_error")
    error = f"max error {float(maximum):.3g}" if isinstance(maximum, int | float) else "max error unknown"
    exact = "exact" if evidence.get("exact") is True else "inexact"
    exceptional = int(evidence.get("nan_count", 0)) + int(evidence.get("inf_count", 0))
    return f"{exact}; {error}" + (f"; {exceptional} NaN/Inf" if exceptional else "")


def _candidate_note(candidate: Candidate) -> str:
    record = candidate.record
    kernels = _underlying_kernels(candidate)
    fallback = f"{record.get('operation', 'kernel')} using {record.get('layout', 'unknown layout')}."
    note = IMPLEMENTATION_NOTES.get(kernels[0], fallback) if kernels else fallback
    if "vc7.w8a8_dequantize" in kernels:
        note += " QPU int32-to-FP32 dequantization epilogue."
    elif "cpu-dequant" in str(record.get("name", "")):
        note += " FP32 dequantization runs on the CPU."
    reason = str(record.get("reason", "")).strip()
    return f"{note} {reason}".strip()


def _partition(record: dict[str, Any]) -> str:
    partition = record.get("partition")
    if not isinstance(partition, dict):
        return "-"
    return f"{partition.get('axis', '?')} {partition.get('qpu_units', '?')}/{partition.get('total_units', '?')} QPU"


def _optional_median_ms(performance: dict[str, Any], field: str) -> str:
    values = performance.get(field)
    if not isinstance(values, list) or not values:
        return "-"
    try:
        return _format_ms(float(median(float(value) for value in values)))
    except (TypeError, ValueError):
        return "-"


def _relative_link(target: Path, output: Path, *, line: int | None = None) -> str:
    relative = os.path.relpath(target, output.parent)
    suffix = f"#L{line}" if line is not None else ""
    return f"[{target.name}]({relative}{suffix})"


def _inventory_speedups(kernel: str, candidates: list[Candidate]) -> str:
    speedups = [
        timing[2] for item in candidates if kernel in _underlying_kernels(item) if (timing := _timings(item.record))
    ]
    if not speedups:
        return "not recorded"
    if len(speedups) == 1:
        return _format_speedup(speedups[0])
    return f"{_format_speedup(min(speedups))}-{_format_speedup(max(speedups))} ({len(speedups)} shapes)"


def _render(
    root: Path,
    output: Path,
    archive_paths: tuple[Path, ...],
    candidates: list[Candidate],
    warnings: list[str],
) -> str:
    hashes = _source_hashes(root)
    packaged = _discover_packaged_kernels(root)
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# QPU Kernel Scoreboard",
        "",
        f"Updated automatically at `{generated}` from {len(archive_paths)} candidate archive(s).",
        "",
        "Speedup is `median reference time / median candidate time`; values above 1.0x are faster. "
        "Candidate timing is the archive's recorded comparison timing; current W8A8 archives use steady-state total, "
        "not kernel-only timing.",
        "",
        "## Evaluated candidates",
        "",
        "| Candidate | Placement | Partition | Kernel | Shape | Dtype | Reference | Ref ms | Total ms | "
        "Prep ms | Kernel ms | Speedup | Status | Correctness | Evidence | Implementation |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    if not candidates:
        lines.append("| _No structured candidate records found_ | | | | | | | | | | | | | | | |")
    for item in candidates:
        record = item.record
        timing = _timings(record)
        raw_performance = record.get("performance")
        performance: dict[str, Any] = raw_performance if isinstance(raw_performance, dict) else {}
        source_hash = record.get("source_hash")
        evidence = "current hash" if isinstance(source_hash, str) and source_hash in hashes else "stale/unmatched hash"
        kernel_name = " + ".join(_underlying_kernels(item)) or "unknown"
        ref_ms, candidate_ms, speedup = timing if timing is not None else (None, None, None)
        candidate_cells = (
            f"`{record.get('name', 'unknown')}`",
            record.get("placement", "qpu"),
            _partition(record),
            f"`{kernel_name}`",
            f"`{record.get('shape_class', 'unknown')}`",
            f"`{record.get('dtype', 'unknown')}`",
            performance.get("cpu_reference", "not recorded"),
            _format_ms(ref_ms) if ref_ms is not None else "-",
            _format_ms(candidate_ms) if candidate_ms is not None else "-",
            _optional_median_ms(performance, "host_prep_seconds"),
            _optional_median_ms(performance, "kernel_seconds"),
            _format_speedup(speedup) if speedup is not None else "-",
            record.get("status", "unknown"),
            _correctness(record),
            f"{evidence}; {_relative_link(item.archive, output)}",
            _candidate_note(item),
        )
        lines.append("| " + " | ".join(_escape(cell) for cell in candidate_cells) + " |")

    lines.extend(
        [
            "",
            "## Packaged kernel inventory",
            "",
            "This is a static inventory of public kernel descriptors. A missing speedup means no matching structured "
            "candidate record exists yet.",
            "",
            "| Kernel | Symbol | Recorded speedup | Source | Implementation |",
            "|---|---|---:|---|---|",
        ]
    )
    for packaged_kernel in packaged:
        inventory_cells = (
            f"`{packaged_kernel.name}`",
            f"`{packaged_kernel.symbol}`",
            _inventory_speedups(packaged_kernel.name, candidates),
            _relative_link(packaged_kernel.source, output, line=packaged_kernel.line),
            IMPLEMENTATION_NOTES.get(
                packaged_kernel.name,
                packaged_kernel.name.removeprefix("vc7.").replace("_", " ").capitalize() + ".",
            ),
        )
        lines.append("| " + " | ".join(_escape(cell) for cell in inventory_cells) + " |")
    if not packaged:
        lines.append("| _No packaged kernels discovered_ | | | | |")

    if warnings:
        lines.extend(["", "## Monitor warnings", ""])
        lines.extend(f"- {_escape(warning)}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The evaluated table keeps only the newest archive record for each candidate name. Older archives remain "
            "unchanged as benchmark history. Source-hash matching detects many stale results, but an archive can only "
            "cover files included by its benchmark's hash policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(output: Path, content: str) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output.read_text(encoding="utf-8") == content:
            return False
    except FileNotFoundError:
        pass
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output)
    return True


def _signature(root: Path, logs: Path) -> tuple[tuple[str, int, int], ...]:
    watched = (*_candidate_paths(logs), *((root / "src").rglob("*.py")))
    signature: list[tuple[str, int, int]] = []
    for path in sorted(set(watched)):
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _update(root: Path, logs: Path, output: Path) -> None:
    paths = _candidate_paths(logs)
    candidates, warnings = _load_candidates(paths)
    content = _render(root, output, paths, candidates, warnings)
    changed = _write_atomic(output, content)
    action = "updated" if changed else "checked"
    print(f"{action} {output} ({len(candidates)} evaluated candidates)", flush=True)


def main() -> None:
    """Generate once, or continuously monitor inputs when ``--watch`` is set."""
    args = _parse_args()
    root = args.root.resolve()
    logs = (args.logs or root / "experiment_logs").resolve()
    output = (args.output or logs / "KERNEL_SCOREBOARD.md").resolve()
    _update(root, logs, output)
    if args.watch is None:
        return
    interval = max(args.watch, 0.25)
    previous = _signature(root, logs)
    try:
        while True:
            time.sleep(interval)
            current = _signature(root, logs)
            if current != previous:
                _update(root, logs, output)
                previous = current
    except KeyboardInterrupt:
        print("scoreboard monitor stopped", flush=True)


if __name__ == "__main__":
    main()
