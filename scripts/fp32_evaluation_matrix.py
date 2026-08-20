#!/usr/bin/env python3
"""Render FP32 projection and post-op JSON logs as one evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} has no results list")
    return [record for record in results if isinstance(record, dict)]


def _median_ms(values: object) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    return float(median(float(value) for value in values)) * 1_000


def _torch_ms(record: dict[str, Any]) -> float | None:
    candidates = [
        _median_ms(record.get("torch_cpu_seconds")),
        _median_ms(record.get("torch_seconds")),
        _median_ms(record.get("torch_complex_seconds")),
        _median_ms(record.get("torch_pairwise_seconds")),
    ]
    backend_seconds = record.get("cpu_backend_seconds")
    if isinstance(backend_seconds, dict):
        candidates.extend(
            _median_ms(samples) for name, samples in backend_seconds.items() if str(name).startswith("torch")
        )
    present = [value for value in candidates if value is not None]
    return min(present) if present else None


def _numpy_ms(record: dict[str, Any]) -> float | None:
    candidates = [
        _median_ms(record.get("numpy_openblas_seconds")),
        _median_ms(record.get("numpy_cpu_seconds")),
    ]
    backend_seconds = record.get("cpu_backend_seconds")
    if isinstance(backend_seconds, dict):
        candidates.extend(
            _median_ms(samples) for name, samples in backend_seconds.items() if str(name).startswith("numpy")
        )
    present = [value for value in candidates if value is not None]
    return min(present) if present else None


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    """Load benchmark logs and write the consolidated Markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    cpu_by_projection: dict[tuple[str, str], tuple[str, float, float | None, float | None]] = {}
    for path in args.inputs:
        for record in _load(path):
            case = record.get("case")
            case_name = case.get("name", "unknown") if isinstance(case, dict) else "unknown"
            projection = record.get("projection")
            if isinstance(projection, str) and record.get("placement") == "qpu":
                numpy_ms = _numpy_ms(record)
                torch_ms = _torch_ms(record)
                cpu_ms = min(value for value in (numpy_ms, torch_ms) if value is not None)
                cpu_by_projection[(case_name, projection)] = (
                    str(record.get("cpu_reference")),
                    cpu_ms,
                    numpy_ms,
                    torch_ms,
                )
            rows.append({**record, "path": path, "case": case_name})

    lines = [
        "# FP32 Evaluation Matrix",
        "",
        "Steady-state whole-operation timings. Speedup is fastest CPU reference divided by candidate time.",
        "",
        "NumPy is explicitly labeled by the BLAS linked into the benchmark environment. For non-BLAS "
        "operators, the NumPy column is the corresponding NumPy implementation from that same build.",
        "",
        "| Case | Operation | Shape | Placement | Partition | NumPy/OpenBLAS ms | Torch ms | "
        "CPU reference | CPU ms | Candidate ms | "
        "Speedup | Max error | Status | Log |",
        "|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for record in sorted(
        rows,
        key=lambda item: (
            str(item["case"]),
            str(item.get("projection", item.get("operation", ""))),
            str(item.get("placement")),
            str(item.get("partition")),
        ),
    ):
        projection = record.get("projection")
        operation = str(projection if projection is not None else record.get("operation", "unknown"))
        shape = record.get("shape", "-")
        shape_text = "x".join(str(value) for value in shape) if isinstance(shape, list) else "-"
        candidate_ms = _median_ms(record.get("candidate_seconds")) or _median_ms(record.get("qpu_seconds"))
        if projection is not None:
            cpu_name, cpu_ms, numpy_ms, torch_ms = cpu_by_projection[(str(record["case"]), str(projection))]
        else:
            cpu_name = str(record.get("cpu_reference", "unknown"))
            cpu_ms = _median_ms(record.get("cpu_seconds"))
            numpy_ms = _numpy_ms(record)
            torch_ms = _torch_ms(record)
        partition = record.get("partition")
        partition_text = "-"
        if isinstance(partition, dict):
            partition_text = f"{partition.get('axis')} {partition.get('qpu_units')}/{partition.get('total_units')}"
        speedup = float(record.get("speedup", record.get("speedup_over_cpu", 0.0)))
        correctness = record.get("correctness")
        nested_error = correctness.get("max_abs_error", 0.0) if isinstance(correctness, dict) else 0.0
        error = float(record.get("max_abs_error", nested_error))
        relative = Path(record["path"]).name
        lines.append(
            f"| {record['case']} | {operation} | {shape_text} | {record.get('placement')} | {partition_text} | "
            f"{_fmt_ms(numpy_ms)} | {_fmt_ms(torch_ms)} | {cpu_name} | {cpu_ms:.3f} | "
            f"{candidate_ms:.3f} | {speedup:.3f}x | {error:.3g} | "
            f"{record.get('status')} | [{relative}]({relative}) |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
