#!/usr/bin/env python3
"""Replay an exact captured llama.cpp Q4_0 node through native CPU/QPU candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    utc_now,
    write_json_atomic,
)

OUTPUT_TILE = 16


def aligned_suffix_partitions(output_columns: int) -> list[tuple[int, int]]:
    """Return unique 1/12 through 11/12 QPU suffixes on 16-column boundaries."""
    if output_columns < 2 * OUTPUT_TILE or output_columns % OUTPUT_TILE:
        return []
    tile_count = output_columns // OUTPUT_TILE
    counts: set[int] = set()
    for numerator in range(1, 12):
        qpu_tiles = min(max((tile_count * numerator + 6) // 12, 1), tile_count - 1)
        counts.add(qpu_tiles * OUTPUT_TILE)
    return [(output_columns - count, count) for count in sorted(counts)]


def calculate_error(reference: np.ndarray, actual: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    """Calculate the retained differential and greedy-logit contracts."""
    if reference.shape != actual.shape:
        raise ValueError(f"output shapes differ: {reference.shape} != {actual.shape}")
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    absolute = np.abs(actual64 - reference64)
    relative = absolute / np.maximum(np.abs(reference64), 1e-12)
    threshold = atol + rtol * np.abs(reference64)
    finite = np.isfinite(actual64)
    worst = int(np.nanargmax(absolute)) if absolute.size else 0
    reference_argmax = int(np.argmax(reference))
    actual_argmax = int(np.argmax(actual))
    violation_count = int(np.count_nonzero(~finite | (absolute > threshold)))
    return {
        "elements": int(reference.size),
        "atol": atol,
        "rtol": rtol,
        "max_absolute": float(np.max(absolute)) if absolute.size else 0.0,
        "max_relative": float(np.max(relative)) if relative.size else 0.0,
        "mean_absolute": float(np.mean(absolute)) if absolute.size else 0.0,
        "p99_absolute": float(np.quantile(absolute, 0.99)) if absolute.size else 0.0,
        "tolerance_violation_count": violation_count,
        "nan_count": int(np.count_nonzero(np.isnan(actual64))),
        "inf_count": int(np.count_nonzero(np.isinf(actual64))),
        "worst_index": worst,
        "worst_reference": float(reference[worst]) if reference.size else None,
        "worst_actual": float(actual[worst]) if actual.size else None,
        "bitwise_equal_elements": int(np.count_nonzero(reference.view("<u4") == actual.view("<u4"))),
        "reference_argmax": reference_argmax,
        "reference_argmax_value": float(reference[reference_argmax]),
        "actual_argmax": actual_argmax,
        "actual_argmax_value": float(actual[actual_argmax]),
        "greedy_argmax_matches": reference_argmax == actual_argmax,
        "passed": violation_count == 0 and reference_argmax == actual_argmax,
    }


def materialize_native_weight(fixture: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Copy only the manifest-selected native weight range into a temporary fixture."""
    weight = fixture["weight"]
    model = Path(weight["model"]["path"])
    remaining = int(weight["bytes"])
    with model.open("rb") as source, destination.open("wb") as output:
        source.seek(int(weight["file_offset"]))
        while remaining:
            chunk = source.read(min(remaining, 8 << 20))
            if not chunk:
                raise ValueError("GGUF ended before the complete native weight was read")
            output.write(chunk)
            remaining -= len(chunk)
    return {
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source_model_offset": int(weight["file_offset"]),
        "retained_after_replay": False,
    }


def build_native_command(
    binary: Path,
    fixture: dict[str, Any],
    *,
    weight_path: Path,
    output_path: Path,
    mode: str,
    cpu_threads: int,
    qpu_column_start: int,
    qpu_column_count: int,
) -> list[str]:
    """Build one untimed correctness replay command with every placement explicit."""
    dimensions = fixture["dimensions"]
    return [
        str(binary),
        "--weights",
        str(weight_path),
        "--activation-f32",
        str(fixture["activation_f32"]["path"]),
        "--mode",
        mode,
        "--input-columns",
        str(dimensions["input_columns"]),
        "--output-columns",
        str(dimensions["output_columns"]),
        "--rows",
        str(dimensions["rows"]),
        "--qpu-column-start",
        str(qpu_column_start),
        "--qpu-column-count",
        str(qpu_column_count),
        "--cpu-threads",
        str(cpu_threads),
        "--warmups",
        "0",
        "--samples",
        "1",
        "--output-bin",
        str(output_path),
    ]


def build_cpu_repack_command(
    binary: Path,
    fixture: dict[str, Any],
    *,
    weight_path: Path,
    output_path: Path,
    cpu_threads: int,
) -> list[str]:
    """Build the exact pinned llama.cpp CPU_REPACK graph-node command."""
    dimensions = fixture["dimensions"]
    return [
        str(binary),
        "--weights",
        str(weight_path),
        "--activation-f32",
        str(fixture["activation_f32"]["path"]),
        "--output-bin",
        str(output_path),
        "--input-columns",
        str(dimensions["input_columns"]),
        "--output-columns",
        str(dimensions["output_columns"]),
        "--rows",
        str(dimensions["rows"]),
        "--cpu-threads",
        str(cpu_threads),
        "--warmups",
        "0",
        "--samples",
        "1",
    ]


def _candidate_specs(output_columns: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "cpu-only",
            "mode": "cpu",
            "qpu_column_start": output_columns,
            "qpu_column_count": 0,
        },
        {
            "name": "qpu-only",
            "mode": "qpu",
            "qpu_column_start": 0,
            "qpu_column_count": output_columns,
        },
    ]
    for start, count in aligned_suffix_partitions(output_columns):
        specs.append(
            {
                "name": f"hybrid-qpu-suffix-{count}-of-{output_columns}",
                "mode": "hybrid",
                "qpu_column_start": start,
                "qpu_column_count": count,
            }
        )
    return specs


def main() -> None:
    """Materialize the bounded weight, replay all placements, and retain comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_manifest", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_q4_0_bench",
    )
    parser.add_argument(
        "--cpu-repack-binary",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_cpu_repack_bench",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.fixture_manifest.is_file():
        parser.error(f"fixture manifest not found: {args.fixture_manifest}")
    if not args.benchmark_binary.is_file():
        parser.error(f"native benchmark not found: {args.benchmark_binary}")
    if not args.cpu_repack_binary.is_file():
        parser.error(f"CPU_REPACK benchmark not found: {args.cpu_repack_binary}")
    if args.cpu_threads <= 0:
        parser.error("CPU thread count must be positive")

    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    if fixture.get("kind") != "llama-cpp-qpu-exact-node-fixture":
        parser.error("input is not an exact node fixture manifest")
    for field in ("activation_f32", "reference_output_f32"):
        path = Path(fixture[field]["path"])
        if not path.is_file() or sha256_file(path) != fixture[field]["sha256"]:
            parser.error(f"{field} is missing or its hash has changed")
    model = Path(fixture["weight"]["model"]["path"])
    if not model.is_file() or sha256_file(model) != fixture["weight"]["model"]["sha256"]:
        parser.error("GGUF model is missing or its hash has changed")

    dimensions = fixture["dimensions"]
    output_elements = int(dimensions["rows"]) * int(dimensions["output_columns"])
    reference = np.fromfile(fixture["reference_output_f32"]["path"], dtype="<f4")
    if reference.size != output_elements:
        parser.error("captured reference output has an unexpected element count")
    atol = float(fixture["replay_contract"]["atol"])
    rtol = float(fixture["replay_contract"]["rtol"])
    binary = args.benchmark_binary.resolve()
    cpu_repack_binary = args.cpu_repack_binary.resolve()
    before = collect_environment()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="llama-qpu-node-replay-") as temporary:
        temporary_root = Path(temporary)
        weight_path = temporary_root / "weight.q4_0.bin"
        output_path = temporary_root / "candidate.f32.bin"
        extracted_weight = materialize_native_weight(fixture, weight_path)
        for spec in _candidate_specs(int(dimensions["output_columns"])):
            if spec["mode"] == "cpu":
                command = build_cpu_repack_command(
                    cpu_repack_binary,
                    fixture,
                    weight_path=weight_path,
                    output_path=output_path,
                    cpu_threads=args.cpu_threads,
                )
            else:
                command = build_native_command(
                    binary,
                    fixture,
                    weight_path=weight_path,
                    output_path=output_path,
                    mode=spec["mode"],
                    cpu_threads=args.cpu_threads,
                    qpu_column_start=spec["qpu_column_start"],
                    qpu_column_count=spec["qpu_column_count"],
                )
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            native_result: dict[str, Any] | None = None
            correctness: dict[str, Any] | None = None
            candidate_output: dict[str, Any] | None = None
            if completed.returncode == 0 and output_path.is_file():
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                if lines:
                    native_result = json.loads(lines[-1])
                    actual = np.fromfile(output_path, dtype="<f4")
                    correctness = calculate_error(reference, actual, atol=atol, rtol=rtol)
                    candidate_output = {
                        "bytes": output_path.stat().st_size,
                        "sha256": sha256_file(output_path),
                        "retained_after_replay": False,
                    }
            records.append(
                {
                    **spec,
                    "cpu_threads": args.cpu_threads,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "native_operator_result": native_result,
                    "candidate_output": candidate_output,
                    "production_graph_correctness": correctness,
                }
            )
    after = collect_environment()
    failures = [
        record
        for record in records
        if record["returncode"] != 0
        or record["production_graph_correctness"] is None
        or not record["production_graph_correctness"]["passed"]
    ]
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-exact-node-replay",
        "created_utc": utc_now(),
        "fixture_manifest": {
            "path": str(args.fixture_manifest.resolve()),
            "sha256": sha256_file(args.fixture_manifest),
        },
        "benchmark_binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "cpu_repack_binary": {
            "path": str(cpu_repack_binary),
            "sha256": sha256_file(cpu_repack_binary),
        },
        "extracted_weight": extracted_weight,
        "records": records,
        "environment_before": before,
        "environment_after": after,
        "validation": {
            "all_candidates_passed": not failures,
            "candidate_count": len(records),
            "failure_count": len(failures),
            "timing_eligible_for_promotion": False,
            "timing_ineligibility_reasons": [
                "each subprocess prepares weights and retains only one untimed correctness sample",
                (
                    "QPU/hybrid subprocesses use the direct block-dot only as an internal oracle; "
                    "the separate cpu-only record uses the production CPU_REPACK graph node"
                ),
                "the source graph callback serializes node execution",
            ],
        },
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(records)} candidates, {len(failures)} failures")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
