#!/usr/bin/env python3
"""Evaluate the exact GGML GEGLU QPU prefill candidate over Gemma shapes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
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


def integer_list(value: str) -> tuple[int, ...]:
    """Parse a duplicate-free comma-separated positive integer list."""
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {value!r}") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer lists must be positive and duplicate-free")
    return result


def bootstrap_speedup(
    baseline: list[int],
    candidate: list[int],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    """Bootstrap the ratio between independent baseline and candidate medians."""
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        baseline_median = np.median(
            rng.choice(baseline_values, baseline_values.size, replace=True)
        )
        candidate_median = np.median(
            rng.choice(candidate_values, candidate_values.size, replace=True)
        )
        ratios[index] = baseline_median / candidate_median
    return {
        "resamples": resamples,
        "seed": seed,
        "median_speedup": float(median(baseline) / median(candidate)),
        "bootstrap_95_low": float(np.quantile(ratios, 0.025)),
        "bootstrap_95_high": float(np.quantile(ratios, 0.975)),
    }


def parse_benchmark_stdout(stdout: str) -> dict[str, Any]:
    """Return the one structured inline-GEGLU benchmark record."""
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("kind") == "qpu-ggml-inline-samples":
            records.append(value)
    if len(records) != 1:
        raise RuntimeError(f"expected one GEGLU result record, found {len(records)}")
    return records[0]


def main() -> None:
    """Run the sampled shape grid and retain correctness and speedup evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_ggml_inline_smoke",
    )
    parser.add_argument(
        "--inline-hook",
        "--plugin",
        dest="inline_hook",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/libggml-qpu-inline.so",
    )
    parser.add_argument(
        "--rows",
        type=integer_list,
        default=integer_list("1,2,4,8,16,17,32,33,64,128,129,256,257"),
    )
    parser.add_argument("--columns", type=integer_list, default=integer_list("6144,12288"))
    parser.add_argument("--cpu-threads", type=integer_list, default=integer_list("1,2,3,4"))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.warmups = 2
        args.samples = 5
        args.bootstrap_resamples = 500
    if args.warmups < 0 or args.samples <= 0 or args.bootstrap_resamples <= 0:
        parser.error("warmups must be non-negative and sample counts must be positive")
    for path in (args.benchmark, args.inline_hook):
        if not path.is_file():
            parser.error(f"required artifact does not exist: {path}")

    before = collect_environment()
    results: list[dict[str, Any]] = []
    raw_commands: list[dict[str, Any]] = []
    calibration_rows = set(args.rows[::2])
    for columns in args.columns:
        for rows in args.rows:
            if rows * columns % 768:
                continue
            for threads in args.cpu_threads:
                command = [
                    str(args.benchmark.resolve()),
                    "--rows",
                    str(rows),
                    "--columns",
                    str(columns),
                    "--cpu-threads",
                    str(threads),
                    "--warmups",
                    str(args.warmups),
                    "--samples",
                    str(args.samples),
                ]
                environment = os.environ.copy()
                environment.update(
                    {
                        "LD_PRELOAD": str(args.inline_hook.resolve()),
                        "GGML_QPU_TELEMETRY": "0",
                        "GGML_QPU_CPU_INLINE_FORCE_MULTITHREAD": "1",
                    }
                )
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    env=environment,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"GEGLU benchmark failed ({completed.returncode}): {' '.join(command)}\n"
                        f"{completed.stderr}"
                    )
                record = parse_benchmark_stdout(completed.stdout)
                total_interval = bootstrap_speedup(
                    record["cpu_complete_ns"],
                    record["qpu_inline_complete_ns"],
                    seed=args.seed + rows * 101 + columns * 7 + threads,
                    resamples=args.bootstrap_resamples,
                )
                correct = bool(record["bitwise_exact"] and record["passed"])
                measured_win = (
                    correct
                    and total_interval["median_speedup"] >= 1.05
                    and total_interval["bootstrap_95_low"] > 1.0
                )
                results.append(
                    {
                        "phase": "calibration" if rows in calibration_rows else "heldout",
                        "shape": {"m": rows, "n": columns, "elements": rows * columns},
                        "cpu_threads": threads,
                        "correctness": {
                            "max_absolute_error": record["max_absolute_error"],
                            "bitwise_equal": record["bitwise_exact"],
                            "dispatch_count_verified": record["passed"],
                        },
                        "cpu_complete_ns": record["cpu_complete_ns"],
                        "qpu_inline_complete_ns": record["qpu_inline_complete_ns"],
                        "same_boundary_speedup": total_interval,
                        "status": "supported-operator-win" if measured_win else "correct-slower",
                    }
                )
                raw_commands.append(
                    {
                        "command": command,
                        "environment": {
                            "LD_PRELOAD": str(args.inline_hook.resolve()),
                            "GGML_QPU_TELEMETRY": "0",
                            "GGML_QPU_CPU_INLINE_FORCE_MULTITHREAD": "1",
                        },
                        "returncode": completed.returncode,
                        "stderr": completed.stderr,
                    }
                )
    after = collect_environment()
    gemma_layer_counts = {6144: 15, 12288: 20}
    model_region_results: list[dict[str, Any]] = []
    for rows in args.rows:
        for threads in args.cpu_threads:
            cells = {
                result["shape"]["n"]: result
                for result in results
                if result["shape"]["m"] == rows and result["cpu_threads"] == threads
            }
            if set(cells) != set(gemma_layer_counts):
                continue
            retained_samples = min(
                len(cells[columns]["cpu_complete_ns"])
                for columns in gemma_layer_counts
            )
            cpu_region_ns = [
                sum(
                    cells[columns]["cpu_complete_ns"][index] * layer_count
                    for columns, layer_count in gemma_layer_counts.items()
                )
                for index in range(retained_samples)
            ]
            qpu_region_ns = [
                sum(
                    cells[columns]["qpu_inline_complete_ns"][index] * layer_count
                    for columns, layer_count in gemma_layer_counts.items()
                )
                for index in range(retained_samples)
            ]
            interval = bootstrap_speedup(
                cpu_region_ns,
                qpu_region_ns,
                seed=args.seed + rows * 107 + threads,
                resamples=args.bootstrap_resamples,
            )
            correct = all(cells[columns]["correctness"]["bitwise_equal"] for columns in cells)
            measured_win = (
                correct
                and interval["median_speedup"] >= 1.05
                and interval["bootstrap_95_low"] > 1.0
            )
            model_region_results.append(
                {
                    "phase": "calibration" if rows in calibration_rows else "heldout",
                    "m": rows,
                    "cpu_threads": threads,
                    "layer_counts_by_ffn_columns": gemma_layer_counts,
                    "cpu_geglu_region_ns": cpu_region_ns,
                    "qpu_inline_geglu_region_ns": qpu_region_ns,
                    "same_boundary_speedup": interval,
                    "bitwise_exact": correct,
                    "status": "supported-model-region-win" if measured_win else "correct-slower",
                }
            )
    winning_shapes = [
        {
            **result["shape"],
            "cpu_threads": result["cpu_threads"],
            "phase": result["phase"],
            "median_speedup": result["same_boundary_speedup"]["median_speedup"],
            "bootstrap_95_low": result["same_boundary_speedup"]["bootstrap_95_low"],
        }
        for result in results
        if result["status"] == "supported-operator-win"
    ]
    winning_model_regions = [
        {
            "m": result["m"],
            "cpu_threads": result["cpu_threads"],
            "phase": result["phase"],
            "median_speedup": result["same_boundary_speedup"]["median_speedup"],
            "bootstrap_95_low": result["same_boundary_speedup"]["bootstrap_95_low"],
        }
        for result in model_region_results
        if result["status"] == "supported-model-region-win"
    ]
    retained_reasons: list[str] = []
    governors = {entry["governor"] for entry in before["cpu_frequency"]}
    if governors != {"performance"}:
        retained_reasons.append(f"CPU governors were {sorted(governors)}")
    if before["commands"].get("llama_servers", {}).get("stdout", "").strip():
        retained_reasons.append("a pre-existing llama-server was active")
    output = {
        "schema_version": 1,
        "kind": "llama-qpu-geglu-prefill-evaluation",
        "created_utc": utc_now(),
        "artifacts": {
            "benchmark": str(args.benchmark.resolve()),
            "benchmark_sha256": sha256_file(args.benchmark),
            "inline_hook": str(args.inline_hook.resolve()),
            "inline_hook_sha256": sha256_file(args.inline_hook),
            "program_manifest": str(
                (ROOT / "integrations/llama_cpp/generated/manifest.json").resolve()
            ),
            "program_manifest_sha256": sha256_file(
                ROOT / "integrations/llama_cpp/generated/manifest.json"
            ),
        },
        "contract": {
            "cpu_reference": "pinned GGML CPU GEGLU with configured batch threads",
            "candidate_boundary": (
                "the same GGML CPU node boundary, including cached DMA input copies, "
                "QPU execution, and cached DMA output copy"
            ),
            "correctness": "bitwise equality to pinned GGML CPU_REPACK build",
            "win_gate": "median >=1.05x and bootstrap 95% lower bound >1.0",
            "calibration_rows": sorted(calibration_rows),
            "heldout_rows": [row for row in args.rows if row not in calibration_rows],
        },
        "configuration": {
            "rows": args.rows,
            "columns": args.columns,
            "cpu_threads": args.cpu_threads,
            "warmups": args.warmups,
            "samples": args.samples,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "results": results,
        "model_region_results": model_region_results,
        "policy_candidates": winning_shapes,
        "model_region_policy_candidates": winning_model_regions,
        "validation": {
            "retained": not retained_reasons,
            "rejection_reasons": retained_reasons,
        },
        "environment_before": before,
        "environment_after": after,
        "raw_commands": raw_commands,
    }
    write_json_atomic(args.output, output)
    print(
        f"wrote {args.output}: {len(results)} cases, "
        f"{len(winning_shapes)} operator wins, retained={not retained_reasons}"
    )


if __name__ == "__main__":
    main()
