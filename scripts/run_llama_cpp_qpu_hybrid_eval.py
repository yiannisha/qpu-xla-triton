#!/usr/bin/env python3
"""Calibrate and evaluate concurrent CPU/QPU GEGLU row partitions."""

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
    command_stdout,
    sha256_file,
    swap_used_bytes,
    utc_now,
    write_json_atomic,
)

GEMMA_LAYER_COUNTS = {6144: 15, 12288: 20}


def integer_list(value: str) -> tuple[int, ...]:
    """Parse a duplicate-free comma-separated positive integer list."""
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {value!r}") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be positive and duplicate-free")
    return result


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


def bootstrap_session_speedup(
    cpu_session_ns: list[int],
    candidate_session_ns: list[int],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Bootstrap a ratio after reducing each independent session to one median."""
    if len(cpu_session_ns) != len(candidate_session_ns) or not cpu_session_ns:
        raise ValueError("CPU and candidate session vectors must be nonempty and paired")
    cpu = np.asarray(cpu_session_ns, dtype=np.float64)
    candidate = np.asarray(candidate_session_ns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selection = rng.integers(0, cpu.size, size=cpu.size)
        ratios[index] = np.median(cpu[selection]) / np.median(candidate[selection])
    return {
        "resamples": resamples,
        "seed": seed,
        "session_count": len(cpu_session_ns),
        "cpu_median_of_session_medians_ns": int(median(cpu_session_ns)),
        "candidate_median_of_session_medians_ns": int(median(candidate_session_ns)),
        "median_speedup": float(median(cpu_session_ns) / median(candidate_session_ns)),
        "session_speedups": [
            float(cpu_value / candidate_value)
            for cpu_value, candidate_value in zip(
                cpu_session_ns, candidate_session_ns, strict=True
            )
        ],
        "bootstrap_95_low": float(np.quantile(ratios, 0.025)),
        "bootstrap_95_high": float(np.quantile(ratios, 0.975)),
    }


def environment_rejection_reasons(environment: dict[str, Any]) -> list[str]:
    """Apply the retained operator-session machine-state contract."""
    reasons: list[str] = []
    governors = {entry.get("governor") for entry in environment.get("cpu_frequency", [])}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(item) for item in governors)}")
    used_swap = swap_used_bytes(environment)
    if used_swap is None:
        reasons.append("swap usage was unavailable")
    elif used_swap != 0:
        reasons.append(f"swap usage was {used_swap} bytes")
    commands = environment.get("commands", {})
    if command_stdout(commands.get("llama_servers", {})):
        reasons.append("a pre-existing llama-server was active")
    throttling = command_stdout(commands.get("throttling", {}))
    if throttling not in {"throttled=0x0", "throttled=0"}:
        reasons.append(f"throttling state was {throttling or 'unavailable'}")
    return reasons


def summarize_partition_sessions(
    sessions: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Reduce raw cell samples to the real 15/20-layer Gemma region."""
    cpu_region_ns = [int(session["cpu_region_median_ns"]) for session in sessions]
    candidate_region_ns = [
        int(session["candidate_region_median_ns"]) for session in sessions
    ]
    exact = all(bool(session["bitwise_exact"]) for session in sessions)
    interval = bootstrap_session_speedup(
        cpu_region_ns,
        candidate_region_ns,
        seed=seed,
        resamples=resamples,
    )
    return {
        "bitwise_exact": exact,
        "cpu_region_session_medians_ns": cpu_region_ns,
        "candidate_region_session_medians_ns": candidate_region_ns,
        "same_boundary_speedup": interval,
    }


def choose_partition(calibration: dict[int, dict[str, Any]]) -> int:
    """Choose the highest calibration median speedup with a deterministic tie break."""
    if not calibration:
        raise ValueError("at least one calibration partition is required")
    return max(
        calibration,
        key=lambda qpu_rows: (
            calibration[qpu_rows]["same_boundary_speedup"]["median_speedup"],
            -qpu_rows,
        ),
    )


def main() -> None:
    """Tune on calibration processes and gate fresh held-out processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_ggml_inline_smoke",
    )
    parser.add_argument(
        "--inline-hook",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/libggml-qpu-inline.so",
    )
    parser.add_argument("--rows", type=integer_list, default=integer_list("17,33,129,257"))
    parser.add_argument(
        "--qpu-rows", type=integer_list, default=integer_list("1,5,9")
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=64)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--calibration-sessions", type=int, default=3)
    parser.add_argument("--heldout-sessions", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.warmups = 2
        args.samples = 5
        args.calibration_sessions = 1
        args.heldout_sessions = 2
        args.bootstrap_resamples = 500
    if (
        args.cpu_threads < 2
        or args.warmups < 0
        or args.samples <= 0
        or args.calibration_sessions <= 0
        or args.heldout_sessions <= 0
        or args.bootstrap_resamples <= 0
    ):
        parser.error("thread, warmup, sample, and session counts are invalid")
    for path in (args.benchmark, args.inline_hook):
        if not path.is_file():
            parser.error(f"required artifact does not exist: {path}")

    raw_commands: list[dict[str, Any]] = []

    def run_session(*, phase: str, rows: int, qpu_rows: int, session: int) -> dict[str, Any]:
        cells: list[dict[str, Any]] = []
        for columns, layer_count in GEMMA_LAYER_COUNTS.items():
            command = [
                str(args.benchmark.resolve()),
                "--rows",
                str(rows),
                "--columns",
                str(columns),
                "--qpu-rows",
                str(qpu_rows),
                "--cpu-threads",
                str(args.cpu_threads),
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
                }
            )
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            raw_commands.append(
                {
                    "phase": phase,
                    "m": rows,
                    "qpu_rows": qpu_rows,
                    "session": session,
                    "command": command,
                    "environment": {
                        "LD_PRELOAD": str(args.inline_hook.resolve()),
                        "GGML_QPU_TELEMETRY": "0",
                    },
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                }
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"hybrid GEGLU benchmark failed ({completed.returncode}): "
                    f"{' '.join(command)}\n{completed.stderr}"
                )
            record = parse_benchmark_stdout(completed.stdout)
            cells.append(
                {
                    "n": columns,
                    "layer_count": layer_count,
                    "cpu_median_ns": int(median(record["cpu_complete_ns"])),
                    "candidate_median_ns": int(
                        median(record["qpu_inline_complete_ns"])
                    ),
                    "record": record,
                }
            )
        return {
            "phase": phase,
            "session": session,
            "m": rows,
            "qpu_rows": qpu_rows,
            "cpu_threads": args.cpu_threads,
            "cells": cells,
            "cpu_region_median_ns": sum(
                cell["cpu_median_ns"] * cell["layer_count"] for cell in cells
            ),
            "candidate_region_median_ns": sum(
                cell["candidate_median_ns"] * cell["layer_count"] for cell in cells
            ),
            "bitwise_exact": all(
                cell["record"]["bitwise_exact"] and cell["record"]["passed"]
                for cell in cells
            ),
        }

    environment_before = collect_environment()
    calibration_results: dict[int, dict[int, dict[str, Any]]] = {}
    selected_partitions: dict[int, int] = {}
    for rows in args.rows:
        partitions = [qpu_rows for qpu_rows in args.qpu_rows if qpu_rows < rows]
        if not partitions:
            parser.error(f"no qpu_rows value is smaller than M={rows}")
        calibration_results[rows] = {}
        for qpu_rows in partitions:
            sessions = [
                run_session(
                    phase="calibration",
                    rows=rows,
                    qpu_rows=qpu_rows,
                    session=session,
                )
                for session in range(1, args.calibration_sessions + 1)
            ]
            calibration_results[rows][qpu_rows] = {
                "sessions": sessions,
                **summarize_partition_sessions(
                    sessions,
                    seed=args.seed + rows * 101 + qpu_rows,
                    resamples=args.bootstrap_resamples,
                ),
            }
        selected_partitions[rows] = choose_partition(calibration_results[rows])

    heldout_results: dict[int, dict[str, Any]] = {}
    for rows, qpu_rows in selected_partitions.items():
        sessions = [
            run_session(
                phase="heldout",
                rows=rows,
                qpu_rows=qpu_rows,
                session=session,
            )
            for session in range(1, args.heldout_sessions + 1)
        ]
        heldout_results[rows] = {
            "qpu_rows": qpu_rows,
            "sessions": sessions,
            **summarize_partition_sessions(
                sessions,
                seed=args.seed + rows * 1009 + qpu_rows,
                resamples=args.bootstrap_resamples,
            ),
        }
    environment_after = collect_environment()

    rejection_reasons = [
        *(f"before: {reason}" for reason in environment_rejection_reasons(environment_before)),
        *(f"after: {reason}" for reason in environment_rejection_reasons(environment_after)),
    ]
    if args.quick:
        rejection_reasons.append("quick mode does not satisfy the retention contract")
    if args.warmups < 64:
        rejection_reasons.append("fewer than 64 warmups were used")
    if args.samples < 31:
        rejection_reasons.append("fewer than 31 samples were retained per cell")
    if args.heldout_sessions < 5:
        rejection_reasons.append("fewer than five independent held-out sessions were used")
    retained = not rejection_reasons

    policy_candidates: list[dict[str, Any]] = []
    for rows, result in heldout_results.items():
        interval = result["same_boundary_speedup"]
        statistical_win = bool(
            result["bitwise_exact"]
            and interval["median_speedup"] >= 1.05
            and interval["bootstrap_95_low"] > 1.0
        )
        result["statistical_win"] = statistical_win
        result["promoted"] = retained and statistical_win
        result["status"] = (
            "promoted"
            if result["promoted"]
            else "environment-rejected-win"
            if statistical_win
            else "correct-no-stable-win"
        )
        if result["promoted"]:
            policy_candidates.append(
                {
                    "m": rows,
                    "qpu_rows": result["qpu_rows"],
                    "cpu_threads": args.cpu_threads,
                    "median_speedup": interval["median_speedup"],
                    "bootstrap_95_low": interval["bootstrap_95_low"],
                }
            )

    output = {
        "schema_version": 1,
        "kind": "llama-qpu-geglu-hybrid-evaluation",
        "created_utc": utc_now(),
        "artifacts": {
            "benchmark": str(args.benchmark.resolve()),
            "benchmark_sha256": sha256_file(args.benchmark),
            "inline_hook": str(args.inline_hook.resolve()),
            "inline_hook_sha256": sha256_file(args.inline_hook),
        },
        "contract": {
            "cpu_reference": "pinned GGML GEGLU with four CPU batch threads",
            "candidate_boundary": (
                "the same GGML node with disjoint CPU/QPU row execution, including "
                "QPU staging, asynchronous submission, wait, and output copy"
            ),
            "model_region": "15 GEGLU nodes at N=6144 plus 20 at N=12288",
            "correctness": "bitwise equality and exact QPU dispatch accounting",
            "selection": "qpu_rows is selected only from calibration processes",
            "promotion": (
                "at least five fresh held-out sessions, median speedup >=1.05x, "
                "paired-session bootstrap 95% lower bound >1.0, and retained environment"
            ),
        },
        "configuration": {
            "rows": args.rows,
            "qpu_rows": args.qpu_rows,
            "cpu_threads": args.cpu_threads,
            "warmups": args.warmups,
            "samples": args.samples,
            "calibration_sessions": args.calibration_sessions,
            "heldout_sessions": args.heldout_sessions,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "calibration_results": calibration_results,
        "selected_partitions": selected_partitions,
        "heldout_results": heldout_results,
        "policy_candidates": policy_candidates,
        "validation": {
            "retained": retained,
            "rejection_reasons": rejection_reasons,
        },
        "environment_before": environment_before,
        "environment_after": environment_after,
        "raw_commands": raw_commands,
    }
    write_json_atomic(args.output, output)
    print(
        f"wrote {args.output}: selected={selected_partitions}, "
        f"promoted={len(policy_candidates)}, retained={retained}"
    )


if __name__ == "__main__":
    main()
