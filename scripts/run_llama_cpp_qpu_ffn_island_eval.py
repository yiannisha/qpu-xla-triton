#!/usr/bin/env python3
"""Evaluate the integrated CPU_REPACK/QPU full-FFN channel island."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
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
    swap_used_bytes,
    utc_now,
    write_json_atomic,
)

ISLAND_PREFIX = "qpu_llama_candidate_json:"
WEIGHT_PREFIX = "qpu_llama_weight_json:"
ISLAND_ENVIRONMENT_KEYS = (
    "LD_PRELOAD",
    "GGML_QPU_FFN_ISLAND",
    "GGML_QPU_FFN_ISLAND_FRACTION",
    "GGML_QPU_FFN_ISLAND_MIN_ROWS",
    "GGML_QPU_FFN_ISLAND_MAX_ROWS",
    "GGML_QPU_FFN_ISLAND_WGS",
    "GGML_QPU_FFN_ISLAND_VERIFY",
    "GGML_QPU_FFN_ISLAND_FAIL_STAGE",
    "GGML_QPU_FFN_ISLAND_FAIL_LAYER",
    "GGML_QPU_TELEMETRY",
)


def integer_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {value!r}") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer lists must be positive and duplicate-free")
    return result


def fraction_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid fraction list {value!r}") from exc
    if not result or any(item < 0.03125 or item > 0.5 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("fractions must be unique values in [0.03125, 0.5]")
    return result


def current_temperature_c() -> float | None:
    values: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(path.read_text(encoding="utf-8").strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def wait_until_cool(maximum_c: float, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    initial = current_temperature_c()
    current = initial
    while current is not None and current > maximum_c:
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"board remained at {current:.1f} C above {maximum_c:.1f} C")
        time.sleep(2.0)
        current = current_temperature_c()
    return {
        "maximum_start_c": maximum_c,
        "initial_c": initial,
        "ready_c": current,
        "wait_seconds": time.monotonic() - started,
    }


def parse_prefixed_json(stderr: str, prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        if not line.startswith(prefix):
            continue
        value = json.loads(line.split(":", 1)[1])
        if not isinstance(value, dict):
            raise RuntimeError(f"{prefix} record was not an object")
        result.append(value)
    return result


def manifest_programs(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = ("ggml-q4-0-q8-wordscale-mx", "ggml-geglu-q8-0-split")
    result = {entry["name"]: entry for entry in payload["programs"] if entry.get("name") in names}
    if set(result) != set(names):
        raise ValueError(f"program manifest does not contain {names}")
    for entry in result.values():
        binary = path.parent / entry["binary"]
        if sha256_file(binary) != entry["binary_sha256"]:
            raise ValueError(f"binary hash mismatch: {binary}")
    return result


def validate_candidate(
    events: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    *,
    rows: int,
    minimum_rows: int,
    maximum_rows: int,
    programs: dict[str, dict[str, Any]],
    allow_fallback_layer: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    full_batches, remainder = divmod(rows, 512)
    expected_m = ({512} if full_batches and minimum_rows <= 512 <= maximum_rows else set()) | (
        {remainder} if minimum_rows <= remainder <= maximum_rows else set()
    )
    layers = sorted({int(event.get("layer", -1)) for event in events})
    expected_layers = list(range(35))
    if layers != expected_layers:
        errors.append(f"observed layers were {layers}, expected 0..34")
    counts = {layer: sum(int(event.get("layer", -1)) == layer for event in events) for layer in expected_layers}
    if not events or len(set(counts.values())) != 1:
        errors.append("candidate event counts were not balanced across all layers")
    expected_programs = {
        name: (entry["source_hash"], entry["binary_sha256"])
        for name, entry in programs.items()
    }
    fallback_events = [event for event in events if event.get("fallback") is True]
    if allow_fallback_layer is None and fallback_events:
        errors.append(f"observed {len(fallback_events)} unexpected fallbacks")
    if allow_fallback_layer is not None and (
        not fallback_events
        or {int(event.get("layer", -1)) for event in fallback_events} != {allow_fallback_layer}
    ):
        errors.append("failure injection did not fall back exclusively on the requested layer")
    for event in events:
        if event.get("operation") != "ffn_island" or event.get("strategy") != "complete_channel_partition":
            errors.append("candidate operation or strategy mismatch")
            break
        if int(event.get("m", -1)) not in expected_m:
            errors.append(
                f"candidate physical-batch M was not in {sorted(expected_m)} for logical M={rows}"
            )
            break
        if not all(bool(value) for value in event.get("cpu_path", {}).values()):
            errors.append("one or more CPU path restrictions were not observed")
            break
        partition = event.get("partition", {})
        if int(partition.get("cpu_columns", -1)) + int(partition.get("qpu_columns", -1)) != int(
            event.get("intermediate_columns", -2)
        ):
            errors.append("CPU/QPU columns did not cover the intermediate dimension")
            break
        observed_programs = {
            item.get("name"): (item.get("source_hash"), item.get("binary_sha256"))
            for item in event.get("programs", [])
        }
        if observed_programs != expected_programs:
            errors.append("runtime program hashes did not match the retained manifest")
            break
        expected_dispatches = 0 if event.get("fallback") and event.get("failure_stage") == "gate" else 4
        if int(event.get("dispatch_count", -1)) != expected_dispatches:
            errors.append("dispatch count did not match the completed island stages")
            break
    weight_roles = {(int(item.get("layer", -1)), item.get("role")) for item in weights}
    expected_weight_roles = {(layer, role) for layer in expected_layers for role in ("gate", "up", "down")}
    if weight_roles != expected_weight_roles or len(weights) != len(expected_weight_roles):
        errors.append("resident gate/up/down weight telemetry was incomplete or duplicated")
    return {
        "valid": not errors,
        "errors": errors,
        "event_count": len(events),
        "passes": counts.get(0, 0),
        "layers": layers,
        "logical_rows": rows,
        "observed_physical_rows": sorted({int(event.get("m", -1)) for event in events}),
        "dispatch_count": sum(int(event.get("dispatch_count", 0)) for event in events),
        "fallback_count": len(fallback_events),
        "all_cpu_paths_restricted": bool(events) and all(
            all(bool(value) for value in event.get("cpu_path", {}).values()) for event in events
        ),
        "resident_weight_count": len(weights),
        "resident_weight_bytes": sum(int(item.get("resident_bytes", 0)) for item in weights),
        "median_timing_ns": {
            key: int(median(int(event["timing_ns"][key]) for event in events)) if events else None
            for key in (
                "input_pack_and_copy",
                "gate",
                "up",
                "geglu",
                "down",
                "qpu_complete",
                "cpu_overlap_before_wait",
                "exposed_wait",
                "join",
            )
        },
    }


def run_benchmark(
    *,
    binary: Path,
    model: Path,
    plugin: Path,
    programs: dict[str, dict[str, Any]],
    rows: int,
    threads: int,
    repetitions: int,
    fraction: float | None,
    minimum_rows: int,
    maximum_rows: int,
    wgs: int,
    cooldown_c: float,
    verify: bool = False,
    fail_layer: int | None = None,
) -> dict[str, Any]:
    cooldown = wait_until_cool(cooldown_c, 600.0)
    command = [
        str(binary.resolve()), "-m", str(model.resolve()), "-p", str(rows), "-n", "0",
        "-t", str(threads), "-r", str(repetitions), "-o", "json",
    ]
    environment = os.environ.copy()
    for key in ISLAND_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    retained_environment: dict[str, str] = {}
    if fraction is not None:
        retained_environment = {
            "LD_PRELOAD": str(plugin.resolve()),
            "GGML_QPU_FFN_ISLAND": "1",
            "GGML_QPU_FFN_ISLAND_FRACTION": str(fraction),
            "GGML_QPU_FFN_ISLAND_MIN_ROWS": str(minimum_rows),
            "GGML_QPU_FFN_ISLAND_MAX_ROWS": str(maximum_rows),
            "GGML_QPU_FFN_ISLAND_WGS": str(wgs),
            "GGML_QPU_TELEMETRY": "1",
        }
        if verify:
            retained_environment["GGML_QPU_FFN_ISLAND_VERIFY"] = "1"
        if fail_layer is not None:
            retained_environment.update(
                {
                    "GGML_QPU_FFN_ISLAND_FAIL_STAGE": "gate",
                    "GGML_QPU_FFN_ISLAND_FAIL_LAYER": str(fail_layer),
                }
            )
        environment.update(retained_environment)
    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    wall_ns = time.monotonic_ns() - started_ns
    try:
        benchmark = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama-bench produced invalid JSON: {exc}\n{completed.stdout}") from exc
    if completed.returncode != 0 or not isinstance(benchmark, list) or len(benchmark) != 1:
        raise RuntimeError(f"llama-bench failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    events = parse_prefixed_json(completed.stderr, ISLAND_PREFIX)
    weights = parse_prefixed_json(completed.stderr, WEIGHT_PREFIX)
    nontelemetry = "\n".join(
        line
        for line in completed.stderr.splitlines()
        if not line.startswith((ISLAND_PREFIX, WEIGHT_PREFIX))
    )
    validation = None
    if fraction is not None:
        validation = validate_candidate(
            events, weights, rows=rows, minimum_rows=minimum_rows,
            maximum_rows=maximum_rows, programs=programs,
            allow_fallback_layer=fail_layer
        )
        if not validation["valid"]:
            raise RuntimeError(f"invalid QPU execution: {validation['errors']}")
    elif events or weights:
        raise RuntimeError("CPU baseline unexpectedly emitted QPU FFN-island telemetry")
    return {
        "kind": "candidate" if fraction is not None else "cpu-repack",
        "rows": rows,
        "fraction": fraction,
        "verify": verify,
        "fail_layer": fail_layer,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_ns": wall_ns,
        "cooldown": cooldown,
        "command": command,
        "process_environment": retained_environment,
        "returncode": completed.returncode,
        "benchmark": benchmark[0],
        "samples_ns": [int(value) for value in benchmark[0]["samples_ns"]],
        "candidate_validation": validation,
        "candidate_events": events,
        "weight_events": weights,
        "stderr_nontelemetry": nontelemetry,
    }


def speedup_interval(
    baseline: list[int], candidate: list[int], *, seed: int, resamples: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    ratios = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        ratios[index] = np.median(rng.choice(left, left.size, replace=True)) / np.median(
            rng.choice(right, right.size, replace=True)
        )
    return {
        "baseline_median_ns": int(median(baseline)),
        "candidate_median_ns": int(median(candidate)),
        "median_speedup": float(median(baseline) / median(candidate)),
        "bootstrap_95_low": float(np.quantile(ratios, 0.025)),
        "bootstrap_95_high": float(np.quantile(ratios, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def aggregate_candidate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    validations = [run["candidate_validation"] for run in runs]
    events = [event for run in runs for event in run["candidate_events"]]
    errors = [
        f"process {index}: {error}"
        for index, validation in enumerate(validations)
        for error in validation["errors"]
    ]
    return {
        "valid": bool(validations) and all(validation["valid"] for validation in validations),
        "errors": errors,
        "independent_processes": len(runs),
        "event_count": sum(int(validation["event_count"]) for validation in validations),
        "passes": sum(int(validation["passes"]) for validation in validations),
        "layers": sorted({layer for validation in validations for layer in validation["layers"]}),
        "observed_physical_rows": sorted(
            {row for validation in validations for row in validation["observed_physical_rows"]}
        ),
        "dispatch_count": sum(int(validation["dispatch_count"]) for validation in validations),
        "fallback_count": sum(int(validation["fallback_count"]) for validation in validations),
        "all_cpu_paths_restricted": all(
            validation["all_cpu_paths_restricted"] for validation in validations
        ),
        "resident_weight_count_per_process": [
            int(validation["resident_weight_count"]) for validation in validations
        ],
        "resident_weight_bytes_per_process": [
            int(validation["resident_weight_bytes"]) for validation in validations
        ],
        "median_timing_ns": {
            key: int(median(int(event["timing_ns"][key]) for event in events)) if events else None
            for key in (
                "input_pack_and_copy",
                "gate",
                "up",
                "geglu",
                "down",
                "qpu_complete",
                "cpu_overlap_before_wait",
                "exposed_wait",
                "join",
            )
        },
    }


def git_record(path: Path) -> dict[str, Any]:
    def capture(arguments: list[str]) -> str:
        result = subprocess.run(arguments, text=True, capture_output=True, check=False)
        return result.stdout.strip()

    return {
        "path": str(path.resolve()),
        "commit": capture(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "status": capture(["git", "-C", str(path), "status", "--short"]),
        "diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "-C", str(path), "diff", "--binary"],
                capture_output=True,
                check=False,
            ).stdout
        ).hexdigest(),
    }


def retention(before: dict[str, Any], after: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    governors = {item.get("governor") for item in before["cpu_frequency"]}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(item) for item in governors)}")
    if before["commands"]["llama_servers"].get("stdout", "").strip():
        reasons.append("a pre-existing llama-server process was active")
    if swap_used_bytes(before) != 0 or swap_used_bytes(after) != 0:
        reasons.append("swap use was unavailable or nonzero")
    for label, environment in (("before", before), ("after", after)):
        text = environment["commands"]["throttling"].get("stdout", "")
        try:
            current_flags = int(text.split("=", 1)[1], 0) & 0xFFFF
        except (IndexError, ValueError):
            current_flags = -1
        if current_flags != 0:
            reasons.append(f"{label} current throttling flags were {current_flags}")
    if any(run["returncode"] != 0 for run in runs):
        reasons.append("one or more benchmark processes failed")
    return {"retained": not reasons, "rejection_reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bench", type=Path, default=Path("/home/yiannis/side/llama.cpp/build/bin/llama-bench"))
    parser.add_argument("--model", type=Path, default=Path("/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"))
    parser.add_argument("--plugin", type=Path, default=ROOT / "build/llama-qpu-runtime/libggml-qpu-inline.so")
    parser.add_argument("--program-manifest", type=Path, default=ROOT / "integrations/llama_cpp/generated/manifest.json")
    parser.add_argument("--rows", type=integer_list, default=integer_list("65,129,257,513"))
    parser.add_argument("--fractions", type=fraction_list, default=fraction_list("0.125,0.1875,0.25"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="independent cooled processes per CPU/QPU configuration",
    )
    parser.add_argument("--minimum-rows", type=int, default=64)
    parser.add_argument("--maximum-rows", type=int, default=528)
    parser.add_argument("--wgs", type=int, default=24)
    parser.add_argument("--cooldown-c", type=float, default=65.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--verification-row", type=int, default=65)
    parser.add_argument("--verification-fraction", type=float, default=0.125)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.repetitions = 1
        args.bootstrap_resamples = 500
    if args.repetitions <= 0 or args.threads <= 0 or args.minimum_rows <= 0:
        parser.error("repetitions/threads/minimum-rows must be positive")
    for path in (args.llama_bench, args.model, args.plugin, args.program_manifest):
        if not path.is_file():
            parser.error(f"required artifact not found: {path}")

    programs = manifest_programs(args.program_manifest)
    before = collect_environment()
    configurations = [(row, None, process_index) for row in args.rows for process_index in range(args.repetitions)] + [
        (row, fraction, process_index)
        for row in args.rows
        for fraction in args.fractions
        for process_index in range(args.repetitions)
    ]
    random.Random(args.seed).shuffle(configurations)
    runs: list[dict[str, Any]] = []
    for row, fraction, process_index in configurations:
        print(
            f"running M={row} process={process_index} "
            f"{'CPU_REPACK' if fraction is None else f'QPU fraction={fraction:g}'}",
            flush=True,
        )
        run = run_benchmark(
            binary=args.llama_bench,
            model=args.model,
            plugin=args.plugin,
            programs=programs,
            rows=row,
            threads=args.threads,
            repetitions=1,
            fraction=fraction,
            minimum_rows=args.minimum_rows,
            maximum_rows=args.maximum_rows,
            wgs=args.wgs,
            cooldown_c=args.cooldown_c,
        )
        run["process_index"] = process_index
        runs.append(run)
        write_json_atomic(args.output, {"schema_version": 1, "kind": "partial-ffn-island-evaluation", "runs": runs})

    verification = None
    if not args.skip_verification:
        print(f"running online numerical verification at M={args.verification_row}", flush=True)
        verification = run_benchmark(
            binary=args.llama_bench,
            model=args.model,
            plugin=args.plugin,
            programs=programs,
            rows=args.verification_row,
            threads=args.threads,
            repetitions=1,
            fraction=args.verification_fraction,
            minimum_rows=args.minimum_rows,
            maximum_rows=args.maximum_rows,
            wgs=args.wgs,
            cooldown_c=args.cooldown_c,
            verify=True,
        )
        values = [event["verification"] for event in verification["candidate_events"]]
        verification["numerical_summary"] = {
            "verified_event_count": sum(bool(item.get("enabled")) for item in values),
            "maximum_absolute_error": max(float(item["max_absolute_error"]) for item in values),
            "maximum_mean_absolute_error": max(float(item["mean_absolute_error"]) for item in values),
            "passed": bool(values)
            and all(bool(item.get("enabled")) for item in values)
            and max(float(item["max_absolute_error"]) for item in values) <= 0.002
            and max(float(item["mean_absolute_error"]) for item in values) <= 2.0e-5,
        }

    fallback = None
    if not args.skip_fallback:
        print(f"running layer-0 failure injection at M={args.verification_row}", flush=True)
        fallback = run_benchmark(
            binary=args.llama_bench,
            model=args.model,
            plugin=args.plugin,
            programs=programs,
            rows=args.verification_row,
            threads=args.threads,
            repetitions=1,
            fraction=args.verification_fraction,
            minimum_rows=args.minimum_rows,
            maximum_rows=args.maximum_rows,
            wgs=args.wgs,
            cooldown_c=args.cooldown_c,
            fail_layer=0,
        )

    comparisons: list[dict[str, Any]] = []
    for row in args.rows:
        baseline_runs = sorted(
            (run for run in runs if run["rows"] == row and run["fraction"] is None),
            key=lambda run: run["process_index"],
        )
        for fraction in args.fractions:
            candidate_runs = sorted(
                (run for run in runs if run["rows"] == row and run["fraction"] == fraction),
                key=lambda run: run["process_index"],
            )
            interval = speedup_interval(
                [run["samples_ns"][0] for run in baseline_runs],
                [run["samples_ns"][0] for run in candidate_runs],
                seed=args.seed + row * 101 + round(float(fraction) * 10_000),
                resamples=args.bootstrap_resamples,
            )
            comparisons.append(
                {
                    "rows": row,
                    "fraction": fraction,
                    "speedup": interval,
                    "candidate_validation": aggregate_candidate_runs(candidate_runs),
                }
            )
    best = [
        max((item for item in comparisons if item["rows"] == row), key=lambda item: item["speedup"]["median_speedup"])
        for row in args.rows
    ]
    after = collect_environment()
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-complete-ffn-island-evaluation",
        "created_utc": utc_now(),
        "contract": {
            "baseline": "patched llama.cpp with QPU flags unset; native CPU_REPACK Q4_0 x Q8_0 FFN path",
            "candidate": "concurrent CPU prefix plus QPU gate/up/GEGLU-to-Q8/down suffix and hidden-size F32 join",
            "partition_axis": "intermediate channels",
            "timed_boundary": "llama-bench prompt processing; model loading and resident packing excluded",
            "correctness": "online QPU suffix compared with scalar Q4_0 x Q8_0 reference on real activations",
        },
        "artifacts": {
            "llama_bench": str(args.llama_bench.resolve()),
            "llama_bench_sha256": sha256_file(args.llama_bench),
            "model": str(args.model.resolve()),
            "model_size": args.model.stat().st_size,
            "model_sha256": sha256_file(args.model),
            "plugin": str(args.plugin.resolve()),
            "plugin_sha256": sha256_file(args.plugin),
            "program_manifest": str(args.program_manifest.resolve()),
            "program_manifest_sha256": sha256_file(args.program_manifest),
            "programs": programs,
        },
        "source": {
            "py_videocore7": git_record(ROOT),
            "llama_cpp": git_record(Path("/home/yiannis/side/llama.cpp")),
        },
        "configuration": {
            "rows": args.rows,
            "fractions": args.fractions,
            "threads": args.threads,
            "independent_processes_per_configuration": args.repetitions,
            "llama_bench_repetitions_per_process": 1,
            "minimum_rows": args.minimum_rows,
            "maximum_rows": args.maximum_rows,
            "wgs": args.wgs,
            "cooldown_c": args.cooldown_c,
            "seed": args.seed,
        },
        "runs": runs,
        "comparisons": comparisons,
        "best_by_rows": best,
        "online_numerical_verification": verification,
        "failure_injection": fallback,
        "environment_before": before,
        "environment_after": after,
        "retention": retention(before, after, runs + [item for item in (verification, fallback) if item]),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "retention": payload["retention"], "best_by_rows": best}, indent=2))


if __name__ == "__main__":
    main()
