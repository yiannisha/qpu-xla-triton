#!/usr/bin/env python3
"""Calibrate and evaluate the overlapped QPU ``ffn_up`` suffix in fresh pairs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from scripts.run_llama_cpp_qpu_evaluation import (  # noqa: E402
    normalize_case,
    run_server_case,
    validate_candidate_evidence,
    validate_session,
    workload_semantics_sha256,
)
from scripts.run_llama_cpp_qpu_hybrid_eval import (  # noqa: E402
    bootstrap_session_speedup,
)

EXPECTED_FFN_COLUMNS = {6144: 15, 12288: 20}


def integer_list(value: str) -> tuple[int, ...]:
    """Parse a duplicate-free comma-separated positive integer list."""
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {value!r}") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be positive and duplicate-free")
    return result


def fraction_list(value: str) -> tuple[float, ...]:
    """Parse unique, positive output-column fractions."""
    try:
        result = tuple(float(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid fraction list {value!r}") from exc
    if not result or any(item < 0.0625 or item > 0.5 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("fractions must be unique values in the closed interval [0.0625, 0.5]")
    return result


def _case_key(case: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(case.get("context_tokens_target", 0)),
        int(case.get("suffix_tokens_target", 0)),
        str(case.get("placement", "cpu-only")),
    )


def load_case_pairs(path: Path) -> dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]]:
    """Load exactly one CPU and hybrid case for every prefix/suffix cell."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    normalized = [normalize_case(case, index=index) for index, case in enumerate(raw_cases)]
    by_key = {_case_key(case): case for case in normalized}
    cells = {(key[0], key[1]) for key in by_key}
    result: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for prefix, suffix in cells:
        cpu = by_key.get((prefix, suffix, "cpu-only"))
        candidate = by_key.get((prefix, suffix, "hybrid"))
        if cpu is not None and candidate is not None:
            validate_candidate_evidence(candidate)
            result[(prefix, suffix)] = (cpu, candidate)
    return result


def candidate_at_fraction(
    case: dict[str, Any], fraction: float, *, weight_mode: str = "exact", wgs: int = 24
) -> dict[str, Any]:
    """Clone a generated candidate while preserving its exact program evidence."""
    candidate = copy.deepcopy(case)
    candidate["name"] = f"{case['name']}-fraction-{fraction:g}"
    candidate["partition"] = {"axis": "output_columns", "fraction": fraction}
    environment = dict(candidate.get("process_environment", {}))
    environment["GGML_QPU_UP_FRACTION"] = str(fraction)
    environment["GGML_QPU_UP_MAX_FRACTION"] = str(fraction)
    environment["GGML_QPU_UP_WEIGHT_MODE"] = weight_mode
    environment["GGML_QPU_UP_WGS"] = str(wgs)
    candidate["process_environment"] = environment
    programs = {
        "exact": (
            "ggml-q4-0-q8-0-mx",
            {"weight": "Q4_0", "activation": "CPU_REPACK_Q8_0x4", "output": "F32", "tile": [16, 16]},
        ),
        "column-w8": (
            "ggml-column-w8-q8-0-mx",
            {"weight": "APPROX_COLUMN_W8", "activation": "CPU_REPACK_Q8_0x4", "output": "F32", "tile": [16, 16]},
        ),
        "rowcol-w8a8": (
            "tiled-w8a8-gemm-dequantize",
            {"weight": "APPROX_COLUMN_W8", "activation": "APPROX_ROW_W8", "output": "F32", "tile": [16, 16]},
        ),
    }
    program, exact_shape = programs[weight_mode]
    manifest_path = Path(candidate["candidate_evidence"]["program_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["programs"] if entry["name"] == program]
    if len(entries) != 1:
        raise ValueError(f"program {program!r} was not unique in {manifest_path}")
    entry = entries[0]
    candidate["candidate_evidence"] = {
        "program_manifest_path": str(manifest_path.resolve()),
        "program_manifest_sha256": sha256_file(manifest_path),
        "program": program,
        "source_hash": entry["source_hash"],
        "binary_sha256": entry["binary_sha256"],
        "exact_shape": exact_shape,
    }
    validate_candidate_evidence(candidate)
    return candidate


def sample_prompt_ns(sample: dict[str, Any]) -> int | None:
    """Read llama.cpp's internal prompt time as integer nanoseconds."""
    value = sample.get("metrics", {}).get("timings", {}).get("prompt_ms")
    return round(float(value) * 1_000_000) if value is not None else None


def token_outputs_identical(cpu: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Require complete raw token IDs and all user-visible greedy semantics to match."""
    left = cpu.get("response_semantics", {})
    right = candidate.get("response_semantics", {})
    return bool(
        left.get("token_ids")
        and left.get("token_ids") == right.get("token_ids")
        and left.get("content") == right.get("content")
        and left.get("stop_reason") == right.get("stop_reason")
        and left.get("tool_calls") == right.get("tool_calls")
    )


def current_temperature_c() -> float | None:
    """Return the hottest readable Linux thermal zone."""
    values: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(path.read_text(encoding="utf-8").strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def wait_until_cool(maximum_c: float, timeout_seconds: float) -> dict[str, Any]:
    """Hold a process launch until the board returns to a repeatable temperature."""
    started = time.monotonic()
    initial = current_temperature_c()
    if initial is None:
        raise RuntimeError("no readable thermal zone is available")
    current = initial
    while current > maximum_c:
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"board remained at {current:.1f} C above {maximum_c:.1f} C")
        time.sleep(2.0)
        value = current_temperature_c()
        if value is None:
            raise RuntimeError("thermal zone disappeared during cooldown")
        current = value
    return {
        "maximum_start_c": maximum_c,
        "initial_c": initial,
        "ready_c": current,
        "wait_seconds": time.monotonic() - started,
    }


def qpu_execution_summary(sample: dict[str, Any], suffix_tokens: int) -> dict[str, Any]:
    """Validate every model-layer dispatch and report overlap measurements."""
    execution = sample.get("candidate_execution", {})
    events = execution.get("events", [])
    weight_events: list[dict[str, Any]] = []
    for line in str(sample.get("server_log", "")).splitlines():
        if not line.startswith("qpu_llama_weight_json:"):
            continue
        try:
            event = json.loads(line.split(":", 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            weight_events.append(event)
    expected_m = suffix_tokens + 1
    observed_counts = {columns: sum(event.get("n") == columns for event in events) for columns in EXPECTED_FFN_COLUMNS}
    resident_counts = {
        columns: sum(event.get("n") == columns for event in weight_events) for columns in EXPECTED_FFN_COLUMNS
    }
    fallbacks = sum(bool(event.get("fallback")) for event in events)
    valid = bool(
        execution.get("valid")
        and execution.get("dispatch_count") == sum(EXPECTED_FFN_COLUMNS.values())
        and observed_counts == EXPECTED_FFN_COLUMNS
        and events
        and len(weight_events) == sum(EXPECTED_FFN_COLUMNS.values())
        and resident_counts == EXPECTED_FFN_COLUMNS
        and len({event.get("tensor") for event in weight_events}) == len(weight_events)
        and all(
            event.get("operation") == "ffn_up"
            and int(event.get("k", -1)) == 1536
            and int(event.get("max_m", -1)) >= expected_m
            and int(event.get("resident_column_start", -1)) >= 0
            and int(event.get("resident_columns", 0)) > 0
            and int(event.get("resident_column_start", -1)) + int(event.get("resident_columns", 0))
            == int(event.get("n", -2))
            and int(event.get("resident_bytes", 0)) > 0
            for event in weight_events
        )
        and {event.get("m") for event in events} == {expected_m}
        and fallbacks == 0
        and all(
            int(event.get("qpu_columns", -1)) + int(event.get("cpu_columns", -1)) == int(event.get("n", -2))
            for event in events
        )
    )
    timing_keys = (
        "input_copy_ns",
        "submit_wait_ns",
        "overlap_before_wait_ns",
        "output_copy_ns",
        "complete_ns",
    )
    timing_medians = {key: int(median(int(event[key]) for event in events)) if events else None for key in timing_keys}
    return {
        "valid": valid,
        "expected_m": expected_m,
        "observed_m": sorted({event.get("m") for event in events}),
        "dispatch_count": execution.get("dispatch_count", 0),
        "dispatches_by_n": observed_counts,
        "fallback_count": fallbacks,
        "resident_weight_count": len(weight_events),
        "resident_weights_by_n": resident_counts,
        "resident_dma_bytes": sum(int(event.get("resident_bytes", 0)) for event in weight_events),
        "timing_medians_ns": timing_medians,
    }


def run_pair(
    *,
    server: Path,
    cpu_case: dict[str, Any],
    candidate_case: dict[str, Any],
    phase: str,
    session: int,
    fraction: float,
    rng: random.Random,
    startup_timeout: float,
    request_timeout: float,
    cooldown_c: float,
    cooldown_timeout: float,
) -> dict[str, Any]:
    """Run randomized fresh CPU/QPU processes and retain a paired observation."""
    order = ["cpu", "candidate"]
    rng.shuffle(order)
    cases = {"cpu": cpu_case, "candidate": candidate_case}
    before: dict[str, Any] | None = None
    samples: dict[str, dict[str, Any]] = {}
    cooldowns: dict[str, dict[str, Any]] = {}
    for placement in order:
        cooldowns[placement] = wait_until_cool(cooldown_c, cooldown_timeout)
        if before is None:
            before = collect_environment()
        samples[placement] = run_server_case(
            server,
            cases[placement],
            session,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
        )
    after = collect_environment()
    assert before is not None
    cpu = samples["cpu"]
    candidate = samples["candidate"]
    suffix_tokens = int(cpu_case["suffix_tokens_target"])
    qpu_summary = qpu_execution_summary(candidate, suffix_tokens)
    environment_validation = validate_session(before, after, [cpu, candidate])
    semantic_exact = token_outputs_identical(cpu, candidate)
    process_valid = all(
        sample.get("returncode") == 0
        and sample.get("context_population", {}).get("valid")
        and sample.get("generation_population", {}).get("valid")
        for sample in samples.values()
    )
    cpu_wall_ns = cpu.get("request_wall_ns")
    candidate_wall_ns = candidate.get("request_wall_ns")
    cpu_prompt_ns = sample_prompt_ns(cpu)
    candidate_prompt_ns = sample_prompt_ns(candidate)
    measurement_valid = all(
        isinstance(value, int) and value > 0
        for value in (cpu_wall_ns, candidate_wall_ns, cpu_prompt_ns, candidate_prompt_ns)
    )
    return {
        "phase": phase,
        "session": session,
        "prefix_tokens": int(cpu_case["context_tokens_target"]),
        "suffix_tokens": suffix_tokens,
        "observed_up_m": suffix_tokens + 1,
        "qpu_output_fraction": fraction,
        "randomized_order": order,
        "pre_process_cooldown": cooldowns,
        "cpu_case": cpu_case,
        "candidate_case": candidate_case,
        "cpu_sample": cpu,
        "candidate_sample": candidate,
        "cpu_request_wall_ns": cpu_wall_ns,
        "candidate_request_wall_ns": candidate_wall_ns,
        "cpu_prompt_ns": cpu_prompt_ns,
        "candidate_prompt_ns": candidate_prompt_ns,
        "request_wall_speedup": (float(cpu_wall_ns / candidate_wall_ns) if measurement_valid else None),
        "prompt_speedup": (float(cpu_prompt_ns / candidate_prompt_ns) if measurement_valid else None),
        "peak_rss_delta_bytes": (
            int(candidate.get("process_memory", {}).get("peak_rss_bytes") or 0)
            - int(cpu.get("process_memory", {}).get("peak_rss_bytes") or 0)
        ),
        "token_outputs_identical": semantic_exact,
        "qpu_execution": qpu_summary,
        "process_valid": process_valid,
        "measurement_valid": measurement_valid,
        "environment_before": before,
        "environment_after": after,
        "environment_validation": environment_validation,
        "correct": semantic_exact and process_valid and measurement_valid and qpu_summary["valid"],
        "retained": (
            semantic_exact
            and process_valid
            and measurement_valid
            and qpu_summary["valid"]
            and environment_validation["retained"]
        ),
    }


def summarize_pairs(pairs: list[dict[str, Any]], *, seed: int, resamples: int) -> dict[str, Any]:
    """Bootstrap paired fresh-process request and llama prompt speedups."""
    usable = [pair for pair in pairs if pair["correct"]]
    if not usable:
        return {
            "pair_count": len(pairs),
            "usable_pair_count": 0,
            "retained_pair_count": 0,
            "all_correct": False,
            "request_wall_speedup": None,
            "prompt_speedup": None,
        }
    wall = bootstrap_session_speedup(
        [int(pair["cpu_request_wall_ns"]) for pair in usable],
        [int(pair["candidate_request_wall_ns"]) for pair in usable],
        seed=seed,
        resamples=resamples,
    )
    prompt = bootstrap_session_speedup(
        [int(pair["cpu_prompt_ns"]) for pair in usable],
        [int(pair["candidate_prompt_ns"]) for pair in usable],
        seed=seed + 1,
        resamples=resamples,
    )
    return {
        "pair_count": len(pairs),
        "usable_pair_count": len(usable),
        "retained_pair_count": sum(pair["retained"] for pair in usable),
        "all_correct": len(usable) == len(pairs),
        "all_retained": all(pair["retained"] for pair in usable) and len(usable) == len(pairs),
        "request_wall_speedup": wall,
        "prompt_speedup": prompt,
        "peak_rss_delta_median_bytes": int(median(pair["peak_rss_delta_bytes"] for pair in usable)),
        "resident_dma_median_bytes": int(median(pair["qpu_execution"]["resident_dma_bytes"] for pair in usable)),
        "total_memory_overhead_median_bytes": int(
            median(
                max(0, pair["peak_rss_delta_bytes"]) + pair["qpu_execution"]["resident_dma_bytes"] for pair in usable
            )
        ),
    }


def choose_fraction(calibration: dict[float, dict[str, Any]]) -> float:
    """Choose the best correct calibration median with a smaller-fraction tie break."""
    usable = {
        fraction: summary
        for fraction, summary in calibration.items()
        if summary.get("all_correct") and summary.get("request_wall_speedup") is not None
    }
    if not usable:
        raise ValueError("no calibration fraction produced correct comparable pairs")
    return max(
        usable,
        key=lambda fraction: (
            usable[fraction]["request_wall_speedup"]["median_speedup"],
            -fraction,
        ),
    )


def write_calibration_failure_record(
    output: Path,
    *,
    args: argparse.Namespace,
    suffix: int,
    error: ValueError,
    selected: dict[int, float],
    calibration_summaries: dict[int, dict[float, dict[str, Any]]],
    calibration_pairs: dict[int, dict[float, list[dict[str, Any]]]],
) -> None:
    """Persist completed calibration evidence before reporting selection failure."""
    write_json_atomic(
        output,
        {
            "schema_version": 1,
            "kind": "llama-cpp-qpu-agentic-prefill-calibration-failure",
            "created_utc": utc_now(),
            "design": {
                "calibration_prefix": args.calibration_prefix,
                "suffixes": args.suffixes,
                "fractions": args.fractions,
                "calibration_sessions": args.calibration_sessions,
                "weight_mode": args.weight_mode,
                "wgs_per_supergroup": args.wgs,
                "seed": args.seed,
            },
            "failure": {
                "stage": "calibration-selection",
                "suffix_tokens": suffix,
                "message": str(error),
            },
            "selected_fraction_by_suffix": selected,
            "calibration_summaries": calibration_summaries,
            "calibration_pairs": calibration_pairs,
        },
    )


def main() -> None:
    """Tune only on calibration pairs, then evaluate fresh held-out pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-file",
        type=Path,
        default=ROOT / "integrations/llama_cpp/eval/agentic_cases.json",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "integrations/llama_cpp/eval/agentic_prefill_spec.json",
    )
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=Path("/home/yiannis/side/llama.cpp/build/bin/llama-server"),
    )
    parser.add_argument("--calibration-prefix", type=int, default=512)
    parser.add_argument("--heldout-prefixes", type=integer_list, default=integer_list("512,4096"))
    parser.add_argument("--suffixes", type=integer_list, default=integer_list("64,128,256"))
    parser.add_argument(
        "--fractions",
        type=fraction_list,
        default=fraction_list("0.0625,0.125,0.1875,0.25"),
    )
    parser.add_argument("--calibration-sessions", type=int, default=3)
    parser.add_argument("--heldout-sessions", type=int, default=7)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--cooldown-c", type=float, default=60.0)
    parser.add_argument("--cooldown-timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--weight-mode",
        choices=("exact", "column-w8", "rowcol-w8a8"),
        default="exact",
    )
    parser.add_argument("--wgs", type=int, default=24)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.quick:
        args.suffixes = (64,)
        args.heldout_prefixes = (512,)
        args.fractions = (0.0625, 0.125)
        args.calibration_sessions = 1
        args.heldout_sessions = 2
        args.bootstrap_resamples = 500
    if (
        args.calibration_prefix <= 0
        or args.calibration_sessions <= 0
        or args.heldout_sessions <= 0
        or args.bootstrap_resamples <= 0
        or args.cooldown_c <= 0.0
        or args.cooldown_timeout <= 0.0
        or not 1 <= args.wgs <= 255
    ):
        parser.error("prefix, session, and bootstrap counts must be positive")
    for path in (args.case_file, args.spec, args.llama_server):
        if not path.is_file():
            parser.error(f"required artifact not found: {path}")
    pairs = load_case_pairs(args.case_file)
    required_cells = {
        (prefix, suffix) for prefix in {args.calibration_prefix, *args.heldout_prefixes} for suffix in args.suffixes
    }
    missing = sorted(required_cells - pairs.keys())
    if missing:
        parser.error(f"case file lacks required prefix/suffix cells: {missing}")

    rng = random.Random(args.seed)
    calibration_pairs: dict[int, dict[float, list[dict[str, Any]]]] = {}
    calibration_summaries: dict[int, dict[float, dict[str, Any]]] = {}
    selected: dict[int, float] = {}
    for suffix in args.suffixes:
        cpu_case, base_candidate = pairs[(args.calibration_prefix, suffix)]
        calibration_pairs[suffix] = {}
        calibration_summaries[suffix] = {}
        for fraction in args.fractions:
            candidate = candidate_at_fraction(base_candidate, fraction, weight_mode=args.weight_mode, wgs=args.wgs)
            observations = [
                run_pair(
                    server=args.llama_server.resolve(),
                    cpu_case=cpu_case,
                    candidate_case=candidate,
                    phase="calibration",
                    session=session,
                    fraction=fraction,
                    rng=rng,
                    startup_timeout=args.startup_timeout,
                    request_timeout=args.request_timeout,
                    cooldown_c=args.cooldown_c,
                    cooldown_timeout=args.cooldown_timeout,
                )
                for session in range(1, args.calibration_sessions + 1)
            ]
            calibration_pairs[suffix][fraction] = observations
            calibration_summaries[suffix][fraction] = summarize_pairs(
                observations,
                seed=args.seed + suffix * 101 + round(fraction * 10_000),
                resamples=args.bootstrap_resamples,
            )
        try:
            selected[suffix] = choose_fraction(calibration_summaries[suffix])
        except ValueError as exc:
            write_calibration_failure_record(
                args.output,
                args=args,
                suffix=suffix,
                error=exc,
                selected=selected,
                calibration_summaries=calibration_summaries,
                calibration_pairs=calibration_pairs,
            )
            raise RuntimeError(f"{exc}; wrote calibration failure record to {args.output}") from exc

    heldout_pairs: dict[int, dict[int, list[dict[str, Any]]]] = {}
    heldout_summaries: dict[int, dict[int, dict[str, Any]]] = {}
    for prefix in args.heldout_prefixes:
        heldout_pairs[prefix] = {}
        heldout_summaries[prefix] = {}
        for suffix in args.suffixes:
            cpu_case, base_candidate = pairs[(prefix, suffix)]
            fraction = selected[suffix]
            candidate = candidate_at_fraction(base_candidate, fraction, weight_mode=args.weight_mode, wgs=args.wgs)
            observations = [
                run_pair(
                    server=args.llama_server.resolve(),
                    cpu_case=cpu_case,
                    candidate_case=candidate,
                    phase="heldout",
                    session=session,
                    fraction=fraction,
                    rng=rng,
                    startup_timeout=args.startup_timeout,
                    request_timeout=args.request_timeout,
                    cooldown_c=args.cooldown_c,
                    cooldown_timeout=args.cooldown_timeout,
                )
                for session in range(1, args.heldout_sessions + 1)
            ]
            heldout_pairs[prefix][suffix] = observations
            heldout_summaries[prefix][suffix] = summarize_pairs(
                observations,
                seed=args.seed + prefix * 1009 + suffix,
                resamples=args.bootstrap_resamples,
            )

    promotion_cells: list[dict[str, Any]] = []
    for prefix, by_suffix in heldout_summaries.items():
        for suffix, summary in by_suffix.items():
            speedup = summary.get("request_wall_speedup") or {}
            passed = bool(
                summary.get("all_correct")
                and summary.get("all_retained")
                and speedup.get("median_speedup", 0.0) >= 1.05
                and speedup.get("bootstrap_95_low", 0.0) > 1.0
                and summary.get("total_memory_overhead_median_bytes", 1 << 60) <= 734_003_200
            )
            promotion_cells.append(
                {
                    "prefix_tokens": prefix,
                    "suffix_tokens": suffix,
                    "observed_up_m": suffix + 1,
                    "selected_fraction": selected[suffix],
                    "passed": passed,
                }
            )
    model_path = Path(next(iter(pairs.values()))[0]["base_model"])
    plugin_path = Path(next(iter(pairs.values()))[1]["process_environment"]["LD_PRELOAD"])
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-agentic-prefill-calibration-heldout",
        "created_utc": utc_now(),
        "design": {
            "unit": "one CPU process and one candidate process with randomized order",
            "calibration_prefix": args.calibration_prefix,
            "heldout_prefixes": args.heldout_prefixes,
            "suffixes": args.suffixes,
            "observed_up_m": {suffix: suffix + 1 for suffix in args.suffixes},
            "fractions": args.fractions,
            "calibration_sessions": args.calibration_sessions,
            "heldout_sessions": args.heldout_sessions,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "weight_mode": args.weight_mode,
            "wgs_per_supergroup": args.wgs,
            "maximum_process_start_temperature_c": args.cooldown_c,
            "cooldown_timeout_seconds": args.cooldown_timeout,
        },
        "artifacts": {
            "case_file": {"path": str(args.case_file.resolve()), "sha256": sha256_file(args.case_file)},
            "spec": {"path": str(args.spec.resolve()), "sha256": sha256_file(args.spec)},
            "llama_server": {
                "path": str(args.llama_server.resolve()),
                "sha256": sha256_file(args.llama_server),
            },
            "plugin": {"path": str(plugin_path.resolve()), "sha256": sha256_file(plugin_path)},
            "model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
        },
        "workload_semantics_sha256": {
            f"p{prefix}-s{suffix}": workload_semantics_sha256(pairs[(prefix, suffix)][0])
            for prefix, suffix in sorted(required_cells)
        },
        "selected_fraction_by_suffix": selected,
        "calibration_summaries": calibration_summaries,
        "heldout_summaries": heldout_summaries,
        "calibration_pairs": calibration_pairs,
        "heldout_pairs": heldout_pairs,
        "promotion": {
            "cells": promotion_cells,
            "passed": bool(promotion_cells) and all(cell["passed"] for cell in promotion_cells),
            "rule": (
                "every held-out cell: exact outputs, 35 attested dispatches, zero "
                "fallbacks, retained environment, median request speedup >=1.05, "
                "bootstrap lower bound >1.0, RSS delta <=700 MiB"
            ),
        },
        "record_sha256": hashlib.sha256(
            json.dumps(
                {
                    "case_file": sha256_file(args.case_file),
                    "spec": sha256_file(args.spec),
                    "seed": args.seed,
                    "selected": selected,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: selected={selected}, promotion={payload['promotion']['passed']}")


if __name__ == "__main__":
    main()
