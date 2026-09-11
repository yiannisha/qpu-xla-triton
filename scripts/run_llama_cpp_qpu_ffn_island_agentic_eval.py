#!/usr/bin/env python3
"""Evaluate the complete FFN island on cached post-tool llama-server requests."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import collect_environment, sha256_file, utc_now, write_json_atomic  # noqa: E402
from scripts.run_llama_cpp_qpu_evaluation import normalize_case, run_server_case  # noqa: E402
from scripts.run_llama_cpp_qpu_ffn_island_eval import (  # noqa: E402
    ISLAND_PREFIX,
    WEIGHT_PREFIX,
    git_record,
    integer_list,
    manifest_programs,
    paired_speedup_interval,
    parse_prefixed_json,
    require_no_competing_workloads,
    retention,
    run_correctness_preflight,
    validate_candidate,
    wait_until_cool,
)


def semantic_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left.get("token_ids")
        and left.get("token_ids") == right.get("token_ids")
        and left.get("content") == right.get("content")
        and left.get("stop_reason") == right.get("stop_reason")
        and left.get("tool_calls") == right.get("tool_calls")
    )


def prompt_ns(sample: dict[str, Any]) -> int:
    milliseconds = sample.get("metrics", {}).get("timings", {}).get("prompt_ms")
    if milliseconds is None:
        raise RuntimeError("server response did not report prompt_ms")
    return round(float(milliseconds) * 1_000_000)


def base_case(model: Path, prefix: int, suffix: int, threads: int) -> dict[str, Any]:
    return normalize_case(
        {
            "name": f"agentic-p{prefix}-s{suffix}",
            "model": "gemma-4-e2b",
            "mode": "plain",
            "surface": "server",
            "base_model": str(model.resolve()),
            "prompt": "Use the newly returned tool evidence to answer with one short token.",
            "context_filler": "A stable cached agent transcript records deterministic prior reasoning. ",
            "suffix_filler": "Tool result: deterministic JSON evidence was returned successfully. ",
            "workload": "decode",
            "request_surface": "completion",
            "predict_tokens": 1,
            "context_size": max(2048, prefix + suffix + 512),
            "context_tokens_target": prefix,
            "suffix_tokens_target": suffix,
            "threads": threads,
            "threads_batch": threads,
            "flash_attention": True,
            "seed": 1234,
            "temperature": 0.0,
            "placement": "cpu-only",
            # Keep the baseline CPU-only even when the caller's interactive
            # shell contains variables from an earlier QPU run.
            "process_environment": {
                "LD_PRELOAD": "",
                "GGML_QPU_FFN_ISLAND": "0",
                "GGML_QPU_TELEMETRY": "0",
            },
        },
        index=0,
    )


def candidate_case(
    baseline: dict[str, Any],
    *,
    plugin: Path,
    manifest: Path,
    programs: dict[str, dict[str, Any]],
    fraction: float,
    minimum_rows: int,
    maximum_rows: int,
    wgs: int,
) -> dict[str, Any]:
    result = copy.deepcopy(baseline)
    result["name"] += "-qpu-ffn-island"
    result["placement"] = "hybrid"
    result["partition"] = {"axis": "intermediate_columns", "fraction": fraction}
    linear = programs["ggml-q4-0-q8-wordscale-mx"]
    # run_server_case requires single-program evidence for a non-CPU label.
    # This wrapper replaces its generic validation result with the complete
    # two-program island contract after the process exits.
    result["candidate_evidence"] = {
        "program_manifest_path": str(manifest.resolve()),
        "program_manifest_sha256": sha256_file(manifest),
        "program": linear["name"],
        "source_hash": linear["source_hash"],
        "binary_sha256": linear["binary_sha256"],
        "exact_shape": {
            "weight": "Q4_0",
            "activation": "CPU_REPACK_Q8_0x4",
            "intermediate": "Q8_0_SPLIT",
            "output": "F32_PARTIAL",
        },
    }
    result["process_environment"] = {
        "LD_PRELOAD": str(plugin.resolve()),
        "GGML_QPU_FFN_ISLAND": "1",
        "GGML_QPU_FFN_ISLAND_FRACTION": str(fraction),
        "GGML_QPU_FFN_ISLAND_MIN_ROWS": str(minimum_rows),
        "GGML_QPU_FFN_ISLAND_MAX_ROWS": str(maximum_rows),
        "GGML_QPU_FFN_ISLAND_WGS": str(wgs),
        "GGML_QPU_TELEMETRY": "1",
    }
    return result


def execute(
    server: Path,
    case: dict[str, Any],
    sample_index: int,
    *,
    cooldown_c: float,
    programs: dict[str, dict[str, Any]],
    minimum_rows: int,
    maximum_rows: int,
) -> dict[str, Any]:
    require_no_competing_workloads("before agentic cooldown")
    cooldown = wait_until_cool(cooldown_c, 600.0)
    require_no_competing_workloads("before agentic server launch")
    sample = run_server_case(
        server,
        case,
        sample_index,
        startup_timeout=180.0,
        request_timeout=300.0,
    )
    require_no_competing_workloads("after agentic server completion")
    sample["cooldown"] = cooldown
    if sample["returncode"] != 0:
        raise RuntimeError(f"server case failed: {sample.get('error')}\n{sample.get('server_log', '')}")
    events = parse_prefixed_json(sample["server_log"], ISLAND_PREFIX)
    weights = parse_prefixed_json(sample["server_log"], WEIGHT_PREFIX)
    # The generic evaluator knows only a single-program placement. Its result
    # is inapplicable to a four-stage/two-program island; all raw events remain
    # in server_log and are validated below.
    sample.pop("candidate_execution", None)
    if case["placement"] == "cpu-only":
        if events or weights:
            raise RuntimeError("CPU-only server emitted FFN-island telemetry")
        execution = {"valid": True, "event_count": 0}
    else:
        expected_m = int(case["suffix_tokens_target"]) + 1
        execution = validate_candidate(
            events,
            weights,
            rows=expected_m,
            minimum_rows=minimum_rows,
            maximum_rows=maximum_rows,
            programs=programs,
        )
        if not execution["valid"]:
            raise RuntimeError(f"invalid FFN-island server execution: {execution['errors']}")
    sample["candidate_execution"] = execution
    sample["ffn_island_execution"] = execution
    return sample


def agentic_retention(
    before: dict[str, Any],
    after: dict[str, Any],
    samples: list[dict[str, Any]],
    pair_results: list[dict[str, Any]],
) -> dict[str, Any]:
    result = retention(
        before,
        after,
        [{"returncode": sample["returncode"]} for sample in samples],
    )
    reasons = list(result["rejection_reasons"])
    if not all(sample["context_population"]["valid"] for sample in samples):
        reasons.append("one or more cached-context populations were invalid")
    if not all(sample["candidate_execution"]["valid"] for sample in samples):
        reasons.append("one or more CPU/QPU execution contracts were invalid")
    if not all(pair["all_greedy_outputs_identical"] for pair in pair_results):
        reasons.append("one or more paired greedy outputs differed")
    if not all(pair["all_qpu_executions_valid"] for pair in pair_results):
        reasons.append("one or more paired QPU executions were invalid")
    return {**result, "retained": not reasons, "rejection_reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=Path("/home/yiannis/side/llama.cpp/build/bin/llama-server"))
    parser.add_argument("--model", type=Path, default=Path("/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"))
    parser.add_argument("--plugin", type=Path, default=ROOT / "build/llama-qpu-runtime/libggml-qpu-inline.so")
    parser.add_argument("--program-manifest", type=Path, default=ROOT / "integrations/llama_cpp/generated/manifest.json")
    parser.add_argument(
        "--stride-smoke",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_repack_stride_smoke",
    )
    parser.add_argument(
        "--graph-smoke",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_ffn_island_graph_smoke",
    )
    parser.add_argument(
        "--model-smoke",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_ffn_island_model_smoke",
    )
    parser.add_argument("--prefix", type=int, default=512)
    parser.add_argument("--suffixes", type=integer_list, default=integer_list("64,128"))
    parser.add_argument("--fraction", type=float, default=0.125)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-rows", type=int, default=64)
    parser.add_argument("--maximum-rows", type=int, default=144)
    parser.add_argument("--wgs", type=int, default=24)
    parser.add_argument("--cooldown-c", type=float, default=65.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.samples = 1
        args.bootstrap_resamples = 500
    if args.samples <= 0 or args.prefix <= args.maximum_rows or args.maximum_rows < max(args.suffixes) + 1:
        parser.error("samples must be positive; prefix must exceed max rows; max rows must cover suffix+1")
    for path in (
        args.server,
        args.model,
        args.plugin,
        args.program_manifest,
        args.stride_smoke,
        args.graph_smoke,
        args.model_smoke,
    ):
        if not path.is_file():
            parser.error(f"required artifact not found: {path}")

    programs = manifest_programs(args.program_manifest)
    correctness_preflight = run_correctness_preflight(
        (
            (args.stride_smoke,),
            (args.graph_smoke,),
            (args.model_smoke, args.model),
        )
    )
    before = collect_environment()
    samples: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    schedule = [
        (suffix, sample_index, placement)
        for suffix in args.suffixes
        for sample_index in range(args.samples)
        for placement in ("cpu", "qpu")
    ]
    random.Random(args.seed).shuffle(schedule)
    for suffix, sample_index, placement in schedule:
        cpu = base_case(args.model, args.prefix, suffix, args.threads)
        case = cpu if placement == "cpu" else candidate_case(
            cpu,
            plugin=args.plugin,
            manifest=args.program_manifest,
            programs=programs,
            fraction=args.fraction,
            minimum_rows=args.minimum_rows,
            maximum_rows=args.maximum_rows,
            wgs=args.wgs,
        )
        print(f"running suffix={suffix} sample={sample_index} placement={placement}", flush=True)
        sample = execute(
            args.server,
            case,
            sample_index,
            cooldown_c=args.cooldown_c,
            programs=programs,
            minimum_rows=args.minimum_rows,
            maximum_rows=args.maximum_rows,
        )
        sample["case"] = case
        sample["suffix"] = suffix
        sample["placement"] = placement
        samples.append(sample)
        write_json_atomic(args.output, {"schema_version": 1, "kind": "partial-agentic-ffn-island", "samples": samples})

    for suffix in args.suffixes:
        cpu_samples = sorted(
            (item for item in samples if item["suffix"] == suffix and item["placement"] == "cpu"),
            key=lambda item: item["sample_index"],
        )
        qpu_samples = sorted(
            (item for item in samples if item["suffix"] == suffix and item["placement"] == "qpu"),
            key=lambda item: item["sample_index"],
        )
        semantic_matches = [
            semantic_identity(cpu["response_semantics"], qpu["response_semantics"])
            for cpu, qpu in zip(cpu_samples, qpu_samples, strict=True)
        ]
        prompt_interval = paired_speedup_interval(
            [prompt_ns(item) for item in cpu_samples],
            [prompt_ns(item) for item in qpu_samples],
            seed=args.seed + suffix * 17,
            resamples=args.bootstrap_resamples,
        )
        request_interval = paired_speedup_interval(
            [int(item["request_wall_ns"]) for item in cpu_samples],
            [int(item["request_wall_ns"]) for item in qpu_samples],
            seed=args.seed + suffix * 19,
            resamples=args.bootstrap_resamples,
        )
        pair_results.append(
            {
                "suffix_tokens": suffix,
                "expected_physical_m": suffix + 1,
                "all_greedy_outputs_identical": all(semantic_matches),
                "greedy_output_matches": semantic_matches,
                "prompt_processing_speedup": prompt_interval,
                "request_wall_speedup": request_interval,
                "all_context_populations_valid": all(item["context_population"]["valid"] for item in cpu_samples + qpu_samples),
                "all_qpu_executions_valid": all(item["ffn_island_execution"]["valid"] for item in qpu_samples),
            }
        )
    after = collect_environment()
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-complete-ffn-island-agentic-evaluation",
        "created_utc": utc_now(),
        "contract": {
            "scenario": "cached agent transcript followed by an exact-size tool-result suffix",
            "timed_boundary": "post-tool server request through the first greedy generated token",
            "baseline": "llama.cpp CPU_REPACK",
            "candidate": "complete QPU FFN channel island at the uncached suffix only",
            "correctness": (
                "preflight stride, joined-FFN, and full-vocabulary differentials "
                "plus exact raw greedy token IDs, bytes, stop reason, and "
                "tool-call structure"
            ),
        },
        "configuration": {
            "prefix_tokens": args.prefix,
            "suffix_tokens": args.suffixes,
            "fraction": args.fraction,
            "samples": args.samples,
            "threads": args.threads,
            "minimum_rows": args.minimum_rows,
            "maximum_rows": args.maximum_rows,
            "wgs": args.wgs,
            "seed": args.seed,
        },
        "artifacts": {
            "server": str(args.server.resolve()),
            "server_sha256": sha256_file(args.server),
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
        "correctness_preflight": correctness_preflight,
        "samples": samples,
        "pair_results": pair_results,
        "environment_before": before,
        "environment_after": after,
        "retention": agentic_retention(before, after, samples, pair_results),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "retention": payload["retention"], "pair_results": pair_results}, indent=2))


if __name__ == "__main__":
    main()
