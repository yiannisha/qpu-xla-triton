#!/usr/bin/env python3
"""Generate the compact llama.cpp QPU latency matrix from retained raw sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import sha256_file, utc_now, write_json_atomic  # noqa: E402


def _milliseconds(nanoseconds: float | int | None) -> str:
    if nanoseconds is None:
        return "—"
    return f"{float(nanoseconds) / 1e6:.3f}"


def _number(value: float | int | None, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def operator_rows(session: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Flatten one native operator session without discarding raw timing categories."""
    retained = bool(session["validation"]["retained"])
    rows: list[dict[str, Any]] = []
    for record in session["records"]:
        result = record["result"]
        correctness = result.get("exact_cpu_repack_correctness", result["correctness"])
        speedup = result.get("speedup_vs_fastest_cpu", {})
        correct = (
            correctness["nan_count"] == 0
            and correctness["inf_count"] == 0
            and correctness["tolerance_violation_count"] == 0
        )
        median_speedup = speedup.get("median_speedup")
        ci_low = speedup.get("bootstrap_95_low")
        placement_eligible = bool(
            result["mode"] != "hybrid"
            or session.get("measurement_contract", {}).get("hybrid_cpu_side_exact", True)
        )
        operator_win = bool(
            retained
            and correct
            and placement_eligible
            and median_speedup is not None
            and median_speedup >= 1.05
            and ci_low is not None
            and ci_low > 1.0
        )
        if result["mode"] == "cpu":
            status = "baseline"
        elif operator_win:
            status = "operator-win-pending-end-to-end"
        elif correct and not placement_eligible:
            status = "experimental-nonproduction-cpu-prefix"
        elif correct:
            status = "experimental"
        else:
            status = "failed-correctness"
        rows.append(
            {
                "session": session["session_id"],
                "source": str(source),
                "model_sha256": session["manifest"].get("model_sha256"),
                "tensor": record["tensor"]["name"],
                "operator": record["tensor"]["owner"]["operator"],
                "ggml_type": record["tensor"]["ggml_type"],
                "m": result["rows"],
                "k": result["input_columns"],
                "n": result["output_columns"],
                "placement": result["mode"],
                "cpu_threads": result["cpu_threads"],
                "qpu_fraction": result["qpu_column_count"] / result["output_columns"],
                "complete_median_ns": result["summaries"]["complete_ns"]["median_ns"],
                "quantize_median_ns": result["summaries"]["quantize_ns"]["median_ns"],
                "input_copy_median_ns": result["summaries"]["qpu_input_copy_ns"]["median_ns"],
                "submit_wait_median_ns": result["summaries"]["qpu_submit_wait_ns"]["median_ns"],
                "output_copy_median_ns": result["summaries"]["qpu_output_copy_ns"]["median_ns"],
                "speedup": median_speedup,
                "speedup_ci95_low": ci_low,
                "speedup_ci95_high": speedup.get("bootstrap_95_high"),
                "max_absolute_error": correctness["max_absolute"],
                "correct": correct,
                "resident_bytes": result["resident_bytes"],
                "session_retained": retained,
                "placement_eligible": placement_eligible,
                "status": status,
                "rejection_reasons": session["validation"]["rejection_reasons"],
            }
        )
    return rows


def e2e_rows(session: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Flatten request samples and retain prompt, decode, acceptance, and cycle measures."""
    retained = bool(session["validation"]["retained"])
    semantic_lookup = {
        (item["candidate_case"], item["sample_index"]): item
        for item in session.get("greedy_semantic_comparisons", [])
    }
    rows: list[dict[str, Any]] = []
    for result in session["results"]:
        case = result["case"]
        for sample in result["samples"]:
            timings = sample.get("metrics", {}).get("timings", {})
            speculative = sample.get("metrics", {}).get("speculative", {})
            process_memory = sample.get("process_memory", {})
            candidate_execution = sample.get("candidate_execution", {})
            context_population = sample.get("context_population", {})
            semantic = semantic_lookup.get((case["name"], sample["sample_index"]))
            prompt_only = case["mode"] == "prompt"
            decode_timing_valid = not prompt_only and int(case.get("predict_tokens", 0)) > 1
            rows.append(
                {
                    "session": session["session_id"],
                    "source": str(source),
                    "case": case["name"],
                    "model": case.get("model"),
                    "mode": case["mode"],
                    "workload": case.get(
                        "workload", "prompt" if case["mode"] == "prompt" else "decode"
                    ),
                    "request_surface": case.get("request_surface", "completion"),
                    "workload_semantics_sha256": result.get(
                        "workload_semantics_sha256", result.get("prompt_sha256")
                    ),
                    "base_model_sha256": result.get("base_model_sha256"),
                    "draft_model_sha256": result.get("draft_model_sha256"),
                    "server_binary_sha256": session.get("llama_server", {}).get("sha256"),
                    "candidate_evidence": case.get("candidate_evidence"),
                    "partition": case.get("partition"),
                    "placement": case["placement"],
                    "threads": case["threads"],
                    "draft_n_max": case.get("draft_n_max"),
                    "context_tokens_target": case.get("context_tokens_target"),
                    "prompt_tokens_target": case.get("prompt_tokens_target"),
                    "sample_index": sample["sample_index"],
                    "startup_ns": sample.get("startup_ns", sample.get("process_wall_ns")),
                    "request_ns": sample.get("request_wall_ns"),
                    "peak_rss_bytes": process_memory.get("peak_rss_bytes"),
                    "process_swap_bytes": process_memory.get("swap_bytes"),
                    "candidate_execution_valid": candidate_execution.get("valid"),
                    "candidate_dispatch_count": candidate_execution.get("dispatch_count"),
                    "prefill_ns": context_population.get("prefill_wall_ns"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "prompt_tokens": timings.get("prompt_n"),
                    "prompt_tokens_per_second": timings.get("prompt_per_second"),
                    "decode_ms": timings.get("predicted_ms") if decode_timing_valid else None,
                    "decode_tokens": timings.get("predicted_n") if decode_timing_valid else None,
                    "tokens_per_second": (
                        timings.get("predicted_per_second") if decode_timing_valid else None
                    ),
                    "mean_accepted_length": speculative.get("mean_accepted_length"),
                    "cycle_seconds": speculative.get("cycle_seconds"),
                    "content_sha256": sample.get("content_sha256", sample.get("stdout_sha256")),
                    "greedy_identical": (
                        semantic.get("identical")
                        if semantic and semantic.get("token_ids_available") is True
                        else None
                    ),
                    "tool_call_valid": (
                        sample.get("tool_call_validation", {}).get("valid")
                        if sample.get("tool_call_validation") is not None
                        else None
                    ),
                    "success": sample["returncode"] == 0,
                    "session_retained": retained,
                    "status": "retained" if retained and sample["returncode"] == 0 else "diagnostic",
                    "rejection_reasons": session["validation"]["rejection_reasons"],
                }
            )
    return rows


def _configuration_key(row: dict[str, Any]) -> str:
    configuration = {
        "placement": row["placement"],
        "threads": row["threads"],
        "draft_n_max": row["draft_n_max"],
        "server_binary_sha256": row.get("server_binary_sha256"),
        "candidate_evidence": row.get("candidate_evidence"),
        "partition": row.get("partition"),
    }
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def _workload_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model"),
        row.get("mode"),
        row.get("workload"),
        row.get("request_surface"),
        row.get("context_tokens_target"),
        row.get("prompt_tokens_target"),
        row.get("workload_semantics_sha256"),
        row.get("base_model_sha256"),
        row.get("draft_model_sha256"),
    )


def _metric(row: dict[str, Any]) -> tuple[str, float] | None:
    if row["workload"] == "prompt" or row["workload"] == "structured-tool-call":
        value = row.get("request_ns")
        return ("request_latency_ns", float(value)) if value is not None else None
    if row["mode"] == "mtp":
        value = row.get("cycle_seconds")
        return ("mtp_cycle_seconds", float(value)) if value is not None else None
    request_ns = row.get("request_ns")
    generated_tokens = row.get("decode_tokens")
    if (
        request_ns is None
        or generated_tokens is None
        or float(generated_tokens) <= 0
    ):
        return None
    return (
        "request_seconds_per_generated_token",
        float(request_ns) / 1e9 / float(generated_tokens),
    )


def _session_medians(rows: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
    samples: dict[str, list[float]] = {}
    for row in rows:
        metric = _metric(row)
        if (
            metric is None
            or metric[0] != metric_name
            or not row["session_retained"]
            or not row["success"]
        ):
            continue
        samples.setdefault(str(row["session"]), []).append(metric[1])
    return {session: float(median(values)) for session, values in samples.items() if values}


def _bootstrap_speedup(
    baseline: list[float],
    candidate: list[float],
    *,
    seed: int = 8842,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    baseline_indices = rng.integers(
        0, baseline_array.size, size=(resamples, baseline_array.size)
    )
    candidate_indices = rng.integers(
        0, candidate_array.size, size=(resamples, candidate_array.size)
    )
    ratios = np.median(baseline_array[baseline_indices], axis=1) / np.median(
        candidate_array[candidate_indices], axis=1
    )
    return {
        "median_speedup": float(median(baseline) / median(candidate)),
        "bootstrap_95_low": float(np.quantile(ratios, 0.025)),
        "bootstrap_95_high": float(np.quantile(ratios, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _session_evidence_complete(rows: list[dict[str, Any]]) -> bool:
    eligible = [row for row in rows if row["session_retained"] and row["success"]]
    return bool(
        eligible
        and all(
            _is_sha256(row.get("workload_semantics_sha256"))
            and _is_sha256(row.get("base_model_sha256"))
            and _is_sha256(row.get("server_binary_sha256"))
            and (row.get("mode") != "mtp" or _is_sha256(row.get("draft_model_sha256")))
            and isinstance(row.get("peak_rss_bytes"), int)
            and row["peak_rss_bytes"] > 0
            and row.get("process_swap_bytes") == 0
            and row.get("candidate_execution_valid") is True
            and (
                row.get("placement") == "cpu-only"
                or (
                    isinstance(row.get("candidate_dispatch_count"), int)
                    and row["candidate_dispatch_count"] > 0
                )
            )
            for row in eligible
        )
    )


def _candidate_evidence_complete(rows: list[dict[str, Any]]) -> bool:
    required = {
        "program_manifest_path",
        "program_manifest_sha256",
        "program",
        "source_hash",
        "binary_sha256",
        "exact_shape",
    }
    evidence = [row.get("candidate_evidence") for row in rows]
    hash_fields = ("program_manifest_sha256", "source_hash", "binary_sha256")
    if not evidence:
        return False
    for item in evidence:
        if not isinstance(item, dict) or not required <= item.keys():
            return False
        exact_shape = item["exact_shape"]
        if not isinstance(exact_shape, dict) or not exact_shape:
            return False
        for field in hash_fields:
            value = item[field]
            if not _is_sha256(value):
                return False
    return (
        _session_evidence_complete(rows)
        and len({json.dumps(item, sort_keys=True) for item in evidence}) == 1
    )


def end_to_end_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the five-independent-session end-to-end promotion gate."""
    workloads: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        workloads.setdefault(_workload_key(row), []).append(row)
    comparisons: list[dict[str, Any]] = []
    for workload_key, workload_rows in workloads.items():
        configurations: dict[str, list[dict[str, Any]]] = {}
        for row in workload_rows:
            configurations.setdefault(_configuration_key(row), []).append(row)
        baseline_options: list[tuple[float, str, dict[str, float]]] = []
        for key, configuration_rows in configurations.items():
            if configuration_rows[0]["placement"] != "cpu-only":
                continue
            metric = _metric(configuration_rows[0])
            if metric is None:
                continue
            session_values = _session_medians(configuration_rows, metric[0])
            if len(session_values) >= 5 and _session_evidence_complete(configuration_rows):
                baseline_options.append(
                    (float(median(session_values.values())), key, session_values)
                )
        baseline_options.sort(key=lambda item: item[0])
        for candidate_key, candidate_rows in configurations.items():
            if candidate_rows[0]["placement"] == "cpu-only":
                continue
            metric = _metric(candidate_rows[0])
            candidate_sessions = (
                _session_medians(candidate_rows, metric[0]) if metric is not None else {}
            )
            baseline_key = baseline_options[0][1] if baseline_options else None
            baseline_sessions = baseline_options[0][2] if baseline_options else {}
            retained_candidate_rows = [
                row
                for row in candidate_rows
                if row["session_retained"] and row["success"]
            ]
            correctness = bool(
                retained_candidate_rows
                and all(
                    row.get("greedy_identical") is True
                    and (
                        row.get("workload") != "structured-tool-call"
                        or row.get("tool_call_valid") is True
                    )
                    for row in retained_candidate_rows
                )
            )
            evidence_complete = _candidate_evidence_complete(retained_candidate_rows)
            speedup = (
                _bootstrap_speedup(
                    list(baseline_sessions.values()), list(candidate_sessions.values())
                )
                if baseline_sessions and candidate_sessions
                else None
            )
            if len(baseline_sessions) < 5:
                status = "missing-five-session-baseline"
            elif len(candidate_sessions) < 5:
                status = "missing-five-session-candidate"
            elif not correctness:
                status = "failed-greedy-semantics"
            elif not evidence_complete:
                status = "incomplete-candidate-evidence"
            elif (
                speedup is None
                or speedup["median_speedup"] < 1.05
                or speedup["bootstrap_95_low"] <= 1.0
            ):
                status = "no-end-to-end-win"
            else:
                status = "promoted"
            first = candidate_rows[0]
            comparisons.append(
                {
                    "workload_key": list(workload_key),
                    "model": first.get("model"),
                    "mode": first.get("mode"),
                    "workload": first.get("workload"),
                    "context_tokens_target": first.get("context_tokens_target"),
                    "prompt_tokens_target": first.get("prompt_tokens_target"),
                    "metric": metric[0] if metric is not None else None,
                    "baseline_configuration": baseline_key,
                    "candidate_configuration": candidate_key,
                    "baseline_session_count": len(baseline_sessions),
                    "candidate_session_count": len(candidate_sessions),
                    "baseline_session_medians": baseline_sessions,
                    "candidate_session_medians": candidate_sessions,
                    "speedup": speedup,
                    "greedy_semantics_passed": correctness,
                    "candidate_evidence_complete": evidence_complete,
                    "status": status,
                }
            )
    return comparisons


def attention_rows(session: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Flatten exact-GGML synthetic attention diagnostics without implying retention."""
    rows: list[dict[str, Any]] = []
    for record in session["records"]:
        correctness = record["exact_ggml_cpu_correctness"]
        correct = (
            correctness["nan_count"] == 0
            and correctness["inf_count"] == 0
            and correctness["tolerance_violation_count"] == 0
        )
        rows.append(
            {
                "source": str(source),
                "query_rows": record["query_rows"],
                "query_heads": record["query_heads"],
                "kv_heads": record["kv_heads"],
                "context_rows": record["context_rows"],
                "head_dim": record["head_dim"],
                "cpu_complete_median_ns": record["cpu_exact_ggml_node_summary"]["median_ns"],
                "qpu_complete_median_ns": record["qpu_persistent_execute_summary"]["median_ns"],
                "speedup": record["diagnostic_speedup_vs_exact_ggml_cpu"],
                "max_absolute_error": correctness["max_absolute"],
                "correct": correct,
                "input_bytes": record["native_input_bytes"],
                "dispatch_count": record["dispatch_count"],
                "session_retained": False,
                "status": "experimental" if correct else "failed-correctness",
                "rejection_reasons": session["measurement_contract"]["reasons"],
            }
        )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact, explicit matrix with no implied promotion from diagnostics."""
    lines = [
        "# llama.cpp QPU latency matrix",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "Native operator wins are provisional evidence only. Promotion requires a complete-request "
        "comparison against the fastest tuned CPU configuration, identical greedy semantics, exact "
        "candidate hashes, at least five retained independent sessions per side, median speedup of "
        "at least 1.05x, and a bootstrap 95% CI lower bound above 1.0. Diagnostic rows are never "
        "automatic placements.",
        "",
        "## Native quantized operator matrix",
        "",
        "| Tensor | Type | M×K×N | Placement | CPU threads | QPU share | Complete ms | Quant ms | "
        "Input copy ms | Submit+wait ms | Output copy ms | Speedup (95% low) | Max abs err | "
        "Resident MiB | Retained | Session | Status |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["operator_rows"]:
        operator_speedup_text = (
            "—"
            if row["speedup"] is None
            else f"{row['speedup']:.3f} ({row['speedup_ci95_low']:.3f})"
        )
        lines.append(
            f"| `{row['tensor']}` | {row['ggml_type']} | "
            f"{row['m']}×{row['k']}×{row['n']} | {row['placement']} | "
            f"{row['cpu_threads']} | {row['qpu_fraction']:.3f} | "
            f"{_milliseconds(row['complete_median_ns'])} | {_milliseconds(row['quantize_median_ns'])} | "
            f"{_milliseconds(row['input_copy_median_ns'])} | "
            f"{_milliseconds(row['submit_wait_median_ns'])} | "
            f"{_milliseconds(row['output_copy_median_ns'])} | {operator_speedup_text} | "
            f"{row['max_absolute_error']:.3g} | {row['resident_bytes'] / (1 << 20):.3f} | "
            f"{row['session_retained']} | {row['session']} | {row['status']} |"
        )
    if not payload["operator_rows"]:
        lines.append(
            "| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | no data |"
        )
    attention = payload.get("attention_rows", [])
    lines.extend(
        [
            "",
            "## Fused attention diagnostics",
            "",
            "These rows use the exact pinned GGML CPU node on deterministic synthetic tensors. "
            "They remain nonpromotable until replayed on captured model nodes in an isolated session.",
            "",
            "| M | Q heads / KV heads | KV rows | Head dim | CPU ms | QPU ms | Speedup | "
            "Max abs err | Input MiB | Dispatches | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in attention:
        lines.append(
            f"| {row['query_rows']} | {row['query_heads']} / {row['kv_heads']} | "
            f"{row['context_rows']} | {row['head_dim']} | "
            f"{_milliseconds(row['cpu_complete_median_ns'])} | "
            f"{_milliseconds(row['qpu_complete_median_ns'])} | {row['speedup']:.3f}x | "
            f"{row['max_absolute_error']:.3g} | {row['input_bytes'] / (1 << 20):.3f} | "
            f"{row['dispatch_count']} | {row['status']} |"
        )
    if not attention:
        lines.append("| — | — | — | — | — | — | — | — | — | — | no data |")
    coverage = payload.get("coverage", {})
    lines.extend(
        [
            "",
            "## End-to-end matrix",
            "",
            "| Case | Mode | Workload | Endpoint | Placement | Threads | Depth | Context target | "
            "Prompt target | Startup s | Prefill s | Request s | Prompt ms | Prompt tok/s | "
            "Decode tokens | Decode tok/s | Mean accepted | Cycle s | Greedy identical | "
            "Tool valid | QPU dispatches | Peak RSS MiB | Process swap KiB | Session | Status |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["end_to_end_rows"]:
        lines.append(
            f"| {row['case']} | {row['mode']} | {row['workload']} | "
            f"{row['request_surface']} | {row['placement']} | {row['threads']} | "
            f"{row['draft_n_max'] if row['draft_n_max'] is not None else '—'} | "
            f"{row['context_tokens_target'] if row['context_tokens_target'] is not None else '—'} | "
            f"{row['prompt_tokens_target'] if row['prompt_tokens_target'] is not None else '—'} | "
            f"{_number(None if row['startup_ns'] is None else row['startup_ns'] / 1e9)} | "
            f"{_number(None if row['prefill_ns'] is None else row['prefill_ns'] / 1e9)} | "
            f"{_number(None if row['request_ns'] is None else row['request_ns'] / 1e9)} | "
            f"{_number(row['prompt_ms'])} | {_number(row['prompt_tokens_per_second'])} | "
            f"{_number(row['decode_tokens'], 0)} | {_number(row['tokens_per_second'])} | "
            f"{_number(row['mean_accepted_length'], 2)} | {_number(row['cycle_seconds'])} | "
            f"{row['greedy_identical'] if row['greedy_identical'] is not None else '—'} | "
            f"{row['tool_call_valid'] if row['tool_call_valid'] is not None else '—'} | "
            f"{row['candidate_dispatch_count'] if row['candidate_dispatch_count'] is not None else '—'} | "
            f"{_number(None if row['peak_rss_bytes'] is None else row['peak_rss_bytes'] / (1 << 20))} | "
            f"{_number(None if row['process_swap_bytes'] is None else row['process_swap_bytes'] / 1024)} | "
            f"{row['session']} | {row['status']} |"
        )
    if not payload["end_to_end_rows"]:
        lines.append(
            "| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | "
            "— | — | — | — | — | — | — | — | no data |"
        )
    comparisons = payload.get("end_to_end_comparisons", [])
    lines.extend(
        [
            "",
            "## End-to-end promotion gate",
            "",
            "Each timing value entering this gate is first reduced to one median per independent "
            "session. Candidate rows cannot be promoted from operator timing alone.",
            "",
            "| Model | Mode | Workload | Context | Prompt | Metric | Baseline sessions | "
            "Candidate sessions | Speedup (95% CI) | Greedy exact | Evidence exact | Status |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|",
        ]
    )
    for comparison in comparisons:
        comparison_speedup = comparison.get("speedup")
        speedup_text = (
            "—"
            if comparison_speedup is None
            else f"{comparison_speedup['median_speedup']:.3f} "
            f"({comparison_speedup['bootstrap_95_low']:.3f}–"
            f"{comparison_speedup['bootstrap_95_high']:.3f})"
        )
        lines.append(
            f"| {comparison['model']} | {comparison['mode']} | {comparison['workload']} | "
            f"{comparison['context_tokens_target'] or 0} | "
            f"{comparison['prompt_tokens_target'] or 0} | {comparison['metric'] or '—'} | "
            f"{comparison['baseline_session_count']} | {comparison['candidate_session_count']} | "
            f"{speedup_text} | {comparison['greedy_semantics_passed']} | "
            f"{comparison['candidate_evidence_complete']} | {comparison['status']} |"
        )
    if not comparisons:
        lines.append("| — | — | — | — | — | — | 0 | 0 | — | — | — | no non-CPU sessions |")
    lines.extend(
        [
            "",
            "## Coverage and retention",
            "",
            f"- Operator rows: {len(payload['operator_rows'])}.",
            f"- End-to-end rows: {len(payload['end_to_end_rows'])}.",
            f"- Attention diagnostic rows: {len(attention)}.",
            f"- CPU baseline operator rows: {sum(row['status'] == 'baseline' for row in payload['operator_rows'])}.",
            f"- Provisional operator wins: "
            f"{sum(row['status'] == 'operator-win-pending-end-to-end' for row in payload['operator_rows'])}.",
            f"- Retained end-to-end rows: "
            f"{sum(row['status'] == 'retained' for row in payload['end_to_end_rows'])}.",
            f"- Promoted end-to-end configurations: "
            f"{sum(item['status'] == 'promoted' for item in comparisons)}.",
            f"- Planned end-to-end cases: {coverage.get('planned_case_count', 'not loaded')}.",
            "- Planned workload counts: "
            + (
                ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(
                        coverage.get("planned_workload_counts", {}).items()
                    )
                )
                or "not loaded"
            )
            + ".",
            "- Missing model/workload combinations remain coverage gaps; absence is not a CPU or QPU win.",
        ]
    )
    for gap in coverage.get("coverage_gaps", []):
        lines.append(f"- Coverage gap `{gap['model']}`: {gap['reason']}.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Load raw sessions, flatten them, and atomically emit JSON plus Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-session", type=Path, action="append", default=[])
    parser.add_argument("--end-to-end-session", type=Path, action="append", default=[])
    parser.add_argument("--attention-session", type=Path, action="append", default=[])
    parser.add_argument("--case-matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    operator_records: list[dict[str, Any]] = []
    e2e_records: list[dict[str, Any]] = []
    attention_records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for path in args.operator_session:
        session = json.loads(path.read_text(encoding="utf-8"))
        operator_records.extend(operator_rows(session, path.resolve()))
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "operator"})
    for path in args.end_to_end_session:
        session = json.loads(path.read_text(encoding="utf-8"))
        e2e_records.extend(e2e_rows(session, path.resolve()))
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "end-to-end"})
    for path in args.attention_session:
        session = json.loads(path.read_text(encoding="utf-8"))
        attention_records.extend(attention_rows(session, path.resolve()))
        sources.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "attention"}
        )
    if args.case_matrix is not None:
        case_matrix = json.loads(args.case_matrix.read_text(encoding="utf-8"))
        planned_cases = case_matrix.get("cases", [])
        mode_counts: dict[str, int] = {}
        workload_counts: dict[str, int] = {}
        for case in planned_cases:
            mode = str(case.get("mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            workload = str(case.get("workload", "unknown"))
            workload_counts[workload] = workload_counts.get(workload, 0) + 1
        coverage = {
            "case_matrix": str(args.case_matrix.resolve()),
            "planned_case_count": len(planned_cases),
            "planned_mode_counts": mode_counts,
            "planned_workload_counts": workload_counts,
            "coverage_gaps": case_matrix.get("coverage_gaps", []),
        }
        sources.append(
            {
                "path": str(args.case_matrix.resolve()),
                "sha256": sha256_file(args.case_matrix),
                "kind": "case-matrix",
            }
        )
    comparisons = end_to_end_comparisons(e2e_records)
    payload = {
        "schema_version": 2,
        "kind": "llama-cpp-qpu-latency-matrix",
        "created_utc": utc_now(),
        "sources": sources,
        "operator_rows": operator_records,
        "attention_rows": attention_records,
        "end_to_end_rows": e2e_records,
        "end_to_end_comparisons": comparisons,
        "coverage": coverage,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")
    write_json_atomic(args.json_output or output.with_suffix(".json"), payload)
    print(
        f"wrote {output}: {len(operator_records)} operator rows, "
        f"{len(attention_records)} attention rows, {len(e2e_records)} end-to-end rows, "
        f"{len(comparisons)} promotion comparisons"
    )


if __name__ == "__main__":
    main()
