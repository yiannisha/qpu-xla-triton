#!/usr/bin/env python3
"""Generate the compact llama.cpp QPU latency matrix from retained raw sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
        correctness = result["correctness"]
        speedup = result.get("speedup_vs_fastest_cpu", {})
        correct = (
            correctness["nan_count"] == 0
            and correctness["inf_count"] == 0
            and correctness["tolerance_violation_count"] == 0
        )
        median_speedup = speedup.get("median_speedup")
        ci_low = speedup.get("bootstrap_95_low")
        promoted = bool(
            retained
            and correct
            and median_speedup is not None
            and median_speedup >= 1.05
            and ci_low is not None
            and ci_low > 1.0
        )
        if result["mode"] == "cpu":
            status = "baseline"
        elif promoted:
            status = "promoted"
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
                    "placement": case["placement"],
                    "threads": case["threads"],
                    "draft_n_max": case.get("draft_n_max"),
                    "context_tokens_target": case.get("context_tokens_target"),
                    "prompt_tokens_target": case.get("prompt_tokens_target"),
                    "sample_index": sample["sample_index"],
                    "startup_ns": sample.get("startup_ns", sample.get("process_wall_ns")),
                    "request_ns": sample.get("request_wall_ns"),
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
                    "greedy_identical": semantic.get("identical") if semantic else None,
                    "success": sample["returncode"] == 0,
                    "session_retained": retained,
                    "status": "retained" if retained and sample["returncode"] == 0 else "diagnostic",
                    "rejection_reasons": session["validation"]["rejection_reasons"],
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
        "Only rows whose session is retained, correctness passes, complete-node speedup is at least "
        "1.05x, and the 95% CI lower bound exceeds 1.0 may be promoted. Diagnostic rows are never "
        "automatic placements.",
        "",
        "## Native Q4_0 operator matrix",
        "",
        "| Tensor | M×K×N | Placement | CPU threads | QPU share | Complete ms | Quant ms | "
        "Input copy ms | Submit+wait ms | Output copy ms | Speedup (95% low) | Max abs err | "
        "Resident MiB | Retained | Session | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["operator_rows"]:
        speedup = "—" if row["speedup"] is None else f"{row['speedup']:.3f} ({row['speedup_ci95_low']:.3f})"
        lines.append(
            f"| `{row['tensor']}` | {row['m']}×{row['k']}×{row['n']} | {row['placement']} | "
            f"{row['cpu_threads']} | {row['qpu_fraction']:.3f} | "
            f"{_milliseconds(row['complete_median_ns'])} | {_milliseconds(row['quantize_median_ns'])} | "
            f"{_milliseconds(row['input_copy_median_ns'])} | "
            f"{_milliseconds(row['submit_wait_median_ns'])} | "
            f"{_milliseconds(row['output_copy_median_ns'])} | {speedup} | "
            f"{row['max_absolute_error']:.3g} | {row['resident_bytes'] / (1 << 20):.3f} | "
            f"{row['session_retained']} | {row['session']} | {row['status']} |"
        )
    if not payload["operator_rows"]:
        lines.append(
            "| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | no data |"
        )
    coverage = payload.get("coverage", {})
    lines.extend(
        [
            "",
            "## End-to-end matrix",
            "",
            "| Case | Mode | Placement | Threads | Depth | Context target | Prompt target | Startup s | "
            "Prefill s | Request s | Prompt ms | Prompt tok/s | Decode tok/s | Mean accepted | "
            "Cycle s | Greedy identical | Session | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in payload["end_to_end_rows"]:
        lines.append(
            f"| {row['case']} | {row['mode']} | {row['placement']} | {row['threads']} | "
            f"{row['draft_n_max'] if row['draft_n_max'] is not None else '—'} | "
            f"{row['context_tokens_target'] if row['context_tokens_target'] is not None else '—'} | "
            f"{row['prompt_tokens_target'] if row['prompt_tokens_target'] is not None else '—'} | "
            f"{_number(None if row['startup_ns'] is None else row['startup_ns'] / 1e9)} | "
            f"{_number(None if row['prefill_ns'] is None else row['prefill_ns'] / 1e9)} | "
            f"{_number(None if row['request_ns'] is None else row['request_ns'] / 1e9)} | "
            f"{_number(row['prompt_ms'])} | {_number(row['prompt_tokens_per_second'])} | "
            f"{_number(row['tokens_per_second'])} | "
            f"{_number(row['mean_accepted_length'], 2)} | {_number(row['cycle_seconds'])} | "
            f"{row['greedy_identical'] if row['greedy_identical'] is not None else '—'} | "
            f"{row['session']} | {row['status']} |"
        )
    if not payload["end_to_end_rows"]:
        lines.append(
            "| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | no data |"
        )
    lines.extend(
        [
            "",
            "## Coverage and retention",
            "",
            f"- Operator rows: {len(payload['operator_rows'])}.",
            f"- End-to-end rows: {len(payload['end_to_end_rows'])}.",
            f"- CPU baseline operator rows: {sum(row['status'] == 'baseline' for row in payload['operator_rows'])}.",
            f"- Promoted operator rows: {sum(row['status'] == 'promoted' for row in payload['operator_rows'])}.",
            f"- Retained end-to-end rows: "
            f"{sum(row['status'] == 'retained' for row in payload['end_to_end_rows'])}.",
            f"- Planned end-to-end cases: {coverage.get('planned_case_count', 'not loaded')}.",
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
    parser.add_argument("--case-matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    operator_records: list[dict[str, Any]] = []
    e2e_records: list[dict[str, Any]] = []
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
    if args.case_matrix is not None:
        case_matrix = json.loads(args.case_matrix.read_text(encoding="utf-8"))
        planned_cases = case_matrix.get("cases", [])
        mode_counts: dict[str, int] = {}
        for case in planned_cases:
            mode = str(case.get("mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        coverage = {
            "case_matrix": str(args.case_matrix.resolve()),
            "planned_case_count": len(planned_cases),
            "planned_mode_counts": mode_counts,
            "coverage_gaps": case_matrix.get("coverage_gaps", []),
        }
        sources.append(
            {
                "path": str(args.case_matrix.resolve()),
                "sha256": sha256_file(args.case_matrix),
                "kind": "case-matrix",
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-latency-matrix",
        "created_utc": utc_now(),
        "sources": sources,
        "operator_rows": operator_records,
        "end_to_end_rows": e2e_records,
        "coverage": coverage,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")
    write_json_atomic(args.json_output or output.with_suffix(".json"), payload)
    print(f"wrote {output}: {len(operator_records)} operator rows, {len(e2e_records)} end-to-end rows")


if __name__ == "__main__":
    main()
