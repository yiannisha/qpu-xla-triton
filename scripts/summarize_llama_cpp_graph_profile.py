#!/usr/bin/env python3
"""Summarize serialized llama.cpp node profiles and diagnostic Amdahl bounds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import sha256_file, utc_now, write_json_atomic  # noqa: E402


def percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize raw nanosecond samples without discarding the sample count."""
    if not values:
        raise ValueError("summary requires at least one value")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "samples": len(values),
        "median_ns": median,
        "p05_ns": percentile(values, 0.05),
        "p95_ns": percentile(values, 0.95),
        "mad_ns": statistics.median(deviations),
        "minimum_ns": min(values),
        "maximum_ns": max(values),
    }


def classify_node(node: dict[str, Any]) -> str:
    """Map a measured GGML node to the plan's stable operator families."""
    op = str(node["op"])
    name = str(node.get("name", "")).lower()
    source_names = " ".join(str(source.get("name", "")).lower() for source in node["sources"])
    identity = f"{name} {source_names}"
    if op == "MUL_MAT":
        if "token_embd.weight" in identity or "output.weight" in identity:
            return "lm_head"
        if "ffn_gate.weight" in identity or "ffn_up.weight" in identity:
            return "ffn_gate_up_linear"
        if "ffn_down.weight" in identity:
            return "ffn_down_linear"
        if "attn_" in identity and ".weight" in identity:
            return "attention_linear"
        if "per_layer_model_proj.weight" in identity:
            return "per_layer_projection"
        if "inp_gate.weight" in identity or ".proj.weight" in identity:
            return "auxiliary_linear"
        return "other_linear"
    if op == "FLASH_ATTN_EXT":
        return "attention_core"
    if op == "ROPE":
        return "rope"
    if op == "SET_ROWS":
        return "kv_or_state_update"
    if op == "GET_ROWS":
        return "embedding"
    if op == "RMS_NORM" or "norm.weight" in identity:
        return "normalization"
    if op == "GLU":
        return "swiglu"
    if op in {"ADD", "MUL", "SCALE", "UNARY"}:
        return "elementwise"
    if op in {"VIEW", "RESHAPE", "PERMUTE", "CONT", "TRANSPOSE"}:
        return "layout"
    return "other"


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Aggregate one exact configuration by run, family, and stable node identity."""
    nodes = profile["nodes"]
    decode_wall_by_run = {
        int(call["run_index"]): float(call["wall_ns"])
        for call in profile["decode_calls"]
    }
    run_ids = sorted(decode_wall_by_run)
    if not run_ids:
        raise ValueError("profile has no decode calls")
    family_by_run: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    node_by_identity: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    node_metadata: dict[tuple[int, str, str], dict[str, Any]] = {}
    for node in nodes:
        run_index = int(node["run_index"])
        if run_index not in decode_wall_by_run:
            raise ValueError(f"node references unknown run {run_index}")
        duration = float(node["duration_ns"])
        family = classify_node(node)
        family_by_run[family][run_index] += duration
        identity = (int(node["node_index"]), str(node["name"]), str(node["op"]))
        node_by_identity[identity].append(duration)
        node_metadata[identity] = node

    node_total_by_run = {
        run_index: sum(family_runs[run_index] for family_runs in family_by_run.values())
        for run_index in run_ids
    }
    families: list[dict[str, Any]] = []
    for family, samples_by_run in family_by_run.items():
        duration_samples = [samples_by_run[run_index] for run_index in run_ids]
        fraction_samples = [
            samples_by_run[run_index] / node_total_by_run[run_index]
            for run_index in run_ids
        ]
        median_fraction = statistics.median(fraction_samples)
        elimination_speedup = 1.0 / max(1.0 - median_fraction, 1e-12)
        families.append(
            {
                "family": family,
                "duration_ns": summarize(duration_samples),
                "fraction_of_serialized_node_time": {
                    "median": median_fraction,
                    "p05": percentile(fraction_samples, 0.05),
                    "p95": percentile(fraction_samples, 0.95),
                },
                "diagnostic_elimination_upper_bound": elimination_speedup,
                "clears_diagnostic_5_percent_gate": elimination_speedup >= 1.05,
            }
        )
    families.sort(key=lambda record: record["duration_ns"]["median_ns"], reverse=True)

    top_nodes: list[dict[str, Any]] = []
    for identity, duration_samples in node_by_identity.items():
        metadata = node_metadata[identity]
        top_nodes.append(
            {
                "node_index": identity[0],
                "name": identity[1],
                "op": identity[2],
                "family": classify_node(metadata),
                "shape": metadata["shape"],
                "sources": metadata["sources"],
                "duration_ns": summarize(duration_samples),
            }
        )
    top_nodes.sort(key=lambda record: record["duration_ns"]["median_ns"], reverse=True)
    return {
        "model": profile["model"],
        "configuration": profile["configuration"],
        "measurement_contract": profile["measurement_contract"],
        "timing_eligible_for_promotion": False,
        "ineligibility_reason": (
            "the public eval callback serializes graph nodes and changes production scheduling"
        ),
        "decode_wall_ns": summarize([decode_wall_by_run[index] for index in run_ids]),
        "serialized_node_total_ns": summarize([node_total_by_run[index] for index in run_ids]),
        "families": families,
        "top_nodes": top_nodes[:50],
        "node_records": len(nodes),
        "fixture_records": len(profile.get("fixtures", [])),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render diagnostic rankings while keeping serialized timings visibly ineligible."""
    lines = [
        "# llama.cpp serialized graph profiles",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "These profiles use llama.cpp's public eval callback. It synchronizes after every node, so the "
        "rankings are diagnostic and cannot promote a QPU placement or establish a production Amdahl gate.",
        "",
    ]
    for index, result in enumerate(payload["profiles"], start=1):
        config = result["configuration"]
        lines.extend(
            [
                f"## Profile {index}: batch {config['batch']}, context {config['context_tokens']}",
                "",
                f"- Model: `{result['model']}`",
                f"- Threads: {config['threads']}",
                f"- Serialized decode median: {result['decode_wall_ns']['median_ns'] / 1e6:.3f} ms",
                f"- Node records: {result['node_records']}",
                "",
                "| Family | Median ms | Serialized share | Elimination bound | Diagnostic 5% gate |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for family in result["families"]:
            lines.append(
                f"| {family['family']} | {family['duration_ns']['median_ns'] / 1e6:.3f} | "
                f"{family['fraction_of_serialized_node_time']['median']:.3f} | "
                f"{family['diagnostic_elimination_upper_bound']:.3f}x | "
                f"{family['clears_diagnostic_5_percent_gate']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Load native profile JSON and generate consolidated JSON and Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    profiles: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in args.profile:
        native = json.loads(path.read_text(encoding="utf-8"))
        profiles.append(summarize_profile(native))
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-serialized-graph-profile-summary",
        "created_utc": utc_now(),
        "sources": sources,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload), encoding="utf-8")
    write_json_atomic(args.json_output or args.output.with_suffix(".json"), payload)
    print(f"wrote {args.output}: {len(profiles)} profiles")


if __name__ == "__main__":
    main()
