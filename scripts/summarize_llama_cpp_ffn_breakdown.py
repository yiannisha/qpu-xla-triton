"""Aggregate complete measured FFN passes, preserving per-process denominators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_llama_cpp_qpu_ffn_island_eval import parse_prefixed_json  # noqa: E402


def summarize(record):
    """Aggregate measured passes per process and width, preserving overlap."""
    rows = []
    for run in record["runs"]:
        if run.get("invalid_reason") or run["returncode"] or "benchmark" not in run:
            continue
        if run["candidate"] and not run.get("candidate_validation", {"valid": True})["valid"]:
            continue
        # llama-bench -r 1: first 35 regions are warmup, final 35 are measured.
        regions = run["regions"][-35:]
        assert len(regions) == 35 and {r["layer"] for r in regions} == set(range(35))
        request_ns = run["benchmark"]["samples_ns"][0]
        ffns = sum(r["wall_ns"] for r in regions)
        timelines = parse_prefixed_json(run["stderr"], "qpu_ffn_timeline_json:")[-35:]
        by_layer = {t["layer"]: t for t in timelines}
        for width in [6144, 12288]:
            group = [r for r in regions if r["width"] == width]
            assert len(group) == (15 if width == 6144 else 20)
            item = {
                "rows": run["rows"],
                "pair": run["pair"],
                "candidate": run["candidate"],
                "width": width,
                "layers": len(group),
                "request_ms": request_ns / 1e6,
                "ffn_total_ms": ffns / 1e6,
                "ffn_share_percent": 100 * ffns / request_ns,
                "class_ffn_sum_ms": sum(r["wall_ns"] for r in group) / 1e6,
            }
            if run["candidate"]:
                metrics = {k: [] for k in ["prep_ms", "cpu_branch_ms", "qpu_path_ms", "merge_ms", "handoff_ms"]}
                for r in group:
                    t = by_layer[r["layer"]]
                    assert r["m"] == t["m"] == run["rows"]
                    assert (
                        r["begin_ns"] <= t["begin_ns"] <= t["prepared_ns"] <= t["cpu_start_ns"] <= t["cpu_finish_ns"]
                    )
                    finish = max(t["cpu_finish_ns"], t["qpu_observed_finish_ns"])
                    assert finish <= t["merge_begin_ns"] <= t["merge_end_ns"] <= r["end_ns"]
                    prep = t["prepared_ns"] - r["begin_ns"]
                    merge = t["merge_end_ns"] - t["merge_begin_ns"]
                    handoff = t["merge_begin_ns"] - finish + r["end_ns"] - t["merge_end_ns"]
                    assert r["wall_ns"] == prep + finish - t["prepared_ns"] + merge + handoff
                    metrics["prep_ms"].append(prep / 1e6)
                    metrics["cpu_branch_ms"].append((t["cpu_finish_ns"] - t["cpu_start_ns"]) / 1e6)
                    metrics["qpu_path_ms"].append((t["qpu_observed_finish_ns"] - t["prepared_ns"]) / 1e6)
                    metrics["merge_ms"].append(merge / 1e6)
                    metrics["handoff_ms"].append(handoff / 1e6)
                item.update({key: sum(values) for key, values in metrics.items()})
            rows.append(item)
    aggregates = []
    for m in [129, 257]:
        for width in [6144, 12288]:
            for candidate in [False, True]:
                group = [r for r in rows if (r["rows"], r["width"], r["candidate"]) == (m, width, candidate)]
                if group:
                    aggregates.append(
                        {
                            "rows": m,
                            "width": width,
                            "candidate": candidate,
                            "processes": len(group),
                            **{
                                key: median(r[key] for r in group)
                                for key in group[0]
                                if key.endswith(("_ms", "_percent"))
                            },
                        }
                    )
    return {
        "kind": "diagnostic-process-FFN-sums",
        "complete": record.get("completed", False),
        "power_clean": record.get("power_clean"),
        "per_process_width_class": rows,
        "median_of_process_totals": aggregates,
        "caution": (
            "Branch durations overlap. No per-layer median replaces a request median; instrumented diagnostics only."
        ),
    }


def main():
    """Write process-level summaries and CSV without manufacturing missing runs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(json.loads(a.input.read_text()))
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["per_process_width_class"]:
        with a.output.with_suffix(".csv").open("w") as f:
            keys = sorted({k for row in result["per_process_width_class"] for k in row})
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result["per_process_width_class"])


if __name__ == "__main__":
    main()
