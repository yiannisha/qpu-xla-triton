"""Summarize remaining rebuttal diagnostics without pooling process repetitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


def comparison(reference, candidate):
    """Bootstrap paired blocks, keeping process or within-process scope explicit."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    assert reference.shape == candidate.shape and reference.size > 0
    rng = np.random.default_rng(20261028)
    indices = rng.integers(0, reference.size, (10000, reference.size))
    ratios = np.median(reference[indices], axis=1) / np.median(candidate[indices], axis=1)
    return {
        "reference_median_ms": float(np.median(reference)) / 1e6,
        "candidate_median_ms": float(np.median(candidate)) / 1e6,
        "ratio_reference_over_candidate": float(np.median(reference) / np.median(candidate)),
        "paired_bootstrap_95_interval": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "paired_marginal_candidate_minus_reference_ms": float(np.median(candidate - reference)) / 1e6,
        "reference_range_ms": [float(reference.min()) / 1e6, float(reference.max()) / 1e6],
        "candidate_range_ms": [float(candidate.min()) / 1e6, float(candidate.max()) / 1e6],
    }


def summarize(root):
    """Preserve rejected attempts and distinguish diagnostic timing boundaries."""

    def read(path):
        return json.loads((root / path).read_text())

    ablation = read("todo3/ablation.json")
    assert ablation["completed"] and all(p["bitwise_matches_full"] for p in ablation["preflights"])
    assert all(not r["kernel"] and not r["invalid_reason"] for r in ablation["runs"] + ablation["preflights"])
    times = {
        mode: [
            r["benchmark"]["avg_ns"] for r in sorted(ablation["runs"], key=lambda r: r["round"]) if r["mode"] == mode
        ]
        for mode in ablation["modes"]
    }
    result = {
        "todo3": {
            "scope": "five randomized independent-process blocks, full design versus matched variants; M=129",
            "comparison": {mode: comparison(times["full"], values) for mode, values in times.items()},
            "bitwise_identical_model_logit_checks": 6,
            "all_channel_numerical_preflights": 4,
            "kernel_errors": 0,
            "fallbacks": 0,
            "qualification": (
                "compaction and materialization intervals span parity; this screen does not establish "
                "separate positive gains from avoiding them; serialization is slower"
            ),
        }
    }
    mtp = read("todo4/mtp.json")
    assert mtp["completed"]
    process_times = {
        mode: [median(r["benchmark"]["complete_ns"]) for r in mtp["runs"] if r["mode"] == mode]
        for mode in ("cpu", "qpu")
    }
    qp = [r for r in mtp["runs"] if r["mode"] == "qpu"]
    fields = [
        "input_access_ns",
        "input_pack_ns",
        "input_sync_ns",
        "input_copy_ns",
        "submit_ns",
        "wait_ns",
        "submit_wait_ns",
        "output_sync_ns",
        "output_copy_ns",
        "complete_ns",
    ]
    result["todo4"] = {
        "scope": "five process pairs; five warmups then 31 samples per process; median of process medians",
        "comparison": comparison(process_times["cpu"], process_times["qpu"]),
        "hook_component_median_us": {
            field: median(median(t[field] for t in r["telemetry_including_warmup"][5:]) for r in qp) / 1e3
            for field in fields
        },
        "max_abs_error": max(r["max_abs_error"] for r in mtp["numerical"]),
        "unresolved": mtp["unresolved"],
        "telemetry_dispatches_including_warmup": sum(len(r["telemetry_including_warmup"]) for r in qp),
    }
    conv = []
    for case in ("pointwise", "dense"):
        control = read(f"todo5-{case}-control/process.json")
        assert control["retained_for_latency"]
        data = read(f"todo5-{case}.json")["results"][0]
        samples = data["samples"]

        def span(sample, name=None, category=None):
            return sum(
                e["finished_ns"] - e["started_ns"]
                for e in sample["qpu"]["events"]
                if (name is None or e["name"] == name) and (category is None or e["category"] == category)
            )

        prep = [span(s, name="conv2d_w8a8.prepare_columns") for s in samples]
        layout = [s["qpu"]["layout_window_ns"] for s in samples]
        conv.append(
            {
                "case": data["case"],
                "comparison": comparison([s["cpu_ns"] for s in samples], [s["qpu"]["wall_ns"] for s in samples]),
                "layout_window_ms": median(layout) / 1e6,
                "prepare_total_ms": median(prep) / 1e6,
                "prepare_remainder_ms": median(a - b for a, b in zip(prep, layout, strict=True)) / 1e6,
                "qpu_event_ms": median(span(s, category="qpu") for s in samples) / 1e6,
                "output_restore_ms": median(span(s, name="conv2d_w8a8.restore_nchw") for s in samples) / 1e6,
                "max_abs_error_matching_w8a8": data["max_abs_error_matching_w8a8"],
                "max_abs_error_fp32": data["max_abs_error_fp32"],
            }
        )
    result["todo5"] = {
        "scope": "one process per case, 11 samples after three warmups; within-process intervals only",
        "valid_cases": conv,
        "depthwise_status": (
            "unretained: first all-case attempt had two GPU resets; second had a reset and numerical failure"
        ),
        "rejected_attempts": ["todo5-control/process.json", "todo5-clean-control/process.json"],
        "qualification": (
            "current depthwise specialized reduction differs from historical lowered implementation; "
            "no accepted depthwise latency"
        ),
    }
    for todo in (6, 7):
        data = read(f"todo{todo}.json")
        control = read(f"todo{todo}-control/process.json")
        assert control["completed"] and not control["result"]["kernel"]
        samples = data["samples"]
        result[f"todo{todo}"] = {
            "scope": "one process; paired within-process bootstrap reflects sample variation only",
            "boundary": data["boundary"],
            "comparison": comparison([s["cpu_ns"] for s in samples], [s["qpu_ns"] for s in samples]),
            "samples": len(samples),
        }
        if todo == 6:
            result["todo6"].update(
                {
                    "max_abs_error": 0.0,
                    "cpu_implementation": data["cpu_implementation"],
                    "w8a8_control": "omitted: no suitable optimized same-contract CPU implementation",
                }
            )
        else:
            assert all(
                s["qpu_coverage"]["qpu_dispatches"] == 20 and s["qpu_coverage"]["unexpected_fallbacks"] == 0
                for s in samples
            )
            result["todo7"].update(
                {
                    "max_abs_error": data["max_abs_error"],
                    "input_shape": data["source_shape"],
                    "scope": "one fixed image, seven repeated samples; full preprocessed-input-to-logits boundary",
                    "timed_dispatches": sum(s["qpu_coverage"]["qpu_dispatches"] for s in samples),
                    "fallbacks": 0,
                }
            )
    audit = read("todo8-9.json")
    result["todo8"] = {k: v for k, v in audit["projection"].items() if k not in ("selected", "full_grid")}
    result["todo9"] = audit["selector"]
    return result


def main():
    """Write a machine-readable summary for an already completed campaign."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
