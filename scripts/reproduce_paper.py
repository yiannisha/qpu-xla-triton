#!/usr/bin/env python3
"""Regenerate main paper result data from archived measurements, without hardware."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact import analyze, write_json  # noqa: E402
from scripts.artifact_common import contained, sha256  # noqa: E402


def read(relative: str):
    """Read a contained JSON evidence record."""
    return json.loads(contained(ROOT, relative).read_text())


def regenerate(output: Path, plots: bool = False) -> dict:
    """Keep expected PDF transcription separate from calculated measurement values."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = read("artifact/manuscript.json")
    for item in manifest["tables"] + manifest["figures"]:
        for relative in item["evidence"]:
            if not contained(ROOT, relative).exists():
                raise ValueError(f"missing manuscript evidence: {relative}")
    analyze(output / "analysis")
    checks, tables = [], {}

    def check(location, metric, value, expected, decimals):
        # Compare at the precision actually printed in the supplied PDF.
        checks.append(
            dict(
                location=location,
                metric=metric,
                calculated=value,
                paper=expected,
                decimal_places=decimals,
                agrees=f"{value:.{decimals}f}" == f"{expected:.{decimals}f}",
            )
        )

    def latency(location, label, cpu, candidate, expected):
        check(location, label + ":CPU ms", cpu, expected[0], 3)
        check(location, label + ":candidate ms", candidate, expected[1], 3)
        check(location, label + ":speedup", cpu / candidate, expected[2], 3)
        return dict(label=label, cpu_ms=cpu, candidate_ms=candidate, speedup=cpu / candidate)

    suite = "experiment_logs/20260819-qpu-xla-kernel-suite/"
    selected = []
    for case in ("prefill-h512-t16", "prefill-h1024-t64"):
        path = suite + case + "-fp32-linear.json"
        for row in read(path)["results"]:
            if row["status"] != "supported-win":
                continue
            cpu = min(statistics.median(row[k]) for k in ("numpy_cpu_seconds", "torch_cpu_seconds")) * 1000
            selected.append(
                dict(
                    shape=row["shape"],
                    role=row["projection"],
                    placement=row["placement"],
                    cpu_ms=cpu,
                    candidate_ms=statistics.median(row["candidate_seconds"]) * 1000,
                    max_abs_error=row["max_abs_error"],
                    source=path,
                )
            )
    tables["table8"] = selected
    expected = [
        (16, 1536, 512, "down", 1.750, 1.472, 1.189),
        (16, 512, 32000, "lm_head", 32.774, 24.166, 1.356),
        (64, 2816, 1024, "down", 7.052, 6.165, 1.144),
        (64, 1024, 2816, "gate", 7.398, 6.083, 1.216),
        (64, 1024, 1024, "o", 3.344, 2.484, 1.346),
    ]
    tables["table1"] = []
    for m, k, n, role, *values in expected:
        row = next(r for r in selected if r["shape"] == [m, k, n] and r["role"] == role)
        tables["table1"].append(latency("Table 1", f"{m}x{k}x{n}:{role}", row["cpu_ms"], row["candidate_ms"], values))

    extra = [
        (16, 512, 1536, "gate", 1.516, 1.357, 1.117),
        (16, 512, 1536, "up", 1.505, 1.342, 1.121),
        (64, 1024, 2816, "up", 7.572, 6.002, 1.261),
        (64, 1024, 32000, "lm_head", 81.400, 66.777, 1.219),
        (64, 1024, 1024, "q", 2.988, 2.502, 1.194),
    ]
    for m, k, n, role, *values in extra:
        row = next(r for r in selected if r["shape"] == [m, k, n] and r["role"] == role)
        latency("Table 8", f"{m}x{k}x{n}:{role}", row["cpu_ms"], row["candidate_ms"], values)

    producer = read("experiment_logs/20260825-qpu-next/fused-ffn-retained.json")
    tables["table2"] = []
    for row, expected in zip(
        producer["records"],
        [
            (1.564, 1.697, 4.555, 21.104, 0.216),
            (3.157, 3.005, 9.542, 37.582, 0.254),
            (6.343, 5.568, 18.904, 70.637, 0.268),
        ],
        strict=True,
    ):
        cpu_p, qpu_p, cpu, qpu = [
            statistics.median(row[k]) / 1e6
            for k in (
                "cpu_geglu_q8_0_ns",
                "qpu_fused_geglu_q8_0_ns",
                "cpu_ffn_region_ns",
                "qpu_fused_geglu_q8_0_down_ns",
            )
        ]
        label = f"M={row['rows']}"
        check("Table 2", label + ":producer CPU ms", cpu_p, expected[0], 3)
        check("Table 2", label + ":producer QPU ms", qpu_p, expected[1], 3)
        tables["table2"].append(
            dict(producer_cpu_ms=cpu_p, producer_qpu_ms=qpu_p, **latency("Table 2", label, cpu, qpu, expected[2:]))
        )

    tables["table3"] = []
    for path, prefix, suffix, expected in [
        ("agentic-column-w8-retained.json", "512", "128", (3019.758, 3021.002, 0.9996)),
        ("agentic-column-w8-retained.json", "512", "256", (5603.841, 5515.050, 1.0161)),
        ("agentic-column-w8-m513-ubatch-screen.json", "32", "512", (9053.511, 8925.507, 1.0143)),
    ]:
        pairs = read("experiment_logs/20260825-qpu-next/" + path)["heldout_pairs"][prefix][suffix]
        cpu, candidate = [
            statistics.median(p[k] for p in pairs) / 1e6 for k in ("cpu_request_wall_ns", "candidate_request_wall_ns")
        ]
        label = f"M={int(suffix) + 1}"
        check("Table 3", label + ":CPU ms", cpu, expected[0], 3)
        check("Table 3", label + ":candidate ms", candidate, expected[1], 3)
        check("Table 3", label + ":speedup", cpu / candidate, expected[2], 4)
        tables["table3"].append(
            dict(label=label, cpu_ms=cpu, candidate_ms=candidate, speedup=cpu / candidate, pairs=len(pairs))
        )

    headline = json.loads((output / "analysis/rebuttal.json").read_text())
    tables["table4"] = headline["headline"]
    expected = [
        (2116.016, 2045.965, 1.0342, 1.0090, 1.1104),
        (4451.305, 4420.986, 1.0069, 0.9651, 1.0168),
        (3194.305, 3077.670, 1.0379, 1.0241, 1.0521),
        (3195.809, 3079.077, 1.0379, 1.0241, 1.0520),
    ]
    for (label, row), values in zip(
        [
            (k, headline["headline"][k])
            for k in ("prompt-M129", "prompt-M257", "cached-512-suffix128-prompt", "cached-512-suffix128-request-wall")
        ],
        expected,
        strict=True,
    ):
        for metric, value, paper, decimals in zip(
            ["CPU ms", "candidate ms", "speedup", "CI lower", "CI upper"],
            [
                row["baseline_median_ns"] / 1e6,
                row["candidate_median_ns"] / 1e6,
                row["median_speedup"],
                row["bootstrap_95_low"],
                row["bootstrap_95_high"],
            ],
            values,
            [3, 3, 4, 4, 4],
            strict=True,
        ):
            check("Tables 4/30", label + ":" + metric, value, paper, decimals)

    quality = json.loads((output / "analysis/quality-numerical.json").read_text())
    tables["quality-raw-metrics"] = quality
    for name, expected in [
        ("gemma-mmlu-1000", (0.362, 0.363)),
        ("resnet18-imagenet5000-qpu-fp32", (0.5368, 0.5368)),
        ("resnet18-imagenet5000-hybrid-fp32", (0.5368, 0.5368)),
        ("resnet18-imagenet5000-qpu-w8a8", (0.5368, 0.5276)),
    ]:
        row = quality[name]
        for placement, value in zip(("baseline", "candidate"), expected, strict=True):
            field = "accuracy" if name.startswith("gemma") else "top1"
            check("Tables 6/36", name + ":" + placement + " %", row[placement + "_" + field] * 100, value * 100, 2)
    for name, candidate in [("qpu-fp32", 78.72), ("hybrid-fp32", 78.72), ("qpu-w8a8", 77.86)]:
        row = quality["resnet18-imagenet5000-" + name]
        check("Tables 6/36", name + ":CPU top5 %", row["baseline_top5"] * 100, 78.72, 2)
        check("Tables 6/36", name + ":candidate top5 %", row["candidate_top5"] * 100, candidate, 2)
    check("§5.3", "MMLU answer agreement %", quality["gemma-mmlu-1000"]["answer_agreement"] * 100, 99.2, 1)
    smol = quality["smolvla-replay100-qpu-fp32"]
    check("Table 38", "NRMSE %", smol["normalized_rmse"] * 100, 1.56, 2)
    check("Table 38", "cosine", smol["cosine_similarity"], 0.999881, 6)
    # Report views retain AP/CI methodology; do not present them as recomputation.
    tables["quality-report-views"] = json.loads((output / "analysis/quality-reports.json").read_text())

    figure4 = []
    for length in (4194048, 4194240):
        stripes = length // 64
        workers = max(w for w in range(1, 13) if stripes % w == 0)
        for row in read(suite + f"fp32-minmax-n{length}.json")["results"]:
            ratio = min(
                statistics.median(row[k]) for k in ("numpy_cpu_seconds", "torch_cpu_seconds")
            ) / statistics.median(row["qpu_seconds"])
            expected = {
                (4194048, "minimum"): 1.550,
                (4194048, "maximum"): 1.517,
                (4194240, "minimum"): 0.908,
                (4194240, "maximum"): 0.909,
            }[(length, row["operation"])]
            check("Figure 4", f"N={length}:{row['operation']} speedup", ratio, expected, 3)
            figure4.append(
                dict(
                    length=length,
                    stripes=stripes,
                    workers=workers,
                    path="vector" if workers >= 6 else "scalar",
                    operation=row["operation"],
                    speedup=ratio,
                )
            )
    events = read("experiment_logs/20260824-qpu-agentic-prefill/agentic-up-m257-screen.json")["calibration_pairs"][
        "256"
    ]["0.25"][0]["candidate_sample"]["candidate_execution"]["events"]
    complete = statistics.median(e["complete_ns"] for e in events) / 1e6
    overlap = statistics.median(e["overlap_before_wait_ns"] for e in events) / 1e6
    figure5 = dict(
        qpu_complete_ms=complete, cpu_overlap_ms=overlap, exposed_tail_ms=complete - overlap, layers=len(events)
    )
    for key, value in zip(
        ("qpu_complete_ms", "cpu_overlap_ms", "exposed_tail_ms"), (40.66, 21.43, 19.23), strict=True
    ):
        check("Figure 5 / Table 25", key, figure5[key], value, 2)
    tables["figure4"] = figure4
    tables["figure5"] = figure5
    write_json(output / "paper-data.json", tables)
    # Export every historical FP32 operator result, including slower cases.
    grid = []
    for path in sorted((ROOT / suite).glob("*.json")):
        if path.name.endswith(".candidates.json"):
            continue
        for index, row in enumerate(json.loads(path.read_text()).get("results", [])):
            grid.append(dict(source=str(path.relative_to(ROOT)), index=index, record=row))
    write_json(output / "operator-grid.json", grid)
    stages = [g["record"] for g in grid if g["source"].endswith("-stages.json")]
    check("Table 9", "total configurations", len(stages), 240, 0)
    stage_counts = {}
    for operation in sorted({r["operation"] for r in stages}):
        group = [r for r in stages if r["operation"] == operation]
        stage_counts[operation] = dict(
            configurations=len(group),
            supported_wins=sum(r["status"] == "supported-win" for r in group),
            best_speedup=max(r["speedup_over_cpu"] for r in group),
        )
    expected_stages = {
        "rms-norm": (48, 3, 2.040),
        "rope": (72, 0, 0.803),
        "softmax": (72, 2, 1.240),
        "swiglu": (48, 6, 2.300),
    }
    for operation, (count, wins, best) in expected_stages.items():
        row = stage_counts[operation]
        check("Table 9", operation + ":configurations", row["configurations"], count, 0)
        check("Table 9", operation + ":supported wins", row["supported_wins"], wins, 0)
        check("Table 9", operation + ":best speedup", row["best_speedup"], best, 3)
    write_json(output / "table9.json", stage_counts)

    for name in ("table1", "table2", "table3", "table8", "figure4"):
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tables[name][0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables[name])
    with (output / "agreement.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    mismatches = [c for c in checks if not c["agrees"]]
    result = dict(
        format="artifact-paper-agreement-v1",
        manuscript_sha256=manifest["manuscript"]["sha256"],
        checks=len(checks),
        mismatches=mismatches,
        ok=not mismatches,
        scope=(
            "Printed-value checks of main numerical results and Figures 4/5; all 39 tables are indexed, "
            "not all supplemental values are independently recomputed. Historical data analysis, "
            "not new empirical or independent validation."
        ),
        sources={
            p: sha256(contained(ROOT, p))
            for item in manifest["tables"] + manifest["figures"]
            for p in item["evidence"]
            if contained(ROOT, p).is_file()
        },
    )
    write_json(output / "agreement.json", result)
    lines = [
        "# Manuscript result agreement",
        "",
        result["scope"],
        "",
        f"Checks: {len(checks)}; mismatches: {len(mismatches)}.",
        "",
        (
            "Static memory estimates (Tables 5/39), quality intervals and COCO AP are preserved views, "
            "not regenerated task evaluations."
        ),
        "",
        "| Item | Page | Result | Evidence/workflow |",
        "|---|---:|---|---|",
    ]
    for item in manifest["tables"]:
        lines.append(f"| Table {item['number']} | {item['page']} | {item['title']} | {item['workflow']} |")
    if mismatches:
        lines += ["", "## Disagreements", "", json.dumps(mismatches, indent=2)]
    (output / "PAPER-RESULTS.md").write_text("\n".join(lines) + "\n")
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 3))
        for index, operation in enumerate(("minimum", "maximum")):
            values = [r["speedup"] for r in figure4 if r["operation"] == operation]
            ax.bar([p + index * 0.32 for p in (0, 1)], values, width=0.3, label=operation)
        ax.axhline(1, color="black", linewidth=0.7)
        ax.set_xticks([0.16, 1.16], ["4,194,048\n12 workers; vector", "4,194,240\n5 workers; scalar"])
        ax.set_ylabel("CPU / QPU latency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "figure4.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        ax.barh([0], overlap, label="CPU overlap window")
        ax.barh([0], complete - overlap, left=overlap, label="Exposed QPU tail")
        ax.set_yticks([0], ["QPU complete"])
        ax.set_xlabel("Median observable duration (ms)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=2)
        fig.tight_layout()
        fig.savefig(output / "figure5.pdf")
        plt.close(fig)
    return result


def main():
    """Run the offline agreement workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="also write PDF figures; requires matplotlib")
    args = parser.parse_args()
    result = regenerate(args.output.resolve(), args.plots)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
