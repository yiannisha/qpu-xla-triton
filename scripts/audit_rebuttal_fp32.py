"""Recover the documented FP32 projection denominator and vector cutoff evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Execute the requested experiment and retain its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / "experiment_logs/20260819-qpu-xla-kernel-suite"
    files = [
        directory / name
        for name in (
            "decode-h2048-c512-fp32-linear.json",
            "prefill-h1024-t64-fp32-linear.json",
            "prefill-h512-t16-fp32-linear.json",
        )
    ]
    if any(not path.is_file() for path in files):
        parser.error("historical campaign logs are missing; use the archived audit rather than infer a denominator")
    records = []
    provenance = []
    for path in files:
        data = json.loads(path.read_text())
        provenance.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "count": len(data["results"]),
            }
        )
        records.extend({**record, "source_file": str(path.relative_to(ROOT))} for record in data["results"])
    counts = Counter(record["status"] for record in records)
    selected = [r for r in records if r["status"] == "supported-win"]
    selector_path = ROOT / "src/qpu_xla/kernels/minmax.py"
    selector_source = selector_path.read_text()
    examples = []
    history = subprocess.run(
        ["git", "show", "-s", "--format=%s", "0e7612c15d812c5d7d5edd7689d378cdd9d23dca"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    for elements in (4194048, 4194240):
        assert elements % 64 == 0
        stripes = elements // 64
        workers = max(n for n in range(1, 13) if stripes % n == 0)
        examples.append(
            {
                "elements": elements,
                "64_value_stripes": stripes,
                "largest_divisor_at_most_12": workers,
                "vector_selected": workers >= 6,
            }
        )
    result = {
        "projection": {
            "scope": (
                "README regeneration glob *-fp32-linear.json; "
                "separate per-role reruns excluded; roles counted separately"
            ),
            "files": provenance,
            "configurations": len(records),
            "statuses": dict(counts),
            "median_speedup_gt_1": sum(r["speedup"] > 1 for r in records),
            "max_error_le_1e_minus_3": sum(r["max_abs_error"] <= 1e-3 for r in records),
            "rule": (
                "max_abs_error <= 1e-3 and median whole-operation speedup >= 1.05 "
                "against faster NumPy/OpenBLAS or Torch; no confidence-interval classification"
            ),
            "selected": selected,
            "full_grid": records,
        },
        "selector": {
            "path": str(selector_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(selector_path.read_bytes()).hexdigest(),
            "source_excerpt": "\n".join(selector_source.splitlines()[225:255]),
            "introducing_commit": "0e7612c15d812c5d7d5edd7689d378cdd9d23dca",
            "commit_subject": history.stdout.strip() if history.returncode == 0 else None,
            "history_available": history.returncode == 0,
            "rationale": (
                "Six is an implementation cutoff. No retained evidence establishes it "
                "as a hardware requirement or optimum."
            ),
            "examples": examples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "configurations": len(records),
                "statuses": dict(counts),
                "speedup_gt_1": result["projection"]["median_speedup_gt_1"],
            }
        )
    )


if __name__ == "__main__":
    main()
