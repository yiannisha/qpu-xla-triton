"""Export every planned split, including rejected and unexecuted configurations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(directory: Path) -> dict:
    """Join fraction-specific numerical outcomes with complete-prompt timings."""
    attempt = json.loads((directory / "screen-attempt.json").read_text())
    timing_file = directory / "screen.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    numerical = {entry["fraction"]: entry for entry in attempt["preflights"]}
    comparisons = {(entry["rows"], entry["fraction"]): entry for entry in timing.get("comparisons", [])}
    rows = []
    for m in attempt["rows"]:
        for fraction in attempt["fractions"]:
            validation = numerical.get(fraction, {})
            comparison = comparisons.get((m, fraction), {})
            speedup = comparison.get("speedup", {})
            row = {
                "rows": m,
                "fraction": fraction,
                "numerical_returncode": validation.get("returncode"),
                "numerical_environment_error": validation.get("invalid_reason"),
                "numerically_accepted": validation.get("returncode") == 0 and not validation.get("invalid_reason"),
                "timing_executed": bool(comparison),
                "cpu_ms": speedup.get("baseline_median_ns", 0) / 1e6 if speedup else None,
                "candidate_ms": speedup.get("candidate_median_ns", 0) / 1e6 if speedup else None,
                "median_speedup": speedup.get("median_speedup"),
                "ci95_low": speedup.get("bootstrap_95_low"),
                "ci95_high": speedup.get("bootstrap_95_high"),
                "execution_contract_valid": comparison.get("candidate_validation", {}).get("valid"),
                "timing_environment_retained": timing.get("retention", {}).get("retained", False),
                "power_clean": attempt.get("power_clean"),
                "clears_median_1_05": speedup.get("median_speedup", 0) >= 1.05 if speedup else None,
                "confirmed_for_placement": False,
            }
            rows.append(row)
    return {
        "complete": attempt.get("completed", False),
        "grid": rows,
        "retention": timing.get("retention"),
        "error": attempt.get("error"),
        "scope": "Exploratory screen; every newly selected split needs independent confirmation.",
    }


def main() -> None:
    """Write the complete planned grid as JSON and CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(result["grid"][0]))
        writer.writeheader()
        writer.writerows(result["grid"])


if __name__ == "__main__":
    main()
