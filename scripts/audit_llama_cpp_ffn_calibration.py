"""Recover the integrated historical split grid, preserving its exclusions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(log_root: Path) -> dict:
    """Read the two historical screens and the corrected fixed-policy confirmations."""
    screens = []
    grid = []
    for name in ("full-island-screen.json", "full-island-tuning-screen.json"):
        path = log_root / "20260910-ffn-island" / name
        payload = json.loads(path.read_text())
        screens.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "configuration": payload["configuration"],
                "retention": payload["retention"],
                "artifacts": payload["artifacts"],
            }
        )
        for entry in payload["comparisons"]:
            speedup = entry["speedup"]
            grid.append(
                {
                    "source": str(path),
                    "rows": entry["rows"],
                    "fraction": entry["fraction"],
                    "cpu_ms": speedup["baseline_median_ns"] / 1e6,
                    "candidate_ms": speedup["candidate_median_ns"] / 1e6,
                    "recorded_ratio": speedup["median_speedup"],
                    "execution_contract_valid": entry["candidate_validation"]["valid"],
                    "environment_retained": payload["retention"]["retained"],
                    "numerical_status": "invalidated CPU-stride implementation; not a validated joined output",
                    "eligible_for_selection": False,
                }
            )
    confirmations = []
    for name in ("corrected-independent-processes-v7.json", "corrected-agentic-15-v2.json"):
        path = log_root / "20260910-ffn-island-stride-fix" / name
        data = json.loads(path.read_text())
        confirmations.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "configuration": data["configuration"],
                "retention": data["retention"],
                "comparisons": data.get("comparisons", data.get("pair_results")),
            }
        )
    return {
        "kind": "historical-integrated-FFN-grid-audit",
        "grid": grid,
        "screens": screens,
        "corrected_confirmation": confirmations,
        "selection": (
            "One eighth was fixed before corrected confirmation. The recovered multi-fraction screens predate "
            "the stride fix and have rejected environments; they do not establish a corrected optimum."
        ),
    }


def main() -> None:
    """Export historical measurements with their original provenance and exclusions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_logs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.log_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(result["grid"][0]))
        writer.writeheader()
        writer.writerows(result["grid"])


if __name__ == "__main__":
    main()
