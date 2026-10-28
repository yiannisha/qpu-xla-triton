"""Protect measurement boundaries and process-level FFN aggregation."""

import json
import os
import subprocess
import sys

import pytest

from scripts.run_llama_cpp_ffn_breakdown import capture
from scripts.summarize_llama_cpp_ffn_breakdown import summarize
from scripts.summarize_llama_cpp_ffn_split_screen import summarize as summarize_screen


def make_run(scale=1, request=1000, candidate=False):
    regions = []
    timelines = []
    for layer in range(35):
        begin = layer * 1000
        regions.append(
            {
                "layer": layer,
                "m": 129,
                "width": 6144 if layer < 15 else 12288,
                "begin_ns": begin,
                "end_ns": begin + 10 * scale,
                "wall_ns": 10 * scale,
            }
        )
        timelines.append(
            {
                "layer": layer,
                "m": 129,
                "begin_ns": begin + 1,
                "prepared_ns": begin + 2,
                "cpu_start_ns": begin + 3,
                "cpu_finish_ns": begin + 7,
                "qpu_observed_finish_ns": begin + 8,
                "merge_begin_ns": begin + 8,
                "merge_end_ns": begin + 9,
            }
        )
    # Warmup is deliberately larger and must be excluded from all aggregates.
    warmup = [{**r, "wall_ns": 1000000} for r in regions]
    return {
        "invalid_reason": None,
        "returncode": 0,
        "rows": 129,
        "pair": 0,
        "candidate": candidate,
        "regions": warmup + regions,
        "benchmark": {"samples_ns": [request]},
        "stderr": "\n".join("qpu_ffn_timeline_json:" + json.dumps(t) for t in timelines),
    }


def test_aggregates_process_shares_and_excludes_warmup():
    result = summarize({"runs": [make_run(), make_run(scale=2, request=10000)], "completed": True})
    summary = result["median_of_process_totals"][0]
    assert summary["ffn_share_percent"] == 21  # median(35%, 7%), not a ratio of medians
    assert summary["ffn_total_ms"] == pytest.approx(525 / 1e6)
    assert summary["class_ffn_sum_ms"] == pytest.approx(225 / 1e6)


def test_overlapping_branches_preserve_timestamp_critical_path():
    result = summarize({"runs": [make_run(candidate=True)], "completed": True})
    row = result["per_process_width_class"][0]
    assert row["cpu_branch_ms"] == pytest.approx(15 * 4 / 1e6)
    assert row["qpu_path_ms"] == pytest.approx(15 * 6 / 1e6)
    assert row["class_ffn_sum_ms"] == pytest.approx(15 * 10 / 1e6)


def test_rejects_merge_before_branch_completion():
    run = make_run(candidate=True)
    run["stderr"] = run["stderr"].replace('"merge_begin_ns": 8', '"merge_begin_ns": 6', 1)
    with pytest.raises(AssertionError):
        summarize({"runs": [run]})


def test_invalid_power_run_cannot_enter_summary():
    run = make_run()
    run["invalid_reason"] = "undervoltage"
    assert summarize({"runs": [run]})["per_process_width_class"] == []


def test_invalid_candidate_execution_cannot_enter_summary():
    run = make_run(candidate=True)
    run["candidate_validation"] = {"valid": False}
    assert summarize({"runs": [run]})["per_process_width_class"] == []


@pytest.mark.parametrize("abort", [True, False])
def test_voltage_monitor_abort_and_diagnostic_override(monkeypatch, tmp_path, abort):
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        if command[0] == "sudo":
            command = [
                sys.executable,
                "-u",
                "-c",
                "import time; print('Undervoltage detected!',flush=True); time.sleep(60)",
            ]
        return real_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    command = [sys.executable, "-c", "import time; time.sleep(0.5); print('completed')"]
    result = capture(command, os.environ.copy(), tmp_path / "capture", abort_on_undervoltage=abort)
    assert result["undervoltage_event_count"] == 1
    if abort:
        assert result["invalid_reason"] == "new kernel undervoltage event during diagnostic"
        assert result["returncode"] != 0
    else:
        assert result["invalid_reason"] is None
        assert result["returncode"] == 0
        assert "completed" in result["stdout"]


def test_screen_export_preserves_missing_and_numerically_rejected_cells(tmp_path):
    attempt = {
        "rows": [129, 257],
        "fractions": [0.0625, 0.125, 0.1875],
        "completed": False,
        "preflights": [{"fraction": 0.0625, "returncode": 1, "invalid_reason": None}],
        "power_clean": True,
    }
    (tmp_path / "screen-attempt.json").write_text(json.dumps(attempt))
    result = summarize_screen(tmp_path)
    assert len(result["grid"]) == 6
    assert all(not row["timing_executed"] for row in result["grid"])
    assert all(not row["confirmed_for_placement"] for row in result["grid"])
    assert result["grid"][0]["numerical_returncode"] == 1
    assert result["grid"][1]["numerical_returncode"] is None
