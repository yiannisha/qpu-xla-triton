from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.run_llama_cpp_qpu_agentic_eval import (
    choose_fraction,
    fraction_list,
    qpu_execution_summary,
    summarize_pairs,
    token_outputs_identical,
    write_calibration_failure_record,
)


def test_fraction_list_has_bounded_unique_values() -> None:
    assert fraction_list("0.0625,0.125") == (0.0625, 0.125)
    with pytest.raises(argparse.ArgumentTypeError, match="fractions"):
        fraction_list("0.125,0.125")
    with pytest.raises(argparse.ArgumentTypeError, match="fractions"):
        fraction_list("0.01")


def test_qpu_summary_requires_all_gemma_layers_and_boundary_row() -> None:
    events = []
    for columns, count in ((6144, 15), (12288, 20)):
        events.extend(
            {
                "m": 65,
                "n": columns,
                "qpu_columns": columns // 8,
                "cpu_columns": columns - columns // 8,
                "fallback": False,
                "input_copy_ns": 1,
                "submit_wait_ns": 2,
                "overlap_before_wait_ns": 2,
                "output_copy_ns": 1,
                "complete_ns": 3,
            }
            for _ in range(count)
        )
    sample = {
        "candidate_execution": {
            "valid": True,
            "dispatch_count": 35,
            "events": events,
        },
        "server_log": "\n".join(
            "qpu_llama_weight_json:"
            + json.dumps(
                {
                    "operation": "ffn_up",
                    "tensor": f"blk.{index}.ffn_up.weight",
                    "k": 1536,
                    "n": 6144 if index < 15 else 12288,
                    "max_m": 257,
                    "resident_column_start": 5760 if index < 15 else 11520,
                    "resident_columns": 384 if index < 15 else 768,
                    "resident_bytes": 10,
                }
            )
            for index in range(35)
        ),
    }
    assert qpu_execution_summary(sample, 64)["valid"] is True
    events[0]["m"] = 64
    assert qpu_execution_summary(sample, 64)["valid"] is False


def test_token_comparison_requires_raw_identical_ids() -> None:
    semantics = {
        "token_ids": [7],
        "content": ".",
        "stop_reason": "limit",
        "tool_calls": [],
    }
    assert token_outputs_identical(
        {"response_semantics": semantics},
        {"response_semantics": dict(semantics)},
    )
    assert not token_outputs_identical(
        {"response_semantics": semantics},
        {"response_semantics": {**semantics, "token_ids": [8]}},
    )


def test_summary_bootstraps_paired_processes_and_selects_best_fraction() -> None:
    pairs = [
        {
            "correct": True,
            "retained": True,
            "cpu_request_wall_ns": cpu,
            "candidate_request_wall_ns": candidate,
            "cpu_prompt_ns": cpu - 5,
            "candidate_prompt_ns": candidate - 5,
            "peak_rss_delta_bytes": 100,
            "qpu_execution": {"resident_dma_bytes": 1_000},
        }
        for cpu, candidate in ((110, 100), (111, 100), (109, 99))
    ]
    summary = summarize_pairs(pairs, seed=9, resamples=200)
    assert summary["all_correct"] is True
    assert summary["all_retained"] is True
    assert summary["request_wall_speedup"]["median_speedup"] == pytest.approx(1.1)
    assert summary["total_memory_overhead_median_bytes"] == 1_100
    selected = choose_fraction(
        {
            0.0625: summary,
            0.125: {
                **summary,
                "request_wall_speedup": {"median_speedup": 1.2},
            },
        }
    )
    assert selected == 0.125


def test_calibration_selection_failure_is_written_atomically(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    args = argparse.Namespace(
        calibration_prefix=32,
        suffixes=(512,),
        fractions=(0.1875,),
        calibration_sessions=1,
        weight_mode="column-w8",
        wgs=24,
        seed=9,
    )
    summaries = {512: {0.1875: {"all_correct": False}}}
    pairs = {512: {0.1875: [{"retained": False}]}}
    write_calibration_failure_record(
        output,
        args=args,
        suffix=512,
        error=ValueError("no usable calibration"),
        selected={},
        calibration_summaries=summaries,
        calibration_pairs=pairs,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "llama-cpp-qpu-agentic-prefill-calibration-failure"
    assert payload["failure"] == {
        "stage": "calibration-selection",
        "suffix_tokens": 512,
        "message": "no usable calibration",
    }
    assert payload["calibration_pairs"] == {"512": {"0.1875": [{"retained": False}]}}
