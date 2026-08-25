from __future__ import annotations

import argparse

import pytest

from scripts.generate_llama_cpp_qpu_agentic_cases import fraction_map
from scripts.run_llama_cpp_qpu_hybrid_eval import (
    bootstrap_session_speedup,
    choose_partition,
    environment_rejection_reasons,
    parse_benchmark_stdout,
    summarize_partition_sessions,
)


def test_agentic_fraction_parser_requires_exact_suffix_mapping() -> None:
    assert fraction_map("64:0.125,128:0.25") == {64: 0.125, 128: 0.25}
    with pytest.raises(argparse.ArgumentTypeError, match="fractions"):
        fraction_map("64:0")


def test_parse_hybrid_record_ignores_unstructured_output() -> None:
    parsed = parse_benchmark_stdout(
        "diagnostic\n"
        '{"kind":"qpu-ggml-inline-samples","m":17,"passed":true}\n'
    )
    assert parsed["m"] == 17


def test_session_bootstrap_reduces_each_process_before_gating() -> None:
    interval = bootstrap_session_speedup(
        [110, 112, 111, 109, 113],
        [100, 101, 100, 99, 102],
        seed=7,
        resamples=1_000,
    )
    assert interval["session_count"] == 5
    assert interval["median_speedup"] == pytest.approx(1.11)
    assert interval["bootstrap_95_low"] > 1.0


def test_partition_selection_uses_weighted_calibration_speedup() -> None:
    calibration = {
        1: {"same_boundary_speedup": {"median_speedup": 1.06}},
        5: {"same_boundary_speedup": {"median_speedup": 0.91}},
    }
    assert choose_partition(calibration) == 1


def test_partition_summary_uses_real_gemma_region_medians() -> None:
    sessions = [
        {
            "cpu_region_median_ns": 8_000,
            "candidate_region_median_ns": 7_000,
            "bitwise_exact": True,
        },
        {
            "cpu_region_median_ns": 8_100,
            "candidate_region_median_ns": 7_100,
            "bitwise_exact": True,
        },
    ]
    summary = summarize_partition_sessions(sessions, seed=11, resamples=100)
    assert summary["bitwise_exact"] is True
    assert summary["cpu_region_session_medians_ns"] == [8_000, 8_100]
    assert summary["same_boundary_speedup"]["median_speedup"] > 1.1


def test_retention_rejects_nonperformance_governor() -> None:
    environment = {
        "cpu_frequency": [{"governor": "ondemand"}],
        "commands": {
            "swap_used_bytes": {"returncode": 0, "stdout": "0\n"},
            "llama_servers": {"stdout": ""},
            "throttling": {"stdout": "throttled=0x0"},
        },
    }
    assert environment_rejection_reasons(environment) == [
        "CPU governors were ['ondemand']"
    ]
