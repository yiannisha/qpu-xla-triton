from __future__ import annotations

import re

from scripts.llama_cpp_common import swap_io_pages
from scripts.run_llama_cpp_qpu_ffn_island_eval import (
    COMPETING_WORKLOAD_PATTERN,
    paired_speedup_interval,
    retention,
    speedup_interval,
)


def environment(*, used: int, pswpin: int, pswpout: int) -> dict:
    return {
        "cpu_frequency": [{"governor": "performance"}],
        "commands": {
            "accelerator_workloads": {"stdout": ""},
            "llama_servers": {"stdout": ""},
            "swap_used_bytes": {"returncode": 0, "stdout": f"{used}\n"},
            "throttling": {"stdout": "throttled=0x0"},
            "vmstat": {
                "returncode": 0,
                "stdout": f"nr_free_pages 10\npswpin {pswpin}\npswpout {pswpout}\n",
            },
        },
    }


def test_swap_io_pages_extracts_both_kernel_counters() -> None:
    observed = swap_io_pages(environment(used=1024, pswpin=12, pswpout=34))
    assert observed == {"pswpin": 12, "pswpout": 34}


def test_competing_workload_pattern_matches_executables_not_source_paths() -> None:
    assert re.search(
        COMPETING_WORKLOAD_PATTERN,
        "/repo/.venv/bin/python3 .venv/bin/qpu-model-quality smolvla --resume",
    )
    assert re.search(
        COMPETING_WORKLOAD_PATTERN,
        "/repo/build/llama-qpu-runtime/qpu_ffn_island_graph_smoke",
    )
    assert re.search(COMPETING_WORKLOAD_PATTERN, "/repo/.venv/bin/pytest -q")
    assert re.search(
        COMPETING_WORKLOAD_PATTERN,
        "/repo/.venv/bin/python /repo/.venv/bin/mypy src/qpu_xla/quality",
    )
    assert not re.search(
        COMPETING_WORKLOAD_PATTERN,
        "python check_sources.py src/qpu_xla/models/smolvla/replay.py "
        "src/qpu_xla/quality",
    )


def test_ffn_retention_allows_residual_swap_without_swap_io() -> None:
    before = environment(used=32 << 20, pswpin=12, pswpout=34)
    after = environment(used=32 << 20, pswpin=12, pswpout=34)
    result = retention(before, after, [{"returncode": 0}])
    assert result["retained"] is True
    assert result["swap_used_bytes_before"] == 32 << 20
    assert result["swap_io_delta_pages"] == {"pswpin": 0, "pswpout": 0}


def test_ffn_retention_allows_bounded_read_only_swap_cleanup() -> None:
    before = environment(used=32 << 20, pswpin=12, pswpout=34)
    after = environment(used=31 << 20, pswpin=19, pswpout=34)
    result = retention(before, after, [{"returncode": 0}])
    assert result["retained"] is True
    assert result["swap_io_delta_pages"] == {"pswpin": 7, "pswpout": 0}


def test_ffn_retention_rejects_swap_page_outs() -> None:
    before = environment(used=32 << 20, pswpin=12, pswpout=34)
    after = environment(used=64 << 20, pswpin=19, pswpout=42)
    result = retention(before, after, [{"returncode": 0}])
    assert result["retained"] is False
    assert "swap page-outs occurred" in result["rejection_reasons"][0]


def test_ffn_retention_rejects_excessive_swap_page_ins() -> None:
    before = environment(used=32 << 20, pswpin=12, pswpout=34)
    after = environment(used=31 << 20, pswpin=12 + 1024, pswpout=34)
    result = retention(before, after, [{"returncode": 0}])
    assert result["retained"] is False
    assert "swap page-ins exceeded" in result["rejection_reasons"][0]


def test_ffn_interval_records_independent_resampling() -> None:
    result = speedup_interval(
        [110, 111, 112], [100, 101, 102], seed=7, resamples=100
    )
    assert result["resampling_scheme"] == "independent-two-sample-bootstrap"


def test_agentic_interval_resamples_process_pairs() -> None:
    result = paired_speedup_interval(
        [110, 220, 330], [100, 200, 300], seed=7, resamples=100
    )
    assert result["resampling_scheme"] == "paired-bootstrap"
    assert result["pair_count"] == 3
    assert result["pair_speedups"] == [1.1, 1.1, 1.1]
    assert result["bootstrap_95_low"] == result["bootstrap_95_high"] == 1.1
