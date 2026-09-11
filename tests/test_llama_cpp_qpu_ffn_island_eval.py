from __future__ import annotations

from scripts.llama_cpp_common import swap_io_pages
from scripts.run_llama_cpp_qpu_ffn_island_eval import (
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


def test_ffn_retention_allows_residual_swap_without_swap_io() -> None:
    before = environment(used=32 << 20, pswpin=12, pswpout=34)
    after = environment(used=32 << 20, pswpin=12, pswpout=34)
    result = retention(before, after, [{"returncode": 0}])
    assert result["retained"] is True
    assert result["swap_used_bytes_before"] == 32 << 20


def test_ffn_retention_rejects_swap_io() -> None:
    before = environment(used=32 << 20, pswpin=12, pswpout=34)
    after = environment(used=64 << 20, pswpin=19, pswpout=42)
    result = retention(before, after, [{"returncode": 0}])
    assert result["retained"] is False
    assert "swap I/O occurred" in result["rejection_reasons"][0]


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
