from __future__ import annotations

from examples.benchmark_llama_cpp_qpu_ops import (
    aligned_qpu_partitions,
    bootstrap_speedup_interval,
    default_tensor_names,
    summarize_ns,
)


def test_aligned_partitions_cover_unique_proper_twelfths() -> None:
    partitions = aligned_qpu_partitions(256)
    counts = [int(item["qpu_column_count"]) for item in partitions]
    assert counts == sorted(set(counts))
    assert all(count % 16 == 0 for count in counts)
    assert all(0 < count < 256 for count in counts)
    assert all(int(item["qpu_column_start"]) + int(item["qpu_column_count"]) == 256 for item in partitions)


def test_timing_summaries_and_bootstrap_are_deterministic() -> None:
    summary = summarize_ns([100, 110, 90])
    assert summary["median_ns"] == 100.0
    assert summarize_ns([0, 0])["throughput_per_second"] is None
    first = bootstrap_speedup_interval([100, 101, 99], [50, 51, 49], seed=7, resamples=100)
    second = bootstrap_speedup_interval([100, 101, 99], [50, 51, 49], seed=7, resamples=100)
    assert first == second
    assert first["median_speedup"] == 2.0


def test_default_tensor_excludes_embedding() -> None:
    manifest = {
        "tensors": [
            {
                "name": "token_embd.weight",
                "shape": [256, 2048],
                "ggml_type": "Q4_0",
                "bytes": 10_000,
                "owner": {"operator": "embedding"},
            },
            {
                "name": "blk.0.ffn_gate.weight",
                "shape": [256, 1024],
                "ggml_type": "Q4_0",
                "bytes": 8_000,
                "owner": {"operator": "ffn_gate"},
            },
        ]
    }
    assert default_tensor_names(manifest) == ["blk.0.ffn_gate.weight"]
