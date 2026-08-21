from __future__ import annotations

import numpy as np

from examples.benchmark_llama_cpp_qpu_ops import (
    _session_validation,
    aligned_qpu_partitions,
    bootstrap_speedup_interval,
    compare_exact_cpu_output,
    default_tensor_names,
    summarize_ns,
)


def _retained_environment() -> dict[str, object]:
    return {
        "commands": {
            "throttling": {"stdout": "throttled=0x0"},
            "swap": {"stdout": "stable"},
            "swap_used_bytes": {"returncode": 0, "stdout": "0\n"},
            "llama_servers": {"stdout": ""},
        },
        "cpu_frequency": [{"governor": "performance"}],
    }


def test_aligned_partitions_cover_unique_proper_twelfths() -> None:
    partitions = aligned_qpu_partitions(256)
    counts = [int(item["qpu_column_count"]) for item in partitions]
    assert counts == sorted(set(counts))
    assert all(count % 16 == 0 for count in counts)
    assert all(0 < count < 256 for count in counts)
    assert all(int(item["qpu_column_start"]) + int(item["qpu_column_count"]) == 256 for item in partitions)


def test_operator_retention_enforces_sample_counts_and_zero_swap() -> None:
    environment = _retained_environment()
    assert _session_validation(environment, environment, warmups=5, samples=31)["retained"]
    short = _session_validation(environment, environment, warmups=4, samples=30)
    assert short["retained"] is False
    assert len(short["rejection_reasons"]) == 2

    swapped = {
        **environment,
        "commands": {
            **environment["commands"],
            "swap_used_bytes": {"returncode": 0, "stdout": "8192\n"},
        },
    }
    validation = _session_validation(swapped, swapped, warmups=5, samples=31)
    assert validation["retained"] is False
    assert "swap was in use" in validation["rejection_reasons"][0]


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


def test_default_tensor_accepts_native_q4_k_and_exact_comparison_tracks_argmax() -> None:
    manifest = {
        "tensors": [
            {
                "name": "blk.0.ffn_gate.weight",
                "shape": [2560, 9216],
                "ggml_type": "Q4_K",
                "bytes": 13_271_040,
                "owner": {"operator": "ffn_gate"},
            }
        ]
    }
    assert default_tensor_names(manifest) == ["blk.0.ffn_gate.weight"]
    reference = np.asarray([[0.0, 2.0, -1.0]], dtype="<f4")
    candidate = reference.copy()
    candidate[0, 0] += np.float32(1e-6)
    metrics = compare_exact_cpu_output(reference, candidate)
    assert metrics["argmax_identical"] is True
    assert metrics["tolerance_violation_count"] == 0
    assert metrics["bitwise_equal_count"] == 2

    candidate[0, 1] = np.nan
    invalid = compare_exact_cpu_output(reference, candidate)
    assert invalid["nan_count"] == 1
    assert invalid["tolerance_violation_count"] == 1
    assert np.isinf(invalid["max_absolute"])
