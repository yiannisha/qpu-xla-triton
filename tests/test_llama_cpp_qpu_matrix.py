from __future__ import annotations

from pathlib import Path

from scripts.generate_llama_cpp_qpu_matrix import (
    attention_rows,
    e2e_rows,
    end_to_end_comparisons,
    operator_rows,
    render_markdown,
)


def _operator_result(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "rows": 4,
        "input_columns": 256,
        "output_columns": 2048,
        "cpu_threads": 3,
        "qpu_column_count": 0 if mode == "cpu" else 2048,
        "resident_bytes": 0 if mode == "cpu" else 4096,
        "summaries": {
            "complete_ns": {"median_ns": 1000.0},
            "quantize_ns": {"median_ns": 10.0},
            "qpu_input_copy_ns": {"median_ns": 20.0},
            "qpu_submit_wait_ns": {"median_ns": 30.0},
            "qpu_output_copy_ns": {"median_ns": 40.0},
        },
        "correctness": {
            "nan_count": 0,
            "inf_count": 0,
            "tolerance_violation_count": 0,
            "max_absolute": 1e-7,
        },
        "speedup_vs_fastest_cpu": (
            {} if mode == "cpu" else {"median_speedup": 0.5, "bootstrap_95_low": 0.4}
        ),
    }


def test_operator_rows_label_cpu_reference_as_baseline() -> None:
    session = {
        "session_id": "diagnostic",
        "manifest": {"model_sha256": "abc"},
        "validation": {"retained": False, "rejection_reasons": ["governor"]},
        "records": [
            {
                "tensor": {
                    "name": "blk.0.ffn_gate.weight",
                    "ggml_type": "Q4_0",
                    "owner": {"operator": "ffn_gate"},
                },
                "result": _operator_result("cpu"),
            },
            {
                "tensor": {
                    "name": "blk.0.ffn_gate.weight",
                    "ggml_type": "Q4_0",
                    "owner": {"operator": "ffn_gate"},
                },
                "result": _operator_result("qpu"),
            },
        ],
    }
    rows = operator_rows(session, Path("raw.json"))
    assert [row["status"] for row in rows] == ["baseline", "experimental"]
    assert not rows[0]["session_retained"]


def test_operator_rows_never_promote_nonproduction_hybrid_cpu_prefix() -> None:
    hybrid = _operator_result("hybrid")
    hybrid["speedup_vs_fastest_cpu"] = {
        "median_speedup": 2.0,
        "bootstrap_95_low": 1.5,
        "bootstrap_95_high": 2.5,
    }
    session = {
        "session_id": "invalid-hybrid",
        "manifest": {"model_sha256": "abc"},
        "measurement_contract": {"hybrid_cpu_side_exact": False},
        "validation": {"retained": True, "rejection_reasons": []},
        "records": [
            {
                "tensor": {
                    "name": "blk.0.ffn_gate.weight",
                    "ggml_type": "Q4_0",
                    "owner": {"operator": "ffn_gate"},
                },
                "result": hybrid,
            }
        ],
    }
    row = operator_rows(session, Path("raw.json"))[0]
    assert row["placement_eligible"] is False
    assert row["status"] == "experimental-nonproduction-cpu-prefix"


def test_operator_win_remains_provisional_until_end_to_end_gate() -> None:
    qpu = _operator_result("qpu")
    qpu["speedup_vs_fastest_cpu"] = {
        "median_speedup": 1.25,
        "bootstrap_95_low": 1.10,
        "bootstrap_95_high": 1.40,
    }
    session = {
        "session_id": "operator-win",
        "manifest": {"model_sha256": "abc"},
        "validation": {"retained": True, "rejection_reasons": []},
        "records": [
            {
                "tensor": {
                    "name": "blk.0.ffn_gate.weight",
                    "ggml_type": "Q4_0",
                    "owner": {"operator": "ffn_gate"},
                },
                "result": qpu,
            }
        ],
    }
    row = operator_rows(session, Path("raw.json"))[0]
    assert row["status"] == "operator-win-pending-end-to-end"


def _promotion_row(
    *,
    session: str,
    placement: str,
    request_ns: int,
    candidate_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "session": session,
        "model": "gemma",
        "mode": "plain",
        "workload": "structured-tool-call",
        "request_surface": "chat-completions",
        "workload_semantics_sha256": "1" * 64,
        "base_model_sha256": "2" * 64,
        "draft_model_sha256": None,
        "server_binary_sha256": ("3" if placement == "cpu-only" else "4") * 64,
        "candidate_evidence": candidate_evidence,
        "partition": None,
        "placement": placement,
        "threads": 3,
        "draft_n_max": None,
        "context_tokens_target": 0,
        "prompt_tokens_target": 0,
        "request_ns": request_ns,
        "peak_rss_bytes": 1 << 30,
        "process_swap_bytes": 0,
        "candidate_execution_valid": True,
        "candidate_dispatch_count": 0 if placement == "cpu-only" else 12,
        "tokens_per_second": None,
        "cycle_seconds": None,
        "greedy_identical": True if placement != "cpu-only" else None,
        "tool_call_valid": True,
        "success": True,
        "session_retained": True,
    }


def test_end_to_end_gate_requires_five_sessions_and_exact_evidence() -> None:
    evidence: dict[str, object] = {
        "program_manifest_path": "/evidence/manifest.json",
        "program_manifest_sha256": "a" * 64,
        "program": "ggml-q4-0-q8-0-m1",
        "source_hash": "b" * 64,
        "binary_sha256": "c" * 64,
        "exact_shape": {"operator": "ffn_gate", "m": 1, "k": 256, "n": 2048},
    }
    baseline = [
        _promotion_row(session=f"baseline-{index}", placement="cpu-only", request_ns=100)
        for index in range(5)
    ]
    candidate = [
        _promotion_row(
            session=f"candidate-{index}",
            placement="qpu-only",
            request_ns=80,
            candidate_evidence=evidence,
        )
        for index in range(5)
    ]
    comparison = end_to_end_comparisons(baseline + candidate)[0]
    assert comparison["status"] == "promoted"
    assert comparison["baseline_session_count"] == 5
    assert comparison["candidate_session_count"] == 5
    assert comparison["speedup"]["median_speedup"] == 1.25
    assert comparison["speedup"]["bootstrap_95_low"] > 1.0

    four_session = end_to_end_comparisons(baseline + candidate[:4])[0]
    assert four_session["status"] == "missing-five-session-candidate"

    missing_evidence = [dict(row, candidate_evidence=None) for row in candidate]
    incomplete = end_to_end_comparisons(baseline + missing_evidence)[0]
    assert incomplete["status"] == "incomplete-candidate-evidence"

    no_execution_proof = [dict(row, candidate_execution_valid=False) for row in candidate]
    unattested = end_to_end_comparisons(baseline + no_execution_proof)[0]
    assert unattested["status"] == "incomplete-candidate-evidence"


def test_prompt_rows_hide_server_sentinel_decode_timing_and_render_prompt_rate() -> None:
    session = {
        "session_id": "prompt",
        "validation": {"retained": True, "rejection_reasons": []},
        "greedy_semantic_comparisons": [],
        "results": [
            {
                "case": {
                    "name": "prompt-128",
                    "model": "gemma",
                    "mode": "prompt",
                    "placement": "cpu-only",
                    "threads": 2,
                    "predict_tokens": 0,
                    "prompt_tokens_target": 128,
                    "context_tokens_target": 0,
                },
                "samples": [
                    {
                        "sample_index": 0,
                        "returncode": 0,
                        "startup_ns": 1_000_000_000,
                        "request_wall_ns": 2_000_000_000,
                        "content_sha256": "def",
                        "metrics": {
                            "timings": {
                                "prompt_ms": 125.0,
                                "prompt_n": 128,
                                "prompt_per_second": 1024.0,
                                "predicted_ms": 0.001,
                                "predicted_n": 1,
                                "predicted_per_second": 1_000_000.0,
                            },
                            "speculative": {},
                        },
                    }
                ],
            }
        ],
    }
    rows = e2e_rows(session, Path("raw.json"))
    assert rows[0]["tokens_per_second"] is None
    assert rows[0]["decode_tokens"] is None
    assert rows[0]["prompt_tokens_per_second"] == 1024.0
    assert rows[0]["workload"] == "prompt"
    markdown = render_markdown(
        {
            "created_utc": "now",
            "operator_rows": [],
            "end_to_end_rows": rows,
            "coverage": {
                "planned_case_count": 39,
                "coverage_gaps": [{"model": "qwen", "reason": "model unavailable"}],
            },
        }
    )
    assert "Prompt target" in markdown
    assert "1024.000" in markdown
    assert "Planned end-to-end cases: 39" in markdown
    assert "Coverage gap `qwen`" in markdown


def test_structured_tool_rows_keep_endpoint_token_count_and_validation() -> None:
    session = {
        "session_id": "tool",
        "validation": {"retained": True, "rejection_reasons": []},
        "greedy_semantic_comparisons": [
            {
                "candidate_case": "tool40",
                "sample_index": 0,
                "token_ids_available": True,
                "identical": True,
            }
        ],
        "results": [
            {
                "case": {
                    "name": "tool40",
                    "model": "gemma",
                    "mode": "plain",
                    "workload": "structured-tool-call",
                    "request_surface": "chat-completions",
                    "placement": "cpu-only",
                    "threads": 3,
                    "predict_tokens": 40,
                    "context_tokens_target": 0,
                },
                "samples": [
                    {
                        "sample_index": 0,
                        "returncode": 0,
                        "request_wall_ns": 1_000_000,
                        "tool_call_validation": {"valid": True},
                        "metrics": {
                            "timings": {"predicted_n": 31, "predicted_per_second": 10.0},
                            "speculative": {},
                        },
                    }
                ],
            }
        ],
    }
    row = e2e_rows(session, Path("raw.json"))[0]
    assert row["workload"] == "structured-tool-call"
    assert row["request_surface"] == "chat-completions"
    assert row["decode_tokens"] == 31
    assert row["tool_call_valid"] is True
    assert row["greedy_identical"] is True


def test_attention_rows_are_always_diagnostic() -> None:
    session = {
        "measurement_contract": {"reasons": ["synthetic"]},
        "records": [
            {
                "query_rows": 1,
                "query_heads": 8,
                "kv_heads": 1,
                "context_rows": 4096,
                "head_dim": 256,
                "native_input_bytes": 123,
                "dispatch_count": 1,
                "cpu_exact_ggml_node_summary": {"median_ns": 100},
                "qpu_persistent_execute_summary": {"median_ns": 500},
                "diagnostic_speedup_vs_exact_ggml_cpu": 0.2,
                "exact_ggml_cpu_correctness": {
                    "nan_count": 0,
                    "inf_count": 0,
                    "tolerance_violation_count": 0,
                    "max_absolute": 1e-6,
                },
            }
        ],
    }
    row = attention_rows(session, Path("attention.json"))[0]
    assert row["correct"]
    assert row["status"] == "experimental"
    assert row["session_retained"] is False
