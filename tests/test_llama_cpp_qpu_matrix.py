from __future__ import annotations

from pathlib import Path

from scripts.generate_llama_cpp_qpu_matrix import e2e_rows, operator_rows, render_markdown


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
