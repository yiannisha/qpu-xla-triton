from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_llama_cpp_qpu_evaluation as evaluation
from scripts.run_llama_cpp_qpu_evaluation import (
    _exact_prompt,
    _server_speculative_metrics,
    build_command,
    build_server_command,
    compare_greedy_semantics,
    normalize_case,
    parse_llama_metrics,
    select_cases,
)


def test_parse_llama_metrics_keeps_phase_and_speculative_statistics() -> None:
    output = (
        "llama_perf_context_print: prompt eval time = 120.00 ms / 12 tokens "
        "(10.00 ms per token, 100.00 tokens per second)\n"
        "llama_perf_context_print: eval time = 200.00 ms / 10 runs "
        "(20.00 ms per token, 50.00 tokens per second)\n"
        "statistics draft-mtp: #calls(b,g,a) = 1 10 10, #gen drafts = 20, "
        "#acc drafts = 8, #gen tokens = 20, #acc tokens = 12, #mean acc len = 2.20, "
        "#acc rate/pos = (0.800, 0.400)\n"
    )
    metrics = parse_llama_metrics(output)
    assert metrics["performance"]["prompt_eval_time"]["milliseconds"] == 120.0
    assert metrics["performance"]["eval_time"]["tokens_per_second"] == 50.0
    assert metrics["speculative"][0]["acceptance_by_position"] == [0.8, 0.4]
    assert metrics["speculative"][0]["cycle_seconds"] == pytest.approx(0.044)


def test_mtp_command_keeps_semantic_controls_explicit() -> None:
    case = normalize_case(
        {
            "name": "mtp",
            "mode": "mtp",
            "base_model": "/model.gguf",
            "draft_model": "/draft.gguf",
            "draft_n_max": 2,
            "threads": 3,
        },
        index=0,
    )
    command = build_command(Path("/llama-cli"), case)
    assert command[0] == "/llama-cli"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--spec-draft-n-max") + 1] == "2"
    assert command[command.index("--threads") + 1] == "3"
    assert command[command.index("--temp") + 1] == "0.0"
    server_command = build_server_command(Path("/llama-server"), case, 8081)
    assert server_command[server_command.index("--parallel") + 1] == "1"
    assert server_command[server_command.index("--spec-draft-n-max") + 1] == "2"
    assert server_command[server_command.index("--port") + 1] == "8081"


def test_mtp_case_requires_depth() -> None:
    with pytest.raises(ValueError, match="draft_n_max"):
        normalize_case({"mode": "mtp", "base_model": "/model.gguf"}, index=0)


def test_case_selection_preserves_file_order_and_rejects_unknown_names() -> None:
    cases = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
    assert [case["name"] for case in select_cases(cases, ["third", "first"])] == [
        "first",
        "third",
    ]
    with pytest.raises(ValueError, match="missing"):
        select_cases(cases, ["missing"])


def test_prompt_case_requires_zero_decode_and_exclusive_exact_size() -> None:
    normalized = normalize_case(
        {
            "mode": "prompt",
            "base_model": "/model.gguf",
            "predict_tokens": 0,
            "prompt_tokens_target": 128,
        },
        index=0,
    )
    assert normalized["prompt_tokens_target"] == 128
    with pytest.raises(ValueError, match="predict_tokens to zero"):
        normalize_case(
            {
                "mode": "prompt",
                "base_model": "/model.gguf",
                "predict_tokens": 1,
            },
            index=0,
        )
    with pytest.raises(ValueError, match="cannot combine"):
        normalize_case(
            {
                "mode": "prompt",
                "base_model": "/model.gguf",
                "predict_tokens": 0,
                "prompt_tokens_target": 128,
                "context_tokens_target": 64,
            },
            index=0,
        )


def test_exact_prompt_uses_native_token_ids_and_exact_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post_json(
        port: int, path: str, payload: dict[str, object], timeout: float
    ) -> tuple[bytes, dict[str, object]]:
        assert port == 8080
        assert path == "/tokenize"
        assert payload["add_special"] is True
        assert timeout == 5.0
        return b"", {"tokens": list(range(20))}

    monkeypatch.setattr(evaluation, "_post_json", fake_post_json)
    assert _exact_prompt(
        8080,
        {"prompt": "deterministic", "prompt_tokens_target": 7},
        5.0,
    ) == list(range(7))


def test_server_acceptance_log_produces_cycle_measure() -> None:
    log = "draft acceptance = 0.37, mean len = 2.10\nacc per pos = (0.68, 0.49, 0.24)\n"
    metrics = _server_speculative_metrics(log, 10.0)
    assert metrics["mean_accepted_length"] == 2.1
    assert metrics["acceptance_by_position"] == [0.68, 0.49, 0.24]
    assert metrics["cycle_seconds"] == pytest.approx(0.21)


def test_greedy_semantic_comparison_checks_tokens_text_count_and_stop() -> None:
    case = {
        "model": "gemma",
        "prompt": "prompt",
        "seed": 1,
        "temperature": 0.0,
        "predict_tokens": 2,
        "context_tokens_target": 0,
        "placement": "cpu-only",
    }
    baseline = {
        "case": {**case, "name": "plain", "mode": "plain"},
        "samples": [
            {
                "content_sha256": "same",
                "response": {"tokens": [1, 2], "tokens_predicted": 2, "stop_type": "limit"},
            }
        ],
    }
    mtp = {
        "case": {**case, "name": "mtp", "mode": "mtp"},
        "samples": [
            {
                "content_sha256": "same",
                "response": {"tokens": [1, 2], "tokens_predicted": 2, "stop_type": "limit"},
            }
        ],
    }
    comparisons = compare_greedy_semantics([baseline, mtp])
    assert comparisons[-1]["candidate_case"] == "mtp"
    assert comparisons[-1]["identical"] is True
