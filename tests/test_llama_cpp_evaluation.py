from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_llama_cpp_qpu_evaluation as evaluation
from scripts.run_llama_cpp_qpu_evaluation import (
    _build_server_request,
    _exact_prompt,
    _expected_timed_suffix_tokens,
    _parse_process_memory,
    _response_semantics,
    _server_speculative_metrics,
    _validate_structured_tool_response,
    build_command,
    build_server_command,
    compare_greedy_semantics,
    normalize_case,
    parse_llama_metrics,
    select_cases,
    validate_candidate_evidence,
    validate_candidate_execution,
    validate_session,
    workload_semantics_sha256,
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


def test_fixed_structured_tool_request_retains_tokens_and_validates_schema() -> None:
    expected = {"sensor_id": "pi5-lab-1", "temperature_c": 32.4}
    case = normalize_case(
        {
            "name": "tool40",
            "mode": "plain",
            "workload": "structured-tool-call",
            "request_surface": "chat-completions",
            "surface": "server",
            "base_model": "/model.gguf",
            "predict_tokens": 40,
            "messages": [{"role": "user", "content": "record it"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "record_sensor_reading", "parameters": {}},
                }
            ],
            "expected_tool_call": {
                "name": "record_sensor_reading",
                "arguments": expected,
            },
        },
        index=0,
    )
    command = build_server_command(Path("/llama-server"), case, 8081)
    assert command[command.index("--tools") + 1] == "all"
    assert command[command.index("--reasoning") + 1] == "off"
    endpoint, request = _build_server_request(case)
    assert endpoint == "/v1/chat/completions"
    assert request["max_tokens"] == 40
    assert request["return_tokens"] is True
    assert request["verbose"] is True

    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "nondeterministic-id",
                            "type": "function",
                            "function": {
                                "name": "record_sensor_reading",
                                "arguments": '{"temperature_c":32.4,"sensor_id":"pi5-lab-1"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"completion_tokens": 4, "prompt_tokens": 12},
        "timings": {"predicted_n": 4},
        "__verbose": {"tokens": [10, 11, 12, 13], "stop_type": "eos"},
    }
    semantics = _response_semantics(response, "chat-completions")
    assert semantics["token_ids"] == [10, 11, 12, 13]
    assert semantics["tool_calls"][0]["arguments"] == expected
    assert _validate_structured_tool_response(case, semantics)["valid"] is True


def test_structured_tool_case_rejects_non_fixed_cap() -> None:
    with pytest.raises(ValueError, match="fixed 40-token cap"):
        normalize_case(
            {
                "mode": "plain",
                "workload": "structured-tool-call",
                "request_surface": "chat-completions",
                "surface": "server",
                "base_model": "/model.gguf",
                "predict_tokens": 39,
                "messages": [{"role": "user", "content": "record it"}],
                "tools": [{"type": "function"}],
                "expected_tool_call": {"name": "record", "arguments": {}},
            },
            index=0,
        )


def test_workload_semantics_hash_ignores_tuning_but_changes_visible_input() -> None:
    base = normalize_case(
        {
            "name": "tool40",
            "model": "gemma",
            "base_model": "/models/gemma.gguf",
            "mode": "plain",
            "workload": "structured-tool-call",
            "surface": "server",
            "request_surface": "chat-completions",
            "predict_tokens": 40,
            "threads": 2,
            "messages": [{"role": "user", "content": "Record the reading."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "record",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "expected_tool_call": {"name": "record", "arguments": {}},
        },
        index=0,
    )
    tuned = dict(base, threads=4, placement="qpu-only", draft_n_max=7)
    changed_prompt = dict(
        base,
        messages=[{"role": "user", "content": "Record another reading."}],
    )
    changed_schema = dict(base)
    changed_schema["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "parameters": {"type": "object", "required": ["temperature"]},
            },
        }
    ]
    assert workload_semantics_sha256(base) == workload_semantics_sha256(tuned)
    assert workload_semantics_sha256(base) != workload_semantics_sha256(changed_prompt)
    assert workload_semantics_sha256(base) != workload_semantics_sha256(changed_schema)


def test_non_cpu_candidate_evidence_resolves_to_exact_manifest_bytes(tmp_path: Path) -> None:
    source = tmp_path / "kernel.py"
    binary = tmp_path / "kernel.bin"
    manifest = tmp_path / "manifest.json"
    source.write_text("def kernel():\n    pass\n", encoding="utf-8")
    binary.write_bytes(b"qpu-program")
    source_file_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    source_hash = "5" * 64
    manifest.write_text(
        json.dumps(
            {
                "programs": [
                    {
                        "name": "test-program",
                        "source": str(source),
                        "source_file_sha256": source_file_sha256,
                        "source_hash": source_hash,
                        "binary": binary.name,
                        "binary_sha256": binary_sha256,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    case = normalize_case(
        {
            "name": "candidate",
            "base_model": "/models/gemma.gguf",
            "placement": "qpu-only",
            "candidate_evidence": {
                "program_manifest_path": str(manifest),
                "program_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "program": "test-program",
                "source_hash": source_hash,
                "binary_sha256": binary_sha256,
                "exact_shape": {"m": 1, "k": 256, "n": 2048},
            },
        },
        index=0,
    )
    validate_candidate_evidence(case)
    assert case["candidate_evidence"]["program_manifest_path"] == str(manifest.resolve())

    binary.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binary bytes do not match"):
        validate_candidate_evidence(case)


def test_retention_requires_zero_swap_usage() -> None:
    environment = {
        "commands": {
            "throttling": {"stdout": "throttled=0x0"},
            "swap": {"stdout": "stable"},
            "swap_used_bytes": {"returncode": 0, "stdout": "0\n"},
            "llama_servers": {"stdout": ""},
        },
        "cpu_frequency": [{"governor": "performance"}],
    }
    sample = {
        "returncode": 0,
        "surface": "llama-server-/completion",
        "context_population": {"valid": True},
        "prompt_population": {"valid": True},
        "generation_population": {"valid": True},
        "tool_call_validation": None,
        "process_memory": {
            "rss_bytes": 1024,
            "peak_rss_bytes": 2048,
            "swap_bytes": 0,
        },
        "candidate_execution": {"valid": True, "dispatch_count": 0},
    }
    assert validate_session(environment, environment, [sample])["retained"] is True

    swapped = {
        **environment,
        "commands": {
            **environment["commands"],
            "swap_used_bytes": {"returncode": 0, "stdout": "4096\n"},
        },
    }
    validation = validate_session(swapped, swapped, [sample])
    assert validation["retained"] is False
    assert "swap was in use" in validation["rejection_reasons"][0]


def test_retention_allows_historical_but_not_current_throttling_flags() -> None:
    environment = {
        "commands": {
            "throttling": {"stdout": "throttled=0x80000"},
            "swap": {"stdout": "stable"},
            "swap_configuration": {"stdout": "/dev/zram0 partition 1 100"},
            "swap_used_bytes": {"returncode": 0, "stdout": "0\n"},
            "llama_servers": {"stdout": ""},
        },
        "cpu_frequency": [{"governor": "performance"}],
    }
    sample = {
        "returncode": 0,
        "surface": "llama-server-/completion",
        "context_population": {"valid": True},
        "prompt_population": {"valid": True},
        "generation_population": {"valid": True},
        "process_memory": {"peak_rss_bytes": 1, "swap_bytes": 0},
        "candidate_execution": {"valid": True},
    }
    assert validate_session(environment, environment, [sample])["retained"] is True
    active = {
        **environment,
        "commands": {
            **environment["commands"],
            "throttling": {"stdout": "throttled=0x80008"},
        },
    }
    assert validate_session(active, active, [sample])["retained"] is False


def test_process_memory_parser_retains_peak_rss_and_swap() -> None:
    parsed = _parse_process_memory(
        "Name:\tllama-server\nVmHWM:\t2048 kB\nVmRSS:\t1536 kB\nVmSwap:\t4 kB\n"
    )
    assert parsed == {
        "rss_bytes": 1536 * 1024,
        "peak_rss_bytes": 2048 * 1024,
        "swap_bytes": 4 * 1024,
    }


def test_candidate_execution_requires_exact_positive_dispatch_telemetry() -> None:
    evidence = {
        "program": "test-program",
        "source_hash": "a" * 64,
        "binary_sha256": "b" * 64,
        "exact_shape": {"m": 1, "k": 256, "n": 2048},
    }
    case = {
        "placement": "qpu-only",
        "partition": None,
        "candidate_evidence": evidence,
    }
    event = {
        **evidence,
        "placement": "qpu-only",
        "partition": None,
        "dispatch_count": 12,
    }
    log = f"unrelated\nqpu_llama_candidate_json: {json.dumps(event)}\n"
    validation = validate_candidate_execution(case, log)
    assert validation["valid"] is True
    assert validation["dispatch_count"] == 12

    assert validate_candidate_execution(case, "no telemetry")["valid"] is False
    wrong = dict(event, binary_sha256="c" * 64, dispatch_count=0)
    invalid = validate_candidate_execution(
        case,
        f"qpu_llama_candidate_json: {json.dumps(wrong)}",
    )
    assert invalid["valid"] is False
    assert any("binary_sha256" in error for error in invalid["errors"])

    cpu = {"placement": "cpu-only"}
    assert validate_candidate_execution(cpu, "ordinary server log")["valid"] is True
    assert validate_candidate_execution(cpu, log)["valid"] is False


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


def test_exact_suffix_accounts_for_native_kv_cache_replay() -> None:
    assert _expected_timed_suffix_tokens(512, 507, 16) == (5, 21)
    assert _expected_timed_suffix_tokens(4096, 4096, 128) == (0, 128)


def test_candidate_server_command_retains_explicit_device_arguments_and_environment() -> None:
    case = normalize_case(
        {
            "name": "qpu",
            "mode": "plain",
            "base_model": "/model.gguf",
            "surface": "server",
            "placement": "hybrid",
            "process_environment": {"LD_PRELOAD": "/lib/libggml-qpu-inline.so"},
            "server_extra_arguments": ["--device", "QPU0"],
        },
        index=0,
    )
    command = build_server_command(Path("/llama-server"), case, 8081)
    assert command[-2:] == ["--device", "QPU0"]
    assert case["process_environment"]["LD_PRELOAD"].endswith("libggml-qpu-inline.so")


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
