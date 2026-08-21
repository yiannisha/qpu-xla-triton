from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.generate_llama_cpp_evaluation_cases import (
    gemma_cases,
    parse_integer_list,
    qwen_cases,
)


def test_gemma_matrix_covers_threads_contexts_depths_and_exact_prompts(tmp_path: Path) -> None:
    cases = gemma_cases(tmp_path / "base.gguf", tmp_path / "draft.gguf")
    assert len(cases) == 48
    decode = [case for case in cases if case["workload"] == "decode"]
    prompts = [case for case in cases if case["mode"] == "prompt"]
    applications = [case for case in cases if case["workload"] == "structured-tool-call"]
    assert {case["threads"] for case in cases} == {2, 3, 4}
    assert {case["context_tokens_target"] for case in decode} == {0, 512, 2048}
    assert {case["draft_n_max"] for case in decode if case["mode"] == "mtp"} == {2, 3}
    assert {case["prompt_tokens_target"] for case in prompts} == {10, 128, 512, 2048}
    assert all(case["predict_tokens"] == 0 for case in prompts)
    assert len(applications) == 9
    assert all(case["predict_tokens"] == 40 for case in applications)
    assert all(case["request_surface"] == "chat-completions" for case in applications)
    assert all(case["context_tokens_target"] == 0 for case in applications)


def test_qwen_matrix_covers_depth_zero_and_unified_kv(tmp_path: Path) -> None:
    cases = qwen_cases(tmp_path / "qwen.gguf")
    assert len(cases) == 102
    mtp = [case for case in cases if case["mode"] == "mtp"]
    assert {case["draft_n_max"] for case in mtp} == {0, 1, 2, 3, 7}
    assert {case["context_tokens_target"] for case in mtp} == {0, 512, 2048, 4096}
    assert all(case["server_extra_arguments"] == ["--kv-unified"] for case in cases)
    applications = [case for case in cases if case["workload"] == "structured-tool-call"]
    assert len(applications) == 18
    assert {case["draft_n_max"] for case in applications if case["mode"] == "mtp"} == {
        0,
        1,
        2,
        3,
        7,
    }


def test_integer_list_parser_rejects_duplicates_and_zero_when_positive() -> None:
    assert parse_integer_list("0,512,2048") == (0, 512, 2048)
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        parse_integer_list("2,2", allow_zero=False)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_integer_list("0,2", allow_zero=False)
