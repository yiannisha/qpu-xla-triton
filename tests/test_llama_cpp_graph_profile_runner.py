from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_llama_cpp_graph_profiles import build_command, normalize_case, select_cases


def test_profile_case_command_is_explicit() -> None:
    case = normalize_case({"name": "m4", "model": "/model.gguf", "batch": 4}, 0)
    command = build_command(Path("/profiler"), case, Path("/out.json"))
    assert command[command.index("--batch") + 1] == "4"
    assert command[command.index("--threads") + 1] == "3"
    assert command[command.index("--context-type") + 1] == "default"
    assert "--quiet" in command

    fixture = normalize_case(
        {
            "model": "/model.gguf",
            "fixture_dir": "/fixtures",
            "fixture_pattern": "FLASH_ATTN",
            "fixture_all_sources": True,
        },
        0,
    )
    assert "--fixture-all-sources" in build_command(
        Path("/profiler"), fixture, Path("/out.json")
    )

    mtp = normalize_case(
        {
            "name": "draft",
            "model": "/draft.gguf",
            "other_model": "/base.gguf",
            "context_type": "mtp",
        },
        0,
    )
    mtp_command = build_command(Path("/profiler"), mtp, Path("/out.json"))
    assert mtp_command[mtp_command.index("--other-model") + 1] == "/base.gguf"


def test_profile_case_rejects_overfull_context_and_partial_fixture_config() -> None:
    with pytest.raises(ValueError, match="exceeds context_size"):
        normalize_case(
            {"model": "/model.gguf", "batch": 8, "context_tokens": 8, "context_size": 16},
            0,
        )
    with pytest.raises(ValueError, match="fixture_dir and fixture_pattern"):
        normalize_case({"model": "/model.gguf", "fixture_dir": "/tmp/fixture"}, 0)


def test_profile_case_selection_rejects_typos() -> None:
    cases = [{"name": "m1"}, {"name": "m4"}]
    assert select_cases(cases, ["m4"]) == [{"name": "m4"}]
    with pytest.raises(ValueError, match="unknown"):
        select_cases(cases, ["unknown"])
