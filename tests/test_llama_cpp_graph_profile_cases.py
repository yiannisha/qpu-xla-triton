from __future__ import annotations

from pathlib import Path

from scripts.generate_llama_cpp_graph_profile_cases import (
    base_profile_cases,
    configure_gemma_attention_fixtures,
    drafter_profile_cases,
)


def test_base_profile_cases_cover_required_widths_and_long_contexts(tmp_path: Path) -> None:
    cases = base_profile_cases(tmp_path / "base.gguf")
    assert len(cases) == 12
    assert {case["batch"] for case in cases if case["context_tokens"] == 0} == {
        1,
        2,
        3,
        4,
        5,
        8,
    }
    assert {
        (case["batch"], case["context_tokens"])
        for case in cases
        if case["context_tokens"] != 0
    } == {(batch, context) for batch in (1, 4) for context in (512, 2048, 4096)}


def test_drafter_profile_cases_use_mtp_context_for_initial_widths(tmp_path: Path) -> None:
    cases = drafter_profile_cases(tmp_path / "draft.gguf", tmp_path / "base.gguf")
    assert {case["batch"] for case in cases} == {1}
    assert all(case["context_type"] == "mtp" for case in cases)
    assert cases[0]["other_model"].endswith("base.gguf")


def test_attention_fixtures_cover_local_and_global_required_cases(tmp_path: Path) -> None:
    cases = base_profile_cases(tmp_path / "base.gguf")
    configure_gemma_attention_fixtures(cases, tmp_path / "fixtures")
    configured = [case for case in cases if case.get("fixture_dir")]
    assert {(case["batch"], case["context_tokens"]) for case in configured} == {
        (batch, context) for batch in (1, 4) for context in (0, 512, 2048, 4096)
    }
    assert all(case["fixture_all_sources"] for case in configured)
    assert all("node_30" in case["fixture_pattern"] for case in configured)
    assert all("node_235" in case["fixture_pattern"] for case in configured)
