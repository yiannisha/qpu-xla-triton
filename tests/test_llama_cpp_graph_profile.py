from __future__ import annotations

import pytest

from scripts.summarize_llama_cpp_graph_profile import (
    classify_node,
    percentile,
    summarize_profile,
)


def _node(
    index: int,
    op: str,
    name: str,
    duration_ns: int,
    source_name: str = "",
) -> dict[str, object]:
    return {
        "run_index": 0,
        "node_index": index,
        "op": op,
        "name": name,
        "duration_ns": duration_ns,
        "shape": [16, 1, 1, 1],
        "sources": [{"name": source_name}],
    }


def test_node_classification_uses_weight_owner_and_op() -> None:
    assert classify_node(_node(0, "MUL_MAT", "node", 1, "blk.0.ffn_gate.weight")) == "ffn_gate_up_linear"
    assert classify_node(_node(0, "MUL_MAT", "node", 1, "token_embd.weight")) == "lm_head"
    assert classify_node(_node(0, "FLASH_ATTN_EXT", "attn", 1)) == "attention_core"


def test_percentile_interpolates_small_samples() -> None:
    assert percentile([0.0, 10.0], 0.5) == 5.0
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)


def test_profile_summary_computes_per_run_family_fraction_and_marks_ineligible() -> None:
    profile = {
        "model": "/model.gguf",
        "configuration": {"batch": 1, "context_tokens": 0, "threads": 2},
        "measurement_contract": {"serialization": "per node"},
        "decode_calls": [{"run_index": 0, "wall_ns": 120}],
        "nodes": [
            _node(0, "MUL_MAT", "linear", 60, "blk.0.ffn_gate.weight"),
            _node(1, "RMS_NORM", "norm", 40),
        ],
        "fixtures": [],
    }
    result = summarize_profile(profile)
    assert result["serialized_node_total_ns"]["median_ns"] == 100
    assert result["families"][0]["family"] == "ffn_gate_up_linear"
    assert result["families"][0]["fraction_of_serialized_node_time"]["median"] == 0.6
    assert result["timing_eligible_for_promotion"] is False
