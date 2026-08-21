from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_llama_cpp_gguf_manifest import classify_operator, ggml_strides, tensor_owner
from scripts.llama_cpp_common import parse_cmake_cache, sha256_file, write_json_atomic


def test_ggml_strides_for_quantized_matrix() -> None:
    assert ggml_strides([256, 2048], block_size=32, type_size=18) == [18, 144]


def test_ggml_strides_for_fp32_tensor() -> None:
    assert ggml_strides([256, 4, 2], block_size=1, type_size=4) == [4, 1024, 4096]


def test_tensor_classification_preserves_layer_and_component() -> None:
    assert tensor_owner("blk.17.ffn_gate.weight", "base") == {
        "component": "base",
        "layer": 17,
        "operator": "ffn_gate",
    }
    assert classify_operator("token_embd.weight") == "embedding"
    assert classify_operator("blk.3.ssm_conv1d.weight") == "deltanet_convolution"
    assert classify_operator("blk.3.layer_output_scale.weight") == "output_scaling"
    assert tensor_owner("blk.32.ffn_gate.weight", "base", embedded_mtp_layer_start=32) == {
        "component": "mtp_head",
        "layer": 32,
        "operator": "ffn_gate",
    }


def test_atomic_json_and_hash(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    write_json_atomic(output, {"b": 2, "a": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert sha256_file(output) == "080d51f49b27c73d17f51f3b808515a425d16218aa40021eed2ca1d204e59224"


def test_parse_cmake_cache(tmp_path: Path) -> None:
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("// comment\nA:BOOL=ON\nB:STRING=value=with=equals\n", encoding="utf-8")
    assert parse_cmake_cache(cache) == {"A": "ON", "B": "value=with=equals"}
