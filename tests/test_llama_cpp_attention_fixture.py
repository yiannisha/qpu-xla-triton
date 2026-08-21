from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from qpu_xla.kernels.ggml_flash_attn import ggml_flash_attn_ext_reference
from scripts.build_llama_cpp_attention_fixture_manifest import (
    build_attention_fixture_manifest,
)
from scripts.llama_cpp_common import sha256_file
from scripts.replay_llama_cpp_attention_fixture import (
    calculate_attention_error,
    replay_attention_fixture,
)


def _strides(shape: tuple[int, int, int, int], itemsize: int) -> list[int]:
    result = [itemsize]
    for extent in shape[:-1]:
        result.append(result[-1] * extent)
    return result


def _capture(
    tmp_path: Path,
    role: str,
    name: str,
    values: np.ndarray,
) -> dict[str, Any]:
    contiguous = np.asfortranarray(values)
    path = tmp_path / f"{role}.bin"
    path.write_bytes(contiguous.tobytes(order="F"))
    tensor_type = "f16" if values.dtype == np.dtype(np.float16) else "f32"
    return {
        "bytes": path.stat().st_size,
        "captured_bytes": path.stat().st_size,
        "name": name,
        "path": str(path),
        "role": role,
        "sha256": sha256_file(path),
        "shape": list(values.shape),
        "strides": _strides(values.shape, values.dtype.itemsize),
        "truncated": False,
        "type": tensor_type,
    }


def test_build_and_replay_exact_flash_attention_fixture(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    native = tmp_path / "native.json"
    native.write_text("{}\n", encoding="utf-8")
    case_file = tmp_path / "cases.json"
    case_file.write_text("{}\n", encoding="utf-8")
    binary = tmp_path / "profiler"
    binary.write_bytes(b"binary")
    session_path = tmp_path / "session.json"

    query = np.array([[[[1.0], [0.25]]], [[[0.5], [1.0]]]], dtype=np.float32)
    key = np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=np.float16)
    value = np.array(
        [[[[2.0]], [[6.0]]], [[[4.0]], [[8.0]]], [[[1.0]], [[3.0]]]],
        dtype=np.float16,
    )
    mask = np.zeros((2, 1, 1, 1), dtype=np.float16)
    scale = 0.5
    expected = ggml_flash_attn_ext_reference(
        query,
        key,
        value,
        mask,
        scale=scale,
        max_bias=0.0,
        logit_softcap=0.0,
    )
    arrays = [query, key, value, mask]
    names = ["q", "k", "v", "mask"]
    captures = [
        _capture(tmp_path, f"src{index}", names[index], array)
        for index, array in enumerate(arrays)
    ]
    captures.append(_capture(tmp_path, "output", "attention", expected))
    sources = []
    for index, capture in enumerate(captures[:4]):
        sources.append(
            {
                field: capture[field]
                for field in ("bytes", "name", "shape", "strides", "type")
            }
            | {"source_index": index}
        )
    raw_params = struct.pack("<fffi", scale, 0.0, 0.0, 0) + bytes(48)
    op_params = {
        "byte_count": 64,
        "raw_little_endian_hex": raw_params.hex(),
        "parsed": {
            "scale": scale,
            "max_bias": 0.0,
            "logit_softcap": 0.0,
            "precision": 0,
            "has_sinks": False,
        },
    }
    node = {
        "bytes": captures[-1]["bytes"],
        "duration_ns": 123,
        "name": "node_7",
        "node_index": 7,
        "op": "FLASH_ATTN_EXT",
        "op_params": op_params,
        "run_index": 0,
        "shape": captures[-1]["shape"],
        "sources": sources,
        "strides": captures[-1]["strides"],
        "type": "f32",
    }
    session: dict[str, Any] = {
        "case_file": {"path": str(case_file), "sha256": sha256_file(case_file)},
        "model_hashes": {str(model.resolve()): sha256_file(model)},
        "profile_binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "records": [
            {
                "case": {"name": "attention", "model": str(model)},
                "native_output": {"path": str(native), "sha256": sha256_file(native)},
                "profile": {
                    "fixtures": [
                        {
                            "node": {"name": "node_7"},
                            "node_index": 7,
                            "run_index": 0,
                            "tensors": captures,
                        }
                    ],
                    "model": str(model),
                    "nodes": [node],
                },
                "returncode": 0,
            }
        ],
    }
    session_path.write_text(json.dumps(session), encoding="utf-8")
    fixture = build_attention_fixture_manifest(
        session,
        session_path=session_path,
        case_name="attention",
        node_name="node_7",
    )
    assert fixture["dimensions"]["query_heads"] == 2
    assert fixture["node"]["op_params"]["scale"] == scale
    actual, correctness = replay_attention_fixture(fixture)
    np.testing.assert_array_equal(actual, expected)
    assert correctness["passed"]


def test_attention_error_retains_nonfinite_failures() -> None:
    reference = np.zeros((1, 1, 1, 1), dtype=np.float32)
    actual = np.full_like(reference, np.nan)
    correctness = calculate_attention_error(reference, actual, atol=1e-3, rtol=1e-3)
    assert not correctness["passed"]
    assert correctness["nan_count"] == 1
    assert correctness["tolerance_violation_count"] == 1
    assert np.isinf(correctness["max_absolute"])
