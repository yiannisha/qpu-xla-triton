from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_llama_cpp_node_fixture_manifest import build_fixture_manifest
from scripts.llama_cpp_common import sha256_file


def _fixture_inputs(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"header" + bytes(18 * 16))
    activation = tmp_path / "activation.bin"
    activation.write_bytes(bytes(32 * 4))
    output = tmp_path / "output.bin"
    output.write_bytes(bytes(16 * 4))
    native = tmp_path / "native.json"
    native.write_text("{}\n", encoding="utf-8")
    case_file = tmp_path / "cases.json"
    case_file.write_text("{}\n", encoding="utf-8")
    binary = tmp_path / "profiler"
    binary.write_bytes(b"binary")
    session_path = tmp_path / "session.json"
    manifest_path = tmp_path / "manifest.json"
    model_hash = sha256_file(model)
    node = {
        "bytes": 64,
        "duration_ns": 123,
        "name": "result_output",
        "node_index": 7,
        "op": "MUL_MAT",
        "run_index": 0,
        "shape": [16, 1, 1, 1],
        "sources": [
            {
                "bytes": 18 * 16,
                "name": "token_embd.weight",
                "shape": [32, 16, 1, 1],
                "strides": [18, 18],
                "type": "q4_0",
            },
            {
                "bytes": 128,
                "name": "result_norm",
                "shape": [32, 1, 1, 1],
                "strides": [4, 128, 128, 128],
                "type": "f32",
            },
        ],
        "strides": [4, 64, 64, 64],
        "type": "f32",
    }
    capture = {
        "node": {
            "name": "result_output",
        },
        "node_index": 7,
        "run_index": 0,
        "tensors": [
            {
                "bytes": 128,
                "captured_bytes": 128,
                "name": "result_norm",
                "path": str(activation),
                "role": "src1",
                "sha256": sha256_file(activation),
                "shape": [32, 1, 1, 1],
                "strides": [4, 128, 128, 128],
                "truncated": False,
                "type": "f32",
            },
            {
                "bytes": 64,
                "captured_bytes": 64,
                "name": "result_output",
                "path": str(output),
                "role": "output",
                "sha256": sha256_file(output),
                "shape": [16, 1, 1, 1],
                "strides": [4, 64, 64, 64],
                "truncated": False,
                "type": "f32",
            },
        ],
    }
    session: dict[str, object] = {
        "case_file": {"path": str(case_file), "sha256": sha256_file(case_file)},
        "model_hashes": {str(model.resolve()): model_hash},
        "profile_binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "records": [
            {
                "case": {"name": "case", "model": str(model), "prompt": "fixed"},
                "native_output": {"path": str(native), "sha256": sha256_file(native)},
                "profile": {
                    "fixtures": [capture],
                    "model": str(model),
                    "nodes": [node],
                },
                "returncode": 0,
            }
        ],
    }
    manifest: dict[str, object] = {
        "model": {
            "path": str(model),
            "sha256": model_hash,
            "size_bytes": model.stat().st_size,
        },
        "tensors": [
            {
                "bytes": 18 * 16,
                "data_relative_offset": 0,
                "file_offset": 6,
                "ggml_nb_bytes": [18, 18],
                "ggml_type": "Q4_0",
                "is_aligned": True,
                "name": "token_embd.weight",
                "required_alignment": 1,
                "shape": [32, 16],
            }
        ],
    }
    session_path.write_text(json.dumps(session), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return session, manifest, session_path, manifest_path


def test_build_fixture_manifest_uses_native_weight_reference(tmp_path: Path) -> None:
    session, manifest, session_path, manifest_path = _fixture_inputs(tmp_path)
    result = build_fixture_manifest(
        session,
        manifest,
        session_path=session_path,
        gguf_manifest_path=manifest_path,
        case_name="case",
        node_name="result_output",
        weight_name="token_embd.weight",
    )
    assert result["dimensions"] == {"rows": 1, "input_columns": 32, "output_columns": 16}
    assert result["weight"]["file_offset"] == 6
    assert result["weight"]["materialized"] is False
    assert result["activation_f32"]["materialized"] is True
    assert result["measurement_contract"]["timing_eligible_for_promotion"] is False


def test_build_fixture_manifest_rejects_changed_capture(tmp_path: Path) -> None:
    session, manifest, session_path, manifest_path = _fixture_inputs(tmp_path)
    activation = Path(session["records"][0]["profile"]["fixtures"][0]["tensors"][0]["path"])
    activation.write_bytes(b"changed")
    with pytest.raises(ValueError, match="unexpected byte count"):
        build_fixture_manifest(
            session,
            manifest,
            session_path=session_path,
            gguf_manifest_path=manifest_path,
            case_name="case",
            node_name="result_output",
            weight_name="token_embd.weight",
        )
