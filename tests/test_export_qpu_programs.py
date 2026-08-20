from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.export_qpu_programs import PROGRAMS, export_program


def test_exported_program_is_deterministic_and_self_describing(tmp_path: Path) -> None:
    spec = PROGRAMS["w8a8-gemv"]
    first = export_program(spec, tmp_path / "first")
    second = export_program(spec, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/w8a8-gemv.bin").read_bytes() == (
        tmp_path / "second/w8a8-gemv.bin"
    ).read_bytes()
    assert first["uniform_word_count"] == 5
    binary = tmp_path / "first/w8a8-gemv.bin"
    assert first["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    header = (tmp_path / "first/w8a8-gemv.h").read_text(encoding="utf-8")
    assert first["source_hash"] in header
    assert first["binary_sha256"] in header


def test_export_manifest_shape_is_json_safe(tmp_path: Path) -> None:
    record = export_program(PROGRAMS["tiled-w8a8-gemm"], tmp_path)
    assert json.loads(json.dumps(record))["launch"]["local_invocation"] == [16, 1, 1]


def test_native_q4_export_uses_llama_cpp_q8_0_block_size(tmp_path: Path) -> None:
    record = export_program(PROGRAMS["ggml-q4-0-q8-0-m4"], tmp_path)
    q8_uniform = next(item for item in record["uniforms"] if item["name"] == "q8_0_block_bytes")
    assert q8_uniform["value"] == "34"
