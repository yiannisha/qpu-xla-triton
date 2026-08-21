#!/usr/bin/env python3
"""Validate an all-source FLASH_ATTN_EXT capture and emit a replay contract."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import sha256_file, write_json_atomic  # noqa: E402


def _one(values: list[dict[str, Any]], description: str) -> dict[str, Any]:
    if len(values) != 1:
        raise ValueError(f"expected one {description}, found {len(values)}")
    return values[0]


def _verified_capture(tensor: dict[str, Any], role: str) -> dict[str, Any]:
    path = Path(tensor["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"captured {role} file does not exist: {path}")
    expected_bytes = int(tensor["bytes"])
    if tensor.get("truncated") or int(tensor["captured_bytes"]) != expected_bytes:
        raise ValueError(f"captured {role} is truncated")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"captured {role} has an unexpected byte count")
    digest = sha256_file(path)
    if tensor.get("sha256") != digest:
        raise ValueError(f"captured {role} hash does not match the profile session")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": expected_bytes,
        "type": str(tensor["type"]),
        "shape": [int(value) for value in tensor["shape"]],
        "strides": [int(value) for value in tensor["strides"]],
        "name": str(tensor["name"]),
    }


def _required_span(tensor: dict[str, Any]) -> int:
    element_bytes = {"f16": 2, "f32": 4}.get(str(tensor["type"]))
    if element_bytes is None:
        raise ValueError(f"unsupported captured tensor type {tensor['type']!r}")
    shape = [int(value) for value in tensor["shape"]]
    strides = [int(value) for value in tensor["strides"]]
    if len(shape) != 4 or len(strides) != 4 or any(value <= 0 for value in shape):
        raise ValueError("attention tensors must expose four positive GGML dimensions")
    return element_bytes + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def _validate_tensor_storage(tensor: dict[str, Any], role: str) -> None:
    if _required_span(tensor) > int(tensor["bytes"]):
        raise ValueError(f"captured {role} does not span its declared strided view")


def _source_map(node: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for source in node["sources"]:
        if "source_index" not in source:
            raise ValueError("profile lacks source_index metadata; recapture with the current profiler")
        index = int(source["source_index"])
        if index in result:
            raise ValueError(f"duplicate attention source index {index}")
        result[index] = source
    return result


def _parse_op_params(node: dict[str, Any]) -> dict[str, Any]:
    params = node.get("op_params")
    if not isinstance(params, dict) or int(params.get("byte_count", -1)) != 64:
        raise ValueError("profile lacks the exact 64-byte GGML op-parameter record")
    try:
        raw = bytes.fromhex(str(params["raw_little_endian_hex"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("profile GGML op parameters are not valid hexadecimal") from exc
    if len(raw) != 64:
        raise ValueError("profile GGML op-parameter hex has an unexpected length")
    scale, max_bias, logit_softcap, precision = struct.unpack_from("<fffi", raw)
    parsed = params.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError("profile lacks parsed FLASH_ATTN_EXT parameters")
    expected = {
        "scale": scale,
        "max_bias": max_bias,
        "logit_softcap": logit_softcap,
        "precision": precision,
    }
    for field, value in expected.items():
        if parsed.get(field) != value:
            raise ValueError(f"parsed FLASH_ATTN_EXT {field} disagrees with raw bytes")
    return {
        "raw_little_endian_hex": raw.hex(),
        **expected,
        "has_sinks": bool(parsed.get("has_sinks")),
    }


def build_attention_fixture_manifest(
    session: dict[str, Any],
    *,
    session_path: Path,
    case_name: str,
    node_name: str,
) -> dict[str, Any]:
    """Build one exact, self-checking FLASH_ATTN_EXT replay manifest."""
    record = _one(
        [entry for entry in session["records"] if entry["case"]["name"] == case_name],
        f"profile case named {case_name!r}",
    )
    if record.get("returncode") != 0 or not record.get("profile"):
        raise ValueError(f"profile case {case_name!r} did not complete successfully")
    profile = record["profile"]
    fixture = _one(
        [entry for entry in profile.get("fixtures", []) if entry["node"]["name"] == node_name],
        f"captured node named {node_name!r}",
    )
    node = _one(
        [
            entry
            for entry in profile["nodes"]
            if entry["name"] == node_name
            and int(entry["node_index"]) == int(fixture["node_index"])
            and int(entry["run_index"]) == int(fixture["run_index"])
        ],
        f"profile node named {node_name!r} at the captured run/index",
    )
    if node["op"] != "FLASH_ATTN_EXT" or node["type"] != "f32":
        raise ValueError("fixture node must be FLASH_ATTN_EXT with FP32 output")
    sources = _source_map(node)
    if set(sources) not in ({0, 1, 2, 3}, {0, 1, 2, 3, 4}):
        raise ValueError("FLASH_ATTN_EXT fixture must have Q, K, V, mask, and optional sinks")
    expected_types = {0: "f32", 1: "f16", 2: "f16", 3: "f16", 4: "f32"}
    for index, source in sources.items():
        if source["type"] != expected_types[index]:
            raise ValueError(
                f"attention src{index} must be {expected_types[index]}, got {source['type']}"
            )

    captures = [entry for entry in fixture["tensors"] if entry is not None]
    captured: dict[int | str, dict[str, Any]] = {}
    for role in [f"src{index}" for index in sorted(sources)] + ["output"]:
        capture = _one(
            [entry for entry in captures if entry["role"] == role],
            f"captured attention {role}",
        )
        item = _verified_capture(capture, role)
        _validate_tensor_storage(item, role)
        captured[role if role == "output" else int(role[3:])] = item
    for index, source in sources.items():
        item = captured[index]
        for field in ("name", "type", "shape", "strides", "bytes"):
            if item[field] != source[field]:
                raise ValueError(f"captured src{index} {field} differs from graph metadata")

    q_shape = captured[0]["shape"]
    k_shape = captured[1]["shape"]
    v_shape = captured[2]["shape"]
    mask_shape = captured[3]["shape"]
    output_shape = captured["output"]["shape"]
    dk, query_rows, query_heads, batches = q_shape
    key_dk, kv_rows, key_heads, key_batches = k_shape
    dv, value_rows, value_heads, value_batches = v_shape
    if dk != key_dk or kv_rows != value_rows:
        raise ValueError("captured Q/K or K/V dimensions disagree")
    if query_heads % key_heads or query_heads % value_heads:
        raise ValueError("captured K/V heads do not broadcast to query heads")
    if batches % key_batches or batches % value_batches:
        raise ValueError("captured K/V batches do not broadcast to query batches")
    if mask_shape[0:2] != [kv_rows, query_rows]:
        raise ValueError("captured attention mask row dimensions disagree")
    if query_heads % mask_shape[2] or batches % mask_shape[3]:
        raise ValueError("captured attention mask does not broadcast to Q")
    if output_shape != [dv, query_heads, query_rows, batches]:
        raise ValueError("captured attention output shape is inconsistent")
    if 4 in captured and captured[4]["shape"][0] != query_heads:
        raise ValueError("captured attention sinks do not match query heads")

    op_params = _parse_op_params(node)
    if op_params["has_sinks"] != (4 in sources):
        raise ValueError("attention sink source disagrees with op parameters")
    model_path = Path(profile["model"]).resolve()
    if Path(record["case"]["model"]).resolve() != model_path:
        raise ValueError("profile case and native result refer to different model paths")
    model_hash = session["model_hashes"].get(str(model_path))
    if model_hash is None or not model_path.is_file() or sha256_file(model_path) != model_hash:
        raise ValueError("live model does not match the retained profile hash")
    native_output = record.get("native_output")
    if native_output is None:
        raise ValueError("profile session does not reference its native output JSON")
    native_path = Path(native_output["path"]).resolve()
    if not native_path.is_file() or sha256_file(native_path) != native_output["sha256"]:
        raise ValueError("native profile JSON is missing or its hash has changed")

    return {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-exact-flash-attention-fixture",
        "fixture_id": f"{case_name}-run{fixture['run_index']}-node{fixture['node_index']}",
        "source": {
            "graph_profile_session": {
                "path": str(session_path.resolve()),
                "sha256": sha256_file(session_path),
            },
            "native_profile": {"path": str(native_path), "sha256": native_output["sha256"]},
            "profile_binary": dict(session["profile_binary"]),
            "case_file": dict(session["case_file"]),
            "model": {"path": str(model_path), "sha256": model_hash},
        },
        "case": dict(record["case"]),
        "node": {
            "name": node["name"],
            "index": int(node["node_index"]),
            "run_index": int(node["run_index"]),
            "op": node["op"],
            "shape": output_shape,
            "strides": captured["output"]["strides"],
            "serialized_profile_duration_ns": int(node["duration_ns"]),
            "op_params": op_params,
        },
        "dimensions": {
            "query_embedding": dk,
            "value_embedding": dv,
            "query_rows": query_rows,
            "kv_rows": kv_rows,
            "query_heads": query_heads,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "batches": batches,
        },
        "query": captured[0],
        "key": captured[1],
        "value": captured[2],
        "mask": captured[3],
        "sinks": captured.get(4),
        "reference_output": captured["output"],
        "replay_contract": {
            "axis_order": "native GGML ne order",
            "semantics": "independent fused streaming QK-softmax-V reference",
            "atol": 0.005,
            "rtol": 0.005,
            "finite_outputs_required": True,
        },
        "measurement_contract": {
            "correctness_eligible": True,
            "timing_eligible_for_promotion": False,
            "reason": "the eval callback serializes graph nodes; fixture replay is untimed correctness evidence",
        },
    }


def main() -> None:
    """Validate CLI inputs and write one exact attention fixture manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-session", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.profile_session.is_file():
        parser.error(f"profile session not found: {args.profile_session}")
    session = json.loads(args.profile_session.read_text(encoding="utf-8"))
    try:
        payload = build_attention_fixture_manifest(
            session,
            session_path=args.profile_session,
            case_name=args.case,
            node_name=args.node,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {payload['fixture_id']}")


if __name__ == "__main__":
    main()
