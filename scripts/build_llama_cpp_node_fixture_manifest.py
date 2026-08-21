#!/usr/bin/env python3
"""Join a captured llama.cpp graph node with its native GGUF weight reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import sha256_file, write_json_atomic  # noqa: E402

Q4_0_BLOCK_ELEMENTS = 32
Q4_0_BLOCK_BYTES = 18
F32_BYTES = 4


def _one(values: list[dict[str, Any]], description: str) -> dict[str, Any]:
    if len(values) != 1:
        raise ValueError(f"expected one {description}, found {len(values)}")
    return values[0]


def _shape_without_unit_tail(shape: list[int]) -> list[int]:
    result = [int(value) for value in shape]
    while len(result) > 2 and result[-1] == 1:
        result.pop()
    return result


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
        "type": tensor["type"],
        "shape": [int(value) for value in tensor["shape"]],
        "strides": [int(value) for value in tensor["strides"]],
        "name": tensor["name"],
        "materialized": True,
    }


def build_fixture_manifest(
    session: dict[str, Any],
    gguf_manifest: dict[str, Any],
    *,
    session_path: Path,
    gguf_manifest_path: Path,
    case_name: str,
    node_name: str,
    weight_name: str,
) -> dict[str, Any]:
    """Build and validate one exact replay contract without copying model weights."""
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
    if node["op"] != "MUL_MAT" or node["type"] != "f32" or len(node["sources"]) != 2:
        raise ValueError("fixture node must be a two-source MUL_MAT with FP32 output")

    weight_source = node["sources"][0]
    activation_source = node["sources"][1]
    if weight_source["name"] != weight_name or weight_source["type"] != "q4_0":
        raise ValueError("profile source 0 is not the requested native Q4_0 weight")
    if activation_source["type"] != "f32":
        raise ValueError("profile source 1 is not an FP32 activation")

    weight = _one(
        [tensor for tensor in gguf_manifest["tensors"] if tensor["name"] == weight_name],
        f"GGUF tensor named {weight_name!r}",
    )
    if weight["ggml_type"] != "Q4_0" or len(weight["shape"]) != 2:
        raise ValueError("GGUF weight must be a rank-two Q4_0 tensor")
    input_columns, output_columns = map(int, weight["shape"])
    if input_columns % Q4_0_BLOCK_ELEMENTS:
        raise ValueError("GGUF weight input dimension is not Q4_0 block aligned")
    expected_weight_bytes = (
        output_columns * (input_columns // Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_BYTES
    )
    if int(weight["bytes"]) != expected_weight_bytes:
        raise ValueError("GGUF weight byte count is inconsistent with native Q4_0")
    if _shape_without_unit_tail(weight_source["shape"]) != [input_columns, output_columns]:
        raise ValueError("profile and GGUF weight shapes differ")
    if int(weight_source["bytes"]) != expected_weight_bytes:
        raise ValueError("profile and GGUF weight byte counts differ")

    captures = fixture["tensors"]
    activation_capture = _one(
        [tensor for tensor in captures if tensor["role"] == "src1"],
        "captured source-1 activation",
    )
    output_capture = _one(
        [tensor for tensor in captures if tensor["role"] == "output"],
        "captured node output",
    )
    if activation_capture["name"] != activation_source["name"]:
        raise ValueError("captured activation name does not match profile source 1")
    if _shape_without_unit_tail(activation_source["shape"])[0] != input_columns:
        raise ValueError("activation width does not match the weight input dimension")
    rows = int(activation_source["shape"][1])
    if rows not in {1, 4}:
        raise ValueError("captured row count is not supported by the native QPU kernels")
    if int(activation_capture["bytes"]) != rows * input_columns * F32_BYTES:
        raise ValueError("captured activation byte count is inconsistent with its shape")
    if _shape_without_unit_tail(node["shape"]) != [output_columns, rows]:
        raise ValueError("profile output shape is inconsistent with MUL_MAT dimensions")
    if int(output_capture["bytes"]) != rows * output_columns * F32_BYTES:
        raise ValueError("captured output byte count is inconsistent with its shape")

    model = gguf_manifest["model"]
    model_path = Path(model["path"]).resolve()
    case_model_path = Path(record["case"]["model"]).resolve()
    profile_model_path = Path(profile["model"]).resolve()
    if model_path != case_model_path or model_path != profile_model_path:
        raise ValueError("GGUF manifest and graph profile refer to different model paths")
    if not model_path.is_file() or model_path.stat().st_size != int(model["size_bytes"]):
        raise ValueError("GGUF model is missing or has an unexpected byte count")
    if int(weight["file_offset"]) + expected_weight_bytes > int(model["size_bytes"]):
        raise ValueError("GGUF weight range extends beyond the model file")
    model_hash = session["model_hashes"].get(str(model_path))
    if model_hash != model["sha256"]:
        raise ValueError("GGUF manifest and profile session model hashes differ")
    if sha256_file(model_path) != model["sha256"]:
        raise ValueError("live GGUF model hash does not match the retained manifest")

    activation = _verified_capture(activation_capture, "activation")
    reference_output = _verified_capture(output_capture, "CPU output")
    native_output = record.get("native_output")
    if native_output is None:
        raise ValueError("profile session does not reference its native output JSON")
    native_output_path = Path(native_output["path"]).resolve()
    if (
        not native_output_path.is_file()
        or sha256_file(native_output_path) != native_output["sha256"]
    ):
        raise ValueError("native profile JSON is missing or its hash has changed")

    return {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-exact-node-fixture",
        "fixture_id": f"{case_name}-run{fixture['run_index']}-node{fixture['node_index']}",
        "source": {
            "graph_profile_session": {
                "path": str(session_path.resolve()),
                "sha256": sha256_file(session_path),
            },
            "native_profile": {
                "path": str(native_output_path),
                "sha256": native_output["sha256"],
            },
            "gguf_manifest": {
                "path": str(gguf_manifest_path.resolve()),
                "sha256": sha256_file(gguf_manifest_path),
            },
            "profile_binary": dict(session["profile_binary"]),
            "case_file": dict(session["case_file"]),
        },
        "case": dict(record["case"]),
        "node": {
            "name": node["name"],
            "index": int(node["node_index"]),
            "run_index": int(node["run_index"]),
            "op": node["op"],
            "type": node["type"],
            "shape": [int(value) for value in node["shape"]],
            "strides": [int(value) for value in node["strides"]],
            "serialized_profile_duration_ns": int(node["duration_ns"]),
        },
        "dimensions": {
            "rows": rows,
            "input_columns": input_columns,
            "output_columns": output_columns,
        },
        "weight": {
            "name": weight["name"],
            "ggml_type": weight["ggml_type"],
            "shape": [int(value) for value in weight["shape"]],
            "ggml_nb_bytes": [int(value) for value in weight["ggml_nb_bytes"]],
            "bytes": expected_weight_bytes,
            "file_offset": int(weight["file_offset"]),
            "data_relative_offset": int(weight["data_relative_offset"]),
            "required_alignment": int(weight["required_alignment"]),
            "is_aligned": bool(weight["is_aligned"]),
            "model": {
                "path": str(model_path),
                "sha256": model["sha256"],
                "size_bytes": int(model["size_bytes"]),
            },
            "materialized": False,
        },
        "activation_f32": activation,
        "reference_output_f32": reference_output,
        "replay_contract": {
            "weight_layout": "native GGUF Q4_0 blocks, one output column per contiguous row",
            "activation_layout": "row-major FP32; native operator quantizes each row to Q8_0",
            "output_layout": "row-major FP32 logits",
            "reference_backend": "pinned llama.cpp CPU graph output captured after result_output",
            "supported_candidate_rows": [1, 4],
            "atol": 0.0002,
            "rtol": 0.00002,
            "greedy_argmax_must_match": True,
        },
        "measurement_contract": {
            "correctness_eligible": True,
            "timing_eligible_for_promotion": False,
            "reason": (
                "the eval callback serializes graph nodes; its node duration is diagnostic only"
            ),
            "weight_materialization": (
                "replay tools seek this bounded tensor range from the hashed GGUF at runtime"
            ),
        },
    }


def main() -> None:
    """Validate the requested capture and write its replay manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-session", type=Path, required=True)
    parser.add_argument("--gguf-manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--node", default="result_output")
    parser.add_argument("--weight", default="token_embd.weight")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for description, path in (
        ("profile session", args.profile_session),
        ("GGUF manifest", args.gguf_manifest),
    ):
        if not path.is_file():
            parser.error(f"{description} not found: {path}")
    session = json.loads(args.profile_session.read_text(encoding="utf-8"))
    gguf_manifest = json.loads(args.gguf_manifest.read_text(encoding="utf-8"))
    try:
        payload = build_fixture_manifest(
            session,
            gguf_manifest,
            session_path=args.profile_session,
            gguf_manifest_path=args.gguf_manifest,
            case_name=args.case,
            node_name=args.node,
            weight_name=args.weight,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_json_atomic(args.output, payload)
    print(
        f"wrote {args.output}: {payload['node']['name']} "
        f"{payload['dimensions']['rows']}x{payload['dimensions']['input_columns']}x"
        f"{payload['dimensions']['output_columns']}"
    )


if __name__ == "__main__":
    main()
