#!/usr/bin/env python3
"""Extract a native-layout GGUF tensor manifest for llama.cpp QPU calibration."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import sha256_file, utc_now, write_json_atomic  # noqa: E402

LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")


def classify_operator(name: str) -> str:
    """Map a GGUF tensor name to a stable operator family."""
    lowered = name.lower()
    if lowered.startswith("output.") or lowered == "output.weight":
        return "output_head"
    if "layer_output_scale" in lowered:
        return "output_scaling"
    patterns = (
        ("ffn_gate", "ffn_gate"),
        ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"),
        ("attn_qkv", "attention_qkv"),
        ("attn_q", "attention_q"),
        ("attn_k", "attention_k"),
        ("attn_v", "attention_v"),
        ("attn_output", "attention_output"),
        ("ssm_conv", "deltanet_convolution"),
        ("conv1d", "deltanet_convolution"),
        ("ssm", "deltanet_scan"),
        ("norm", "normalization"),
        ("rope", "rope"),
        ("token_embd", "embedding"),
        ("position", "position_embedding"),
        ("class_embedding", "class_embedding"),
        ("patch", "vision_patch"),
        ("proj", "projection"),
    )
    for marker, family in patterns:
        if marker in lowered:
            return family
    if lowered.endswith((".bias", ".weight")):
        return "other_parameter"
    return "other"


def tensor_owner(name: str, component: str) -> dict[str, Any]:
    """Return model component, layer, and operator ownership."""
    match = LAYER_RE.search(name)
    return {
        "component": component,
        "layer": int(match.group(1)) if match else None,
        "operator": classify_operator(name),
    }


def ggml_strides(shape: list[int], block_size: int, type_size: int) -> list[int]:
    """Compute GGML's nb[] values for a contiguous native tensor."""
    if not shape:
        return []
    strides = [type_size]
    if len(shape) > 1:
        if shape[0] % block_size:
            raise ValueError(f"dimension {shape[0]} is not divisible by block size {block_size}")
        strides.append(type_size * shape[0] // block_size)
    for dimension in shape[1:-1]:
        strides.append(strides[-1] * dimension)
    return strides


def _json_metadata_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, list):
        return [_json_metadata_value(item) for item in value]
    return value


def _architecture_metadata(reader: Any) -> dict[str, Any]:
    architecture_field = reader.fields.get("general.architecture")
    architecture = str(architecture_field.contents()) if architecture_field is not None else ""
    prefixes = ("general.", f"{architecture}.")
    metadata: dict[str, Any] = {}
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or not name.startswith(prefixes):
            continue
        metadata[name] = _json_metadata_value(field.contents())
    return metadata


def build_manifest(model: Path, component: str, llama_root: Path, *, include_hash: bool = True) -> dict[str, Any]:
    """Build a complete tensor inventory while keeping tensor payloads memory-mapped."""
    gguf_python = llama_root / "gguf-py"
    if not gguf_python.is_dir():
        raise FileNotFoundError(f"llama.cpp gguf-py not found: {gguf_python}")
    sys.path.insert(0, str(gguf_python))
    from gguf import GGML_QUANT_SIZES, GGUFReader  # type: ignore[import-not-found]

    reader = GGUFReader(model, "r")
    tensors: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_operator: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_component: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tensor in reader.tensors:
        shape = [int(value) for value in tensor.shape.tolist()]
        block_size, type_size = GGML_QUANT_SIZES[tensor.tensor_type]
        owner = tensor_owner(tensor.name, component)
        absolute_offset = int(tensor.data_offset)
        record = {
            "name": tensor.name,
            "shape": shape,
            "ggml_type": tensor.tensor_type.name,
            "block_elements": int(block_size),
            "block_bytes": int(type_size),
            "ggml_nb_bytes": ggml_strides(shape, int(block_size), int(type_size)),
            "elements": int(tensor.n_elements),
            "bytes": int(tensor.n_bytes),
            "file_offset": absolute_offset,
            "data_relative_offset": absolute_offset - int(reader.data_offset),
            "required_alignment": int(reader.alignment),
            "is_aligned": absolute_offset % int(reader.alignment) == 0,
            "owner": owner,
            "matrix_macs_per_activation_row": int(tensor.n_elements) if len(shape) == 2 else 0,
        }
        tensors.append(record)
        for aggregate_key, aggregate_value in (
            ("by_tensor_type", tensor.tensor_type.name),
            ("by_operator", owner["operator"]),
            ("by_component", owner["component"]),
        ):
            target = {
                "by_tensor_type": by_type,
                "by_operator": by_operator,
                "by_component": by_component,
            }[aggregate_key][str(aggregate_value)]
            target["tensor_count"] += 1
            target["elements"] += int(tensor.n_elements)
            target["bytes"] += int(tensor.n_bytes)
            if len(shape) == 2:
                target["matrix_macs_per_activation_row"] += int(tensor.n_elements)

    return {
        "schema_version": 1,
        "kind": "llama-cpp-gguf-tensor-manifest",
        "created_utc": utc_now(),
        "model": {
            "path": str(model.resolve()),
            "filename": model.name,
            "size_bytes": model.stat().st_size,
            "sha256": sha256_file(model) if include_hash else None,
            "component": component,
            "gguf_alignment": int(reader.alignment),
            "gguf_data_offset": int(reader.data_offset),
        },
        "architecture": _architecture_metadata(reader),
        "tensors": tensors,
        "aggregates": {
            "by_tensor_type": {name: dict(values) for name, values in sorted(by_type.items())},
            "by_operator": {name: dict(values) for name, values in sorted(by_operator.items())},
            "by_component": {name: dict(values) for name, values in sorted(by_component.items())},
            "total": {
                "tensor_count": len(tensors),
                "elements": sum(record["elements"] for record in tensors),
                "bytes": sum(record["bytes"] for record in tensors),
                "matrix_macs_per_activation_row": sum(
                    record["matrix_macs_per_activation_row"] for record in tensors
                ),
            },
        },
    }


def main() -> None:
    """Parse command-line arguments and write the requested manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--component", choices=("base", "mtp_head", "vision_encoder", "audio_encoder", "projector"))
    parser.add_argument("--llama-root", type=Path, default=Path.home() / "side/llama.cpp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    component = args.component
    if component is None:
        component = "mtp_head" if args.model.name.lower().startswith("mtp-") else "base"
    payload = build_manifest(
        args.model.resolve(),
        component,
        args.llama_root.resolve(),
        include_hash=not args.skip_sha256,
    )
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(payload['tensors'])} tensors")


if __name__ == "__main__":
    main()
