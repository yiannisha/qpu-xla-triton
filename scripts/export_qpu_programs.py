#!/usr/bin/env python3
"""Deterministically export selected QPU assembly as binaries, headers, and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from _videocore7.assembler import assemble  # noqa: E402
from qpu_xla.kernels.gemm_int8 import qpu_tiled_w8a8_gemm  # noqa: E402
from qpu_xla.kernels.gemv_int8 import qpu_w8a8_gemv  # noqa: E402
from qpu_xla.kernels.ggml_flash_attn import qpu_ggml_gemma_flash_attn_f16_m1  # noqa: E402
from qpu_xla.kernels.ggml_geglu import qpu_ggml_geglu_split_fp32  # noqa: E402
from qpu_xla.kernels.ggml_geglu_q8 import qpu_ggml_geglu_q8_0  # noqa: E402
from qpu_xla.kernels.ggml_q4_0 import (  # noqa: E402
    qpu_ggml_q4_0_q8_0_linear,
    qpu_ggml_q4_0_q8_0_tiled_gemm,
)
from qpu_xla.kernels.ggml_q4_k import qpu_ggml_q4_k_q8_k_linear_m4  # noqa: E402
from qpu_xla.kernels.ggml_q6_k import qpu_ggml_q6_k_q8_k_linear_m4  # noqa: E402
from qpu_xla.kernels.ggml_q8_0 import qpu_ggml_q8_0_q8_0_linear_m4  # noqa: E402
from scripts.llama_cpp_common import sha256_file, write_json_atomic  # noqa: E402


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    """Assembly entry point plus the native launch contract it requires."""

    name: str
    symbol: str
    function: Callable[..., None]
    assembly_kwargs: dict[str, object]
    uniforms: tuple[dict[str, str], ...]
    launch: dict[str, object]
    description: str
    data_alignment: int = 4
    code_alignment: int = 8


PROGRAMS = {
    "ggml-geglu-q8-0": ProgramSpec(
        name="ggml-geglu-q8-0",
        symbol="qpu_ggml_geglu_q8_0",
        function=qpu_ggml_geglu_q8_0,
        assembly_kwargs={},
        uniforms=(
            {"name": "blocks_per_row", "type": "uint32"},
            {"name": "gate_address", "type": "gpu_address"},
            {"name": "up_address", "type": "gpu_address"},
            {"name": "q8_0_destination_address", "type": "gpu_address"},
            {"name": "ggml_gelu_fp16_table_address", "type": "gpu_address"},
            {"name": "inverse_127", "type": "float32"},
            {"name": "minimum_maximum", "type": "float32"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["columns / 32", "rows", 1],
            "wgs_per_sg": 24,
            "thread": "rows * columns / 32",
            "constraint": "columns % 32 == 0",
        },
        description=(
            "Exact lookup-table GGML GEGLU fused with native per-block Q8_0 "
            "quantization for direct consumption by a down projection."
        ),
    ),
    "ggml-geglu-split-fp32": ProgramSpec(
        name="ggml-geglu-split-fp32",
        symbol="qpu_ggml_geglu_split_fp32",
        function=qpu_ggml_geglu_split_fp32,
        assembly_kwargs={},
        uniforms=(
            {"name": "iterations_per_workgroup", "type": "uint32"},
            {"name": "gate_address", "type": "gpu_address"},
            {"name": "up_address", "type": "gpu_address"},
            {"name": "destination_address", "type": "gpu_address"},
            {"name": "ggml_gelu_fp16_table_address", "type": "gpu_address"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": [12, 1, 1],
            "wgs_per_sg": 24,
            "thread": 12,
            "constraint": "element_count % 768 == 0",
        },
        description=(
            "Native GGML split GEGLU over equal contiguous FP32 tensors, using "
            "four words per SIMD lane and all twelve VideoCore VII QPUs."
        ),
    ),
    "ggml-gemma-flash-attn-f16-m1": ProgramSpec(
        name="ggml-gemma-flash-attn-f16-m1",
        symbol="qpu_ggml_gemma_flash_attn_f16_m1",
        function=qpu_ggml_gemma_flash_attn_f16_m1,
        assembly_kwargs={},
        uniforms=(
            {"name": "kv_row_pairs", "type": "uint32"},
            {"name": "query_address", "type": "gpu_address"},
            {"name": "query_head_stride_bytes", "type": "uint32"},
            {"name": "key_address", "type": "gpu_address"},
            {"name": "key_row_stride_bytes", "type": "uint32"},
            {"name": "value_address", "type": "gpu_address"},
            {"name": "value_row_stride_bytes", "type": "uint32"},
            {"name": "mask_address", "type": "gpu_address"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "output_head_stride_bytes", "type": "uint32"},
            {"name": "scale", "type": "float32"},
            {"name": "log2_e", "type": "float32"},
            {"name": "negative_infinity", "type": "float32"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["query_heads", 1, 1],
            "wgs_per_sg": 48,
            "thread": "query_heads",
        },
        description=(
            "Experimental fused Gemma M=1 attention for 256-wide F32 query heads, "
            "one native F16 KV head/mask, and no ALiBi, softcap, or sinks."
        ),
    ),
    "ggml-gemma-flash-attn-f16-mx": ProgramSpec(
        name="ggml-gemma-flash-attn-f16-mx",
        symbol="qpu_ggml_gemma_flash_attn_f16_mx",
        function=qpu_ggml_gemma_flash_attn_f16_m1,
        assembly_kwargs={"batched_query": True},
        uniforms=(
            {"name": "kv_row_pairs", "type": "uint32"},
            {"name": "query_address", "type": "gpu_address"},
            {"name": "query_head_stride_bytes", "type": "uint32"},
            {"name": "query_row_stride_bytes", "type": "uint32"},
            {"name": "key_address", "type": "gpu_address"},
            {"name": "key_row_stride_bytes", "type": "uint32"},
            {"name": "value_address", "type": "gpu_address"},
            {"name": "value_row_stride_bytes", "type": "uint32"},
            {"name": "mask_address", "type": "gpu_address"},
            {"name": "mask_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "output_head_stride_bytes", "type": "uint32"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "scale", "type": "float32"},
            {"name": "log2_e", "type": "float32"},
            {"name": "negative_infinity", "type": "float32"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["8 * query_rows", 1, 1],
            "wgs_per_sg": 48,
            "thread": "8 * query_rows",
        },
        description=(
            "Experimental fused Gemma batched-prefill attention for Mx8x256 F32 "
            "queries, one native F16 KV head/mask, and no ALiBi, softcap, or sinks."
        ),
    ),
    "ggml-q8-0-q8-0-m4": ProgramSpec(
        name="ggml-q8-0-q8-0-m4",
        symbol="qpu_ggml_q8_0_q8_0_m4",
        function=qpu_ggml_q8_0_q8_0_linear_m4,
        assembly_kwargs={},
        uniforms=(
            {"name": "reduction_blocks", "type": "uint32"},
            {"name": "activation_row_stride_bytes", "type": "uint32"},
            {"name": "activation_address", "type": "gpu_address"},
            {"name": "weight_row_stride_bytes", "type": "uint32"},
            {"name": "weight_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "weight_column_offset", "type": "uint32"},
            {"name": "output_column_offset", "type": "uint32"},
            {"name": "q8_0_block_bytes", "type": "uint32", "value": "34"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", 1, 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x",
        },
        description="Native GGML Q8_0 by Q8_0 four-row linear with FP32 output.",
    ),
    "ggml-q6-k-q8-k-m4": ProgramSpec(
        name="ggml-q6-k-q8-k-m4",
        symbol="qpu_ggml_q6_k_q8_k_m4",
        function=qpu_ggml_q6_k_q8_k_linear_m4,
        assembly_kwargs={},
        uniforms=(
            {"name": "reduction_superblocks", "type": "uint32"},
            {"name": "activation_row_stride_bytes", "type": "uint32"},
            {"name": "activation_address", "type": "gpu_address"},
            {"name": "weight_row_stride_bytes", "type": "uint32"},
            {"name": "weight_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "weight_column_offset", "type": "uint32"},
            {"name": "output_column_offset", "type": "uint32"},
            {"name": "nibble_mask", "type": "uint32", "value": "0x0f0f0f0f"},
            {"name": "two_bit_mask", "type": "uint32", "value": "0x03030303"},
            {"name": "signed_byte_ones", "type": "uint32", "value": "0x01010101"},
            {"name": "q6_k_block_bytes", "type": "uint32", "value": "210"},
            {"name": "q8_k_block_bytes", "type": "uint32", "value": "292"},
            {"name": "q6_k_ql_half_bytes", "type": "uint32", "value": "64"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", 1, 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x",
        },
        description="Native GGML Q6_K by Q8_K four-row linear with FP32 output.",
    ),
    "ggml-q4-k-q8-k-m4": ProgramSpec(
        name="ggml-q4-k-q8-k-m4",
        symbol="qpu_ggml_q4_k_q8_k_m4",
        function=qpu_ggml_q4_k_q8_k_linear_m4,
        assembly_kwargs={},
        uniforms=(
            {"name": "reduction_superblocks", "type": "uint32"},
            {"name": "activation_row_stride_bytes", "type": "uint32"},
            {"name": "activation_address", "type": "gpu_address"},
            {"name": "weight_row_stride_bytes", "type": "uint32"},
            {"name": "weight_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "weight_column_offset", "type": "uint32"},
            {"name": "output_column_offset", "type": "uint32"},
            {"name": "nibble_mask", "type": "uint32", "value": "0x0f0f0f0f"},
            {"name": "signed_byte_ones", "type": "uint32", "value": "0x01010101"},
            {"name": "six_bit_mask", "type": "uint32", "value": "0x0000003f"},
            {"name": "q4_k_block_bytes", "type": "uint32", "value": "144"},
            {"name": "q8_k_block_bytes", "type": "uint32", "value": "292"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", 1, 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x",
        },
        description="Native GGML Q4_K by Q8_K four-row linear with FP32 output.",
    ),
    "ggml-q4-0-q8-0-m1": ProgramSpec(
        name="ggml-q4-0-q8-0-m1",
        symbol="qpu_ggml_q4_0_q8_0_m1",
        function=qpu_ggml_q4_0_q8_0_linear,
        assembly_kwargs={"rows": 1},
        uniforms=(
            {"name": "reduction_blocks", "type": "uint32"},
            {"name": "activation_row_stride_bytes", "type": "uint32"},
            {"name": "activation_address", "type": "gpu_address"},
            {"name": "weight_row_stride_bytes", "type": "uint32"},
            {"name": "weight_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "weight_column_offset", "type": "uint32"},
            {"name": "output_column_offset", "type": "uint32"},
            {"name": "nibble_mask", "type": "uint32", "value": "0x0f0f0f0f"},
            {"name": "nibble_sign_bit", "type": "uint32", "value": "0x08080808"},
            {"name": "q4_0_block_bytes", "type": "uint32", "value": "18"},
            {"name": "q8_0_block_bytes", "type": "uint32", "value": "34"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", 1, 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x",
        },
        description=(
            "Native GGML Q4_0 by Q8_0 logical one-row linear with FP32 output; "
            "the C adapter expands input and staging to the four-row physical pipeline."
        ),
    ),
    "ggml-q4-0-q8-0-m4": ProgramSpec(
        name="ggml-q4-0-q8-0-m4",
        symbol="qpu_ggml_q4_0_q8_0_m4",
        function=qpu_ggml_q4_0_q8_0_linear,
        assembly_kwargs={"rows": 4},
        uniforms=(
            {"name": "reduction_blocks", "type": "uint32"},
            {"name": "activation_row_stride_bytes", "type": "uint32"},
            {"name": "activation_address", "type": "gpu_address"},
            {"name": "weight_row_stride_bytes", "type": "uint32"},
            {"name": "weight_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "weight_column_offset", "type": "uint32"},
            {"name": "output_column_offset", "type": "uint32"},
            {"name": "nibble_mask", "type": "uint32", "value": "0x0f0f0f0f"},
            {"name": "nibble_sign_bit", "type": "uint32", "value": "0x08080808"},
            {"name": "q4_0_block_bytes", "type": "uint32", "value": "18"},
            {"name": "q8_0_block_bytes", "type": "uint32", "value": "34"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", 1, 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x",
        },
        description="Native GGML Q4_0 by Q8_0 four-row linear with FP32 output.",
    ),
    "ggml-q4-0-q8-0-mx": ProgramSpec(
        name="ggml-q4-0-q8-0-mx",
        symbol="qpu_ggml_q4_0_q8_0_mx",
        function=qpu_ggml_q4_0_q8_0_tiled_gemm,
        assembly_kwargs={},
        uniforms=(
            {"name": "activation_q_row_stride_bytes", "type": "uint32"},
            {"name": "activation_q_address", "type": "gpu_address"},
            {"name": "weight_q_row_stride_bytes", "type": "uint32"},
            {"name": "weight_q_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "reduction_blocks", "type": "uint32"},
            {"name": "activation_scale_row_stride_bytes", "type": "uint32"},
            {"name": "activation_scale_address", "type": "gpu_address"},
            {"name": "weight_scale_block_stride_bytes", "type": "uint32"},
            {"name": "weight_scale_address", "type": "gpu_address"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", "ceil(output_rows / 16)", 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x * workgroup_y",
        },
        description=(
            "Native GGML Q4_0 by Q8_0 arbitrary-row 16x16 tiled linear with "
            "padded rows, persistent weights, and FP32 output."
        ),
    ),
    "ggml-column-w8-q8-0-mx": ProgramSpec(
        name="ggml-column-w8-q8-0-mx",
        symbol="qpu_ggml_column_w8_q8_0_mx",
        function=qpu_ggml_q4_0_q8_0_tiled_gemm,
        assembly_kwargs={"column_weight_scale": True},
        uniforms=(
            {"name": "activation_q_row_stride_bytes", "type": "uint32"},
            {"name": "activation_q_address", "type": "gpu_address"},
            {"name": "weight_q_row_stride_bytes", "type": "uint32"},
            {"name": "weight_q_address", "type": "gpu_address"},
            {"name": "output_row_stride_bytes", "type": "uint32"},
            {"name": "output_address", "type": "gpu_address"},
            {"name": "reduction_blocks", "type": "uint32"},
            {"name": "activation_scale_row_stride_bytes", "type": "uint32"},
            {"name": "activation_scale_address", "type": "gpu_address"},
            {"name": "column_weight_scale_address", "type": "gpu_address"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["selected_output_columns / 16", "ceil(output_rows / 16)", 1],
            "wgs_per_sg": 48,
            "thread": "workgroup_x * workgroup_y",
        },
        description=(
            "Approximate per-column W8 by block-Q8_0 arbitrary-row 16x16 "
            "tiled linear with fused FP32 scaling."
        ),
    ),
    "w8a8-gemv": ProgramSpec(
        name="w8a8-gemv",
        symbol="qpu_w8a8_gemv",
        function=qpu_w8a8_gemv,
        assembly_kwargs={},
        uniforms=(
            {"name": "q_words", "type": "uint32"},
            {"name": "source_address", "type": "gpu_address"},
            {"name": "weight_row_stride_bytes", "type": "uint32"},
            {"name": "weight_address", "type": "gpu_address"},
            {"name": "destination_address", "type": "gpu_address"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["output_columns / 16", 1, 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x",
        },
        description="One-row packed signed W8A8 GEMV with INT32 output.",
    ),
    "tiled-w8a8-gemm": ProgramSpec(
        name="tiled-w8a8-gemm",
        symbol="qpu_tiled_w8a8_gemm",
        function=qpu_tiled_w8a8_gemm,
        assembly_kwargs={"dequantize": False},
        uniforms=(
            {"name": "left_row_stride_bytes", "type": "uint32"},
            {"name": "left_address", "type": "gpu_address"},
            {"name": "right_row_stride_bytes", "type": "uint32"},
            {"name": "right_address", "type": "gpu_address"},
            {"name": "destination_row_stride_bytes", "type": "uint32"},
            {"name": "destination_address", "type": "gpu_address"},
            {"name": "q_words", "type": "uint32"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["output_columns / 16", "output_rows / 16", 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x * workgroup_y",
        },
        description="16x16x16 packed signed W8A8 GEMM with INT32 output.",
    ),
    "tiled-w8a8-gemm-dequantize": ProgramSpec(
        name="tiled-w8a8-gemm-dequantize",
        symbol="qpu_tiled_w8a8_gemm_dequantize",
        function=qpu_tiled_w8a8_gemm,
        assembly_kwargs={"dequantize": True},
        uniforms=(
            {"name": "left_row_stride_bytes", "type": "uint32"},
            {"name": "left_address", "type": "gpu_address"},
            {"name": "right_row_stride_bytes", "type": "uint32"},
            {"name": "right_address", "type": "gpu_address"},
            {"name": "destination_row_stride_bytes", "type": "uint32"},
            {"name": "destination_address", "type": "gpu_address"},
            {"name": "q_words", "type": "uint32"},
            {"name": "row_scales_address", "type": "gpu_address"},
            {"name": "column_scales_address", "type": "gpu_address"},
        ),
        launch={
            "local_invocation": [16, 1, 1],
            "workgroup": ["output_columns / 16", "output_rows / 16", 1],
            "wgs_per_sg": 24,
            "thread": "workgroup_x * workgroup_y",
        },
        description="16x16x16 packed signed W8A8 GEMM with fused FP32 scaling.",
    ),
}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _source_identity(spec: ProgramSpec, source_sha256: str) -> str:
    payload = {
        "assembly_kwargs": spec.assembly_kwargs,
        "function": f"{spec.function.__module__}.{spec.function.__name__}",
        "launch": spec.launch,
        "source_sha256": source_sha256,
        "uniforms": spec.uniforms,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _header(spec: ProgramSpec, words: list[int], source_hash: str, binary_hash: str) -> str:
    guard = f"QPU_PROGRAM_{spec.symbol.upper()}_H"
    values = "\n".join(f"    UINT64_C(0x{word:016x})," for word in words)
    return f"""#ifndef {guard}
#define {guard}

#include <stddef.h>
#include <stdint.h>

static const uint64_t {spec.symbol}[] = {{
{values}
}};
static const size_t {spec.symbol}_instruction_count = {len(words)};
static const char {spec.symbol}_source_hash[] = "{source_hash}";
static const char {spec.symbol}_binary_hash[] = "{binary_hash}";

#endif
"""


def export_program(spec: ProgramSpec, output_dir: Path) -> dict[str, Any]:
    """Assemble and export one program with hashes covering source and variant metadata."""
    source_path = Path(sys.modules[spec.function.__module__].__file__ or "").resolve()
    source_sha256 = sha256_file(source_path)
    source_hash = _source_identity(spec, source_sha256)
    words = assemble(spec.function, **spec.assembly_kwargs)
    binary = b"".join(struct.pack("<Q", word) for word in words)
    binary_hash = hashlib.sha256(binary).hexdigest()
    binary_name = f"{spec.name}.bin"
    header_name = f"{spec.name}.h"
    _atomic_write(output_dir / binary_name, binary)
    _atomic_write(output_dir / header_name, _header(spec, words, source_hash, binary_hash).encode())
    return {
        "name": spec.name,
        "symbol": spec.symbol,
        "description": spec.description,
        "source": str(source_path.relative_to(ROOT)),
        "source_file_sha256": source_sha256,
        "source_hash": source_hash,
        "binary_sha256": binary_hash,
        "binary": binary_name,
        "header": header_name,
        "instruction_count": len(words),
        "size_bytes": len(binary),
        "code_alignment": spec.code_alignment,
        "data_alignment": spec.data_alignment,
        "uniform_word_count": len(spec.uniforms),
        "uniforms": list(spec.uniforms),
        "launch": spec.launch,
        "assembly_kwargs": spec.assembly_kwargs,
    }


def main() -> None:
    """Export selected programs in stable name order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--program", action="append", choices=(*PROGRAMS, "all"), default=[])
    args = parser.parse_args()
    requested = args.program or ["all"]
    names = sorted(PROGRAMS) if "all" in requested else sorted(set(requested))
    output_dir = args.output_dir.resolve()
    records = [export_program(PROGRAMS[name], output_dir) for name in names]
    manifest = {
        "schema_version": 1,
        "kind": "videocore-vii-qpu-program-export",
        "endianness": "little",
        "instruction_word_bits": 64,
        "programs": records,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(f"exported {len(records)} QPU programs to {output_dir}")


if __name__ == "__main__":
    main()
