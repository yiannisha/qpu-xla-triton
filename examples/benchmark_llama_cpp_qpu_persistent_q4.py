#!/usr/bin/env python3
"""Compare conventional and persistent exact tiled Q4_0 execution at M=513."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_xla.kernels.ggml_q4_0 import (  # noqa: E402
    pack_ggml_q4_0_tiled_weights,
    qpu_ggml_q4_0_q8_0_tiled_gemm,
)
from qpu_xla.kernels.ggml_q4_0_persistent import (  # noqa: E402
    qpu_ggml_q4_0_q8_0_tiled_gemm_persistent,
)
from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from videocore7.driver import Driver  # noqa: E402

Q4_BLOCK_ELEMENTS = 32
Q4_BLOCK_BYTES = 18
ROW_TILE = 16
OUTPUT_TILE = 16
PERSISTENT_THREADS = 24


def quantize_q8_0_split(values: np.ndarray[Any, np.dtype[np.float32]]) -> tuple[np.ndarray, np.ndarray]:
    """Quantize F32 rows into low-F16 scale words and dense signed bytes."""
    rows, columns = values.shape
    grouped = values.reshape(rows, columns // Q4_BLOCK_ELEMENTS, Q4_BLOCK_ELEMENTS)
    scales = np.max(np.abs(grouped), axis=2).astype(np.float32) / 127.0
    inverse = np.zeros_like(scales)
    np.divide(1.0, scales, out=inverse, where=scales != 0.0)
    quantized = np.clip(np.rint(grouped * inverse[..., None]), -127, 127).astype(np.int8)
    return (
        np.ascontiguousarray(scales.astype(np.float16).view(np.uint16)),
        np.ascontiguousarray(quantized.reshape(rows, columns).view(np.uint8)),
    )


def summary(samples: list[int]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    center = float(np.median(values))
    return {
        "samples_ns": samples,
        "median_ns": center,
        "minimum_ns": int(np.min(values)),
        "maximum_ns": int(np.max(values)),
        "mad_ns": float(np.median(np.abs(values - center))),
    }


def build_streams(
    drv: Driver,
    *,
    row_tiles: int,
    column_tiles: int,
    activation_q: Any,
    weight_q: Any,
    destinations: tuple[Any, ...],
    reduction_blocks: int,
    activation_scales: Any,
    weight_scales: Any,
) -> tuple[Any, Any]:
    """Build balanced private uniform streams for 24 persistent threads."""
    tasks = [
        ((tile_i & 0xFFFF) << 16) | (tile_j & 0xFFFF) for tile_i in range(row_tiles) for tile_j in range(column_tiles)
    ]
    by_thread: list[list[int]] = [[] for _ in range(PERSISTENT_THREADS)]
    for index, task in enumerate(tasks):
        by_thread[index % PERSISTENT_THREADS].append(task)
    max_tasks = max(map(len, by_thread))
    phase_words = 1 + max_tasks * 12
    streams = drv.alloc((PERSISTENT_THREADS, 1 + len(destinations) * phase_words), dtype=np.uint32)
    uniforms = drv.alloc(2, dtype=np.uint32)
    streams[:] = 0
    for thread, thread_tasks in enumerate(by_thread):
        streams[thread, 0] = len(destinations)
        for phase, destination in enumerate(destinations):
            phase_start = 1 + phase * phase_words
            streams[thread, phase_start] = len(thread_tasks)
            fixed = (
                activation_q.strides[0],
                activation_q.addresses()[0, 0],
                weight_q.strides[0],
                weight_q.addresses()[0, 0],
                destination.strides[0],
                destination.addresses()[0, 0],
                reduction_blocks,
                activation_scales.strides[0],
                activation_scales.addresses()[0, 0],
                weight_scales.strides[0],
                weight_scales.addresses()[0, 0],
            )
            for slot, task in enumerate(thread_tasks):
                start = phase_start + 1 + slot * 12
                streams[thread, start : start + 12] = (task, *fixed)
    uniforms[:] = (streams.addresses()[0, 0], streams.strides[0])
    return streams, uniforms


def run_case(
    *,
    fixture_root: Path,
    weight_name: str,
    activation_name: str,
    output_columns: int,
    qpu_columns: int,
    rows: int,
    baseline_wgs: int,
    phases: int,
    warmups: int,
    samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    input_columns = 1536
    reduction_blocks = input_columns // Q4_BLOCK_ELEMENTS
    weight_path = fixture_root / weight_name
    activation_path = fixture_root / activation_name
    native_weights = np.fromfile(weight_path, dtype=np.uint8).reshape(output_columns, reduction_blocks, Q4_BLOCK_BYTES)
    selected_native_weights = np.ascontiguousarray(native_weights[-qpu_columns:])
    weight_scales_host, weight_q_host = pack_ggml_q4_0_tiled_weights(selected_native_weights)
    activation = np.fromfile(activation_path, dtype="<f4").reshape(rows, input_columns)
    padded_rows = (rows + ROW_TILE - 1) & ~(ROW_TILE - 1)
    padded_activation = np.zeros((padded_rows, input_columns), dtype=np.float32)
    padded_activation[:rows] = activation
    activation_scales_host, activation_q_host = quantize_q8_0_split(padded_activation)
    output_shape = (padded_rows, qpu_columns)
    max_tasks = (padded_rows // ROW_TILE) * (qpu_columns // OUTPUT_TILE)
    stream_bytes = PERSISTENT_THREADS * (1 + phases * (1 + ((max_tasks + 23) // 24) * 12)) * 4
    data_area_size = int(
        activation_scales_host.nbytes
        + activation_q_host.nbytes
        + weight_scales_host.nbytes
        + weight_q_host.nbytes
        + 2 * phases * np.prod(output_shape) * 4
        + stream_bytes
        + (4 << 20)
    )

    with Driver(data_area_size=data_area_size) as drv:
        conventional_code = drv.program(qpu_ggml_q4_0_q8_0_tiled_gemm)
        persistent_code = drv.program(qpu_ggml_q4_0_q8_0_tiled_gemm_persistent)
        activation_scales = drv.alloc(activation_scales_host.shape, dtype=np.uint16)
        activation_q = drv.alloc(activation_q_host.shape, dtype=np.uint8)
        weight_scales = drv.alloc(weight_scales_host.shape, dtype=np.uint16)
        weight_q = drv.alloc(weight_q_host.shape, dtype=np.uint32)
        conventional_outputs = tuple(drv.alloc(output_shape, dtype=np.float32) for _ in range(phases))
        persistent_outputs = tuple(drv.alloc(output_shape, dtype=np.float32) for _ in range(phases))
        conventional_uniforms = drv.alloc((phases, 11), dtype=np.uint32)

        activation_scales[:] = activation_scales_host
        activation_q[:] = activation_q_host
        weight_scales[:] = weight_scales_host
        weight_q[:] = weight_q_host
        for phase, conventional_output in enumerate(conventional_outputs):
            conventional_output[:] = np.nan
            conventional_uniforms[phase] = (
                activation_q.strides[0],
                activation_q.addresses()[0, 0],
                weight_q.strides[0],
                weight_q.addresses()[0, 0],
                conventional_output.strides[0],
                conventional_output.addresses()[0, 0],
                reduction_blocks,
                activation_scales.strides[0],
                activation_scales.addresses()[0, 0],
                weight_scales.strides[0],
                weight_scales.addresses()[0, 0],
            )
        for persistent_output in persistent_outputs:
            persistent_output[:] = np.nan
        persistent_streams, persistent_uniforms = build_streams(
            drv,
            row_tiles=padded_rows // ROW_TILE,
            column_tiles=qpu_columns // OUTPUT_TILE,
            activation_q=activation_q,
            weight_q=weight_q,
            destinations=persistent_outputs,
            reduction_blocks=reduction_blocks,
            activation_scales=activation_scales,
            weight_scales=weight_scales,
        )
        # Keep the private stream allocation live for every timed dispatch.
        assert persistent_streams.shape[0] == PERSISTENT_THREADS
        conventional_grid = (
            qpu_columns // OUTPUT_TILE,
            padded_rows // ROW_TILE,
            1,
        )
        conventional_thread_count = conventional_grid[0] * conventional_grid[1]

        def execute_conventional() -> None:
            for phase in range(phases):
                drv.execute(
                    conventional_code,
                    local_invocation=(16, 1, 1),
                    uniforms=conventional_uniforms.addresses()[phase, 0],
                    workgroup=conventional_grid,
                    wgs_per_sg=baseline_wgs,
                    thread=conventional_thread_count,
                )

        def execute_persistent() -> None:
            drv.execute(
                persistent_code,
                local_invocation=(16, 1, 1),
                uniforms=persistent_uniforms.addresses()[0],
                wgs_per_sg=PERSISTENT_THREADS,
                thread=PERSISTENT_THREADS,
            )

        execute_conventional()
        execute_persistent()
        conventional_host = tuple(np.array(output, copy=True) for output in conventional_outputs)
        persistent_host = tuple(np.array(output, copy=True) for output in persistent_outputs)
        differences = [
            np.abs(conventional - persistent)
            for conventional, persistent in zip(conventional_host, persistent_host, strict=True)
        ]
        bitwise_equal = all(
            np.array_equal(conventional.view(np.uint32), persistent.view(np.uint32))
            for conventional, persistent in zip(conventional_host, persistent_host, strict=True)
        )
        if not bitwise_equal:
            raise RuntimeError(
                "persistent output differs: max_absolute="
                f"{max(float(np.max(difference)) for difference in differences)}"
            )

        for _ in range(warmups):
            execute_conventional()
            execute_persistent()
        timings = {"conventional": [], "persistent": []}
        for _ in range(samples):
            order = ["conventional", "persistent"]
            rng.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                (execute_conventional if name == "conventional" else execute_persistent)()
                timings[name].append(time.perf_counter_ns() - start)

        conventional_summary = summary(timings["conventional"])
        persistent_summary = summary(timings["persistent"])
        return {
            "input_columns": input_columns,
            "output_columns": output_columns,
            "qpu_columns": qpu_columns,
            "rows": rows,
            "padded_rows": padded_rows,
            "reduction_blocks": reduction_blocks,
            "tile_count": conventional_thread_count,
            "persistent_thread_count": PERSISTENT_THREADS,
            "phase_count": phases,
            "tasks_per_persistent_thread": {
                "minimum": conventional_thread_count // PERSISTENT_THREADS,
                "maximum": (conventional_thread_count + PERSISTENT_THREADS - 1) // PERSISTENT_THREADS,
            },
            "baseline_wgs_per_sg": baseline_wgs,
            "conventional": conventional_summary,
            "persistent": persistent_summary,
            "persistent_speedup": (conventional_summary["median_ns"] / persistent_summary["median_ns"]),
            "correctness": {
                "bitwise_equal": bitwise_equal,
                "max_absolute": max(float(np.max(difference)) for difference in differences),
                "nan_count": sum(int(np.isnan(value).sum()) for value in persistent_host),
                "inf_count": sum(int(np.isinf(value).sum()) for value in persistent_host),
            },
            "fixtures": {
                "weight": {"path": str(weight_path), "sha256": sha256_file(weight_path)},
                "activation": {
                    "path": str(activation_path),
                    "sha256": sha256_file(activation_path),
                },
            },
            "programs": {
                "conventional_source": {
                    "path": str(ROOT / "src/qpu_xla/kernels/ggml_q4_0.py"),
                    "sha256": sha256_file(ROOT / "src/qpu_xla/kernels/ggml_q4_0.py"),
                },
                "persistent_source": {
                    "path": str(ROOT / "src/qpu_xla/kernels/ggml_q4_0_persistent.py"),
                    "sha256": sha256_file(ROOT / "src/qpu_xla/kernels/ggml_q4_0_persistent.py"),
                },
                "conventional_instruction_count": int(conventional_code.size),
                "persistent_instruction_count": int(persistent_code.size),
                "conventional_binary_sha256": hashlib.sha256(np.asarray(conventional_code).tobytes()).hexdigest(),
                "persistent_binary_sha256": hashlib.sha256(np.asarray(persistent_code).tobytes()).hexdigest(),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=ROOT / "experiment_logs/20260825-qpu-next/fixtures",
    )
    parser.add_argument("--rows", type=int, default=513)
    parser.add_argument("--baseline-wgs", type=int, default=192)
    parser.add_argument("--phases", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows <= 0 or args.phases <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("rows and samples must be positive; warmups must be nonnegative")
    cases = (
        ("blk-0-ffn_up-weight.q4_0.bin", f"blk-0-ffn_up-weight.m{args.rows}.f32.bin", 6144, 1152),
        ("blk-15-ffn_up-weight.q4_0.bin", f"blk-15-ffn_up-weight.m{args.rows}.f32.bin", 12288, 2304),
    )
    for weight, activation, _, _ in cases:
        for path in (args.fixture_root / weight, args.fixture_root / activation):
            if not path.is_file():
                parser.error(f"fixture not found: {path}")

    rng = random.Random(args.seed)
    payload = {
        "schema_version": 1,
        "kind": "llama-qpu-persistent-q4-0-screen",
        "created_utc": utc_now(),
        "design": {
            "comparison": (
                f"{args.phases} exact tiled projection phases: conventional one CSD "
                "per phase vs one 24-thread persistent CSD"
            ),
            "timed_boundary": "Driver.execute submission and completion wait; buffers already resident",
            "warmups": args.warmups,
            "samples": args.samples,
            "seed": args.seed,
        },
        "benchmark": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment_before": collect_environment(),
        "results": [
            run_case(
                fixture_root=args.fixture_root,
                weight_name=weight,
                activation_name=activation,
                output_columns=output_columns,
                qpu_columns=qpu_columns,
                rows=args.rows,
                baseline_wgs=args.baseline_wgs,
                phases=args.phases,
                warmups=args.warmups,
                samples=args.samples,
                rng=rng,
            )
            for weight, activation, output_columns, qpu_columns in cases
        ],
        "environment_after": collect_environment(),
    }
    write_json_atomic(args.output, payload)
    print(
        "persistent speedups: "
        + ", ".join(f"N={item['output_columns']} {item['persistent_speedup']:.3f}x" for item in payload["results"])
    )


if __name__ == "__main__":
    main()
