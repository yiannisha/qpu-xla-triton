#!/usr/bin/env python3
"""Evaluate a resident GEGLU->Q8_0->Q4_0 down-projection QPU chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_xla import Device  # noqa: E402
from qpu_xla.kernels.ggml_geglu_q8 import (  # noqa: E402
    GGML_GEGLU_Q8_0_SPLIT_KERNEL,
    ggml_geglu_q8_0_reference,
    ggml_gelu_fp16_table,
)
from qpu_xla.kernels.ggml_q4_0 import (  # noqa: E402
    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
    pack_ggml_q4_0_blocks,
    pack_ggml_q4_0_tiled_weights,
)
from scripts.llama_cpp_common import collect_environment, utc_now, write_json_atomic  # noqa: E402
from scripts.replay_llama_cpp_attention_fixture import calculate_attention_error  # noqa: E402


def summarize_ns(samples: list[int]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    center = float(np.median(values))
    return {
        "count": len(samples),
        "median_ns": center,
        "p05_ns": float(np.quantile(values, 0.05)),
        "p95_ns": float(np.quantile(values, 0.95)),
        "mad_ns": float(np.median(np.abs(values - center))),
    }


def timed(callable_: Any, warmups: int, samples: int) -> list[int]:
    for _ in range(warmups):
        callable_()
    result: list[int] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        callable_()
        result.append(time.perf_counter_ns() - start)
    return result


def last_json(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("benchmark did not emit a JSON record")
    return json.loads(lines[-1])


def run_checked(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark failed: {' '.join(command)}\n{completed.stderr}")
    return last_json(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, action="append", dest="row_values")
    parser.add_argument("--columns", type=int, default=6144)
    parser.add_argument("--down-outputs", type=int, default=1536)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--seed", type=int, default=4966)
    parser.add_argument(
        "--cpu-geglu-benchmark",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_ggml_geglu_bench",
    )
    parser.add_argument(
        "--cpu-down-benchmark",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_cpu_repack_bench",
    )
    parser.add_argument(
        "--qpu-plugin",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/libggml-qpu.so",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows_values = args.row_values or [65, 129, 257]
    if (
        any(rows <= 0 for rows in rows_values)
        or args.columns <= 0
        or args.columns % 32
        or args.down_outputs <= 0
        or args.down_outputs % 16
        or args.cpu_threads <= 0
        or args.warmups < 0
        or args.samples <= 0
    ):
        parser.error("rows and counts must be positive; columns/down outputs must be aligned")
    for path in (args.cpu_geglu_benchmark, args.cpu_down_benchmark, args.qpu_plugin):
        if not path.is_file():
            parser.error(f"required pinned artifact not found: {path}")

    before = collect_environment()
    rng = np.random.default_rng(args.seed)
    blocks = args.columns // 32
    weight_native = pack_ggml_q4_0_blocks(
        rng.uniform(-0.08, 0.08, size=(args.down_outputs, blocks)).astype(np.float16),
        rng.integers(0, 16, size=(args.down_outputs, blocks, 32), dtype=np.uint8),
    )
    weight_scale_host, weight_q_host = pack_ggml_q4_0_tiled_weights(weight_native)
    table_host = ggml_gelu_fp16_table()
    data_area_size = int(
        sum(
            2 * padded_rows * args.columns * 4
            + padded_rows * blocks * 4
            + padded_rows * args.columns
            + padded_rows * args.down_outputs * 4
            for padded_rows in ((rows + 15) & ~15 for rows in rows_values)
        )
        + weight_scale_host.nbytes
        + weight_q_host.nbytes
        + table_host.nbytes
        + (8 << 20)
    )
    records: list[dict[str, Any]] = []

    with (
        tempfile.TemporaryDirectory(prefix="llama-qpu-fused-ffn-") as temporary,
        Device.open(data_area_size=data_area_size) as device,
        device.queue() as queue,
    ):
        temporary_root = Path(temporary)
        weight_path = temporary_root / "down-weight.q4_0.bin"
        weight_path.write_bytes(weight_native.tobytes(order="C"))
        table = device.tensor(table_host.shape, np.uint16)
        weight_scales = device.tensor(weight_scale_host.shape, np.uint16)
        weight_q = device.tensor(weight_q_host.shape, np.uint32)
        table.numpy()[:] = table_host
        weight_scales.numpy()[:] = weight_scale_host
        weight_q.numpy()[:] = weight_q_host

        for rows in rows_values:
            padded_rows = (rows + 15) & ~15
            gate_host = np.zeros((padded_rows, args.columns), dtype=np.float32)
            up_host = np.zeros_like(gate_host)
            gate_host[:rows] = rng.normal(0.0, 0.8, size=(rows, args.columns))
            up_host[:rows] = rng.normal(0.0, 0.8, size=(rows, args.columns))
            expected_q8 = ggml_geglu_q8_0_reference(gate_host, up_host)
            table_values = table_host.view(np.float16)
            activated = (
                table_values[gate_host[:rows].astype(np.float16).view(np.uint16)].astype(
                    np.float32
                )
                * up_host[:rows]
            )
            activation_path = temporary_root / f"geglu-{rows}.f32.bin"
            cpu_output_path = temporary_root / f"down-{rows}.f32.bin"
            activation_path.write_bytes(activated.tobytes(order="C"))

            cpu_geglu_command = [
                str(args.cpu_geglu_benchmark.resolve()),
                "--plugin",
                str(args.qpu_plugin.resolve()),
                "--rows",
                str(rows),
                "--columns",
                str(args.columns),
                "--cpu-threads",
                str(args.cpu_threads),
                "--warmups",
                str(args.warmups),
                "--samples",
                str(args.samples),
            ]
            cpu_geglu = run_checked(cpu_geglu_command)
            cpu_down_command = [
                str(args.cpu_down_benchmark.resolve()),
                "--weights",
                str(weight_path),
                "--activation-f32",
                str(activation_path),
                "--output-bin",
                str(cpu_output_path),
                "--weight-type",
                "q4_0",
                "--input-columns",
                str(args.columns),
                "--output-columns",
                str(args.down_outputs),
                "--rows",
                str(rows),
                "--cpu-threads",
                str(args.cpu_threads),
                "--warmups",
                str(args.warmups),
                "--samples",
                str(args.samples),
            ]
            cpu_down = run_checked(cpu_down_command)
            cpu_down_output = np.fromfile(cpu_output_path, dtype="<f4").reshape(
                rows, args.down_outputs
            )

            gate = device.tensor(gate_host.shape, np.float32)
            up = device.tensor(up_host.shape, np.float32)
            activation_scales = device.tensor((padded_rows, blocks), np.uint32)
            activation_q = device.tensor(gate_host.shape, np.uint8)
            destination = device.tensor((padded_rows, args.down_outputs), np.float32)
            gate.numpy()[:] = gate_host
            up.numpy()[:] = up_host

            def fused_producer() -> None:
                queue.submit(
                    GGML_GEGLU_Q8_0_SPLIT_KERNEL,
                    (gate, up, activation_scales, activation_q, table),
                    grid=(blocks, padded_rows, 1),
                ).wait()

            def fused_chain() -> None:
                fused_producer()
                queue.submit(
                    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
                    (activation_q, activation_scales, weight_q, weight_scales, destination),
                    grid=(args.down_outputs // 16, padded_rows // 16, 1),
                ).wait()

            fused_producer()
            actual_scales = np.array(activation_scales.numpy(), copy=True)
            actual_q = np.array(activation_q.numpy(), copy=True)
            scale_correct = np.array_equal(
                actual_scales.astype(np.uint16),
                expected_q8[..., :2].copy().view(np.uint16).reshape(padded_rows, blocks),
            )
            q_correct = np.array_equal(
                actual_q.reshape(padded_rows, blocks, 32), expected_q8[..., 2:]
            )
            producer_samples = timed(fused_producer, args.warmups, args.samples)
            chain_samples = timed(fused_chain, args.warmups, args.samples)
            qpu_down_output = np.array(destination.numpy(), copy=True)[:rows]
            correctness = calculate_attention_error(
                cpu_down_output, qpu_down_output, atol=2.0e-4, rtol=2.0e-5
            )
            cpu_geglu_samples = [int(value) for value in cpu_geglu["cpu_complete_ns"]]
            cpu_quantize_samples = [
                int(value) for value in cpu_geglu["cpu_q8_0_quantize_ns"]
            ]
            cpu_geglu_q8_samples = [
                int(value) for value in cpu_geglu["cpu_geglu_q8_0_ns"]
            ]
            cpu_down_samples = [int(value) for value in cpu_down["complete_ns"]]
            cpu_region_samples = [
                geglu + down
                for geglu, down in zip(cpu_geglu_samples, cpu_down_samples, strict=True)
            ]
            records.append(
                {
                    "rows": rows,
                    "padded_rows": padded_rows,
                    "intermediate_columns": args.columns,
                    "down_outputs": args.down_outputs,
                    "cpu_threads": args.cpu_threads,
                    "cpu_geglu_ns": cpu_geglu_samples,
                    "cpu_q8_0_quantize_ns": cpu_quantize_samples,
                    "cpu_geglu_q8_0_ns": cpu_geglu_q8_samples,
                    "cpu_down_complete_ns": cpu_down_samples,
                    "cpu_ffn_region_ns": cpu_region_samples,
                    "qpu_fused_geglu_q8_0_ns": producer_samples,
                    "qpu_fused_geglu_q8_0_down_ns": chain_samples,
                    "cpu_ffn_region_summary": summarize_ns(cpu_region_samples),
                    "qpu_fused_producer_summary": summarize_ns(producer_samples),
                    "qpu_fused_chain_summary": summarize_ns(chain_samples),
                    "producer_speedup_vs_cpu_geglu_q8_0": float(
                        np.median(cpu_geglu_q8_samples) / np.median(producer_samples)
                    ),
                    "region_speedup_vs_cpu": float(
                        np.median(cpu_region_samples) / np.median(chain_samples)
                    ),
                    "q8_scale_bytes_exact": scale_correct,
                    "q8_value_bytes_exact": q_correct,
                    "down_correctness": correctness,
                    "cpu_geglu_command": cpu_geglu_command,
                    "cpu_down_command": cpu_down_command,
                }
            )

    after = collect_environment()
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-fused-ffn-diagnostic",
        "created_utc": utc_now(),
        "seed": args.seed,
        "warmups": args.warmups,
        "samples": args.samples,
        "records": records,
        "environment_before": before,
        "environment_after": after,
        "measurement_contract": {
            "timing_eligible_for_promotion": False,
            "reasons": [
                "operator tensors are deterministic synthetic data at exact Gemma dimensions",
                "the fused chain is not yet selected by the llama.cpp graph planner",
                "retained end-to-end sessions remain the promotion gate",
            ],
            "qpu_boundary": (
                "persistent mapped gate/up and down weights; one GEGLU-Q8_0 dispatch "
                "followed by one tiled Q4_0 down dispatch; no intermediate host access"
            ),
        },
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(records)} fused FFN shapes")


if __name__ == "__main__":
    main()
