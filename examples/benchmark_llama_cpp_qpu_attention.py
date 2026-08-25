#!/usr/bin/env python3
"""Diagnose fused Gemma decode and batched-prefill attention at required KV lengths."""

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
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_xla import Device  # noqa: E402
from qpu_xla.kernels.ggml_flash_attn import (  # noqa: E402
    GEMMA_HEAD_DIM,
    GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL,
    GGML_GEMMA_FLASH_ATTN_F16_MX_KERNEL,
    ggml_flash_attn_ext_reference,
)
from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    utc_now,
    write_json_atomic,
)
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


def _time(callable_: Any, warmups: int, samples: int) -> list[int]:
    for _ in range(warmups):
        callable_()
    result: list[int] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        callable_()
        result.append(time.perf_counter_ns() - started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, action="append", dest="contexts")
    parser.add_argument("--query-rows", type=int, action="append", dest="query_rows")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=4962)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument(
        "--cpu-benchmark",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_cpu_flash_attn_bench",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts = args.contexts or [256, 512, 2048, 4096]
    query_rows_values = args.query_rows or [1, 17, 65, 129, 257]
    if (
        not contexts
        or any(context <= 0 or context % 2 for context in contexts)
        or not query_rows_values
        or any(rows <= 0 for rows in query_rows_values)
        or args.heads <= 0
        or args.heads > 12
        or args.cpu_threads <= 0
        or args.warmups < 0
        or args.samples <= 0
    ):
        parser.error("contexts must be positive/even; heads 1..12; warmups/samples valid")
    if any(rows > 1 for rows in query_rows_values) and args.heads != 8:
        parser.error("batched-query attention currently requires exactly eight query heads")
    if not args.cpu_benchmark.is_file():
        parser.error(f"pinned GGML CPU benchmark not found: {args.cpu_benchmark}")

    before = collect_environment()
    rng = np.random.default_rng(args.seed)
    scale = float(GEMMA_HEAD_DIM**-0.5)
    records: list[dict[str, Any]] = []
    largest = max(contexts)
    largest_query = max(query_rows_values)
    data_bytes = (
        largest_query * args.heads * GEMMA_HEAD_DIM * 8
        + largest * GEMMA_HEAD_DIM * 4
        + largest_query * largest * 2
        + (4 << 20)
    )
    with (
        Device.open(data_area_size=data_bytes) as device,
        device.queue() as queue,
        tempfile.TemporaryDirectory(prefix="llama-qpu-attention-") as temporary,
    ):
        temporary_root = Path(temporary)
        cases = (
            (context, query_rows)
            for context in contexts
            for query_rows in query_rows_values
        )
        for context, query_rows in cases:
            query_host = rng.normal(
                0.0, 0.15, size=(query_rows, args.heads, GEMMA_HEAD_DIM)
            ).astype(np.float32)
            key_host = rng.normal(
                0.0, 0.15, size=(context, GEMMA_HEAD_DIM)
            ).astype(np.float16)
            value_host = rng.normal(
                0.0, 0.2, size=(context, GEMMA_HEAD_DIM)
            ).astype(np.float16)
            mask_host = np.zeros((query_rows, context), dtype=np.float16)
            qpu_query_host = query_host[0] if query_rows == 1 else query_host
            qpu_mask_host = mask_host[0] if query_rows == 1 else mask_host
            query = device.tensor(qpu_query_host.shape, np.float32)
            key = device.tensor(key_host.shape, np.float16)
            value = device.tensor(value_host.shape, np.float16)
            mask = device.tensor(qpu_mask_host.shape, np.float16)
            destination = device.tensor(qpu_query_host.shape, np.float32)
            query.numpy()[:] = qpu_query_host
            key.numpy()[:] = key_host
            value.numpy()[:] = value_host
            mask.numpy()[:] = qpu_mask_host

            def cpu_reference() -> npt.NDArray[np.float32]:
                return ggml_flash_attn_ext_reference(
                    query_host.transpose(2, 0, 1)[:, :, :, None],
                    key_host.T[:, :, None, None],
                    value_host.T[:, :, None, None],
                    mask_host.T[:, :, None, None],
                    scale=scale,
                    max_bias=0.0,
                    logit_softcap=0.0,
                )[:, :, :, 0].transpose(2, 1, 0)

            expected = cpu_reference()

            input_paths = {
                "query": temporary_root / f"query-{context}-{query_rows}.f32.bin",
                "key": temporary_root / f"key-{context}-{query_rows}.f16.bin",
                "value": temporary_root / f"value-{context}-{query_rows}.f16.bin",
                "mask": temporary_root / f"mask-{context}-{query_rows}.f16.bin",
                "output": temporary_root / f"cpu-output-{context}-{query_rows}.f32.bin",
            }
            for array, name in (
                (query_host.transpose(1, 0, 2), "query"),
                (key_host, "key"),
                (value_host, "value"),
                (mask_host, "mask"),
            ):
                input_paths[name].write_bytes(array.tobytes(order="C"))
            cpu_command = [
                str(args.cpu_benchmark.resolve()),
                "--query-f32",
                str(input_paths["query"]),
                "--key-f16",
                str(input_paths["key"]),
                "--value-f16",
                str(input_paths["value"]),
                "--mask-f16",
                str(input_paths["mask"]),
                "--output-bin",
                str(input_paths["output"]),
                "--heads",
                str(args.heads),
                "--query-rows",
                str(query_rows),
                "--kv-rows",
                str(context),
                "--head-dim",
                str(GEMMA_HEAD_DIM),
                "--cpu-threads",
                str(args.cpu_threads),
                "--warmups",
                str(args.warmups),
                "--samples",
                str(args.samples),
                "--scale",
                repr(scale),
                "--max-bias",
                "0",
                "--logit-softcap",
                "0",
            ]
            completed = subprocess.run(
                cpu_command, text=True, capture_output=True, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"pinned GGML CPU attention failed for context {context}: {completed.stderr}"
                )
            cpu_result = json.loads(
                [line for line in completed.stdout.splitlines() if line.strip()][-1]
            )
            cpu_output = np.fromfile(input_paths["output"], dtype="<f4").reshape(
                query_rows, args.heads, GEMMA_HEAD_DIM
            )

            def qpu_execute() -> None:
                kernel = (
                    GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL
                    if query_rows == 1
                    else GGML_GEMMA_FLASH_ATTN_F16_MX_KERNEL
                )
                grid = (args.heads * query_rows, 1, 1)
                queue.submit(
                    kernel,
                    (query, key, value, mask, destination, scale),
                    grid=grid,
                ).wait()

            qpu_samples = _time(qpu_execute, args.warmups, args.samples)
            actual = np.array(destination.numpy(), copy=True).reshape(
                query_rows, args.heads, GEMMA_HEAD_DIM
            )
            cpu_samples = [int(value) for value in cpu_result["complete_ns"]]
            correctness = calculate_attention_error(
                cpu_output, actual, atol=0.005, rtol=0.005
            )
            oracle_correctness = calculate_attention_error(
                cpu_output, expected, atol=0.005, rtol=0.005
            )
            records.append(
                {
                    "context_rows": context,
                    "query_rows": query_rows,
                    "query_heads": args.heads,
                    "kv_heads": 1,
                    "head_dim": GEMMA_HEAD_DIM,
                    "scale": scale,
                    "native_input_bytes": int(
                        query_host.nbytes
                        + key_host.nbytes
                        + value_host.nbytes
                        + mask_host.nbytes
                    ),
                    "output_bytes": int(actual.nbytes),
                    "dispatch_count": 1,
                    "cpu_command": cpu_command,
                    "cpu_exact_ggml_node_ns": cpu_samples,
                    "qpu_persistent_execute_ns": qpu_samples,
                    "cpu_exact_ggml_node_summary": summarize_ns(cpu_samples),
                    "qpu_persistent_execute_summary": summarize_ns(qpu_samples),
                    "diagnostic_speedup_vs_exact_ggml_cpu": float(
                        np.median(cpu_samples) / np.median(qpu_samples)
                    ),
                    "exact_ggml_cpu_correctness": correctness,
                    "independent_oracle_correctness": oracle_correctness,
                }
            )
    after = collect_environment()
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-gemma-flash-attention-diagnostic",
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
                (
                    "the exact CPU comparator is a standalone pinned GGML node over "
                    "deterministic synthetic tensors, not a captured model node"
                ),
                (
                    "the candidate supports only 256-wide, one-KV-head, "
                    "no-ALiBi/no-softcap/no-sinks records; batched mode requires eight query heads"
                ),
                "environment retention must be established by an isolated multi-session runner",
            ],
            "qpu_timing": "persistent mapped inputs, one fused dispatch for all query rows and heads, no host copies",
        },
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(records)} contexts")


if __name__ == "__main__":
    main()
