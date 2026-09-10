#!/usr/bin/env python3
"""Benchmark a concurrent CPU/QPU intermediate-channel FFN partition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_xla import Device  # noqa: E402
from qpu_xla.kernels.ggml_geglu_q8 import (  # noqa: E402
    GGML_GEGLU_Q8_0_SPLIT_KERNEL,
    ggml_gelu_fp16_table,
)
from qpu_xla.kernels.ggml_q4_0 import (  # noqa: E402
    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
    pack_ggml_q4_0_tiled_weights,
)
from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from scripts.replay_llama_cpp_attention_fixture import calculate_attention_error  # noqa: E402

Q4_0_BLOCK_ELEMENTS = 32
Q4_0_BLOCK_BYTES = 18
OUTPUT_TILE = 16
ROW_TILE = 16
CORRECTNESS_ATOL = 7.0e-3
CORRECTNESS_RTOL = 5.0e-4
PROMOTION_SPEEDUP = 1.10


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


def quantize_q8_0_split(values: np.ndarray[Any, np.dtype[np.float32]]) -> tuple[np.ndarray, np.ndarray]:
    """Quantize row-major FP32 into scale words and contiguous signed Q8 bytes."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % Q4_0_BLOCK_ELEMENTS:
        raise ValueError("Q8_0 input must be rank two with 32-aligned columns")
    rows, columns = values.shape
    blocks = columns // Q4_0_BLOCK_ELEMENTS
    grouped = values.reshape(rows, blocks, Q4_0_BLOCK_ELEMENTS)
    scales = np.max(np.abs(grouped), axis=2).astype(np.float32) / 127.0
    inverse = np.zeros_like(scales)
    np.divide(1.0, scales, out=inverse, where=scales != 0.0)
    quantized = np.rint(grouped * inverse[..., None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    scale_words = np.zeros((rows, blocks), dtype=np.uint32)
    scale_words.view(np.uint16).reshape(rows, blocks, 2)[..., 0] = (
        scales.astype(np.float16).view(np.uint16)
    )
    return scale_words, np.ascontiguousarray(quantized.reshape(rows, columns).view(np.uint8))


def q4_blocks(payload: bytes, input_columns: int, output_columns: int) -> np.ndarray:
    expected = output_columns * (input_columns // Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_BYTES
    if len(payload) != expected:
        raise ValueError(f"Q4_0 payload has {len(payload)} bytes, expected {expected}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(
        output_columns, input_columns // Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES
    ).copy()


def find_tensor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [tensor for tensor in manifest["tensors"] if tensor["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor named {name!r}, found {len(matches)}")
    tensor = matches[0]
    if tensor["ggml_type"] != "Q4_0" or len(tensor["shape"]) != 2:
        raise ValueError(f"{name!r} is not a rank-two Q4_0 tensor")
    return tensor


def read_tensor(model: Path, tensor: dict[str, Any]) -> bytes:
    with model.open("rb") as source:
        source.seek(int(tensor["file_offset"]))
        payload = source.read(int(tensor["bytes"]))
    if len(payload) != int(tensor["bytes"]):
        raise ValueError(f"short model read for {tensor['name']}")
    return payload


def write_bytes(path: Path, values: bytes | np.ndarray) -> None:
    data = values if isinstance(values, bytes) else values.tobytes(order="C")
    path.write_bytes(data)


def last_json(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("CPU FFN benchmark produced no JSON")
    return json.loads(lines[-1])


def cpu_command(
    binary: Path,
    gate: Path,
    up: Path,
    down: Path,
    activation: Path,
    output: Path,
    *,
    input_columns: int,
    intermediate_columns: int,
    output_columns: int,
    rows: int,
    cpu_backend: str,
    cpu_threads: int,
    warmups: int,
    samples: int,
    serve: bool = False,
) -> list[str]:
    command = [
        str(binary.resolve()),
        "--gate",
        str(gate.resolve()),
        "--up",
        str(up.resolve()),
        "--down",
        str(down.resolve()),
        "--activation-f32",
        str(activation.resolve()),
        "--input-columns",
        str(input_columns),
        "--intermediate-columns",
        str(intermediate_columns),
        "--output-columns",
        str(output_columns),
        "--rows",
        str(rows),
        "--backend",
        cpu_backend,
        "--cpu-threads",
        str(cpu_threads),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--output-bin",
        str(output.resolve()),
    ]
    if serve:
        command.append("--serve")
    return command


def run_cpu(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"CPU FFN benchmark failed: {' '.join(command)}\n{completed.stderr}")
    return last_json(completed.stdout)


def server_request(process: subprocess.Popen[str], command: str) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("CPU FFN server has no pipes")
    process.stdin.write(command + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"CPU FFN server exited unexpectedly: {stderr}")
    record = json.loads(line)
    if "error" in record:
        raise RuntimeError(f"CPU FFN server error: {record['error']}")
    return record


def start_cpu_server(command: list[str]) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        command,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    ready = server_line(process.stdout)
    if not ready.get("ready"):
        process.terminate()
        raise RuntimeError("CPU FFN server did not become ready")
    return process


def server_line(stdout: TextIO | None) -> dict[str, Any]:
    if stdout is None:
        raise RuntimeError("CPU FFN server has no stdout")
    line = stdout.readline()
    if not line:
        raise RuntimeError("CPU FFN server exited before becoming ready")
    return json.loads(line)


def stop_cpu_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            server_request(process, "quit")
        except (BrokenPipeError, RuntimeError):
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiment_logs/20260819-llama-cpp-qpu/gemma-base-gguf-manifest.json",
    )
    parser.add_argument("--layer", type=int, action="append", dest="layers")
    parser.add_argument("--rows", type=int, action="append", dest="row_values")
    parser.add_argument("--fraction", type=float, action="append", dest="fractions")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--cpu-backend",
        choices=("cpu-repack", "openblas"),
        default="cpu-repack",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument(
        "--cpu-benchmark",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_cpu_ffn_bench",
    )
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = args.layers or [0, 15]
    rows_values = args.row_values or [257]
    fractions = args.fractions or [0.125, 0.1875, 0.25]
    if (
        any(layer < 0 for layer in layers)
        or any(rows <= 0 for rows in rows_values)
        or any(not 0.0 < fraction < 1.0 for fraction in fractions)
        or args.cpu_threads <= 0
        or args.warmups < 0
        or args.samples <= 0
        or not args.manifest.is_file()
        or not args.cpu_benchmark.is_file()
    ):
        parser.error("invalid layer, rows, fraction, count, manifest, or CPU benchmark")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = Path(manifest["model"]["path"])
    if not model.is_file():
        parser.error(f"model not found: {model}")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.fixture_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="llama-qpu-ffn-island-")
        fixture_root = Path(temporary.name)
    else:
        fixture_root = args.fixture_dir.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)

    before = collect_environment()
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    try:
        for layer in layers:
            tensors = {
                role: find_tensor(manifest, f"blk.{layer}.ffn_{role}.weight")
                for role in ("gate", "up", "down")
            }
            input_columns, intermediate_columns = map(int, tensors["gate"]["shape"])
            if list(map(int, tensors["up"]["shape"])) != [input_columns, intermediate_columns]:
                raise ValueError("gate and up shapes differ")
            down_input, output_columns = map(int, tensors["down"]["shape"])
            if down_input != intermediate_columns:
                raise ValueError("down reduction extent does not match gate/up output")
            payloads = {role: read_tensor(model, tensor) for role, tensor in tensors.items()}
            native_gate = q4_blocks(payloads["gate"], input_columns, intermediate_columns)
            native_up = q4_blocks(payloads["up"], input_columns, intermediate_columns)
            native_down = q4_blocks(payloads["down"], intermediate_columns, output_columns)
            layer_root = fixture_root / f"layer-{layer}"
            layer_root.mkdir(parents=True, exist_ok=True)
            full_paths = {role: layer_root / f"full-{role}.q4_0.bin" for role in payloads}
            for role, path in full_paths.items():
                write_bytes(path, payloads[role])

            for rows in rows_values:
                activation = rng.normal(0.0, 0.75, size=(rows, input_columns)).astype("<f4")
                activation_path = layer_root / f"input-m{rows}.f32.bin"
                full_output_path = layer_root / f"full-output-m{rows}.f32.bin"
                write_bytes(activation_path, activation)
                full_command = cpu_command(
                    args.cpu_benchmark,
                    full_paths["gate"],
                    full_paths["up"],
                    full_paths["down"],
                    activation_path,
                    full_output_path,
                    input_columns=input_columns,
                    intermediate_columns=intermediate_columns,
                    output_columns=output_columns,
                    rows=rows,
                    cpu_backend=args.cpu_backend,
                    cpu_threads=args.cpu_threads,
                    warmups=args.warmups,
                    samples=args.samples,
                )
                full_cpu = run_cpu(full_command)
                full_output = np.fromfile(full_output_path, dtype="<f4").reshape(
                    rows, output_columns
                )
                padded_rows = (rows + ROW_TILE - 1) & ~(ROW_TILE - 1)
                padded_activation = np.zeros((padded_rows, input_columns), dtype=np.float32)
                padded_activation[:rows] = activation
                input_scales_host, input_q_host = quantize_q8_0_split(padded_activation)

                for fraction in fractions:
                    qpu_columns = int(round(intermediate_columns * fraction / Q4_0_BLOCK_ELEMENTS))
                    qpu_columns = max(1, min(qpu_columns, intermediate_columns // Q4_0_BLOCK_ELEMENTS - 1))
                    qpu_columns *= Q4_0_BLOCK_ELEMENTS
                    cpu_columns = intermediate_columns - qpu_columns
                    actual_fraction = qpu_columns / intermediate_columns
                    cpu_blocks = cpu_columns // Q4_0_BLOCK_ELEMENTS
                    split_name = f"m{rows}-f{qpu_columns}-of-{intermediate_columns}"
                    split_root = layer_root / split_name
                    split_root.mkdir(parents=True, exist_ok=True)
                    prefix_paths = {
                        "gate": split_root / "cpu-gate.q4_0.bin",
                        "up": split_root / "cpu-up.q4_0.bin",
                        "down": split_root / "cpu-down.q4_0.bin",
                    }
                    write_bytes(prefix_paths["gate"], native_gate[:cpu_columns])
                    write_bytes(prefix_paths["up"], native_up[:cpu_columns])
                    write_bytes(prefix_paths["down"], native_down[:, :cpu_blocks])
                    prefix_output_path = split_root / "cpu-output.f32.bin"
                    prefix_command = cpu_command(
                        args.cpu_benchmark,
                        prefix_paths["gate"],
                        prefix_paths["up"],
                        prefix_paths["down"],
                        activation_path,
                        prefix_output_path,
                        input_columns=input_columns,
                        intermediate_columns=cpu_columns,
                        output_columns=output_columns,
                        rows=rows,
                        cpu_backend=args.cpu_backend,
                        cpu_threads=args.cpu_threads,
                        warmups=args.warmups,
                        samples=args.samples,
                        serve=True,
                    )
                    process = start_cpu_server(prefix_command)
                    try:
                        # The output tensor is undefined until the graph has run.  Do one
                        # untimed initialization even when the requested warmup count is zero.
                        server_request(process, "run")
                        server_request(process, f"dump {prefix_output_path.resolve()}")
                        cpu_output = np.fromfile(prefix_output_path, dtype="<f4").reshape(
                            rows, output_columns
                        ).copy()
                        qpu_gate_native = native_gate[cpu_columns:]
                        qpu_up_native = native_up[cpu_columns:]
                        qpu_down_native = native_down[:, cpu_blocks:]
                        gate_scales_host, gate_q_host = pack_ggml_q4_0_tiled_weights(
                            qpu_gate_native
                        )
                        up_scales_host, up_q_host = pack_ggml_q4_0_tiled_weights(
                            qpu_up_native
                        )
                        down_scales_host, down_q_host = pack_ggml_q4_0_tiled_weights(
                            qpu_down_native
                        )
                        qpu_blocks = qpu_columns // Q4_0_BLOCK_ELEMENTS
                        table_host = ggml_gelu_fp16_table()
                        data_area_size = int(
                            input_scales_host.nbytes
                            + input_q_host.nbytes
                            + gate_scales_host.nbytes
                            + gate_q_host.nbytes
                            + up_scales_host.nbytes
                            + up_q_host.nbytes
                            + down_scales_host.nbytes
                            + down_q_host.nbytes
                            + table_host.nbytes
                            + 2 * padded_rows * qpu_columns * 4
                            + padded_rows * qpu_blocks * 4
                            + padded_rows * qpu_columns
                            + padded_rows * output_columns * 4
                            + (16 << 20)
                        )
                        with Device.open(data_area_size=data_area_size) as device, device.queue() as queue:
                            input_scales = device.tensor(input_scales_host.shape, np.uint32)
                            input_q = device.tensor(input_q_host.shape, np.uint8)
                            gate_scales = device.tensor(gate_scales_host.shape, np.uint16)
                            gate_q = device.tensor(gate_q_host.shape, np.uint32)
                            up_scales = device.tensor(up_scales_host.shape, np.uint16)
                            up_q = device.tensor(up_q_host.shape, np.uint32)
                            down_scales = device.tensor(down_scales_host.shape, np.uint16)
                            down_q = device.tensor(down_q_host.shape, np.uint32)
                            table = device.tensor(table_host.shape, np.uint16)
                            gate_output = device.tensor((padded_rows, qpu_columns), np.float32)
                            up_output = device.tensor((padded_rows, qpu_columns), np.float32)
                            intermediate_scales = device.tensor((padded_rows, qpu_blocks), np.uint32)
                            intermediate_q = device.tensor((padded_rows, qpu_columns), np.uint8)
                            qpu_output = device.tensor((padded_rows, output_columns), np.float32)
                            gate_scales.numpy()[:] = gate_scales_host
                            gate_q.numpy()[:] = gate_q_host
                            up_scales.numpy()[:] = up_scales_host
                            up_q.numpy()[:] = up_q_host
                            down_scales.numpy()[:] = down_scales_host
                            down_q.numpy()[:] = down_q_host
                            table.numpy()[:] = table_host
                            qpu_output_host = np.empty((rows, output_columns), dtype=np.float32)
                            combined = np.empty_like(qpu_output_host)

                            def qpu_chain() -> int:
                                start = time.perf_counter_ns()
                                input_scales.numpy()[:] = input_scales_host
                                input_q.numpy()[:] = input_q_host
                                queue.submit(
                                    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
                                    (input_q, input_scales, gate_q, gate_scales, gate_output),
                                    grid=(qpu_columns // OUTPUT_TILE, padded_rows // ROW_TILE, 1),
                                ).wait()
                                queue.submit(
                                    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
                                    (input_q, input_scales, up_q, up_scales, up_output),
                                    grid=(qpu_columns // OUTPUT_TILE, padded_rows // ROW_TILE, 1),
                                ).wait()
                                queue.submit(
                                    GGML_GEGLU_Q8_0_SPLIT_KERNEL,
                                    (
                                        gate_output,
                                        up_output,
                                        intermediate_scales,
                                        intermediate_q,
                                        table,
                                    ),
                                    grid=(qpu_blocks, padded_rows, 1),
                                ).wait()
                                queue.submit(
                                    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
                                    (
                                        intermediate_q,
                                        intermediate_scales,
                                        down_q,
                                        down_scales,
                                        qpu_output,
                                    ),
                                    grid=(output_columns // OUTPUT_TILE, padded_rows // ROW_TILE, 1),
                                ).wait()
                                np.copyto(qpu_output_host, qpu_output.numpy()[:rows])
                                return time.perf_counter_ns() - start

                            for _ in range(args.warmups):
                                server_request(process, "run")
                                qpu_chain()
                            cpu_samples: list[int] = []
                            qpu_samples: list[int] = []
                            add_samples: list[int] = []
                            candidate_samples: list[int] = []
                            for _ in range(args.samples):
                                wall_start = time.perf_counter_ns()
                                if process.stdin is None:
                                    raise RuntimeError("CPU FFN server stdin closed")
                                process.stdin.write("run\n")
                                process.stdin.flush()
                                qpu_ns = qpu_chain()
                                cpu_record = server_line(process.stdout)
                                add_start = time.perf_counter_ns()
                                np.add(cpu_output, qpu_output_host, out=combined)
                                add_ns = time.perf_counter_ns() - add_start
                                candidate_ns = time.perf_counter_ns() - wall_start
                                cpu_samples.append(int(cpu_record["complete_ns"]))
                                qpu_samples.append(qpu_ns)
                                add_samples.append(add_ns)
                                candidate_samples.append(candidate_ns)
                            correctness = calculate_attention_error(
                                full_output,
                                combined,
                                atol=CORRECTNESS_ATOL,
                                rtol=CORRECTNESS_RTOL,
                            )
                            full_samples = [int(value) for value in full_cpu["complete_ns"]]
                            records.append(
                                {
                                    "layer": layer,
                                    "rows": rows,
                                    "padded_rows": padded_rows,
                                    "input_columns": input_columns,
                                    "intermediate_columns": intermediate_columns,
                                    "output_columns": output_columns,
                                    "cpu_intermediate_columns": cpu_columns,
                                    "qpu_intermediate_columns": qpu_columns,
                                    "qpu_fraction": actual_fraction,
                                    "cpu_threads": args.cpu_threads,
                                    "cpu_backend": args.cpu_backend,
                                    "cpu_full_placements": full_cpu.get("placements"),
                                    "cpu_full_ns": full_samples,
                                    "cpu_complement_ns": cpu_samples,
                                    "qpu_island_ns": qpu_samples,
                                    "join_add_ns": add_samples,
                                    "candidate_wall_ns": candidate_samples,
                                    "cpu_full_summary": summarize_ns(full_samples),
                                    "cpu_complement_summary": summarize_ns(cpu_samples),
                                    "qpu_island_summary": summarize_ns(qpu_samples),
                                    "join_add_summary": summarize_ns(add_samples),
                                    "candidate_wall_summary": summarize_ns(candidate_samples),
                                    "speedup_vs_cpu": float(
                                        np.median(full_samples) / np.median(candidate_samples)
                                    ),
                                    "correctness": correctness,
                                    "resident_weight_bytes": int(
                                        gate_scales_host.nbytes
                                        + gate_q_host.nbytes
                                        + up_scales_host.nbytes
                                        + up_q_host.nbytes
                                        + down_scales_host.nbytes
                                        + down_q_host.nbytes
                                    ),
                                    "programs": [
                                        "ggml-q4-0-q8-wordscale-mx",
                                        "ggml-geglu-q8-0-split",
                                    ],
                                    "cpu_full_command": full_command,
                                    "cpu_complement_command": prefix_command,
                                }
                            )
                    finally:
                        stop_cpu_server(process)
    finally:
        if temporary is not None:
            temporary.cleanup()

    after = collect_environment()
    best_by_shape: list[dict[str, Any]] = []
    for layer in sorted({int(record["layer"]) for record in records}):
        for rows in sorted(
            {int(record["rows"]) for record in records if record["layer"] == layer}
        ):
            candidates = [
                record
                for record in records
                if record["layer"] == layer
                and record["rows"] == rows
                and record["correctness"]["passed"]
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda record: float(record["speedup_vs_cpu"]))
            best_by_shape.append(
                {
                    "layer": layer,
                    "rows": rows,
                    "qpu_fraction": best["qpu_fraction"],
                    "speedup_vs_cpu": best["speedup_vs_cpu"],
                    "correctness": best["correctness"],
                    "promotion_passed": bool(
                        float(best["speedup_vs_cpu"]) >= PROMOTION_SPEEDUP
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-concurrent-ffn-island",
        "created_utc": utc_now(),
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
        "model": {"path": str(model.resolve()), "sha256": sha256_file(model)},
        "seed": args.seed,
        "warmups": args.warmups,
        "samples": args.samples,
        "records": records,
        "best_by_shape": best_by_shape,
        "environment_before": before,
        "environment_after": after,
        "measurement_contract": {
            "cpu": (
                "pinned llama.cpp scheduler with OpenBLAS gate/up/down and CPU GEGLU"
                if args.cpu_backend == "openblas"
                else "pinned llama.cpp CPU_REPACK gate/up/GEGLU/down graph"
            ),
            "candidate": (
                "CPU_REPACK computes the intermediate prefix while four resident QPU stages "
                "compute the suffix gate/up/GEGLU-Q8/down partial; final outputs are added"
            ),
            "intermediate_host_access": False,
            "timing_includes": [
                "QPU input staging",
                "four QPU dispatches and waits",
                "final QPU output access",
                "concurrent CPU complement",
                "final partial-output addition",
            ],
            "correctness_tolerance": {
                "atol": CORRECTNESS_ATOL,
                "rtol": CORRECTNESS_RTOL,
                "rationale": (
                    "the CPU and channel-partitioned paths use identical Q4_0/Q8_0 blocks "
                    "but add independently accumulated down-projection partials"
                ),
            },
            "promotion_gate": (
                f"correct and median full-FFN speedup >= {PROMOTION_SPEEDUP:.2f}"
            ),
        },
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(records)} concurrent FFN island records")


if __name__ == "__main__":
    main()
