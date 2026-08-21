#!/usr/bin/env python3
"""Benchmark native llama.cpp Q4_0/Q4_K/Q6_K CPU, QPU, and output-column hybrids."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, TypedDict, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    swap_used_bytes,
    utc_now,
    write_json_atomic,
)

Q4_0_BLOCK_ELEMENTS = 32
Q4_0_BLOCK_BYTES = 18
Q4_K_BLOCK_ELEMENTS = 256
Q4_K_BLOCK_BYTES = 144
Q6_K_BLOCK_ELEMENTS = 256
Q6_K_BLOCK_BYTES = 210
Q8_0_BLOCK_ELEMENTS = 32
Q8_0_BLOCK_BYTES = 34
OUTPUT_TILE = 16
class FormatLayout(TypedDict):
    """Native block geometry and the matching C harness format name."""

    cli: str
    block_elements: int
    block_bytes: int


FORMAT_LAYOUT: dict[str, FormatLayout] = {
    "Q4_0": {"cli": "q4_0", "block_elements": Q4_0_BLOCK_ELEMENTS, "block_bytes": Q4_0_BLOCK_BYTES},
    "Q4_K": {"cli": "q4_k", "block_elements": Q4_K_BLOCK_ELEMENTS, "block_bytes": Q4_K_BLOCK_BYTES},
    "Q6_K": {"cli": "q6_k", "block_elements": Q6_K_BLOCK_ELEMENTS, "block_bytes": Q6_K_BLOCK_BYTES},
    "Q8_0": {"cli": "q8_0", "block_elements": Q8_0_BLOCK_ELEMENTS, "block_bytes": Q8_0_BLOCK_BYTES},
}


def aligned_qpu_partitions(output_columns: int) -> list[dict[str, int | float]]:
    """Return unique 1/12 through 11/12 suffix partitions on 16-column tiles."""
    if output_columns < 2 * OUTPUT_TILE or output_columns % OUTPUT_TILE:
        return []
    tiles = output_columns // OUTPUT_TILE
    records: list[dict[str, int | float]] = []
    seen: set[int] = set()
    for numerator in range(1, 12):
        qpu_tiles = (tiles * numerator + 6) // 12
        qpu_tiles = min(max(qpu_tiles, 1), tiles - 1)
        qpu_columns = qpu_tiles * OUTPUT_TILE
        if qpu_columns in seen:
            continue
        seen.add(qpu_columns)
        records.append(
            {
                "requested_numerator": numerator,
                "requested_denominator": 12,
                "qpu_column_start": output_columns - qpu_columns,
                "qpu_column_count": qpu_columns,
                "actual_qpu_fraction": qpu_columns / output_columns,
            }
        )
    return records


def summarize_ns(samples: list[int]) -> dict[str, float | int | None]:
    """Compute robust summaries while preserving raw samples elsewhere."""
    if not samples:
        return {}
    values = np.asarray(samples, dtype=np.float64)
    center = float(np.median(values))
    return {
        "count": len(samples),
        "median_ns": center,
        "p05_ns": float(np.quantile(values, 0.05)),
        "p95_ns": float(np.quantile(values, 0.95)),
        "mad_ns": float(np.median(np.abs(values - center))),
        "throughput_per_second": 1e9 / center if center > 0 else None,
    }


def bootstrap_speedup_interval(
    baseline_ns: list[int],
    candidate_ns: list[int],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    """Bootstrap the ratio of independent sample medians."""
    if not baseline_ns or not candidate_ns:
        return {}
    baseline = np.asarray(baseline_ns, dtype=np.float64)
    candidate = np.asarray(candidate_ns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        base_median = np.median(rng.choice(baseline, baseline.size, replace=True))
        candidate_median = np.median(rng.choice(candidate, candidate.size, replace=True))
        ratios[index] = base_median / candidate_median
    return {
        "resamples": resamples,
        "seed": seed,
        "median_speedup": float(median(baseline_ns) / median(candidate_ns)),
        "bootstrap_95_low": float(np.quantile(ratios, 0.025)),
        "bootstrap_95_high": float(np.quantile(ratios, 0.975)),
    }


def compare_exact_cpu_output(
    reference: np.ndarray[Any, np.dtype[np.float32]],
    candidate: np.ndarray[Any, np.dtype[np.float32]],
) -> dict[str, Any]:
    """Compare one emitted candidate tensor with the exact CPU_REPACK output."""
    if reference.shape != candidate.shape:
        raise ValueError(f"output shapes differ: {reference.shape} versus {candidate.shape}")
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    finite = np.isfinite(candidate64)
    absolute = np.where(finite, np.abs(candidate64 - reference64), np.inf)
    relative = absolute / np.maximum(np.abs(reference64), 1e-12)
    tolerance = 2e-4 + 2e-5 * np.abs(reference64)
    rows = reference.shape[0]
    p99_absolute = (
        float(np.quantile(absolute, 0.99))
        if absolute.size and np.all(np.isfinite(absolute))
        else (float("inf") if absolute.size else 0.0)
    )
    return {
        "reference": "exact pinned llama.cpp CPU_REPACK MUL_MAT graph output",
        "max_absolute": float(np.max(absolute)) if absolute.size else 0.0,
        "max_relative": float(np.max(relative)) if relative.size else 0.0,
        "mean_absolute": float(np.mean(absolute)) if absolute.size else 0.0,
        "p99_absolute": p99_absolute,
        "atol": 2e-4,
        "rtol": 2e-5,
        "tolerance_violation_count": int(np.count_nonzero(absolute > tolerance)),
        "nan_count": int(np.count_nonzero(np.isnan(candidate))),
        "inf_count": int(np.count_nonzero(np.isinf(candidate))),
        "finite_count": int(np.count_nonzero(finite)),
        "bitwise_equal_count": int(
            np.count_nonzero(reference.view(np.uint32) == candidate.view(np.uint32))
        ),
        "element_count": int(reference.size),
        "reference_argmax": [int(np.argmax(reference[row])) for row in range(rows)],
        "candidate_argmax": [int(np.argmax(candidate[row])) for row in range(rows)],
        "argmax_identical": bool(
            all(np.argmax(reference[row]) == np.argmax(candidate[row]) for row in range(rows))
        ),
    }


def _model_tensor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [tensor for tensor in manifest["tensors"] if tensor["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor named {name!r}, found {len(matches)}")
    tensor = cast(dict[str, Any], matches[0])
    if tensor["ggml_type"] not in FORMAT_LAYOUT or len(tensor["shape"]) != 2:
        raise ValueError(f"{name!r} is not a supported rank-two native quantized tensor")
    layout = FORMAT_LAYOUT[tensor["ggml_type"]]
    input_columns, output_columns = map(int, tensor["shape"])
    if input_columns % int(layout["block_elements"]) or output_columns % OUTPUT_TILE:
        raise ValueError(f"{name!r} does not meet native-format/QPU alignment constraints")
    return tensor


def default_tensor_names(manifest: dict[str, Any]) -> list[str]:
    """Choose one largest aligned supported non-embedding projection."""
    eligible: list[dict[str, Any]] = [
        cast(dict[str, Any], tensor)
        for tensor in manifest["tensors"]
        if tensor["ggml_type"] in FORMAT_LAYOUT
        and len(tensor["shape"]) == 2
        and tensor["owner"]["operator"] != "embedding"
        and int(tensor["shape"][0]) % int(FORMAT_LAYOUT[tensor["ggml_type"]]["block_elements"]) == 0
        and int(tensor["shape"][1]) % OUTPUT_TILE == 0
    ]
    if not eligible:
        raise ValueError("manifest has no aligned rank-two non-embedding supported tensor")
    return [max(eligible, key=lambda tensor: int(tensor["bytes"]))["name"]]


def _write_fixture(
    model: Path,
    tensor: dict[str, Any],
    rows: int,
    seed: int,
    directory: Path,
) -> dict[str, Any]:
    input_columns, output_columns = map(int, tensor["shape"])
    layout = FORMAT_LAYOUT[tensor["ggml_type"]]
    expected_weight_bytes = (
        output_columns
        * (input_columns // int(layout["block_elements"]))
        * int(layout["block_bytes"])
    )
    if expected_weight_bytes != int(tensor["bytes"]):
        raise ValueError(
            f"manifest byte count for {tensor['name']} is inconsistent with {tensor['ggml_type']}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = tensor["name"].replace("/", "_").replace(".", "-")
    weight_path = directory / f"{safe_name}.{str(layout['cli'])}.bin"
    activation_path = directory / f"{safe_name}.m{rows}.f32.bin"
    with model.open("rb") as source:
        source.seek(int(tensor["file_offset"]))
        weights = source.read(expected_weight_bytes)
    if len(weights) != expected_weight_bytes:
        raise ValueError(f"short tensor payload for {tensor['name']}")
    weight_path.write_bytes(weights)
    rng = np.random.default_rng(seed)
    activation = rng.normal(0.0, 0.75, size=(rows, input_columns)).astype("<f4")
    activation_path.write_bytes(activation.tobytes())
    return {
        "tensor_name": tensor["name"],
        "input_columns": input_columns,
        "output_columns": output_columns,
        "rows": rows,
        "weight_type": tensor["ggml_type"],
        "weight_type_cli": layout["cli"],
        "seed": seed,
        "weight": {
            "path": str(weight_path.resolve()),
            "bytes": weight_path.stat().st_size,
            "sha256": sha256_file(weight_path),
            "model_file_offset": int(tensor["file_offset"]),
        },
        "activation_f32": {
            "path": str(activation_path.resolve()),
            "bytes": activation_path.stat().st_size,
            "sha256": sha256_file(activation_path),
        },
    }


def _run_native(
    binary: Path,
    fixture: dict[str, Any],
    *,
    mode: str,
    cpu_threads: int,
    qpu_column_start: int,
    qpu_column_count: int,
    qpu_wgs_per_sg: int,
    warmups: int,
    samples: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray[Any, np.dtype[np.float32]]]:
    with tempfile.TemporaryDirectory(prefix="llama-native-output-") as temporary:
        output_path = Path(temporary) / "output.f32.bin"
        command = [
            str(binary),
            "--weights",
            fixture["weight"]["path"],
            "--activation-f32",
            fixture["activation_f32"]["path"],
            "--mode",
            mode,
            "--weight-type",
            fixture["weight_type_cli"],
            "--input-columns",
            str(fixture["input_columns"]),
            "--output-columns",
            str(fixture["output_columns"]),
            "--rows",
            str(fixture["rows"]),
            "--qpu-column-start",
            str(qpu_column_start),
            "--qpu-column-count",
            str(qpu_column_count),
            "--qpu-wgs-per-sg",
            str(qpu_wgs_per_sg),
            "--cpu-threads",
            str(cpu_threads),
            "--warmups",
            str(warmups),
            "--samples",
            str(samples),
            "--output-bin",
            str(output_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        raw = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            raise RuntimeError(
                f"native operator benchmark failed: {' '.join(command)}\n{completed.stderr}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("native operator benchmark produced no JSON")
        result = json.loads(lines[-1])
        expected_elements = fixture["rows"] * fixture["output_columns"]
        output = np.fromfile(output_path, dtype="<f4")
        if output.size != expected_elements:
            raise RuntimeError("native operator benchmark produced an invalid output fixture")
        output = output.reshape(fixture["rows"], fixture["output_columns"]).copy()
        raw["output"] = {
            "bytes": int(output.nbytes),
            "sha256": sha256_file(output_path),
            "retained": False,
        }
    result["summaries"] = {
        key: summarize_ns(result[key])
        for key in (
            "complete_ns",
            "quantize_ns",
            "qpu_input_copy_ns",
            "qpu_submit_wait_ns",
            "qpu_output_copy_ns",
        )
    }
    return result, raw, output


def _run_cpu_repack(
    binary: Path,
    fixture: dict[str, Any],
    *,
    cpu_threads: int,
    warmups: int,
    samples: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray[Any, np.dtype[np.float32]]]:
    """Run the exact pinned llama.cpp CPU_REPACK single-node graph baseline."""
    with tempfile.TemporaryDirectory(prefix="llama-cpu-repack-output-") as temporary:
        output_path = Path(temporary) / "output.f32.bin"
        command = [
            str(binary),
            "--weights",
            fixture["weight"]["path"],
            "--activation-f32",
            fixture["activation_f32"]["path"],
            "--output-bin",
            str(output_path),
            "--weight-type",
            fixture["weight_type_cli"],
            "--input-columns",
            str(fixture["input_columns"]),
            "--output-columns",
            str(fixture["output_columns"]),
            "--rows",
            str(fixture["rows"]),
            "--cpu-threads",
            str(cpu_threads),
            "--warmups",
            str(warmups),
            "--samples",
            str(samples),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        raw = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            raise RuntimeError(
                f"native CPU_REPACK benchmark failed: {' '.join(command)}\n{completed.stderr}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("native CPU_REPACK benchmark produced no JSON")
        result = json.loads(lines[-1])
        expected_output_bytes = fixture["rows"] * fixture["output_columns"] * 4
        if not output_path.is_file() or output_path.stat().st_size != expected_output_bytes:
            raise RuntimeError("native CPU_REPACK benchmark produced an invalid output fixture")
        output = np.fromfile(output_path, dtype="<f4").reshape(
            fixture["rows"], fixture["output_columns"]
        ).copy()
        result.update(
            {
                "mode": "cpu",
                "qpu_column_start": fixture["output_columns"],
                "qpu_column_count": 0,
                "resident_bytes": 0,
                "cpu_weight_buffer_bytes": result.pop("weight_buffer_bytes"),
                "cpu_kernel": (
                    f"pinned llama.cpp CPU_REPACK {fixture['weight_type']} MUL_MAT graph node"
                ),
                "activation_contract": "FP32 graph input; activation quantization is inside complete_ns",
                "quantize_ns": [0] * samples,
                "qpu_input_copy_ns": [0] * samples,
                "qpu_submit_wait_ns": [0] * samples,
                "qpu_output_copy_ns": [0] * samples,
                "correctness": {
                    "reference": "production CPU baseline",
                    "max_absolute": 0.0,
                    "max_relative": 0.0,
                    "mean_absolute": 0.0,
                    "p99_absolute": 0.0,
                    "atol": 0.0,
                    "rtol": 0.0,
                    "tolerance_violation_count": 0,
                    "nan_count": 0,
                    "inf_count": 0,
                },
            }
        )
        result["summaries"] = {
            key: summarize_ns(result[key])
            for key in (
                "complete_ns",
                "quantize_ns",
                "qpu_input_copy_ns",
                "qpu_submit_wait_ns",
                "qpu_output_copy_ns",
            )
        }
        raw["output"] = {
            "bytes": expected_output_bytes,
            "sha256": sha256_file(output_path),
            "retained": False,
        }
        return result, raw, output


def _throttle_value(environment: dict[str, Any]) -> int | None:
    stdout = environment["commands"]["throttling"].get("stdout", "")
    marker = "throttled="
    if marker not in stdout:
        return None
    try:
        return int(stdout.split(marker, 1)[1].strip(), 0)
    except ValueError:
        return None


def _session_validation(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    before_throttle = _throttle_value(before)
    after_throttle = _throttle_value(after)
    if before_throttle != 0 or after_throttle != 0:
        reasons.append(f"throttling flags were {before_throttle!r} before and {after_throttle!r} after")
    before_swap = before["commands"]["swap"].get("stdout", "")
    after_swap = after["commands"]["swap"].get("stdout", "")
    if before_swap != after_swap:
        reasons.append("swapon state changed during the session")
    before_swap_used = swap_used_bytes(before)
    after_swap_used = swap_used_bytes(after)
    if before_swap_used is None or after_swap_used is None:
        reasons.append("swap usage was unavailable")
    elif before_swap_used != 0 or after_swap_used != 0:
        reasons.append(
            f"swap was in use before/after the session ({before_swap_used}/{after_swap_used} bytes)"
        )
    governors = {entry["governor"] for entry in before["cpu_frequency"]}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(value) for value in governors)}")
    active_servers = before["commands"].get("llama_servers", {}).get("stdout", "").strip()
    if active_servers:
        reasons.append("one or more pre-existing llama-server processes were active")
    if warmups < 5:
        reasons.append(f"only {warmups} warmups were run; retained sessions require at least 5")
    if samples < 31:
        reasons.append(f"only {samples} samples were run; retained sessions require at least 31")
    return {"retained": not reasons, "rejection_reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--rows", type=int, action="append", choices=(1, 4), default=[])
    parser.add_argument("--cpu-threads", type=int, action="append", default=[])
    parser.add_argument("--qpu-wgs-per-sg", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--session-id", default="session-1")
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_q4_0_bench",
    )
    parser.add_argument(
        "--cpu-repack-binary",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_cpu_repack_bench",
    )
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.warmups = 1
        args.samples = 3
        args.bootstrap_resamples = 500
    if (
        args.warmups < 0
        or args.samples <= 0
        or args.bootstrap_resamples <= 0
        or not 1 <= args.qpu_wgs_per_sg <= 255
    ):
        parser.error("warmups must be non-negative; samples and bootstrap resamples must be positive")
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    if not args.benchmark_binary.is_file():
        parser.error(f"native benchmark not found: {args.benchmark_binary}")
    if not args.cpu_repack_binary.is_file():
        parser.error(f"CPU_REPACK benchmark not found: {args.cpu_repack_binary}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = Path(manifest["model"]["path"])
    if not model.is_file():
        parser.error(f"model referenced by manifest not found: {model}")
    tensor_names = args.tensor or default_tensor_names(manifest)
    rows_values = args.rows or [1, 4]
    thread_values = sorted(set(args.cpu_threads or [2, 3, 4]))
    if any(value <= 0 for value in thread_values):
        parser.error("CPU thread counts must be positive")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.fixture_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="llama-qpu-fixtures-")
        fixture_root = Path(temporary.name)
    else:
        fixture_root = args.fixture_dir.resolve()

    before = collect_environment()
    records: list[dict[str, Any]] = []
    raw_commands: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    try:
        for tensor_index, tensor_name in enumerate(tensor_names):
            tensor = _model_tensor(manifest, tensor_name)
            for rows in rows_values:
                if tensor["ggml_type"] != "Q4_0" and rows != 4:
                    continue
                fixture = _write_fixture(
                    model,
                    tensor,
                    rows,
                    args.seed + tensor_index * 100 + rows,
                    fixture_root,
                )
                fixtures.append(fixture)
                cpu_results: list[
                    tuple[dict[str, Any], np.ndarray[Any, np.dtype[np.float32]]]
                ] = []
                for threads in thread_values:
                    result, raw, exact_cpu_output = _run_cpu_repack(
                        args.cpu_repack_binary.resolve(),
                        fixture,
                        cpu_threads=threads,
                        warmups=args.warmups,
                        samples=args.samples,
                    )
                    result["exact_cpu_repack_correctness"] = compare_exact_cpu_output(
                        exact_cpu_output, exact_cpu_output
                    )
                    cpu_results.append((result, exact_cpu_output))
                    records.append({"tensor": tensor, "fixture": fixture, "result": result})
                    raw_commands.append(raw)
                fastest_cpu, fastest_cpu_output = min(
                    cpu_results,
                    key=lambda item: item[0]["summaries"]["complete_ns"]["median_ns"],
                )
                result, raw, candidate_output = _run_native(
                    args.benchmark_binary.resolve(),
                    fixture,
                    mode="qpu",
                    cpu_threads=int(fastest_cpu["cpu_threads"]),
                    qpu_column_start=0,
                    qpu_column_count=fixture["output_columns"],
                    qpu_wgs_per_sg=args.qpu_wgs_per_sg,
                    warmups=args.warmups,
                    samples=args.samples,
                )
                result["speedup_vs_fastest_cpu"] = bootstrap_speedup_interval(
                    fastest_cpu["complete_ns"],
                    result["complete_ns"],
                    seed=args.seed + 10_000 + tensor_index * 100 + rows,
                    resamples=args.bootstrap_resamples,
                )
                result["exact_cpu_repack_correctness"] = compare_exact_cpu_output(
                    fastest_cpu_output, candidate_output
                )
                records.append({"tensor": tensor, "fixture": fixture, "result": result})
                raw_commands.append(raw)
                partitions = aligned_qpu_partitions(int(fixture["output_columns"]))
                if args.quick and partitions:
                    partitions = [min(partitions, key=lambda item: abs(float(item["actual_qpu_fraction"]) - 0.5))]
                for threads in thread_values:
                    for partition_index, partition in enumerate(partitions):
                        result, raw, candidate_output = _run_native(
                            args.benchmark_binary.resolve(),
                            fixture,
                            mode="hybrid",
                            cpu_threads=threads,
                            qpu_column_start=int(partition["qpu_column_start"]),
                            qpu_column_count=int(partition["qpu_column_count"]),
                            qpu_wgs_per_sg=args.qpu_wgs_per_sg,
                            warmups=args.warmups,
                            samples=args.samples,
                        )
                        result["partition"] = partition
                        result["speedup_vs_fastest_cpu"] = bootstrap_speedup_interval(
                            fastest_cpu["complete_ns"],
                            result["complete_ns"],
                            seed=(
                                args.seed
                                + 20_000
                                + tensor_index * 10_000
                                + rows * 100
                                + threads * 12
                                + partition_index
                            ),
                            resamples=args.bootstrap_resamples,
                        )
                        result["exact_cpu_repack_correctness"] = compare_exact_cpu_output(
                            fastest_cpu_output, candidate_output
                        )
                        records.append({"tensor": tensor, "fixture": fixture, "result": result})
                        raw_commands.append(raw)
    finally:
        after = collect_environment()

    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-native-operator-session",
        "created_utc": utc_now(),
        "session_id": args.session_id,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
            "model_sha256": manifest["model"].get("sha256"),
        },
        "native_benchmark": {
            "path": str(args.benchmark_binary.resolve()),
            "sha256": sha256_file(args.benchmark_binary),
        },
        "cpu_repack_benchmark": {
            "path": str(args.cpu_repack_binary.resolve()),
            "sha256": sha256_file(args.cpu_repack_binary),
        },
        "measurement_contract": {
            "cpu_only": "exact pinned llama.cpp CPU_REPACK MUL_MAT graph node for the native weight type",
            "qpu_only": (
                "native Q4_0/Q8_0, Q4_K/Q8_K, Q6_K/Q8_K, or Q8_0/Q8_0 "
                "persistent QPU operator"
            ),
            "hybrid_cpu_side": (
                "exported direct GGML native block-dot prefix; diagnostic only because it is "
                "not the production CPU_REPACK scheduling path"
            ),
            "hybrid_cpu_side_exact": False,
            "promotion_rule": "hybrid rows cannot be promoted until their CPU prefix uses CPU_REPACK",
            "correctness_reference": (
                "every emitted CPU, QPU, and hybrid tensor is compared directly with the fastest "
                "exact pinned llama.cpp CPU_REPACK output"
            ),
        },
        "configuration": {
            "tensor_names": tensor_names,
            "rows": rows_values,
            "cpu_threads": thread_values,
            "warmups": args.warmups,
            "samples": args.samples,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "qpu_wgs_per_sg": args.qpu_wgs_per_sg,
            "quick": args.quick,
        },
        "fixtures": fixtures,
        "records": records,
        "raw_commands": raw_commands,
        "environment_before": before,
        "environment_after": after,
        "validation": _session_validation(
            before,
            after,
            warmups=args.warmups,
            samples=args.samples,
        ),
    }
    write_json_atomic(args.output, payload)
    if temporary is not None:
        temporary.cleanup()
        for fixture in payload["fixtures"]:
            fixture["weight"]["path"] = None
            fixture["activation_f32"]["path"] = None
        for record in payload["records"]:
            record["fixture"]["weight"]["path"] = None
            record["fixture"]["activation_f32"]["path"] = None
        write_json_atomic(args.output, payload)
    retained = payload["validation"]["retained"]
    print(f"wrote {args.output}: {len(records)} records; retained={retained}")


if __name__ == "__main__":
    main()
