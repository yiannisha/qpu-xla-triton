#!/usr/bin/env python3
"""Benchmark native llama.cpp Q4_0 CPU, QPU, and output-column hybrid nodes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    utc_now,
    write_json_atomic,
)

Q4_0_BLOCK_ELEMENTS = 32
Q4_0_BLOCK_BYTES = 18
OUTPUT_TILE = 16


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


def _model_tensor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [tensor for tensor in manifest["tensors"] if tensor["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor named {name!r}, found {len(matches)}")
    tensor = matches[0]
    if tensor["ggml_type"] != "Q4_0" or len(tensor["shape"]) != 2:
        raise ValueError(f"{name!r} is not a rank-two Q4_0 tensor")
    input_columns, output_columns = map(int, tensor["shape"])
    if input_columns % Q4_0_BLOCK_ELEMENTS or output_columns % OUTPUT_TILE:
        raise ValueError(f"{name!r} does not meet Q4_0/QPU alignment constraints")
    return tensor


def default_tensor_names(manifest: dict[str, Any]) -> list[str]:
    """Choose one largest non-embedding Q4_0 projection as the bounded default."""
    eligible = [
        tensor
        for tensor in manifest["tensors"]
        if tensor["ggml_type"] == "Q4_0"
        and len(tensor["shape"]) == 2
        and tensor["owner"]["operator"] != "embedding"
        and int(tensor["shape"][0]) % Q4_0_BLOCK_ELEMENTS == 0
        and int(tensor["shape"][1]) % OUTPUT_TILE == 0
    ]
    if not eligible:
        raise ValueError("manifest has no aligned rank-two non-embedding Q4_0 tensor")
    return [max(eligible, key=lambda tensor: int(tensor["bytes"]))["name"]]


def _write_fixture(
    model: Path,
    tensor: dict[str, Any],
    rows: int,
    seed: int,
    directory: Path,
) -> dict[str, Any]:
    input_columns, output_columns = map(int, tensor["shape"])
    expected_weight_bytes = output_columns * (input_columns // Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_BYTES
    if expected_weight_bytes != int(tensor["bytes"]):
        raise ValueError(f"manifest byte count for {tensor['name']} is inconsistent with native Q4_0")
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = tensor["name"].replace("/", "_").replace(".", "-")
    weight_path = directory / f"{safe_name}.q4_0.bin"
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
    warmups: int,
    samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        str(binary),
        "--weights",
        fixture["weight"]["path"],
        "--activation-f32",
        fixture["activation_f32"]["path"],
        "--mode",
        mode,
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
        raise RuntimeError(f"native operator benchmark failed: {' '.join(command)}\n{completed.stderr}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("native operator benchmark produced no JSON")
    result = json.loads(lines[-1])
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
    return result, raw


def _throttle_value(environment: dict[str, Any]) -> int | None:
    stdout = environment["commands"]["throttling"].get("stdout", "")
    marker = "throttled="
    if marker not in stdout:
        return None
    try:
        return int(stdout.split(marker, 1)[1].strip(), 0)
    except ValueError:
        return None


def _session_validation(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    before_throttle = _throttle_value(before)
    after_throttle = _throttle_value(after)
    if before_throttle != 0 or after_throttle != 0:
        reasons.append(f"throttling flags were {before_throttle!r} before and {after_throttle!r} after")
    before_swap = before["commands"]["swap"].get("stdout", "")
    after_swap = after["commands"]["swap"].get("stdout", "")
    if before_swap != after_swap:
        reasons.append("swapon state changed during the session")
    governors = {entry["governor"] for entry in before["cpu_frequency"]}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(value) for value in governors)}")
    return {"retained": not reasons, "rejection_reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--rows", type=int, action="append", choices=(1, 4), default=[])
    parser.add_argument("--cpu-threads", type=int, action="append", default=[])
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
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.warmups = 1
        args.samples = 3
        args.bootstrap_resamples = 500
    if args.warmups < 0 or args.samples <= 0 or args.bootstrap_resamples <= 0:
        parser.error("warmups must be non-negative; samples and bootstrap resamples must be positive")
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    if not args.benchmark_binary.is_file():
        parser.error(f"native benchmark not found: {args.benchmark_binary}")
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
                fixture = _write_fixture(
                    model,
                    tensor,
                    rows,
                    args.seed + tensor_index * 100 + rows,
                    fixture_root,
                )
                fixtures.append(fixture)
                cpu_results: list[dict[str, Any]] = []
                for threads in thread_values:
                    result, raw = _run_native(
                        args.benchmark_binary.resolve(),
                        fixture,
                        mode="cpu",
                        cpu_threads=threads,
                        qpu_column_start=fixture["output_columns"],
                        qpu_column_count=0,
                        warmups=args.warmups,
                        samples=args.samples,
                    )
                    cpu_results.append(result)
                    records.append({"tensor": tensor, "fixture": fixture, "result": result})
                    raw_commands.append(raw)
                fastest_cpu = min(
                    cpu_results,
                    key=lambda result: result["summaries"]["complete_ns"]["median_ns"],
                )
                result, raw = _run_native(
                    args.benchmark_binary.resolve(),
                    fixture,
                    mode="qpu",
                    cpu_threads=int(fastest_cpu["cpu_threads"]),
                    qpu_column_start=0,
                    qpu_column_count=fixture["output_columns"],
                    warmups=args.warmups,
                    samples=args.samples,
                )
                result["speedup_vs_fastest_cpu"] = bootstrap_speedup_interval(
                    fastest_cpu["complete_ns"],
                    result["complete_ns"],
                    seed=args.seed + 10_000 + tensor_index * 100 + rows,
                    resamples=args.bootstrap_resamples,
                )
                records.append({"tensor": tensor, "fixture": fixture, "result": result})
                raw_commands.append(raw)
                partitions = aligned_qpu_partitions(int(fixture["output_columns"]))
                if args.quick and partitions:
                    partitions = [min(partitions, key=lambda item: abs(float(item["actual_qpu_fraction"]) - 0.5))]
                for threads in thread_values:
                    for partition_index, partition in enumerate(partitions):
                        result, raw = _run_native(
                            args.benchmark_binary.resolve(),
                            fixture,
                            mode="hybrid",
                            cpu_threads=threads,
                            qpu_column_start=int(partition["qpu_column_start"]),
                            qpu_column_count=int(partition["qpu_column_count"]),
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
        "configuration": {
            "tensor_names": tensor_names,
            "rows": rows_values,
            "cpu_threads": thread_values,
            "warmups": args.warmups,
            "samples": args.samples,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "quick": args.quick,
        },
        "fixtures": fixtures,
        "records": records,
        "raw_commands": raw_commands,
        "environment_before": before,
        "environment_after": after,
        "validation": _session_validation(before, after),
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
