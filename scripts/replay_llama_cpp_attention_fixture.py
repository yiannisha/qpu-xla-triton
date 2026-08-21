#!/usr/bin/env python3
"""Replay a captured FLASH_ATTN_EXT node through the independent FP32 oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_xla.kernels.ggml_flash_attn import ggml_flash_attn_ext_reference  # noqa: E402
from scripts.llama_cpp_common import (  # noqa: E402
    sha256_file,
    utc_now,
    write_json_atomic,
)


def load_strided_tensor(tensor: dict[str, Any]) -> npt.NDArray[np.floating[Any]]:
    """Load one hashed capture as a logical native-GGML strided array."""
    path = Path(tensor["path"])
    if not path.is_file() or sha256_file(path) != tensor["sha256"]:
        raise ValueError(f"captured tensor {tensor['name']!r} is missing or changed")
    raw = path.read_bytes()
    if len(raw) != int(tensor["bytes"]):
        raise ValueError(f"captured tensor {tensor['name']!r} has the wrong byte count")
    if tensor["type"] == "f16":
        dtype: np.dtype[Any] = np.dtype("<f2")
    elif tensor["type"] == "f32":
        dtype = np.dtype("<f4")
    else:
        raise ValueError(f"unsupported captured tensor type {tensor['type']!r}")
    shape = tuple(int(value) for value in tensor["shape"])
    strides = tuple(int(value) for value in tensor["strides"])
    try:
        return np.ndarray(shape=shape, dtype=dtype, buffer=raw, strides=strides)
    except ValueError as exc:
        raise ValueError(f"captured tensor {tensor['name']!r} has an invalid strided view") from exc


def calculate_attention_error(
    reference: npt.NDArray[np.floating[Any]],
    actual: npt.NDArray[np.floating[Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Retain full numerical evidence without imposing a logits argmax contract."""
    if reference.shape != actual.shape:
        raise ValueError(f"attention output shapes differ: {reference.shape} != {actual.shape}")
    reference64 = np.asarray(reference, dtype=np.float64)
    actual64 = np.asarray(actual, dtype=np.float64)
    invalid = ~np.isfinite(actual64)
    absolute = np.where(invalid, np.inf, np.abs(actual64 - reference64))
    relative = absolute / np.maximum(np.abs(reference64), 1e-12)
    threshold = atol + rtol * np.abs(reference64)
    violations = invalid | (absolute > threshold)
    flattened_absolute = absolute.reshape(-1)
    worst_flat = int(np.argmax(flattened_absolute)) if flattened_absolute.size else 0
    worst_index = (
        [int(index) for index in np.unravel_index(worst_flat, absolute.shape)]
        if absolute.size
        else []
    )
    p99_absolute = (
        float(np.quantile(absolute, 0.99))
        if absolute.size and np.all(np.isfinite(absolute))
        else (float("inf") if absolute.size else 0.0)
    )
    return {
        "elements": int(reference.size),
        "atol": atol,
        "rtol": rtol,
        "max_absolute": float(np.max(absolute)) if absolute.size else 0.0,
        "max_relative": float(np.max(relative)) if relative.size else 0.0,
        "mean_absolute": float(np.mean(absolute)) if absolute.size else 0.0,
        "p99_absolute": p99_absolute,
        "tolerance_violation_count": int(np.count_nonzero(violations)),
        "nan_count": int(np.count_nonzero(np.isnan(actual64))),
        "inf_count": int(np.count_nonzero(np.isinf(actual64))),
        "worst_index": worst_index,
        "worst_reference": float(reference[tuple(worst_index)]) if worst_index else None,
        "worst_actual": float(actual[tuple(worst_index)]) if worst_index else None,
        "passed": not np.any(violations),
    }


def replay_attention_fixture(
    fixture: dict[str, Any],
) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
    """Load, evaluate, and compare one validated attention fixture."""
    query = load_strided_tensor(fixture["query"])
    key = load_strided_tensor(fixture["key"])
    value = load_strided_tensor(fixture["value"])
    mask = load_strided_tensor(fixture["mask"])
    sinks = load_strided_tensor(fixture["sinks"]) if fixture.get("sinks") else None
    captured_output = load_strided_tensor(fixture["reference_output"])
    params = fixture["node"]["op_params"]
    actual = ggml_flash_attn_ext_reference(
        query,
        key,
        value,
        mask,
        scale=float(params["scale"]),
        max_bias=float(params["max_bias"]),
        logit_softcap=float(params["logit_softcap"]),
        sinks=sinks,
    )
    contract = fixture["replay_contract"]
    correctness = calculate_attention_error(
        captured_output,
        actual,
        atol=float(contract["atol"]),
        rtol=float(contract["rtol"]),
    )
    return actual, correctness


def main() -> None:
    """Replay one manifest, retain correctness evidence, and fail on mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-output-bin", type=Path)
    args = parser.parse_args()
    if not args.fixture_manifest.is_file():
        parser.error(f"fixture manifest not found: {args.fixture_manifest}")
    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    if fixture.get("kind") != "llama-cpp-qpu-exact-flash-attention-fixture":
        parser.error("input is not an exact FLASH_ATTN_EXT fixture manifest")
    try:
        actual, correctness = replay_attention_fixture(fixture)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    output_record: dict[str, Any] | None = None
    if args.reference_output_bin is not None:
        args.reference_output_bin.parent.mkdir(parents=True, exist_ok=True)
        contiguous = np.asfortranarray(actual)
        args.reference_output_bin.write_bytes(contiguous.tobytes(order="F"))
        output_record = {
            "path": str(args.reference_output_bin.resolve()),
            "bytes": args.reference_output_bin.stat().st_size,
            "sha256": sha256_file(args.reference_output_bin),
        }
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-exact-flash-attention-replay",
        "created_utc": utc_now(),
        "fixture_manifest": {
            "path": str(args.fixture_manifest.resolve()),
            "sha256": sha256_file(args.fixture_manifest),
        },
        "candidate": "independent-fp32-flash-attention-reference",
        "correctness": correctness,
        "reference_output": output_record,
        "measurement_contract": {
            "timing_eligible_for_promotion": False,
            "reason": "this replay establishes semantics and numerical tolerance only",
        },
    }
    write_json_atomic(args.output, payload)
    print(
        f"wrote {args.output}: max_abs={correctness['max_absolute']:.9g}, "
        f"violations={correctness['tolerance_violation_count']}"
    )
    if not correctness["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
