from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from qpu_xla.workloads import LLAMA_DENSE_V1, YOLO_DETECTION_V1

_SCRIPT = Path(__file__).parents[1] / "scripts/w8a8_evaluation_matrix.py"
_SPEC = importlib.util.spec_from_file_location("w8a8_evaluation_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_render_evaluation = _MODULE._render_evaluation
_validate_coverage = _MODULE._validate_coverage
_validate_result = _MODULE._validate_result


def _quality() -> dict[str, object]:
    return {
        "normalized_rmse": 0.01,
        "cosine_similarity": 0.999,
        "max_abs_error": 0.1,
        "mean_abs_error": 0.01,
        "p99_abs_error": 0.05,
        "nan_count": 0,
        "inf_count": 0,
        "top1_agreement": 1.0,
    }


def _dense(case: str, projection: str, execution_form: str) -> dict[str, object]:
    return {
        "case": {"name": case},
        "projection": projection,
        "aliases": [projection],
        "placement": "qpu",
        "partition": None,
        "execution_form": execution_form,
        "numpy_openblas_int32_seconds": [0.002],
        "torch_seconds": [0.003],
        "numpy_openblas_fp32_seconds": [0.004],
        "torch_fp32_seconds": [0.005],
        "qpu_total_seconds": [0.001],
        "host_quantize_pack_seconds": [0.0001],
        "qpu_kernel_only_seconds": [0.0007],
        "dequantization_seconds": [0.0001] if execution_form != "fused-qpu-dequant" else [],
        "speedup_over_deployable_fp32_cpu": 4.0,
        "speedup_over_dynamic_w8a8_cpu": 2.0,
        "correctness": {"max_abs_error": 0.0},
        "quality": _quality(),
        "status": "supported-win",
        "_path": Path("dense.json"),
    }


def _yolo(case: str, axis: str) -> dict[str, object]:
    if axis == "outputs":
        return {
            "case": {"name": case},
            "placement": "hybrid",
            "partition": {"axis": "outputs"},
            "configuration": "hybrid-outputs",
            "status": "unsupported",
            "reason": "per-group alignment",
            "_path": Path("yolo.json"),
        }
    return {
        "case": {"name": case},
        "placement": "hybrid",
        "partition": {"axis": "rows", "qpu_units": 16, "total_units": 32},
        "configuration": "hybrid-r16",
        "numpy_openblas_dynamic_w8a8_seconds": [0.002],
        "torch_dynamic_w8a8_seconds": [0.003],
        "numpy_openblas_lowered_fp32_seconds": [0.004],
        "torch_native_fp32_seconds": [0.005],
        "whole_operation_seconds": [0.001],
        "hybrid_total_seconds": [0.001],
        "host_quantize_pack_seconds": [0.0001],
        "kernel_only_seconds": [0.0007],
        "dequantization_seconds": [],
        "speedup_over_deployable_fp32_cpu": 4.0,
        "speedup_over_dynamic_w8a8_cpu": 2.0,
        "correctness": {"max_abs_error": 0.0},
        "quality": _quality(),
        "status": "supported-win",
        "_path": Path("yolo.json"),
    }


def test_w8a8_matrix_integrity_requires_every_shape_partition_backend_and_quality() -> None:
    rows: list[dict[str, object]] = []
    for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout):
        for projection in ("hidden", "kv", "up", "down", "lm_head"):
            rows.extend(
                _dense(case.name, projection, epilogue)
                for epilogue in ("cpu-dequant", "standalone-qpu-dequant", "fused-qpu-dequant")
            )
            hybrid = _dense(case.name, projection, "fused-qpu-dequant")
            hybrid.update(
                placement="hybrid",
                partition={"axis": "outputs", "qpu_units": 16, "total_units": 32},
                hybrid_total_seconds=[0.001],
            )
            rows.append(hybrid)
    for case in ("square-64x64x64", "square-512x512x512"):
        rows.extend(
            _dense(case, "square", epilogue)
            for epilogue in ("cpu-dequant", "standalone-qpu-dequant", "fused-qpu-dequant")
        )
        for axis in ("rows", "outputs"):
            hybrid = _dense(case, "square", "fused-qpu-dequant")
            hybrid.update(
                placement="hybrid",
                partition={"axis": axis, "qpu_units": 16, "total_units": 32},
                hybrid_total_seconds=[0.001],
            )
            rows.append(hybrid)
    for case in (*YOLO_DETECTION_V1.tuning, *YOLO_DETECTION_V1.holdout):
        rows.extend((_yolo(case.name, "rows"), _yolo(case.name, "outputs")))

    for row in rows:
        _validate_result(row)
    _validate_coverage(rows)
    rendered = _render_evaluation(rows)

    assert "same-contract-win" in rendered
    assert "FP32 evaluation matrix" in rendered
    assert "outputs unsupported" in rendered


def test_w8a8_matrix_rejects_missing_raw_backend_or_quality_evidence() -> None:
    row = _dense("case", "hidden", "cpu-dequant")
    del row["torch_fp32_seconds"]
    with pytest.raises(ValueError, match="torch_fp32_seconds"):
        _validate_result(row)

    row = _dense("case", "hidden", "cpu-dequant")
    del row["quality"]
    with pytest.raises(ValueError, match="quality"):
        _validate_result(row)
