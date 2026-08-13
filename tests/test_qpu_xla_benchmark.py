from __future__ import annotations

import json

import pytest

from qpu_xla import Device
from qpu_xla.benchmark import BenchmarkCategory, BenchmarkReport, collect_metadata


def test_benchmark_report_preserves_raw_categories_and_json(tmp_path) -> None:
    report = BenchmarkReport("unit", {"commit": "test"})
    sample = report.measure(BenchmarkCategory.NUMPY, lambda: sum(range(10)), warmup=0, repeat=2)

    assert len(sample.seconds) == 2
    assert sample.best_seconds > 0
    assert sample.median_seconds > 0
    with pytest.raises(ValueError, match="already recorded"):
        report.measure(BenchmarkCategory.NUMPY, lambda: None)

    path = tmp_path / "benchmark.json"
    report.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["name"] == "unit"
    assert saved["samples"][0]["category"] == "numpy"


def test_metadata_records_backend_and_caller_provenance() -> None:
    with Device.fake() as device:
        metadata = collect_metadata(device, extra={"commit": "abc"})

    assert metadata["backend"] == "FakeBackend"
    assert metadata["commit"] == "abc"
