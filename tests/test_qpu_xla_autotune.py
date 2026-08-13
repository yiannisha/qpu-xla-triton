from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.autotune import (
    CandidateCase,
    CandidateLimits,
    CandidateResultStore,
    CandidateRunner,
    CandidateSource,
    CandidateStatus,
)

_COPY_SOURCE = """
def generated_copy(source, destination):
    offsets = arange(0, 16)
    values = load(source + offsets)
    store(destination + offsets, values)
"""


def test_candidate_runner_accepts_a_canonical_candidate_without_executing_source(tmp_path) -> None:
    case = CandidateCase((np.arange(16, dtype=np.int32),))
    store = CandidateResultStore(tmp_path / "results.jsonl")

    result = CandidateRunner().evaluate(CandidateSource(_COPY_SOURCE), case, store=store)

    assert result.status is CandidateStatus.ACCEPTED
    assert result.kernel_name == "vc7.copy_words"
    assert result.cpu_reference_passed
    assert result.qpu_differential_passed is None
    assert result.qpu_median_seconds is None
    saved = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert saved["status"] == "accepted"
    assert saved["qpu_differential_passed"] is None


def test_candidate_runner_rejects_source_that_tries_to_call_python_or_exceeds_limits() -> None:
    case = CandidateCase((np.arange(16, dtype=np.int32),))

    unsafe = CandidateRunner().evaluate(
        CandidateSource("def generated(source, destination):\n    open('unsafe', 'w')"), case
    )
    oversized_runner = CandidateRunner(limits=CandidateLimits(max_source_bytes=4))
    oversized = oversized_runner.evaluate(CandidateSource(_COPY_SOURCE), case)
    node_limited_runner = CandidateRunner(limits=CandidateLimits(max_ast_nodes=3))
    node_limited = node_limited_runner.evaluate(CandidateSource(_COPY_SOURCE), case)

    assert unsafe.status is CandidateStatus.REJECTED
    assert "only QPU-XLA DSL primitives" in (unsafe.diagnostic or "")
    assert oversized.status is CandidateStatus.REJECTED
    assert "byte limit" in (oversized.diagnostic or "")
    assert node_limited.status is CandidateStatus.REJECTED
    assert "node limit" in (node_limited.diagnostic or "")


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_candidate_runner_hardware_differential_and_timing_for_canonical_copy() -> None:
    case = CandidateCase((np.arange(16, dtype=np.int32),))
    with Device.open(data_area_size=1024 * 1024) as device:
        result = CandidateRunner().evaluate(CandidateSource(_COPY_SOURCE), case, device=device, timeout_seconds=10.0)

    assert result.status is CandidateStatus.ACCEPTED
    assert result.cpu_reference_passed
    assert result.qpu_differential_passed
    assert result.qpu_median_seconds is not None
    assert result.qpu_median_seconds > 0
