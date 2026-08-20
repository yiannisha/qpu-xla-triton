from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import calibrate_matmul, matmul, plan_matmul
from qpu_xla.scheduler import (
    CapabilityRegistry,
    CostModel,
    ExecutionCandidate,
    OperationSpec,
    PartitionSpec,
    Placement,
)


def test_registry_prefers_calibrated_median_and_persists_json(tmp_path) -> None:
    specification = OperationSpec("example", "int32", (16, 16), "row-major")
    registry = CapabilityRegistry(
        (
            ExecutionCandidate("cpu", Placement.CPU, lambda _: True, lambda _: 2.0),
            ExecutionCandidate("qpu", Placement.QPU, lambda _: True, lambda _: 1.0),
        )
    )
    model = CostModel({"device": "test"})
    model.record(specification, "cpu", 0.1)
    model.record(specification, "cpu", 0.3)
    model.record(specification, "qpu", 0.4)

    plan = registry.choose(specification, cost_model=model)
    assert plan.candidate.name == "cpu"
    assert plan.calibrated
    assert plan.estimated_seconds == pytest.approx(0.2)

    path = tmp_path / "cost-model.json"
    model.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["fingerprint"] == {"device": "test"}
    reloaded = CostModel.load(path)
    assert reloaded.fingerprint == {"device": "test"}
    assert registry.choose(specification, cost_model=reloaded).candidate.name == "cpu"


def test_matmul_planner_has_cpu_fallback_and_rejects_forced_unsupported_qpu() -> None:
    with Device.fake() as device:
        left = device.tensor((3, 5), np.int32)
        right = device.tensor((5, 7), np.int32)
        destination = device.tensor((3, 7), np.int32)

        assert plan_matmul(destination, left, right).candidate.placement is Placement.CPU
        with pytest.raises(ValueError, match="no qpu implementation"):
            plan_matmul(destination, left, right, placement=Placement.QPU)


def test_hybrid_candidates_require_calibration_before_auto_selection() -> None:
    specification = OperationSpec("matmul", "float32", (512, 512, 512), "row-major")
    partition = PartitionSpec("rows", 128, 512, 16)
    registry = CapabilityRegistry(
        (
            ExecutionCandidate("cpu", Placement.CPU, lambda _: True, lambda _: 1.0),
            ExecutionCandidate(
                "hybrid.rows.128",
                Placement.HYBRID,
                lambda _: True,
                lambda _: 0.5,
                partition,
                requires_calibration=True,
            ),
        )
    )

    assert registry.choose(specification).candidate.name == "cpu"
    assert registry.choose(specification, preference=Placement.HYBRID).candidate.partition == partition
    model = CostModel()
    model.record(specification, "cpu", 1.0)
    model.record(specification, "hybrid.rows.128", 0.5)
    assert registry.choose(specification, cost_model=model).candidate.name == "hybrid.rows.128"


def test_matmul_can_be_forced_to_the_cpu_path() -> None:
    with Device.fake() as device, device.queue() as queue:
        left = device.tensor((16, 4), np.int32)
        right = device.tensor((4, 16), np.int32)
        destination = device.tensor((16, 16), np.int32)
        left.numpy()[:] = 2
        right.numpy()[:] = 3

        matmul(destination, left, right, queue=queue, placement=Placement.CPU).wait()

        np.testing.assert_array_equal(destination.numpy(), np.full((16, 16), 24, dtype=np.int32))


def test_matmul_calibration_records_only_supported_variants() -> None:
    with Device.fake() as device, device.queue() as queue:
        left = device.tensor((3, 5), np.int32)
        right = device.tensor((5, 7), np.int32)
        destination = device.tensor((3, 7), np.int32)
        left.numpy()[:] = 2
        right.numpy()[:] = 3

        model = calibrate_matmul(destination, left, right, queue=queue, warmup=0, repeat=2)
        specification = OperationSpec("matmul", "int32", (3, 5, 7), "row-major")

        assert model.estimate(specification, "numpy.matmul") is not None
        assert model.estimate(specification, "vc7.tiled_int32_gemm") is None


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_matmul_calibration_records_a_real_qpu_sample() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        left = device.tensor((16, 4), np.int32)
        right = device.tensor((4, 16), np.int32)
        destination = device.tensor((16, 16), np.int32)
        left.numpy()[:] = 2
        right.numpy()[:] = 3

        model = calibrate_matmul(destination, left, right, queue=queue, warmup=0, repeat=1)
        specification = OperationSpec("matmul", "int32", (16, 4, 16), "row-major")

        assert model.estimate(specification, "vc7.tiled_int32_gemm") is not None
