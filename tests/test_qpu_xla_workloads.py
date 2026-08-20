from __future__ import annotations

import json

import pytest

from qpu_xla.workloads import LLAMA_DENSE_V1, YOLO_DETECTION_V1, LlamaWorkload, WorkloadManifest


def test_versioned_workload_manifests_separate_tuning_and_holdout_shapes() -> None:
    assert LLAMA_DENSE_V1.version == YOLO_DETECTION_V1.version == 1
    assert {case.name for case in LLAMA_DENSE_V1.tuning}.isdisjoint(
        {case.name for case in LLAMA_DENSE_V1.holdout}
    )
    assert any(case.phase == "decode" and case.cache_length == 2048 for case in LLAMA_DENSE_V1.tuning)
    assert any(case.groups == case.in_channels for case in YOLO_DETECTION_V1.tuning)
    assert json.loads(json.dumps(LLAMA_DENSE_V1.to_dict()))["name"] == "llama-dense"


def test_workload_manifests_reject_invalid_topologies_or_mixed_families() -> None:
    with pytest.raises(ValueError, match="exactly one token"):
        LlamaWorkload("bad", "decode", 2, 512, 1536, 8, 1, 64, 2, 32_000)
    with pytest.raises(ValueError, match="cannot mix"):
        WorkloadManifest("mixed", 1, (LLAMA_DENSE_V1.tuning[0],), (YOLO_DETECTION_V1.holdout[0],))
