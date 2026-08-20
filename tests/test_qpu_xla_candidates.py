from __future__ import annotations

import pytest

from qpu_xla.benchmark import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PartitionEvidence,
    PerformanceEvidence,
    QualityEvidence,
)


def _correctness() -> CorrectnessEvidence:
    return CorrectnessEvidence("numpy-int64", 8, True, 0.0, 0.0, 0.0, 0.0)


def test_candidate_registry_round_trips_winners_and_slower_candidates(tmp_path) -> None:
    winning = CandidateRecord(
        "packed-gemm-h2048",
        "w8a8-gemm",
        "int8-int32",
        "packed-k4",
        "llama-h2048-prefill",
        "abc123",
        CandidateStatus.SUPPORTED_WIN,
        _correctness(),
        PerformanceEvidence("torch", (0.010, 0.011), (0.008, 0.009)),
    )
    slower = CandidateRecord(
        "packed-gemv-h2048",
        "w8a8-gemv",
        "int8-int32",
        "packed-k4",
        "llama-h2048-decode",
        "def456",
        CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER,
        _correctness(),
        PerformanceEvidence("numpy", (0.001,), (0.002,)),
        "correct but slower on this shape",
    )
    registry = CandidateRegistry((winning, slower))
    path = tmp_path / "candidates.json"
    registry.save(path)
    restored = CandidateRegistry.load(path)

    assert [record.name for record in restored.supported] == ["packed-gemm-h2048"]
    assert [record.name for record in restored.records] == ["packed-gemm-h2048", "packed-gemv-h2048"]
    assert restored.supported_for(
        operation="w8a8-gemm",
        dtype="int8-int32",
        layout="packed-k4",
        shape_class="llama-h2048-prefill",
    ) == (winning,)
    assert not restored.supported_for(
        operation="w8a8-gemv",
        dtype="int8-int32",
        layout="packed-k4",
        shape_class="llama-h2048-decode",
    )
    combined = CandidateRegistry.combine(CandidateRegistry((winning,)), CandidateRegistry((slower,)))
    assert combined.records == (winning, slower)


def test_candidate_registry_round_trips_hybrid_partition_and_quality(tmp_path) -> None:
    record = CandidateRecord(
        "w8a8-hybrid",
        "linear",
        "w8a8-i32-fp32",
        "row-major-packed-k4",
        "64x1024x2816",
        "source",
        CandidateStatus.SUPPORTED_WIN,
        _correctness(),
        PerformanceEvidence(
            "numpy-fp32",
            (0.010,),
            (0.008,),
            host_prep_seconds=(0.001,),
            kernel_seconds=(0.005,),
        ),
        placement="hybrid",
        kernels=("numpy.matmul", "vc7.tiled_w8a8_gemm"),
        partition=PartitionEvidence("rows", 16, 64, 16),
        quality=QualityEvidence("numpy-fp32", 0.08, 0.995),
    )
    registry = CandidateRegistry((record,))
    path = tmp_path / "hybrid.json"
    registry.save(path)
    restored = CandidateRegistry.load(path)

    assert restored.records == (record,)
    assert restored.records[0].quality is not None
    assert restored.records[0].quality.passes_default_gate


def test_candidate_registry_rejects_unsupported_promotion_without_a_win() -> None:
    with pytest.raises(ValueError, match="measured or fused win"):
        CandidateRecord(
            "slow",
            "gemm",
            "int8",
            "packed",
            "shape",
            "hash",
            CandidateStatus.SUPPORTED_WIN,
            _correctness(),
            PerformanceEvidence("torch", (0.001,), (0.002,)),
        )


def test_candidate_registry_distinguishes_same_contract_from_deployable_wins(tmp_path) -> None:
    record = CandidateRecord(
        "same-contract-only",
        "linear",
        "w8a8-i32-fp32",
        "row-major-packed-k4",
        "64x64x64",
        "hash",
        CandidateStatus.SAME_CONTRACT_WIN,
        _correctness(),
        PerformanceEvidence(
            "torch-fp32",
            (0.0005,),
            (0.001,),
            dequantization_seconds=(0.0001,),
            same_contract_reference="numpy-int32-dynamic-w8a8",
            same_contract_seconds=(0.002,),
        ),
        quality=QualityEvidence(
            "numpy-fp32",
            0.02,
            0.999,
            max_abs_error=0.1,
            mean_abs_error=0.01,
            p99_abs_error=0.05,
            top1_agreement=1.0,
        ),
    )
    registry = CandidateRegistry((record,))
    path = tmp_path / "same-contract.json"
    registry.save(path)
    restored = CandidateRegistry.load(path)

    assert restored.records == (record,)
    assert restored.supported == ()
    assert restored.records[0].performance is not None
    assert restored.records[0].performance.same_contract_win
    assert restored.records[0].performance.same_contract_speedup == 2.0
