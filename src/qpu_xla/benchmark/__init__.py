"""Reproducible benchmark metadata, timing categories, and JSON reporting."""

from qpu_xla.benchmark.candidates import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PartitionEvidence,
    PerformanceEvidence,
    QualityEvidence,
)
from qpu_xla.benchmark.core import (
    BenchmarkCategory,
    BenchmarkReport,
    BenchmarkSample,
    collect_metadata,
    numpy_backend_label,
    numpy_blas_metadata,
)

__all__ = [
    "BenchmarkCategory",
    "BenchmarkReport",
    "BenchmarkSample",
    "CandidateRecord",
    "CandidateRegistry",
    "CandidateStatus",
    "CorrectnessEvidence",
    "PerformanceEvidence",
    "PartitionEvidence",
    "QualityEvidence",
    "collect_metadata",
    "numpy_backend_label",
    "numpy_blas_metadata",
]
