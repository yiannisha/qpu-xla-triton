"""Persistent lifecycle records for supported, slower, and quarantined kernels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike
from statistics import median
from typing import Self, cast


class CandidateStatus(Enum):
    """Whether a candidate is dispatchable, retained for tuning, or unsafe."""

    SUPPORTED_WIN = "supported-win"
    SAME_CONTRACT_WIN = "same-contract-win"
    EXPERIMENTAL_CORRECT_SLOWER = "experimental-correct-slower"
    QUARANTINED_INCORRECT = "quarantined-incorrect"
    REFERENCE_ONLY = "reference-only"


@dataclass(frozen=True, slots=True)
class CorrectnessEvidence:
    """Numerical evidence retained with one candidate evaluation."""

    reference: str
    cases: int
    exact: bool
    max_abs_error: float
    mean_abs_error: float
    p99_abs_error: float
    max_relative_error: float
    saturation_count: int = 0
    nan_count: int = 0
    inf_count: int = 0

    def __post_init__(self: Self) -> None:
        """Reject incomplete or nonsensical numerical evidence."""
        if not self.reference or self.cases <= 0:
            raise ValueError("correctness evidence requires a reference and at least one case")
        errors = (self.max_abs_error, self.mean_abs_error, self.p99_abs_error, self.max_relative_error)
        counts = (self.saturation_count, self.nan_count, self.inf_count)
        if any(value < 0 for value in errors) or any(value < 0 for value in counts):
            raise ValueError("correctness errors and counts cannot be negative")


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    """Raw whole-operation samples for deployable and same-contract baselines."""

    cpu_reference: str
    cpu_seconds: tuple[float, ...]
    candidate_seconds: tuple[float, ...]
    fused_model_win: bool = False
    host_prep_seconds: tuple[float, ...] = ()
    kernel_seconds: tuple[float, ...] = ()
    cold_start_seconds: tuple[float, ...] = ()
    dequantization_seconds: tuple[float, ...] = ()
    same_contract_reference: str = ""
    same_contract_seconds: tuple[float, ...] = ()

    def __post_init__(self: Self) -> None:
        """Reject empty references and invalid timing samples."""
        if not self.cpu_reference or not self.cpu_seconds or not self.candidate_seconds:
            raise ValueError("performance evidence requires both CPU and candidate samples")
        samples = (
            *self.cpu_seconds,
            *self.candidate_seconds,
            *self.host_prep_seconds,
            *self.kernel_seconds,
            *self.cold_start_seconds,
            *self.dequantization_seconds,
            *self.same_contract_seconds,
        )
        if any(value <= 0 for value in samples):
            raise ValueError("performance samples must be positive")
        if bool(self.same_contract_reference) != bool(self.same_contract_seconds):
            raise ValueError("same-contract performance requires both a reference and samples")

    @property
    def cpu_median_seconds(self: Self) -> float:
        """Return the median fastest-reference wall time."""
        return float(median(self.cpu_seconds))

    @property
    def candidate_median_seconds(self: Self) -> float:
        """Return the median candidate wall time."""
        return float(median(self.candidate_seconds))

    @property
    def speedup(self: Self) -> float:
        """Return CPU median divided by candidate median."""
        return self.cpu_median_seconds / self.candidate_median_seconds

    @property
    def measured_win(self: Self) -> bool:
        """Return whether the candidate clears the v1 five-percent gate."""
        return self.speedup >= 1.05 or self.fused_model_win

    @property
    def same_contract_speedup(self: Self) -> float | None:
        """Return same-contract CPU median divided by candidate median when recorded."""
        if not self.same_contract_seconds:
            return None
        return float(median(self.same_contract_seconds)) / self.candidate_median_seconds

    @property
    def same_contract_win(self: Self) -> bool:
        """Return whether the candidate clears the same-contract five-percent gate."""
        speedup = self.same_contract_speedup
        return speedup is not None and speedup >= 1.05


@dataclass(frozen=True, slots=True)
class PartitionEvidence:
    """Serializable CPU/QPU split used by one measured hybrid candidate."""

    axis: str
    qpu_units: int
    total_units: int
    alignment: int = 1

    def __post_init__(self: Self) -> None:
        """Apply the same non-overlap rules as the runtime partition contract."""
        if not self.axis or self.total_units <= 0 or self.alignment <= 0:
            raise ValueError("partition evidence requires an axis and positive sizes")
        if self.qpu_units <= 0 or self.qpu_units >= self.total_units:
            raise ValueError("partition evidence must retain work on both CPU and QPU")
        if self.qpu_units % self.alignment:
            raise ValueError("partition evidence does not satisfy its alignment")


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    """Error of a quantized format against its FP32 source-domain reference."""

    reference: str
    normalized_rmse: float
    cosine_similarity: float
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    p99_abs_error: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    top1_agreement: float | None = None

    def __post_init__(self: Self) -> None:
        """Reject incomplete or invalid quality observations."""
        errors = (self.normalized_rmse, self.max_abs_error, self.mean_abs_error, self.p99_abs_error)
        if not self.reference or any(value < 0 for value in errors):
            raise ValueError("quality evidence requires a reference and non-negative NRMSE")
        if not -1.0 <= self.cosine_similarity <= 1.0:
            raise ValueError("quality cosine similarity must be within [-1, 1]")
        if self.nan_count < 0 or self.inf_count < 0:
            raise ValueError("quality NaN and Inf counts cannot be negative")
        if self.top1_agreement is not None and not 0.0 <= self.top1_agreement <= 1.0:
            raise ValueError("quality top-1 agreement must be within [0, 1]")

    @property
    def passes_default_gate(self: Self) -> bool:
        """Return whether quantized output clears the default quality gate."""
        return (
            self.normalized_rmse <= 0.10
            and self.cosine_similarity >= 0.99
            and self.nan_count == 0
            and self.inf_count == 0
        )


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Complete retained status for one kernel and shape-class specialization."""

    name: str
    operation: str
    dtype: str
    layout: str
    shape_class: str
    source_hash: str
    status: CandidateStatus
    correctness: CorrectnessEvidence | None = None
    performance: PerformanceEvidence | None = None
    reason: str = ""
    placement: str = "qpu"
    kernels: tuple[str, ...] = ()
    partition: PartitionEvidence | None = None
    quality: QualityEvidence | None = None

    def __post_init__(self: Self) -> None:
        """Enforce evidence required by each lifecycle state."""
        identity = (self.name, self.operation, self.dtype, self.layout, self.shape_class, self.source_hash)
        if any(not value for value in identity):
            raise ValueError("candidate identity fields must be non-empty")
        if self.placement not in {"cpu", "qpu", "hybrid"}:
            raise ValueError("candidate placement must be cpu, qpu, or hybrid")
        if self.placement == "hybrid" and self.partition is None:
            raise ValueError("hybrid candidate records require partition evidence")
        if self.placement != "hybrid" and self.partition is not None:
            raise ValueError("only hybrid candidate records may include partition evidence")
        if self.status is CandidateStatus.SUPPORTED_WIN:
            if self.correctness is None or self.performance is None or not self.performance.measured_win:
                raise ValueError("supported candidates require correctness and a measured or fused win")
            if self.correctness.nan_count or self.correctness.inf_count:
                raise ValueError("supported candidates cannot produce NaN or Inf outputs")
        if self.status is CandidateStatus.SAME_CONTRACT_WIN:
            if self.correctness is None or self.performance is None or not self.performance.same_contract_win:
                raise ValueError("same-contract winners require correctness and a measured W8A8 CPU win")
            if self.correctness.nan_count or self.correctness.inf_count:
                raise ValueError("same-contract winners cannot produce NaN or Inf outputs")
        if self.status is CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER and self.correctness is None:
            raise ValueError("correct experimental candidates require correctness evidence")
        if self.status is CandidateStatus.QUARANTINED_INCORRECT and not self.reason:
            raise ValueError("quarantined candidates require a failure reason")


class CandidateRegistry:
    """Serializable candidate archive that exposes only measured winners."""

    def __init__(self: Self, records: tuple[CandidateRecord, ...] = ()) -> None:
        """Create a registry and reject duplicate candidate names."""
        self._records: dict[str, CandidateRecord] = {}
        for record in records:
            self.register(record)

    @property
    def records(self: Self) -> tuple[CandidateRecord, ...]:
        """Return every retained candidate in registration order."""
        return tuple(self._records.values())

    @property
    def supported(self: Self) -> tuple[CandidateRecord, ...]:
        """Return only candidates eligible for automatic placement."""
        return tuple(record for record in self._records.values() if record.status is CandidateStatus.SUPPORTED_WIN)

    def supported_for(
        self: Self,
        *,
        operation: str,
        dtype: str,
        layout: str,
        shape_class: str,
    ) -> tuple[CandidateRecord, ...]:
        """Return scheduler-visible winners matching one exact operation class."""
        return tuple(
            record
            for record in self.supported
            if record.operation == operation
            and record.dtype == dtype
            and record.layout == layout
            and record.shape_class == shape_class
        )

    def register(self: Self, record: CandidateRecord) -> None:
        """Retain one uniquely named candidate record."""
        if record.name in self._records:
            raise ValueError(f"candidate {record.name!r} is already registered")
        self._records[record.name] = record

    @classmethod
    def combine(cls: type[Self], *registries: CandidateRegistry) -> Self:
        """Combine disjoint benchmark archives into one placement registry."""
        return cls(tuple(record for registry in registries for record in registry.records))

    def to_dict(self: Self) -> dict[str, object]:
        """Return the stable versioned serialization payload."""
        payload: list[dict[str, object]] = []
        for record in self.records:
            item = asdict(record)
            item["status"] = record.status.value
            payload.append(item)
        return {"schema_version": 3, "records": payload}

    def save(self: Self, path: str | PathLike[str]) -> None:
        """Persist every candidate, including slower and quarantined ones."""
        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, indent=2, sort_keys=True)

    @classmethod
    def load(cls: type[Self], path: str | PathLike[str]) -> Self:
        """Load and validate a supported candidate-archive schema."""
        with open(path, encoding="utf-8") as source:
            payload = cast(object, json.load(source))
        if not isinstance(payload, dict) or payload.get("schema_version") not in (1, 2, 3):
            raise ValueError("unsupported candidate registry schema")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("candidate registry records must be a list")
        records: list[CandidateRecord] = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ValueError("candidate registry records must be objects")
            item = dict(raw)
            try:
                item["status"] = CandidateStatus(item["status"])
                correctness = item.get("correctness")
                performance = item.get("performance")
                if correctness is not None:
                    item["correctness"] = CorrectnessEvidence(**correctness)
                if performance is not None:
                    performance = dict(performance)
                    for field_name in (
                        "cpu_seconds",
                        "candidate_seconds",
                        "host_prep_seconds",
                        "kernel_seconds",
                        "cold_start_seconds",
                        "dequantization_seconds",
                        "same_contract_seconds",
                    ):
                        if field_name in performance:
                            performance[field_name] = tuple(performance[field_name])
                    item["performance"] = PerformanceEvidence(**performance)
                partition = item.get("partition")
                quality = item.get("quality")
                if partition is not None:
                    item["partition"] = PartitionEvidence(**partition)
                if quality is not None:
                    item["quality"] = QualityEvidence(**quality)
                if "kernels" in item:
                    item["kernels"] = tuple(item["kernels"])
                records.append(CandidateRecord(**item))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("candidate registry contains an invalid record") from exc
        return cls(tuple(records))


__all__ = [
    "CandidateRecord",
    "CandidateRegistry",
    "CandidateStatus",
    "CorrectnessEvidence",
    "PerformanceEvidence",
    "PartitionEvidence",
    "QualityEvidence",
]
