"""Small deterministic whole-operator placement layer for CPU and QPU variants."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from statistics import median
from typing import Self, cast


class Placement(Enum):
    """A requested or selected whole-operator execution strategy."""

    AUTO = "auto"
    CPU = "cpu"
    QPU = "qpu"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """Disjoint output partition assigned to the QPU side of a hybrid plan."""

    axis: str
    qpu_units: int
    total_units: int
    alignment: int = 1

    def __post_init__(self: Self) -> None:
        """Reject empty, overlapping, or misaligned partition descriptions."""
        if not self.axis or self.total_units <= 0 or self.alignment <= 0:
            raise ValueError("hybrid partitions require an axis and positive sizes")
        if self.qpu_units <= 0 or self.qpu_units >= self.total_units:
            raise ValueError("hybrid partitions must assign non-empty work to both CPU and QPU")
        if self.qpu_units % self.alignment:
            raise ValueError("QPU partition units must satisfy the kernel alignment")


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Normalized operator identity used for capabilities and cost samples."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    layout: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionCandidate:
    """One executable implementation with a capability check and fallback cost."""

    name: str
    placement: Placement
    supported: Callable[[OperationSpec], bool]
    estimate_seconds: Callable[[OperationSpec], float]
    partition: PartitionSpec | None = None
    requires_calibration: bool = False

    def __post_init__(self: Self) -> None:
        """Reject nonsensical registry entries before scheduling can use them."""
        if self.placement is Placement.AUTO:
            raise ValueError("execution candidates must name CPU or QPU placement")
        if self.placement is Placement.HYBRID and self.partition is None:
            raise ValueError("hybrid execution candidates require a partition")
        if self.placement is not Placement.HYBRID and self.partition is not None:
            raise ValueError("only hybrid execution candidates may carry a partition")


@dataclass(frozen=True, slots=True)
class PlannedExecution:
    """The deterministic selected implementation and its estimated cost."""

    candidate: ExecutionCandidate
    estimated_seconds: float
    calibrated: bool


@dataclass(slots=True)
class CostModel:
    """Median timing samples, serializable per device/software fingerprint."""

    fingerprint: dict[str, str] = field(default_factory=dict)
    _samples: dict[tuple[str, str, tuple[int, ...], str, str], list[float]] = field(default_factory=dict)

    @staticmethod
    def _key(specification: OperationSpec, candidate_name: str) -> tuple[str, str, tuple[int, ...], str, str]:
        """Build a fully normalized in-memory sample key."""
        return (specification.name, specification.dtype, specification.shape, specification.layout, candidate_name)

    def record(self: Self, specification: OperationSpec, candidate_name: str, seconds: float) -> None:
        """Add one positive benchmark sample for a specific implementation."""
        if seconds <= 0:
            raise ValueError("cost samples must be positive seconds")
        values = self._samples.setdefault(self._key(specification, candidate_name), [])
        values.append(seconds)

    def estimate(self: Self, specification: OperationSpec, candidate_name: str) -> float | None:
        """Return the robust median sample for this exact operator identity."""
        values = self._samples.get(self._key(specification, candidate_name))
        return None if not values else float(median(values))

    def to_dict(self: Self) -> dict[str, object]:
        """Serialize this model without relying on opaque tuple-key encodings."""
        records: list[dict[str, object]] = []
        for (name, dtype, shape, layout, candidate), values in self._samples.items():
            records.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "layout": layout,
                    "candidate": candidate,
                    "seconds": values,
                }
            )
        return {"fingerprint": self.fingerprint, "records": records}

    def save(self: Self, path: str | PathLike[str]) -> None:
        """Write a portable JSON cost model for future runs on the same device."""
        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, indent=2, sort_keys=True)

    @classmethod
    def load(cls: type[Self], path: str | PathLike[str]) -> Self:
        """Load and validate a JSON model produced by :meth:`save`."""
        with open(path, encoding="utf-8") as source:
            payload = cast(object, json.load(source))
        if not isinstance(payload, dict):
            raise ValueError("cost model JSON must contain an object")
        fingerprint = payload.get("fingerprint")
        records = payload.get("records")
        valid_fingerprint = isinstance(fingerprint, dict) and all(
            isinstance(key, str) and isinstance(value, str) for key, value in fingerprint.items()
        )
        if not valid_fingerprint:
            raise ValueError("cost model fingerprint must map strings to strings")
        if not isinstance(records, list):
            raise ValueError("cost model records must be a list")
        model = cls(dict(cast(dict[str, str], fingerprint)))
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("cost model record must be an object")
            name, dtype, shape, layout, candidate, seconds = (
                record.get("name"),
                record.get("dtype"),
                record.get("shape"),
                record.get("layout"),
                record.get("candidate"),
                record.get("seconds"),
            )
            if (
                not all(isinstance(value, str) for value in (name, dtype, layout, candidate))
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value > 0 for value in shape)
                or not isinstance(seconds, list)
                or not all(isinstance(value, int | float) and value > 0 for value in seconds)
            ):
                raise ValueError("cost model record has invalid fields")
            specification = OperationSpec(
                cast(str, name),
                cast(str, dtype),
                tuple(cast(list[int], shape)),
                cast(str, layout),
            )
            for seconds_value in cast(list[int | float], seconds):
                model.record(specification, cast(str, candidate), float(seconds_value))
        return model


class CapabilityRegistry:
    """Select the cheapest supported whole-operator implementation."""

    def __init__(self: Self, candidates: Iterable[ExecutionCandidate] = ()) -> None:
        """Create an empty or pre-populated deterministic capability registry."""
        self._candidates: list[ExecutionCandidate] = []
        for candidate in candidates:
            self.register(candidate)

    def register(self: Self, candidate: ExecutionCandidate) -> None:
        """Add a unique named implementation to the registry."""
        if any(existing.name == candidate.name for existing in self._candidates):
            raise ValueError(f"candidate {candidate.name!r} is already registered")
        self._candidates.append(candidate)

    def choose(
        self: Self,
        specification: OperationSpec,
        *,
        preference: Placement = Placement.AUTO,
        cost_model: CostModel | None = None,
    ) -> PlannedExecution:
        """Choose a supported candidate by calibrated median or fallback estimate."""
        eligible = [
            candidate
            for candidate in self._candidates
            if candidate.supported(specification)
            and (preference is Placement.AUTO or candidate.placement is preference)
            and not (
                preference is Placement.AUTO
                and candidate.requires_calibration
                and (cost_model is None or cost_model.estimate(specification, candidate.name) is None)
            )
        ]
        if not eligible:
            raise ValueError(f"no {preference.value} implementation supports {specification.name!r}")

        scored: list[tuple[float, str, ExecutionCandidate, bool]] = []
        for candidate in eligible:
            measured = None if cost_model is None else cost_model.estimate(specification, candidate.name)
            estimate = candidate.estimate_seconds(specification) if measured is None else measured
            if estimate <= 0:
                raise ValueError(f"candidate {candidate.name!r} produced a non-positive cost estimate")
            scored.append((estimate, candidate.name, candidate, measured is not None))
        estimate, _, candidate, calibrated = min(scored, key=lambda score: (score[0], score[1]))
        return PlannedExecution(candidate, estimate, calibrated)
