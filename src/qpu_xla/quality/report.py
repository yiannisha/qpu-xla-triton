"""Versioned machine-readable model-quality report contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class RunIntegrity:
    """Execution integrity only; deliberately not a task-quality gate."""

    expected_samples: int
    processed_samples: int
    finite_outputs: bool
    qpu_dispatches: int = 0
    cpu_operations: int = 0
    expected_fallbacks: int = 0
    unexpected_fallbacks: int = 0
    requires_qpu: bool = False
    requires_cpu: bool = False
    errors: tuple[str, ...] = ()

    @property
    def valid_run(self: Self) -> bool:
        """Return whether the comparison is complete and honestly attributed."""
        return (
            self.expected_samples == self.processed_samples
            and self.expected_samples > 0
            and self.finite_outputs
            and self.unexpected_fallbacks == 0
            and (not self.requires_qpu or self.qpu_dispatches > 0)
            and (not self.requires_cpu or self.cpu_operations > 0)
            and not self.errors
        )


@dataclass(frozen=True, slots=True)
class QualityReport:
    """CPU-versus-candidate measurements with provenance and raw-artifact links."""

    task: str
    baseline: str
    candidate: str
    metrics: dict[str, Any]
    integrity: RunIntegrity
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    format: str = "qpu-model-quality-v1"

    def to_dict(self: Self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        payload = asdict(self)
        payload["integrity"]["valid_run"] = self.integrity.valid_run
        return payload

    def save(self: Self, path: str | PathLike[str]) -> None:
        """Atomically save the report without introducing a quality verdict."""
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        temporary = location.with_suffix(location.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(location)


__all__ = ["QualityReport", "RunIntegrity"]
