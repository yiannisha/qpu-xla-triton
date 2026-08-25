"""Whole-action numerical gates for SmolVLA benchmark reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Self

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class SmolVLAActionMetrics:
    """Error and directional agreement over one or more complete action chunks."""

    max_abs_error: float
    mean_abs_error: float
    p99_abs_error: float
    normalized_rmse: float
    cosine_similarity: float
    nan_count: int
    inf_count: int

    @classmethod
    def calculate(
        cls: type[SmolVLAActionMetrics],
        actual: npt.NDArray[np.float32],
        expected: npt.NDArray[np.float32],
    ) -> SmolVLAActionMetrics:
        """Calculate stable FP64 statistics over equal FP32 action arrays."""
        if actual.dtype != np.dtype(np.float32) or expected.dtype != np.dtype(np.float32):
            raise ValueError("SmolVLA action metrics require FP32 arrays")
        if actual.shape != expected.shape or actual.size == 0:
            raise ValueError("SmolVLA action metrics require equal non-empty arrays")
        actual64 = actual.astype(np.float64)
        expected64 = expected.astype(np.float64)
        finite_actual = np.nan_to_num(actual64)
        finite_expected = np.nan_to_num(expected64)
        difference = finite_actual - finite_expected
        absolute = np.abs(difference)
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        reference_rms = float(np.sqrt(np.mean(np.square(finite_expected))))
        denominator = float(np.linalg.norm(finite_actual.ravel()) * np.linalg.norm(finite_expected.ravel()))
        cosine = (
            1.0 if denominator == 0.0 else float(np.dot(finite_actual.ravel(), finite_expected.ravel()) / denominator)
        )
        return cls(
            max_abs_error=float(np.max(absolute, initial=0.0)),
            mean_abs_error=float(np.mean(absolute)),
            p99_abs_error=float(np.percentile(absolute, 99)),
            normalized_rmse=rmse / max(reference_rms, 1e-12),
            cosine_similarity=max(-1.0, min(1.0, cosine)),
            nan_count=int(np.count_nonzero(np.isnan(actual))),
            inf_count=int(np.count_nonzero(np.isinf(actual))),
        )

    @property
    def passes_fp32_upstream(self: Self) -> bool:
        """Apply the full-action FP32 versus upstream PyTorch gate."""
        return (
            self.normalized_rmse <= 1e-3
            and self.cosine_similarity >= 0.9999
            and self.nan_count == 0
            and self.inf_count == 0
        )

    @property
    def passes_w8a8_contract(self: Self) -> bool:
        """Apply the QPU versus dynamic-W8A8 CPU contract gate."""
        return self.max_abs_error <= 1e-4 and self.nan_count == 0 and self.inf_count == 0

    @property
    def passes_w8a8_quality(self: Self) -> bool:
        """Apply the W8A8 versus FP32 full-action quality gate."""
        return (
            self.normalized_rmse <= 0.10
            and self.cosine_similarity >= 0.99
            and self.nan_count == 0
            and self.inf_count == 0
        )

    def to_dict(self: Self) -> dict[str, float | int]:
        """Return a JSON-compatible representation."""
        return asdict(self)


__all__ = ["SmolVLAActionMetrics"]
