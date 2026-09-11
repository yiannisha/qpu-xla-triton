"""Task-quality metrics and deterministic paired confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """One observed paired delta and its percentile bootstrap interval."""

    delta: float
    low: float
    high: float
    confidence: float = 0.95
    replicates: int = 10_000
    seed: int = 20_260_910

    def to_dict(self: Self) -> dict[str, float | int]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def paired_bootstrap_delta(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    statistic: Callable[[npt.NDArray[np.float64]], float] = np.mean,
    groups: Sequence[str | int] | None = None,
    replicates: int = 10_000,
    seed: int = 20_260_910,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap candidate-minus-baseline on matched samples or matched groups.

    Supplying ``groups`` performs a cluster bootstrap. Every selected group
    contributes all of its observations, which is appropriate for correlated
    robot observations from the same episode.
    """
    before = np.asarray(baseline, dtype=np.float64)
    after = np.asarray(candidate, dtype=np.float64)
    if before.ndim != 1 or after.shape != before.shape or before.size == 0:
        raise ValueError("paired bootstrap requires equal, non-empty one-dimensional samples")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap replicates and confidence are invalid")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("paired bootstrap samples must be finite")
    differences = after - before
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    if groups is None:
        for index in range(replicates):
            selected = rng.integers(0, differences.size, size=differences.size)
            samples[index] = statistic(differences[selected])
    else:
        group_values = np.asarray(tuple(str(value) for value in groups))
        if group_values.shape != before.shape:
            raise ValueError("bootstrap groups must match the paired samples")
        unique = np.unique(group_values)
        members = {group: np.flatnonzero(group_values == group) for group in unique}
        for index in range(replicates):
            chosen = rng.choice(unique, size=unique.size, replace=True)
            selected = np.concatenate([members[group] for group in chosen])
            samples[index] = statistic(differences[selected])
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        delta=float(statistic(differences)),
        low=float(np.quantile(samples, tail)),
        high=float(np.quantile(samples, 1.0 - tail)),
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )


def paired_stratified_bootstrap_delta(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    groups: Sequence[str | int],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_910,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap matched observations within every fixed stratum."""
    before = np.asarray(baseline, dtype=np.float64)
    after = np.asarray(candidate, dtype=np.float64)
    group_values = np.asarray(tuple(str(value) for value in groups))
    if before.ndim != 1 or after.shape != before.shape or group_values.shape != before.shape or before.size == 0:
        raise ValueError("stratified bootstrap inputs must be equal non-empty vectors")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap replicates and confidence are invalid")
    differences = after - before
    members = [np.flatnonzero(group_values == group) for group in np.unique(group_values)]
    rng = np.random.default_rng(seed)
    samples = np.zeros(replicates, dtype=np.float64)
    chunk_size = 1_024
    for indices in members:
        values = differences[indices]
        for start in range(0, replicates, chunk_size):
            stop = min(start + chunk_size, replicates)
            selected = rng.integers(0, values.size, size=(stop - start, values.size))
            samples[start:stop] += np.sum(values[selected], axis=1)
    samples *= np.float64(1.0 / differences.size)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        delta=float(np.mean(differences)),
        low=float(np.quantile(samples, tail)),
        high=float(np.quantile(samples, 1.0 - tail)),
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )


def classification_metrics(
    baseline_logits: npt.ArrayLike,
    candidate_logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    groups: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Compare complete classifier logits and paired top-1/top-5 decisions."""
    baseline = np.asarray(baseline_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    expected = np.asarray(labels, dtype=np.int64)
    if baseline.ndim != 2 or candidate.shape != baseline.shape or expected.shape != (baseline.shape[0],):
        raise ValueError("classification arrays have incompatible shapes")
    if baseline.shape[1] < 5 or not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("classification logits must be finite and contain at least five classes")
    baseline_top5 = np.argpartition(baseline, -5, axis=1)[:, -5:]
    candidate_top5 = np.argpartition(candidate, -5, axis=1)[:, -5:]
    baseline_top1 = np.argmax(baseline, axis=1)
    candidate_top1 = np.argmax(candidate, axis=1)
    baseline_correct1 = baseline_top1 == expected
    candidate_correct1 = candidate_top1 == expected
    baseline_correct5 = np.any(baseline_top5 == expected[:, None], axis=1)
    candidate_correct5 = np.any(candidate_top5 == expected[:, None], axis=1)
    difference = candidate - baseline
    reference_rms = float(np.sqrt(np.mean(np.square(baseline))))
    denominator = float(np.linalg.norm(baseline) * np.linalg.norm(candidate))
    top1_delta = (
        paired_bootstrap_delta(baseline_correct1, candidate_correct1)
        if groups is None
        else paired_stratified_bootstrap_delta(baseline_correct1, candidate_correct1, groups)
    )
    top5_delta = (
        paired_bootstrap_delta(baseline_correct5, candidate_correct5)
        if groups is None
        else paired_stratified_bootstrap_delta(baseline_correct5, candidate_correct5, groups)
    )
    result: dict[str, Any] = {
        "samples": int(expected.size),
        "baseline_top1": float(np.mean(baseline_correct1)),
        "candidate_top1": float(np.mean(candidate_correct1)),
        "top1_delta": top1_delta.to_dict(),
        "baseline_top5": float(np.mean(baseline_correct5)),
        "candidate_top5": float(np.mean(candidate_correct5)),
        "top5_delta": top5_delta.to_dict(),
        "top1_agreement": float(np.mean(baseline_top1 == candidate_top1)),
        "top5_set_agreement": float(
            np.mean(np.all(np.sort(baseline_top5, axis=1) == np.sort(candidate_top5, axis=1), axis=1))
        ),
        "cpu_correct_candidate_wrong": int(np.count_nonzero(baseline_correct1 & ~candidate_correct1)),
        "cpu_wrong_candidate_correct": int(np.count_nonzero(~baseline_correct1 & candidate_correct1)),
        "logit_max_abs_error": float(np.max(np.abs(difference), initial=0.0)),
        "logit_normalized_rmse": float(np.sqrt(np.mean(np.square(difference))) / max(reference_rms, 1e-12)),
        "logit_cosine_similarity": 1.0 if denominator == 0.0 else float(np.sum(baseline * candidate) / denominator),
    }
    if groups is not None:
        group_values = np.asarray(tuple(str(value) for value in groups))
        result["per_group_top1_delta"] = {
            group: float(
                np.mean(candidate_correct1[group_values == group]) - np.mean(baseline_correct1[group_values == group])
            )
            for group in np.unique(group_values)
        }
    return result


def action_metrics(
    baseline_actions: npt.ArrayLike,
    candidate_actions: npt.ArrayLike,
    *,
    episode_ids: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Compare matched batches of complete action chunks without quality gates."""
    baseline = np.asarray(baseline_actions, dtype=np.float64)
    candidate = np.asarray(candidate_actions, dtype=np.float64)
    if baseline.ndim != 3 or candidate.shape != baseline.shape or baseline.size == 0:
        raise ValueError("action metrics require equal observations-by-chunk-by-action arrays")
    finite = np.isfinite(candidate)
    safe_candidate = np.nan_to_num(candidate)
    safe_baseline = np.nan_to_num(baseline)
    difference = safe_candidate - safe_baseline
    absolute = np.abs(difference)
    per_observation_rmse = np.sqrt(np.mean(np.square(difference), axis=(1, 2)))
    reference_rms = np.sqrt(np.mean(np.square(safe_baseline), axis=(1, 2)))
    normalized = per_observation_rmse / np.maximum(reference_rms, 1e-12)
    denominator = float(np.linalg.norm(safe_baseline) * np.linalg.norm(safe_candidate))
    return {
        "observations": int(baseline.shape[0]),
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "mean_abs_error": float(np.mean(absolute)),
        "p99_abs_error": float(np.percentile(absolute, 99)),
        "per_action_mae": np.mean(absolute, axis=(0, 1)).tolist(),
        "per_action_rmse": np.sqrt(np.mean(np.square(difference), axis=(0, 1))).tolist(),
        "normalized_rmse": float(
            np.sqrt(np.mean(np.square(difference))) / max(float(np.sqrt(np.mean(np.square(safe_baseline)))), 1e-12)
        ),
        "normalized_rmse_interval": paired_bootstrap_delta(
            np.zeros_like(normalized), normalized, groups=episode_ids
        ).to_dict(),
        "cosine_similarity": 1.0
        if denominator == 0.0
        else float(np.sum(safe_baseline * safe_candidate) / denominator),
        "sign_agreement": float(np.mean(np.signbit(safe_baseline) == np.signbit(safe_candidate))),
        "worst_observation": int(np.argmax(per_observation_rmse)),
        "nan_count": int(np.count_nonzero(np.isnan(candidate))),
        "inf_count": int(np.count_nonzero(np.isinf(candidate))),
        "finite_fraction": float(np.mean(finite)),
    }


__all__ = [
    "BootstrapInterval",
    "action_metrics",
    "classification_metrics",
    "paired_bootstrap_delta",
    "paired_stratified_bootstrap_delta",
]
