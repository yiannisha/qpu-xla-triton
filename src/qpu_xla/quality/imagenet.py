"""Fixed-subset ImageNet qualification runner."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from qpu_xla.quality.manifest import SampleManifest
from qpu_xla.quality.metrics import classification_metrics


def evaluate_imagenet_pair(
    root: str | PathLike[str],
    manifest: SampleManifest,
    baseline: Callable[[npt.NDArray[np.float32]], npt.ArrayLike],
    candidate: Callable[[npt.NDArray[np.float32]], npt.ArrayLike],
    preprocess: Callable[[str | PathLike[str]], npt.NDArray[np.float32]],
) -> tuple[dict[str, Any], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Evaluate two runtimes on byte-identical, ordered ImageNet inputs."""
    if manifest.task != "imagenet-classification" or len(manifest.samples) != 5_000:
        raise ValueError("ImageNet qualification requires a 5000-item classification manifest")
    directory = Path(root)
    labels = np.asarray([record.label for record in manifest.samples], dtype=np.int64)
    groups = [record.group or str(record.label) for record in manifest.samples]
    baseline_logits: list[npt.NDArray[np.float32]] = []
    candidate_logits: list[npt.NDArray[np.float32]] = []
    for record in manifest.samples:
        values = preprocess(directory / record.relative_path)
        baseline_logits.append(np.asarray(baseline(values), dtype=np.float32).reshape(-1))
        candidate_logits.append(np.asarray(candidate(values), dtype=np.float32).reshape(-1))
    before = np.stack(baseline_logits)
    after = np.stack(candidate_logits)
    return classification_metrics(before, after, labels, groups=groups), before, after


__all__ = ["evaluate_imagenet_pair"]
