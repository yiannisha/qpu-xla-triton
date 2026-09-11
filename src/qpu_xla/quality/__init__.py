"""Reproducible model-quality comparisons for CPU, QPU, and hybrid paths."""

from qpu_xla.quality.manifest import (
    SampleManifest,
    SampleRecord,
    coco_random_manifest,
    imagenet_stratified_manifest,
)
from qpu_xla.quality.metrics import (
    BootstrapInterval,
    action_metrics,
    classification_metrics,
    paired_bootstrap_delta,
    paired_stratified_bootstrap_delta,
)
from qpu_xla.quality.report import QualityReport, RunIntegrity

__all__ = [
    "BootstrapInterval",
    "QualityReport",
    "RunIntegrity",
    "SampleManifest",
    "SampleRecord",
    "action_metrics",
    "classification_metrics",
    "coco_random_manifest",
    "imagenet_stratified_manifest",
    "paired_bootstrap_delta",
    "paired_stratified_bootstrap_delta",
]
