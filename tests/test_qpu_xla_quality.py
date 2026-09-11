from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qpu_xla.quality.cli import _load_coco_batches
from qpu_xla.quality.coco import detection_agreement
from qpu_xla.quality.manifest import SampleManifest, imagenet_stratified_manifest
from qpu_xla.quality.metrics import action_metrics, classification_metrics, paired_bootstrap_delta
from qpu_xla.quality.mmlu import MMLURecord, compare_mmlu, load_mmlu_jsonl
from qpu_xla.quality.report import QualityReport, RunIntegrity


def test_paired_bootstrap_is_deterministic_and_reports_candidate_minus_baseline() -> None:
    baseline = np.asarray([0, 1, 0, 1], dtype=np.float64)
    candidate = np.asarray([1, 1, 1, 1], dtype=np.float64)
    first = paired_bootstrap_delta(baseline, candidate, replicates=200, seed=7)
    second = paired_bootstrap_delta(baseline, candidate, replicates=200, seed=7)
    assert first == second
    assert first.delta == 0.5
    assert first.low <= first.delta <= first.high


def test_cluster_bootstrap_preserves_grouped_observations() -> None:
    interval = paired_bootstrap_delta(
        np.zeros(4),
        np.asarray([1.0, 1.0, 3.0, 3.0]),
        groups=("a", "a", "b", "b"),
        replicates=100,
        seed=11,
    )
    assert interval.delta == 2.0
    assert interval.low == 1.0
    assert interval.high == 3.0


def test_classification_metrics_report_flips_and_logit_difference() -> None:
    baseline = np.asarray([[5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]], dtype=np.float64)
    candidate = np.asarray([[0, 4, 3, 2, 1, 5], [0, 1, 2, 3, 5, 4]], dtype=np.float64)
    metrics = classification_metrics(baseline, candidate, np.asarray([0, 5]), groups=("a", "b"))
    assert metrics["baseline_top1"] == 1.0
    assert metrics["candidate_top1"] == 0.0
    assert metrics["cpu_correct_candidate_wrong"] == 2
    assert metrics["top1_delta"]["delta"] == -1.0
    assert metrics["logit_max_abs_error"] == 5.0


def test_action_metrics_cover_dimensions_direction_and_worst_observation() -> None:
    baseline = np.ones((2, 3, 2), dtype=np.float32)
    candidate = baseline.copy()
    candidate[1, :, 1] += 2
    metrics = action_metrics(baseline, candidate, episode_ids=("first", "second"))
    assert metrics["max_abs_error"] == 2.0
    assert metrics["per_action_mae"] == [0.0, 1.0]
    assert metrics["worst_observation"] == 1
    assert metrics["finite_fraction"] == 1.0


def test_mmlu_records_are_paired_by_stable_source_id(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"task_id": 9, "correct_index": 1, "predicted_index": 1, "log_probs": [-2, -1, -3, -4]},
                {"task_id": 3, "correct_index": 0, "predicted_index": 2, "log_probs": [-2, -3, -1, -4]},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = load_mmlu_jsonl(path)
    candidate = (
        MMLURecord(3, 0, 0, (-1.0, -3.0, -2.0, -4.0)),
        MMLURecord(9, 1, 1, (-2.0, -1.0, -3.0, -4.0)),
    )
    metrics = compare_mmlu(baseline, candidate)
    assert metrics["questions"] == 2
    assert metrics["baseline_accuracy"] == 0.5
    assert metrics["candidate_accuracy"] == 1.0
    assert metrics["cpu_wrong_candidate_correct"] == 1


def test_imagenet_manifest_selects_five_stable_samples_per_class(tmp_path: Path) -> None:
    for class_index in range(1_000):
        directory = tmp_path / f"n{class_index:08d}"
        directory.mkdir()
        for image_index in range(6):
            (directory / f"{image_index}.jpeg").touch()
    first = imagenet_stratified_manifest(tmp_path, hash_contents=False)
    second = imagenet_stratified_manifest(tmp_path, hash_contents=False)
    assert first == second
    assert len(first.samples) == 5_000
    assert len({record.group for record in first.samples}) == 1_000
    location = tmp_path / "manifest.json"
    first.save(location)
    assert SampleManifest.load(location) == first


def test_quality_report_validates_execution_integrity_not_quality_delta(tmp_path: Path) -> None:
    integrity = RunIntegrity(
        10,
        10,
        True,
        qpu_dispatches=1,
        cpu_operations=1,
        requires_qpu=True,
        requires_cpu=True,
    )
    report = QualityReport("task", "cpu", "qpu", {"accuracy_delta": -0.5}, integrity)
    assert integrity.valid_run
    path = tmp_path / "report.json"
    report.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["integrity"]["valid_run"] is True
    assert "passes" not in payload and payload["metrics"]["accuracy_delta"] == -0.5


def test_integrity_requires_declared_qpu_and_cpu_work() -> None:
    assert not RunIntegrity(1, 1, True, requires_qpu=True).valid_run
    assert not RunIntegrity(1, 1, True, qpu_dispatches=1, requires_cpu=True).valid_run


def test_detection_agreement_matches_only_same_image_and_class() -> None:
    baseline = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    candidate = [
        {"image_id": 1, "category_id": 3, "bbox": [1, 1, 10, 10], "score": 0.8},
        {"image_id": 2, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    metrics = detection_agreement(baseline, candidate)
    assert metrics["matched_detections"] == 1
    assert metrics["baseline_match_fraction"] == 0.5
    assert metrics["candidate_match_fraction"] == 0.5
    np.testing.assert_allclose(metrics["mean_matched_iou"], 81 / 119)


def test_coco_checkpoint_records_images_with_zero_detections(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    prediction = {"image_id": 2, "category_id": 3, "bbox": [1, 2, 3, 4], "score": 0.5}
    path.write_text(
        json.dumps({"image_id": 1, "predictions": []})
        + "\n"
        + json.dumps({"image_id": 2, "predictions": [prediction]})
        + "\n",
        encoding="utf-8",
    )
    completed, predictions = _load_coco_batches(path, [{"id": 1}, {"id": 2}, {"id": 3}])
    assert completed == 2
    assert predictions == [prediction]
