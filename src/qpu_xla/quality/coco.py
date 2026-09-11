"""COCO bbox serialization and authoritative pycocotools metrics."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

COCO_METRIC_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL")


def save_coco_predictions(path: str | PathLike[str], predictions: list[dict[str, Any]]) -> None:
    """Write sorted, finite COCO predictions for reproducible evaluation."""
    normalized = sorted(
        predictions, key=lambda item: (int(item["image_id"]), -float(item["score"]), int(item["category_id"]))
    )
    for item in normalized:
        if set(item) != {"image_id", "category_id", "bbox", "score"}:
            raise ValueError("COCO predictions contain unexpected fields")
        if len(item["bbox"]) != 4 or not np.all(np.isfinite([*item["bbox"], item["score"]])):
            raise ValueError("COCO prediction coordinates and score must be finite")
    Path(path).write_text(json.dumps(normalized, separators=(",", ":")) + "\n", encoding="utf-8")


def coco_bbox_metrics(
    annotation: str | PathLike[str],
    predictions: str | PathLike[str] | list[dict[str, Any]],
    *,
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    """Calculate the standard twelve COCO bbox summary statistics."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("COCO evaluation requires the optional pycocotools dependency") from exc
    ground_truth = COCO(str(annotation))
    values = (
        json.loads(Path(predictions).read_text(encoding="utf-8")) if not isinstance(predictions, list) else predictions
    )
    detected = ground_truth.loadRes(values)
    evaluator = COCOeval(ground_truth, detected, "bbox")
    if image_ids is not None:
        evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {name: float(value) for name, value in zip(COCO_METRIC_NAMES, evaluator.stats, strict=True)}


def coco_per_category_ap(
    annotation: str | PathLike[str],
    predictions: str | PathLike[str] | list[dict[str, Any]],
    *,
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    """Calculate AP50:95 for each COCO category from one shared evaluation."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("COCO evaluation requires the optional pycocotools dependency") from exc
    ground_truth = COCO(str(annotation))
    values = (
        json.loads(Path(predictions).read_text(encoding="utf-8")) if not isinstance(predictions, list) else predictions
    )
    evaluator = COCOeval(ground_truth, ground_truth.loadRes(values), "bbox")
    if image_ids is not None:
        evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    categories = {int(item["id"]): str(item["name"]) for item in ground_truth.dataset["categories"]}
    result: dict[str, float] = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values_for_category = precision[:, :, index, 0, -1]
        valid = values_for_category[values_for_category >= 0]
        result[categories[int(category_id)]] = float(np.mean(valid)) if valid.size else float("nan")
    return result


def detection_agreement(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, float | int]:
    """Greedily match same-image, same-class boxes as a diagnostic only."""
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("detection agreement IoU must lie in [0, 1]")
    grouped: list[dict[tuple[int, int], list[dict[str, Any]]]] = [{}, {}]
    for destination, predictions in zip(grouped, (baseline, candidate), strict=True):
        for item in predictions:
            destination.setdefault((int(item["image_id"]), int(item["category_id"])), []).append(item)
    matched_ious: list[float] = []
    confidence_errors: list[float] = []
    for key in grouped[0].keys() | grouped[1].keys():
        left = grouped[0].get(key, [])
        right = grouped[1].get(key, [])
        if not left or not right:
            continue
        left_boxes = np.asarray([item["bbox"] for item in left], dtype=np.float64)
        right_boxes = np.asarray([item["bbox"] for item in right], dtype=np.float64)
        left_xyxy = np.column_stack(
            (
                left_boxes[:, 0],
                left_boxes[:, 1],
                left_boxes[:, 0] + left_boxes[:, 2],
                left_boxes[:, 1] + left_boxes[:, 3],
            )
        )
        right_xyxy = np.column_stack(
            (
                right_boxes[:, 0],
                right_boxes[:, 1],
                right_boxes[:, 0] + right_boxes[:, 2],
                right_boxes[:, 1] + right_boxes[:, 3],
            )
        )
        intersection_min = np.maximum(left_xyxy[:, None, :2], right_xyxy[None, :, :2])
        intersection_max = np.minimum(left_xyxy[:, None, 2:], right_xyxy[None, :, 2:])
        intersection = np.prod(np.maximum(intersection_max - intersection_min, 0.0), axis=2)
        left_area = left_boxes[:, 2] * left_boxes[:, 3]
        right_area = right_boxes[:, 2] * right_boxes[:, 3]
        iou = intersection / np.maximum(left_area[:, None] + right_area[None, :] - intersection, 1e-12)
        while iou.size:
            flat = int(np.argmax(iou))
            left_index, right_index = np.unravel_index(flat, iou.shape)
            value = float(iou[left_index, right_index])
            if value < minimum_iou:
                break
            matched_ious.append(value)
            confidence_errors.append(abs(float(left[left_index]["score"]) - float(right[right_index]["score"])))
            iou[left_index, :] = -1.0
            iou[:, right_index] = -1.0
    matches = len(matched_ious)
    return {
        "baseline_detections": len(baseline),
        "candidate_detections": len(candidate),
        "matched_detections": matches,
        "baseline_match_fraction": matches / max(len(baseline), 1),
        "candidate_match_fraction": matches / max(len(candidate), 1),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_matched_confidence_abs_error": float(np.mean(confidence_errors)) if confidence_errors else 0.0,
        "minimum_iou": minimum_iou,
    }


def paired_coco_bootstrap(
    annotation: str | PathLike[str],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    image_ids: list[int] | None = None,
    replicates: int = 10_000,
    seed: int = 20_260_910,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap true COCO metrics by cloning resampled images and annotations."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("COCO evaluation requires the optional pycocotools dependency") from exc
    dataset = json.loads(Path(annotation).read_text(encoding="utf-8"))
    available_image_ids = {int(item["id"]) for item in dataset["images"]}
    selected_image_ids = sorted(available_image_ids if image_ids is None else image_ids)
    if not selected_image_ids or not set(selected_image_ids) <= available_image_ids:
        raise ValueError("COCO bootstrap image IDs must be a non-empty subset of the annotation")
    bootstrap_image_ids = np.asarray(selected_image_ids, dtype=np.int64)
    images = {int(item["id"]): item for item in dataset["images"]}
    annotations: dict[int, list[dict[str, Any]]] = {int(value): [] for value in bootstrap_image_ids}
    for item in dataset["annotations"]:
        image_id = int(item["image_id"])
        if image_id in annotations:
            annotations[image_id].append(item)
    prediction_maps = []
    for prediction_set in (baseline, candidate):
        mapping: dict[int, list[dict[str, Any]]] = {int(value): [] for value in bootstrap_image_ids}
        for item in prediction_set:
            image_id = int(item["image_id"])
            if image_id in mapping:
                mapping[image_id].append(item)
        prediction_maps.append(mapping)
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, len(COCO_METRIC_NAMES)), dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.choice(bootstrap_image_ids, size=bootstrap_image_ids.size, replace=True)
        sampled_dataset = {
            key: copy.deepcopy(value) for key, value in dataset.items() if key not in {"images", "annotations"}
        }
        sampled_dataset["images"] = []
        sampled_dataset["annotations"] = []
        sampled_predictions: list[list[dict[str, Any]]] = [[], []]
        annotation_id = 1
        for new_image_id, old_image_id_value in enumerate(selected, start=1):
            old_image_id = int(old_image_id_value)
            image = copy.deepcopy(images[old_image_id])
            image["id"] = new_image_id
            sampled_dataset["images"].append(image)
            for old in annotations[old_image_id]:
                item = copy.deepcopy(old)
                item["id"] = annotation_id
                item["image_id"] = new_image_id
                annotation_id += 1
                sampled_dataset["annotations"].append(item)
            for mode, mapping in enumerate(prediction_maps):
                for old in mapping[old_image_id]:
                    item = copy.deepcopy(old)
                    item["image_id"] = new_image_id
                    sampled_predictions[mode].append(item)
        ground_truth = COCO()
        ground_truth.dataset = sampled_dataset
        ground_truth.createIndex()
        stats = []
        for values in sampled_predictions:
            evaluated = COCOeval(ground_truth, ground_truth.loadRes(values), "bbox")
            evaluated.evaluate()
            evaluated.accumulate()
            with contextlib.redirect_stdout(io.StringIO()):
                evaluated.summarize()
            stats.append(np.asarray(evaluated.stats, dtype=np.float64))
        deltas[replicate] = stats[1] - stats[0]
    return {
        name: {
            "low": float(np.quantile(deltas[:, index], 0.025)),
            "high": float(np.quantile(deltas[:, index], 0.975)),
            "replicates": replicates,
            "seed": seed,
        }
        for index, name in enumerate(COCO_METRIC_NAMES)
    }


__all__ = [
    "COCO_METRIC_NAMES",
    "coco_bbox_metrics",
    "coco_per_category_ap",
    "detection_agreement",
    "paired_coco_bootstrap",
    "save_coco_predictions",
]
