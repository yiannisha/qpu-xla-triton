from __future__ import annotations

import json
from pathlib import Path

import pytest

from qpu_xla.quality.coco import coco_bbox_metrics, coco_per_category_ap, paired_coco_bootstrap


def test_real_pycocotools_metrics_and_paired_bootstrap(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [{"id": 1, "width": 20, "height": 20, "file_name": "one.jpg"}],
                "categories": [{"id": 1, "name": "thing", "supercategory": "thing"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [2, 2, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10], "score": 0.9}]
    metrics = coco_bbox_metrics(annotation, predictions)
    categories = coco_per_category_ap(annotation, predictions)
    intervals = paired_coco_bootstrap(annotation, predictions, predictions, replicates=3, seed=2)
    assert metrics["AP"] == pytest.approx(1.0)
    assert categories["thing"] == pytest.approx(1.0)
    assert intervals["AP"]["low"] == pytest.approx(0.0)
    assert intervals["AP"]["high"] == pytest.approx(0.0)
