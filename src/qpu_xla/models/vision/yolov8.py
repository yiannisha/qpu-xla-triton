"""Pinned YOLOv8n graph with every learned convolution routed through QPU-XLA."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import torch

from qpu_xla.device import Device
from qpu_xla.models.vision.runtime import VisionExecutionContext, VisionMode, instrument_torch_convolutions
from qpu_xla.quality.manifest import sha256_file

COCO80_CATEGORY_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


def result_to_coco(result: Any, *, image_id: int) -> list[dict[str, Any]]:
    """Convert one Ultralytics result to standard xywh COCO records."""
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
    predictions: list[dict[str, Any]] = []
    for coordinates, score, label in zip(xyxy, confidence, classes, strict=True):
        x1, y1, x2, y2 = map(float, coordinates)
        predictions.append(
            {
                "image_id": image_id,
                "category_id": COCO80_CATEGORY_IDS[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return predictions


@dataclass(frozen=True, slots=True)
class YoloV8Artifact:
    """Content identity for an Ultralytics YOLOv8n checkpoint."""

    checkpoint: Path
    checkpoint_sha256: str
    source_revision: str

    @classmethod
    def open(
        cls: type[YoloV8Artifact],
        checkpoint: str | PathLike[str],
        *,
        expected_sha256: str | None = None,
        source_revision: str = "ultralytics-8.4.146",
    ) -> YoloV8Artifact:
        """Resolve and optionally pin-check a local checkpoint."""
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError(f"YOLOv8 checkpoint does not exist: {path}")
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("YOLOv8 checkpoint checksum mismatch")
        return cls(path, digest, source_revision)


class YoloV8Runtime:
    """Complete Ultralytics detection graph with QPU convolution adapters."""

    def __init__(
        self: Self,
        artifact: YoloV8Artifact,
        *,
        mode: VisionMode,
        device: Device | None = None,
    ) -> None:
        """Load, fold, instrument, and validate one YOLOv8n graph."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLOv8 runtime requires the optional ultralytics dependency") from exc
        self.artifact = artifact
        self.context = VisionExecutionContext(mode, device)
        try:
            self.model = YOLO(str(artifact.checkpoint))
            predictor_options = {
                "mode": "predict",
                "batch": 1,
                "save": False,
                # Prepared QPU convolution plans require one stable tensor
                # shape across the complete COCO run. Letterbox every source
                # into a square rather than using per-image minimal rectangles.
                "rect": False,
                "imgsz": 640,
                "conf": 0.001,
                "iou": 0.7,
                "max_det": 300,
                "agnostic_nms": False,
                "verbose": False,
                "device": "cpu",
            }
            predictor_type = self.model._smart_load("predictor")
            self.predictor = predictor_type(overrides=predictor_options, _callbacks=self.model.callbacks)
            # setup_model deep-copies and folds the canonical graph. Instrument
            # that retained inference graph, not YOLO.model, so Ultralytics
            # preprocessing/postprocessing cannot silently bypass the adapters.
            self.predictor.setup_model(model=self.model.model, verbose=False)
            backend = getattr(self.predictor.model, "backend", None)
            inference_model = getattr(backend, "model", None)
            if not isinstance(inference_model, torch.nn.Module):
                raise RuntimeError("Ultralytics did not expose its retained PyTorch inference graph")
            instrument_torch_convolutions(inference_model, self.context)
            inference_model.eval()
        except BaseException:
            self.context.close()
            raise

    def predict(
        self: Self,
        source: str | PathLike[str] | npt.NDArray[np.generic],
        *,
        image_id: int = 0,
    ) -> list[dict[str, Any]]:
        """Return standard COCO-style predictions for one image."""
        with torch.inference_mode():
            results = self.predictor(source=source, stream=False)
        if len(results) != 1:
            raise RuntimeError("YOLOv8 batch-one runtime returned an unexpected result count")
        return result_to_coco(results[0], image_id=image_id)

    def close(self: Self) -> None:
        """Release prepared convolution buffers and queues."""
        self.context.close()

    def __enter__(self: Self) -> Self:
        """Enter the runtime lifetime."""
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the runtime at context exit."""
        self.close()


__all__ = ["COCO80_CATEGORY_IDS", "YoloV8Artifact", "YoloV8Runtime", "result_to_coco"]
