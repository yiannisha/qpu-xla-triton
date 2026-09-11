from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qpu_xla import Device
from qpu_xla.models.vision import ResNet18Artifact, ResNet18Runtime, VisionMode, YoloV8Artifact, YoloV8Runtime


def test_resnet18_artifact_runtime_matches_torchvision(tmp_path: Path) -> None:
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(7)
    model = torchvision.models.resnet18(weights=None).eval()
    source = np.random.default_rng(7).standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(source)).numpy()
    artifact = ResNet18Artifact.create(tmp_path / "artifact", model, source="torchvision-random-test")
    with ResNet18Runtime.from_artifact(ResNet18Artifact.open(artifact.root), mode=VisionMode.CPU_FP32) as runtime:
        actual = runtime.predict_logits(source)
        telemetry = runtime.context.telemetry
        assert telemetry.eligible_layers == 20
        assert telemetry.cpu_calls == 20
    np.testing.assert_array_equal(actual, expected)


def test_yolov8_predictor_executes_instrumented_graph(tmp_path: Path) -> None:
    ultralytics = pytest.importorskip("ultralytics")
    checkpoint = tmp_path / "yolov8n-random.pt"
    ultralytics.YOLO("yolov8n.yaml").save(checkpoint)
    artifact = YoloV8Artifact.open(checkpoint)
    landscape = np.zeros((448, 640, 3), dtype=np.uint8)
    portrait = np.zeros((640, 608, 3), dtype=np.uint8)
    with YoloV8Runtime(artifact, mode=VisionMode.CPU_FP32) as runtime:
        runtime.predict(landscape, image_id=41)
        runtime.predict(portrait, image_id=42)
        telemetry = runtime.context.telemetry
        assert telemetry.eligible_layers > 0
        assert telemetry.cpu_calls == 2 * telemetry.eligible_layers


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_resnet18_qpu_executes_complete_instrumented_graph(tmp_path: Path) -> None:
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(17)
    model = torchvision.models.resnet18(weights=None).eval()
    source = np.random.default_rng(17).standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(source)).numpy()
    artifact = ResNet18Artifact.create(tmp_path / "artifact", model, source="torchvision-random-test")
    with Device.open(data_area_size=1024**3) as device:
        with ResNet18Runtime.from_artifact(artifact, mode=VisionMode.QPU_FP32, device=device) as runtime:
            actual = runtime.predict_logits(source)
            telemetry = runtime.context.telemetry
            assert telemetry.qpu_dispatches == telemetry.eligible_layers == 20
            assert telemetry.unexpected_fallbacks == 0
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_yolov8_hybrid_w8a8_executes_complete_instrumented_graph(tmp_path: Path) -> None:
    ultralytics = pytest.importorskip("ultralytics")
    checkpoint = tmp_path / "yolov8n-random.pt"
    ultralytics.YOLO("yolov8n.yaml").save(checkpoint)
    artifact = YoloV8Artifact.open(checkpoint)
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    with Device.open(data_area_size=2 * 1024**3) as device:
        with YoloV8Runtime(artifact, mode=VisionMode.HYBRID_W8A8, device=device) as runtime:
            runtime.predict(image, image_id=41)
            telemetry = runtime.context.telemetry
            assert telemetry.qpu_dispatches == telemetry.eligible_layers
            assert telemetry.hybrid_calls == telemetry.eligible_layers
            assert telemetry.qpu_rows > 0 and telemetry.cpu_rows > 0
            assert telemetry.unexpected_fallbacks == 0
