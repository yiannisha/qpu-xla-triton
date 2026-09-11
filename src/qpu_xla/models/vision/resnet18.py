"""Pinned ResNet-18 artifact and complete QPU-instrumented classifier runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import torch

from qpu_xla.device import Device
from qpu_xla.models.vision.runtime import VisionExecutionContext, VisionMode, instrument_torch_convolutions
from qpu_xla.quality.manifest import sha256_file

ARTIFACT_FORMAT = "qpu-xla-resnet18-v1"
TORCHVISION_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"


def _state_hash(state: dict[str, npt.NDArray[np.generic]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        values = np.ascontiguousarray(state[name])
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResNet18Artifact:
    """Portable Torchvision ResNet-18 parameters with content provenance."""

    root: Path
    state: dict[str, npt.NDArray[np.generic]]
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls: type[ResNet18Artifact],
        destination: str | PathLike[str],
        model: torch.nn.Module,
        *,
        source: str = TORCHVISION_WEIGHTS,
    ) -> ResNet18Artifact:
        """Export an evaluated Torchvision ResNet-18 without pickle payloads."""
        root = Path(destination)
        if root.exists():
            raise ValueError(f"ResNet-18 artifact destination already exists: {root}")
        root.mkdir(parents=True)
        state = {
            name: np.ascontiguousarray(
                value.detach().cpu().numpy(), dtype=np.float32 if value.dtype.is_floating_point else None
            )
            for name, value in model.state_dict().items()
        }
        arrays_path = root / "weights.npz"
        np.savez(arrays_path, **cast(dict[str, Any], state))
        metadata = {
            "format": ARTIFACT_FORMAT,
            "architecture": "resnet18",
            "source": source,
            "state_sha256": _state_hash(state),
            "weights_sha256": sha256_file(arrays_path),
            "preprocessing": {
                "resize_shorter": 256,
                "center_crop": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bilinear",
                "antialias": True,
            },
        }
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root, state, metadata)

    @classmethod
    def open(cls: type[ResNet18Artifact], directory: str | PathLike[str], *, verify: bool = True) -> ResNet18Artifact:
        """Open and verify a native artifact."""
        root = Path(directory)
        metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError("unsupported ResNet-18 artifact format")
        arrays_path = root / "weights.npz"
        if verify and sha256_file(arrays_path) != metadata.get("weights_sha256"):
            raise ValueError("ResNet-18 weight archive checksum mismatch")
        with np.load(arrays_path, allow_pickle=False) as archive:
            state = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if verify and _state_hash(state) != metadata.get("state_sha256"):
            raise ValueError("ResNet-18 state checksum mismatch")
        return cls(root, state, metadata)

    def torch_model(self: Self) -> torch.nn.Module:
        """Reconstruct the canonical graph and load the verified arrays."""
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise RuntimeError("ResNet-18 runtime requires the optional torchvision dependency") from exc
        model = resnet18(weights=None)
        current = model.state_dict()
        missing = sorted(set(current) - set(self.state))
        if missing:
            raise ValueError(f"ResNet-18 artifact is missing tensors: {missing}")
        loaded = {name: torch.from_numpy(self.state[name]).to(dtype=value.dtype) for name, value in current.items()}
        model.load_state_dict(loaded, strict=True)
        return cast(torch.nn.Module, model.eval())


class ResNet18Runtime:
    """Complete ResNet-18 with every convolution routed by one vision mode."""

    def __init__(self: Self, model: torch.nn.Module, context: VisionExecutionContext) -> None:
        """Bind an evaluated canonical graph to one placement context."""
        self.model = instrument_torch_convolutions(model.eval(), context)
        self.context = context

    @classmethod
    def from_artifact(
        cls: type[ResNet18Runtime],
        artifact: ResNet18Artifact,
        *,
        mode: VisionMode,
        device: Device | None = None,
    ) -> ResNet18Runtime:
        """Build a runtime from content-verified weights."""
        return cls(artifact.torch_model(), VisionExecutionContext(mode, device))

    @classmethod
    def from_torchvision(
        cls: type[ResNet18Runtime],
        *,
        mode: VisionMode,
        device: Device | None = None,
    ) -> ResNet18Runtime:
        """Build directly from the pinned Torchvision V1 weights."""
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("ResNet-18 runtime requires the optional torchvision dependency") from exc
        return cls(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(), VisionExecutionContext(mode, device))

    def predict_logits(self: Self, source: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return one batch of 1000-class logits from normalized NCHW input."""
        if source.ndim != 4 or source.shape[0] != 1 or source.shape[1:] != (3, 224, 224):
            raise ValueError("ResNet-18 expects normalized float32 input shaped (1,3,224,224)")
        with torch.inference_mode():
            result = self.model(torch.from_numpy(np.ascontiguousarray(source)))
        values = np.ascontiguousarray(result.detach().cpu().numpy(), dtype=np.float32)
        if values.shape != (1, 1000) or not np.all(np.isfinite(values)):
            raise FloatingPointError("ResNet-18 produced invalid logits")
        return values

    def close(self: Self) -> None:
        """Release all prepared convolution state."""
        self.context.close()

    def __enter__(self: Self) -> Self:
        """Enter the runtime lifetime."""
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the runtime at context exit."""
        self.close()


def torchvision_preprocess(path: str | PathLike[str]) -> npt.NDArray[np.float32]:
    """Apply the exact V1 validation transform to one image path."""
    try:
        from PIL import Image
        from torchvision.models import ResNet18_Weights
    except ImportError as exc:
        raise RuntimeError("ImageNet preprocessing requires Pillow and torchvision") from exc
    with Image.open(path) as image:
        tensor = ResNet18_Weights.IMAGENET1K_V1.transforms()(image.convert("RGB"))
    return np.ascontiguousarray(tensor[None].numpy(), dtype=np.float32)


__all__ = ["ResNet18Artifact", "ResNet18Runtime", "torchvision_preprocess"]
