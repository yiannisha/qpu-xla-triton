"""Streaming SmolVLA checkpoint conversion and memory-mapped artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.models.smolvla.config import (
    UPSTREAM_LEROBOT_REVISION,
    UPSTREAM_SMOLVLA_REPOSITORY,
    UPSTREAM_SMOLVLA_REVISION,
    SmolVLAConfig,
)

ARTIFACT_FORMAT = "qpu-xla-smolvla-v1"


class SmolVLANumerics(Enum):
    """Mutually exclusive model-weight contracts."""

    FP32 = "fp32"
    W8A8 = "w8a8"


@dataclass(frozen=True, slots=True)
class W8A8Weight:
    """Per-output-channel symmetric INT8 weight and FP32 scales."""

    values: npt.NDArray[np.int8]
    scales: npt.NDArray[np.float32]

    def __post_init__(self: Self) -> None:
        """Validate the signed values and per-output scale contract."""
        if self.values.dtype != np.dtype(np.int8) or self.values.ndim not in {2, 4}:
            raise ValueError("W8A8 values must be a rank-2 or rank-4 int8 array")
        if self.scales.dtype != np.dtype(np.float32) or self.scales.shape != (self.values.shape[0],):
            raise ValueError("W8A8 scales must be one FP32 value per output channel")
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("W8A8 scales must be finite and positive")

    @property
    def shape(self: Self) -> tuple[int, ...]:
        """Return the logical unquantized weight shape."""
        return cast(tuple[int, ...], self.values.shape)


Weight = npt.NDArray[np.float32] | W8A8Weight


def is_quantized_weight(name: str, shape: tuple[int, ...]) -> bool:
    """Return whether a checkpoint tensor uses the W8A8 linear/patch contract."""
    if len(shape) not in {2, 4} or not name.endswith(".weight"):
        return False
    return not (
        name.endswith("embed_tokens.weight")
        or name.endswith("position_embedding.weight")
        or name.endswith("lm_head.weight")
    )


def quantize_per_output_channel(values: npt.NDArray[np.float32]) -> W8A8Weight:
    """Quantize a linear or convolution weight without a second model-sized copy."""
    if values.dtype != np.dtype(np.float32) or values.ndim not in {2, 4}:
        raise ValueError("per-output quantization requires rank-2 or rank-4 FP32 weights")
    flattened = values.reshape(values.shape[0], -1)
    maxima = np.max(np.abs(flattened), axis=1)
    scales = np.maximum(maxima / np.float32(127.0), np.float32(1.0 / 127.0)).astype(np.float32)
    quantized = np.rint(flattened / scales[:, None])
    np.clip(quantized, -127, 127, out=quantized)
    return W8A8Weight(np.ascontiguousarray(quantized.reshape(values.shape), dtype=np.int8), scales)


class _TensorMap(Mapping[str, Weight]):
    """Lazy, memory-mapped tensor mapping backed by an artifact manifest."""

    def __init__(self: Self, root: Path, entries: dict[str, dict[str, Any]], numerics: SmolVLANumerics) -> None:
        self._root = root
        self._entries = entries
        self._numerics = numerics
        self._cache: dict[str, Weight] = {}

    def __len__(self: Self) -> int:
        return len(self._entries)

    def __iter__(self: Self) -> Iterator[str]:
        return iter(self._entries)

    def __getitem__(self: Self, name: str) -> Weight:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        entry = self._entries[name]
        if entry["storage"] == "w8a8":
            values = np.load(self._root / entry["values"], mmap_mode="r", allow_pickle=False)
            scales = np.load(self._root / entry["scales"], mmap_mode="r", allow_pickle=False)
            result: Weight = W8A8Weight(values, scales)
        else:
            result = np.load(self._root / entry["values"], mmap_mode="r", allow_pickle=False)
        self._cache[name] = result
        return result


@dataclass(frozen=True, slots=True)
class SmolVLACheckpoint:
    """Validated native checkpoint with lazily mapped tensor storage."""

    config: SmolVLAConfig
    numerics: SmolVLANumerics
    tensors: Mapping[str, Weight]

    def __post_init__(self: Self) -> None:
        """Reject incomplete, extra, or shape-incompatible artifacts."""
        expected = self.config.expected_shapes()
        if set(self.tensors) != set(expected):
            missing = sorted(set(expected) - set(self.tensors))
            unexpected = sorted(set(self.tensors) - set(expected))
            raise ValueError(f"SmolVLA tensor names do not match config; missing={missing}, unexpected={unexpected}")
        for name, shape in expected.items():
            value = self.tensors[name]
            quantized = self.numerics is SmolVLANumerics.W8A8 and is_quantized_weight(name, shape)
            if quantized:
                if not isinstance(value, W8A8Weight) or value.shape != shape:
                    raise ValueError(f"SmolVLA tensor {name!r} must be a W8A8 weight with shape {shape}")
            elif (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"SmolVLA tensor {name!r} must be contiguous FP32 with shape {shape}")

    def fp32(self, name: str) -> npt.NDArray[np.float32]:
        """Return a tensor that is stored in FP32 under both numerical contracts."""
        value = self.tensors[name]
        if isinstance(value, W8A8Weight):
            raise TypeError(f"SmolVLA tensor {name!r} is quantized")
        return value

    def linear_weight(self, name: str) -> Weight:
        """Return one validated projection or patch-embedding weight."""
        value = self.tensors[name]
        if not isinstance(value, W8A8Weight) and value.ndim not in {2, 4}:
            raise TypeError(f"SmolVLA tensor {name!r} is not a linear/convolution weight")
        return value


def _sha256(path: Path, *, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated SafeTensors length prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 0 or header_size > 128 * 1024 * 1024:
                raise ValueError("invalid SafeTensors header size")
            payload = json.loads(stream.read(header_size))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read SafeTensors header {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("SafeTensors header must be an object")
    payload.pop("__metadata__", None)
    entries: dict[str, dict[str, Any]] = {}
    for name, entry in payload.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("SafeTensors header contains an invalid tensor entry")
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if dtype not in {"BF16", "F32"} or not isinstance(shape, list) or not isinstance(offsets, list):
            raise ValueError(f"unsupported SafeTensors entry for {name!r}")
        if len(offsets) != 2 or any(not isinstance(value, int) for value in (*shape, *offsets)):
            raise ValueError(f"invalid SafeTensors shape/offsets for {name!r}")
        entries[name] = {"dtype": dtype, "shape": tuple(shape), "data_offsets": tuple(offsets)}
    return 8 + header_size, entries


def _load_source_tensor(path: Path, data_start: int, entry: dict[str, Any]) -> npt.NDArray[np.float32]:
    shape = cast(tuple[int, ...], entry["shape"])
    start, end = cast(tuple[int, int], entry["data_offsets"])
    count = int(np.prod(shape, dtype=np.int64))
    dtype = entry["dtype"]
    expected_bytes = count * (2 if dtype == "BF16" else 4)
    if end - start != expected_bytes:
        raise ValueError("SafeTensors entry byte length does not match its shape")
    if dtype == "F32":
        mapped = np.memmap(path, dtype="<f4", mode="r", offset=data_start + start, shape=shape)
        return np.ascontiguousarray(mapped, dtype=np.float32)
    mapped_u16 = np.memmap(path, dtype="<u2", mode="r", offset=data_start + start, shape=shape)
    words = np.asarray(mapped_u16, dtype=np.uint32)
    np.left_shift(words, np.uint32(16), out=words)
    return np.ascontiguousarray(words.view(np.float32))


def _safe_filename(index: int) -> str:
    return f"tensors/{index:04d}.npy"


def convert_smolvla_safetensors(
    source: str | PathLike[str],
    destination: str | PathLike[str],
    config: SmolVLAConfig,
    *,
    numerics: SmolVLANumerics = SmolVLANumerics.W8A8,
    tokenizer_directory: str | PathLike[str] | None = None,
) -> Path:
    """Stream a pinned SafeTensors checkpoint into one native runtime artifact.

    Conversion materializes only one source tensor at a time.  The resulting
    directory can be memory-mapped and therefore does not create a second full
    model copy in process memory.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        raise ValueError(f"SmolVLA artifact destination already exists: {destination_path}")
    data_start, entries = _read_safetensors_header(source_path)
    expected = config.expected_shapes()
    if set(entries) != set(expected):
        missing = sorted(set(expected) - set(entries))
        unexpected = sorted(set(entries) - set(expected))
        raise ValueError(f"source checkpoint names mismatch; missing={missing}, unexpected={unexpected}")
    for name, shape in expected.items():
        if entries[name]["shape"] != shape:
            raise ValueError(f"source checkpoint tensor {name!r} has shape {entries[name]['shape']}, expected {shape}")

    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=parent))
    try:
        (temporary / "tensors").mkdir()
        manifest_entries: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(sorted(expected)):
            values = _load_source_tensor(source_path, data_start, entries[name])
            relative = _safe_filename(index)
            target = temporary / relative
            if numerics is SmolVLANumerics.W8A8 and is_quantized_weight(name, expected[name]):
                quantized = quantize_per_output_channel(values)
                scale_relative = f"tensors/{index:04d}.scales.npy"
                np.save(target, quantized.values, allow_pickle=False)
                np.save(temporary / scale_relative, quantized.scales, allow_pickle=False)
                manifest_entries[name] = {
                    "storage": "w8a8",
                    "shape": list(expected[name]),
                    "values": relative,
                    "values_sha256": _sha256(target),
                    "scales": scale_relative,
                    "scales_sha256": _sha256(temporary / scale_relative),
                }
            else:
                np.save(target, values, allow_pickle=False)
                manifest_entries[name] = {
                    "storage": "fp32",
                    "shape": list(expected[name]),
                    "values": relative,
                    "values_sha256": _sha256(target),
                }
            del values

        tokenizer_files: dict[str, str] = {}
        if tokenizer_directory is not None:
            tokenizer_root = Path(tokenizer_directory)
            output_root = temporary / "tokenizer"
            output_root.mkdir()
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "processor_config.json",
                "chat_template.json",
            ):
                source_file = tokenizer_root / filename
                if source_file.is_file():
                    shutil.copyfile(source_file, output_root / filename)
                    tokenizer_files[f"tokenizer/{filename}"] = _sha256(output_root / filename)

        manifest = {
            "format": ARTIFACT_FORMAT,
            "numerics": numerics.value,
            "source": {
                "repository": UPSTREAM_SMOLVLA_REPOSITORY,
                "model_revision": UPSTREAM_SMOLVLA_REVISION,
                "lerobot_revision": UPSTREAM_LEROBOT_REVISION,
                "checkpoint_sha256": _sha256(source_path),
            },
            "config": _config_payload(config),
            "tensors": manifest_entries,
            "tokenizer_files": tokenizer_files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_path


def _config_payload(config: SmolVLAConfig) -> dict[str, Any]:
    """Serialize dataclass fields without relying on pickles or class metadata."""
    result: dict[str, Any] = {}
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        if isinstance(value, tuple):
            result[field_name] = list(value)
        elif isinstance(value, dict):
            result[field_name] = {key: list(shape) for key, shape in value.items()}
        else:
            result[field_name] = value
    return result


def _config_from_payload(payload: Any) -> SmolVLAConfig:
    if not isinstance(payload, dict):
        raise ValueError("SmolVLA artifact config must be an object")
    values = dict(payload)
    if "image_keys" in values:
        values["image_keys"] = tuple(values["image_keys"])
    if "input_features" in values:
        values["input_features"] = {name: tuple(shape) for name, shape in values["input_features"].items()}
    try:
        return SmolVLAConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError("SmolVLA artifact contains an invalid config") from exc


@dataclass(frozen=True, slots=True)
class SmolVLAArtifact:
    """Validated memory-mapped model artifact and source provenance."""

    root: Path
    checkpoint: SmolVLACheckpoint
    source: Mapping[str, str]
    tokenizer_files: Mapping[str, str]

    @classmethod
    def open(cls: type[SmolVLAArtifact], directory: str | PathLike[str], *, verify: bool = True) -> SmolVLAArtifact:
        """Open an artifact, optionally verifying every content checksum."""
        root = Path(directory)
        try:
            payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load SmolVLA artifact manifest from {root}") from exc
        if not isinstance(payload, dict) or payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError("unsupported SmolVLA artifact format")
        try:
            numerics = SmolVLANumerics(payload["numerics"])
        except (KeyError, ValueError) as exc:
            raise ValueError("SmolVLA artifact has invalid numerics") from exc
        config = _config_from_payload(payload.get("config"))
        entries = payload.get("tensors")
        source = payload.get("source")
        tokenizer_files = payload.get("tokenizer_files", {})
        if not isinstance(entries, dict) or not isinstance(source, dict) or not isinstance(tokenizer_files, dict):
            raise ValueError("SmolVLA artifact manifest sections must be objects")
        expected_source = {
            "repository": UPSTREAM_SMOLVLA_REPOSITORY,
            "model_revision": UPSTREAM_SMOLVLA_REVISION,
            "lerobot_revision": UPSTREAM_LEROBOT_REVISION,
        }
        if any(source.get(key) != value for key, value in expected_source.items()):
            raise ValueError("SmolVLA artifact provenance does not match the pinned baseline")
        if not isinstance(source.get("checkpoint_sha256"), str):
            raise ValueError("SmolVLA artifact is missing its source checkpoint checksum")
        if verify:
            for name, entry in entries.items():
                if not isinstance(entry, dict) or not isinstance(entry.get("values"), str):
                    raise ValueError(f"invalid SmolVLA artifact entry {name!r}")
                if _sha256(root / entry["values"]) != entry.get("values_sha256"):
                    raise ValueError(f"SmolVLA artifact checksum mismatch for {name!r}")
                if entry.get("storage") == "w8a8":
                    if not isinstance(entry.get("scales"), str) or (
                        _sha256(root / entry["scales"]) != entry.get("scales_sha256")
                    ):
                        raise ValueError(f"SmolVLA artifact scale checksum mismatch for {name!r}")
            for relative, digest in tokenizer_files.items():
                if not isinstance(relative, str) or not isinstance(digest, str) or _sha256(root / relative) != digest:
                    raise ValueError(f"SmolVLA tokenizer checksum mismatch for {relative!r}")
        tensors = _TensorMap(root, entries, numerics)
        checkpoint = SmolVLACheckpoint(config, numerics, tensors)
        return cls(root, checkpoint, source, tokenizer_files)


__all__ = [
    "ARTIFACT_FORMAT",
    "SmolVLAArtifact",
    "SmolVLACheckpoint",
    "SmolVLANumerics",
    "W8A8Weight",
    "convert_smolvla_safetensors",
    "is_quantized_weight",
    "quantize_per_output_channel",
]
