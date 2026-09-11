"""Content-addressed deterministic qualification sample manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Self


def sha256_file(path: str | PathLike[str]) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One stable sample identity and its expected label/group metadata."""

    sample_id: str
    relative_path: str
    label: int | None = None
    group: str | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SampleManifest:
    """A versioned, ordered set of qualification examples."""

    task: str
    seed: int
    samples: tuple[SampleRecord, ...]
    metadata: dict[str, Any]
    format: str = "qpu-quality-samples-v1"

    def save(self: Self, path: str | PathLike[str]) -> None:
        """Atomically persist this manifest as canonical JSON."""
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        temporary = location.with_suffix(location.suffix + ".tmp")
        payload = asdict(self)
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(location)

    @classmethod
    def load(cls: type[SampleManifest], path: str | PathLike[str]) -> SampleManifest:
        """Load and validate a manifest written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != "qpu-quality-samples-v1":
            raise ValueError("unsupported sample manifest format")
        records = payload.get("samples")
        if not isinstance(records, list) or not records:
            raise ValueError("sample manifest must contain samples")
        samples = tuple(SampleRecord(**record) for record in records)
        if len({record.sample_id for record in samples}) != len(samples):
            raise ValueError("sample manifest IDs must be unique")
        return cls(str(payload["task"]), int(payload["seed"]), samples, dict(payload.get("metadata", {})))


def imagenet_stratified_manifest(
    root: str | PathLike[str],
    *,
    samples_per_class: int = 5,
    seed: int = 20_260_910,
    hash_contents: bool = False,
) -> SampleManifest:
    """Select an unbiased fixed number of ImageFolder images from every class."""
    directory = Path(root)
    classes = sorted(path for path in directory.iterdir() if path.is_dir())
    if len(classes) != 1_000:
        raise ValueError(f"ImageNet validation root must contain 1000 class directories, found {len(classes)}")
    records: list[SampleRecord] = []
    for label, class_path in enumerate(classes):
        files = sorted(path for path in class_path.iterdir() if path.is_file())
        if len(files) < samples_per_class:
            raise ValueError(f"ImageNet class {class_path.name!r} has fewer than {samples_per_class} images")
        ranked = sorted(
            files,
            key=lambda path: hashlib.sha256(f"{seed}:{path.relative_to(directory)}".encode()).digest(),
        )[:samples_per_class]
        for path in ranked:
            relative = path.relative_to(directory).as_posix()
            records.append(
                SampleRecord(
                    sample_id=relative,
                    relative_path=relative,
                    label=label,
                    group=class_path.name,
                    sha256=sha256_file(path) if hash_contents else None,
                )
            )
    return SampleManifest(
        "imagenet-classification",
        seed,
        tuple(records),
        {"classes": 1_000, "samples_per_class": samples_per_class, "count": len(records)},
    )


__all__ = ["SampleManifest", "SampleRecord", "imagenet_stratified_manifest", "sha256_file"]
