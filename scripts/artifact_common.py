"""Read retained campaign files without reconstructing personal working paths."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, and symlinks outside the package."""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes package: {relative}")
    return path


def read_record(root: Path, relative: str) -> dict:
    """Read a live campaign file or its uniquely identified archived equivalent."""
    direct = contained(root, relative)
    if direct.is_file():
        return json.loads(direct.read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    matches = [r for r in provenance["records"] if r["source_path"].endswith("/" + relative)]
    if len(matches) != 1:
        raise ValueError(f"expected one archived record for {relative}, found {len(matches)}")
    record = matches[0]
    path = contained(root, record["archive_path"])
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256_uncompressed"]:
        raise ValueError(f"archive provenance mismatch: {relative}")
    return json.loads(data)


def verify_sums(root: Path, manifest: Path) -> int:
    """Check a SHA256SUMS file relative to its own package root."""
    count = 0
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = contained(root, relative)
        if sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {path}")
        count += 1
    if not count:
        raise ValueError(f"empty checksum manifest: {manifest}")
    return count
