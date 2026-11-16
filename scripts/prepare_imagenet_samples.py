# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==26.0.1"]
# ///
"""Reconstruct the exact retained ImageNet-128 sample files from pinned Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_common import contained, sha256  # noqa: E402

SOURCE_SHA256 = "86cd7452777bda7e907c9a68e6079f16371b4f4c304cf710084e692d0a2fc943"
SOURCE_URL = "https://huggingface.co/datasets/benjamin-paine/imagenet-1k-128x128/resolve/dd3adfd6ecfb3ef3ae26205733bf5d9d5410643b/data/validation-00000-of-00001.parquet"


def prepare(parquet: Path, output: Path, download: bool) -> dict:
    """Write only manifest-selected JPEGs and enforce their original byte hashes."""
    if not parquet.exists() and download:
        parquet.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as source, parquet.open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
    if sha256(parquet) != SOURCE_SHA256:
        raise ValueError("source Parquet SHA256 differs from the retained dataset")
    manifest = json.loads((ROOT / "artifact/evidence/manifests/imagenet-samples-5000.json").read_text())
    expected = {s["relative_path"]: s["sha256"] for s in manifest["samples"]}
    by_digest = {}
    for relative, digest in expected.items():
        by_digest.setdefault(digest, []).append(relative)
    output.mkdir(parents=True, exist_ok=False)
    written = set()
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128, columns=["image", "label"]):
        for row in batch.to_pylist():
            raw = row["image"]["bytes"]
            # Match the original encoded bytes, preserving duplicate content
            # under each distinct sample identity in the retained manifest.
            digest = hashlib.sha256(raw).hexdigest()
            for relative in by_digest.get(digest, []):
                if int(Path(relative).parent.name) != int(row["label"]):
                    continue
                path = contained(output, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                written.add(relative)
    if written != set(expected):
        raise ValueError(f"reconstructed {len(written)}/{len(expected)} selected images; keep the original manifest")
    return {"ok": True, "samples": len(written), "source_sha256": SOURCE_SHA256, "output": str(output)}


def main() -> None:
    """Prepare external inputs separately from runtime dependencies and timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Acquire the pinned 217 MB Parquet if absent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.parquet, args.output, args.download), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(2, f"ImageNet preparation: {error}\n")


if __name__ == "__main__":
    main()
