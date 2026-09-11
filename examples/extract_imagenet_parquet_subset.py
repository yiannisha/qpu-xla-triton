"""Extract a deterministic class-balanced ImageFolder subset from ImageNet parquet."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return bytes(value["bytes"])
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    raise ValueError("ImageNet parquet image column does not contain embedded bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_910)
    args = parser.parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("ImageNet parquet extraction requires pyarrow") from exc

    selected: dict[int, list[tuple[bytes, bytes]]] = defaultdict(list)
    parquet = pq.ParquetFile(args.parquet)
    for batch in parquet.iter_batches(columns=("image", "label"), batch_size=512):
        for row in batch.to_pylist():
            label = int(row["label"])
            if not 0 <= label < 1_000:
                raise ValueError(f"ImageNet label {label} is outside [0, 1000)")
            image = _image_bytes(row["image"])
            rank = hashlib.sha256(str(args.seed).encode() + b":" + image).digest()
            values = selected[label]
            values.append((rank, image))
            values.sort(key=lambda item: item[0])
            del values[args.samples_per_class :]

    missing = [label for label in range(1_000) if len(selected[label]) < args.samples_per_class]
    if missing:
        raise ValueError(f"ImageNet parquet has too few samples for {len(missing)} classes")
    for label in range(1_000):
        directory = args.output / f"{label:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        for rank, image in selected[label]:
            (directory / f"{rank.hex()}.jpg").write_bytes(image)
    print(f"saved {1_000 * args.samples_per_class} images to {args.output}")


if __name__ == "__main__":
    main()
