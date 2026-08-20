"""Benchmark prepared FP32 2x2/stride-2 pooling against NumPy and Torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import torch

from qpu_xla import Device
from qpu_xla.benchmark import collect_metadata
from qpu_xla.ops import PreparedPool2DFP32


def _samples(fn: Any, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        values.append(perf_counter() - start)
    return tuple(values)


def _numpy_pool(source: np.ndarray, mode: str) -> np.ndarray:
    x00, x01 = source[:, :, 0::2, 0::2], source[:, :, 0::2, 1::2]
    x10, x11 = source[:, :, 1::2, 0::2], source[:, :, 1::2, 1::2]
    if mode == "max":
        return np.maximum(np.maximum(x00, x01), np.maximum(x10, x11))
    return ((x00 + x01) + (x10 + x11)) * np.float32(0.25)


def main() -> None:
    """Run both supported FP32 pooling modes and retain all timing samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("fp32-pool2d-benchmark.json"))
    args = parser.parse_args()
    if args.height % 2 or args.width % 2:
        parser.error("height and width must be even")

    input_shape = (args.batch, args.channels, args.height, args.width)
    output_shape = (args.batch, args.channels, args.height // 2, args.width // 2)
    source_value = np.random.default_rng(args.seed).standard_normal(input_shape, dtype=np.float32)
    torch_source = torch.from_numpy(source_value)
    results: list[dict[str, object]] = []
    with Device.open(data_area_size=args.data_area_mib * 1024 * 1024) as device, device.queue() as queue:
        metadata = collect_metadata(
            device,
            extra={
                "semantics": "steady-state-total-prepared-metadata",
                "cpu_reference": "fastest-of-torch-native-and-numpy-vectorized",
                "torch": torch.__version__,
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
            },
        )
        source = device.tensor(input_shape, np.float32)
        destination = device.tensor(output_shape, np.float32)
        source.numpy()[:] = source_value
        with PreparedPool2DFP32(source, destination) as plan:
            for mode in ("max", "avg"):
                expected = _numpy_pool(source_value, mode)
                numpy_seconds = _samples(lambda mode=mode: _numpy_pool(source_value, mode), args.warmup, args.repeat)
                torch_fn = torch.nn.functional.max_pool2d if mode == "max" else torch.nn.functional.avg_pool2d
                torch_seconds = _samples(lambda: torch_fn(torch_source, 2, 2), args.warmup, args.repeat)
                qpu_seconds = _samples(
                    lambda mode=mode: plan.execute(mode=mode, queue=queue).wait(), args.warmup, args.repeat
                )
                actual = np.array(destination.numpy(), copy=True)
                error = float(
                    np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0)
                )
                cpu_seconds = min((numpy_seconds, torch_seconds), key=median)
                speedup = median(cpu_seconds) / median(qpu_seconds)
                results.append(
                    {
                        "mode": mode,
                        "input_shape": list(input_shape),
                        "output_shape": list(output_shape),
                        "numpy_cpu_seconds": list(numpy_seconds),
                        "torch_cpu_seconds": list(torch_seconds),
                        "qpu_seconds": list(qpu_seconds),
                        "speedup": speedup,
                        "max_abs_error": error,
                        "status": (
                            "supported-win" if error <= 1e-6 and speedup >= 1.05 else "experimental-correct-slower"
                        ),
                    }
                )
                print(
                    f"{mode}: NumPy {median(numpy_seconds) * 1e3:.3f} ms, "
                    f"Torch {median(torch_seconds) * 1e3:.3f} ms, QPU {median(qpu_seconds) * 1e3:.3f} ms, "
                    f"{speedup:.3f}x, error {error:g}"
                )
        source.buffer.close()
        destination.buffer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "case": {
                    "name": f"pool2d-n{args.batch}-c{args.channels}-h{args.height}-w{args.width}",
                    "input_shape": list(input_shape),
                    "output_shape": list(output_shape),
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
