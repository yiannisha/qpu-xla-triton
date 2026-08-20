"""Benchmark packaged multi-QPU FP32 minimum/maximum against CPU backends."""

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
from qpu_xla.ops import maximum, minimum


def _samples(fn: Any, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        values.append(perf_counter() - start)
    return tuple(values)


def main() -> None:
    """Measure both word-oriented FP32 min/max specializations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4_194_048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("fp32-minmax-benchmark.json"))
    args = parser.parse_args()
    if args.length <= 0 or args.length % 16:
        parser.error("length must be a positive multiple of 16")

    rng = np.random.default_rng(args.seed)
    left_value = rng.standard_normal(args.length, dtype=np.float32)
    right_value = rng.standard_normal(args.length, dtype=np.float32)
    numpy_output = np.empty_like(left_value)
    torch_left = torch.from_numpy(left_value)
    torch_right = torch.from_numpy(right_value)
    torch_output = torch.empty_like(torch_left)
    results: list[dict[str, object]] = []

    with Device.open(data_area_size=args.data_area_mib * 1024 * 1024) as device, device.queue() as queue:
        metadata = collect_metadata(
            device,
            extra={
                "semantics": "steady-state-total-preallocated-output",
                "cpu_reference": "fastest-of-torch-and-numpy",
                "torch": torch.__version__,
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
            },
        )
        left = device.tensor((args.length,), np.float32)
        right = device.tensor((args.length,), np.float32)
        destination = device.tensor((args.length,), np.float32)
        left.numpy()[:] = left_value
        right.numpy()[:] = right_value

        for operation, numpy_fn, torch_fn, qpu_fn in (
            ("minimum", np.minimum, torch.minimum, minimum),
            ("maximum", np.maximum, torch.maximum, maximum),
        ):
            numpy_seconds = _samples(
                lambda: numpy_fn(left_value, right_value, out=numpy_output), args.warmup, args.repeat
            )
            torch_seconds = _samples(
                lambda: torch_fn(torch_left, torch_right, out=torch_output), args.warmup, args.repeat
            )
            qpu_seconds = _samples(
                lambda: qpu_fn(destination, left, right, queue=queue).wait(), args.warmup, args.repeat
            )
            numpy_fn(left_value, right_value, out=numpy_output)
            qpu_fn(destination, left, right, queue=queue).wait()
            error = float(np.max(np.abs(destination.numpy() - numpy_output), initial=0.0))
            cpu_seconds = min((numpy_seconds, torch_seconds), key=median)
            speedup = median(cpu_seconds) / median(qpu_seconds)
            results.append(
                {
                    "operation": operation,
                    "shape": [args.length],
                    "numpy_cpu_seconds": list(numpy_seconds),
                    "torch_cpu_seconds": list(torch_seconds),
                    "qpu_seconds": list(qpu_seconds),
                    "speedup": speedup,
                    "max_abs_error": error,
                    "status": (
                        "supported-win" if error == 0.0 and speedup >= 1.05 else "experimental-correct-slower"
                    ),
                }
            )
            print(
                f"{operation}: NumPy {median(numpy_seconds) * 1e3:.3f} ms, "
                f"Torch {median(torch_seconds) * 1e3:.3f} ms, QPU {median(qpu_seconds) * 1e3:.3f} ms, "
                f"{speedup:.3f}x, error {error:g}"
            )

        left.buffer.close()
        right.buffer.close()
        destination.buffer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "case": {"name": f"fp32-minmax-n{args.length}", "shape": [args.length]},
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
