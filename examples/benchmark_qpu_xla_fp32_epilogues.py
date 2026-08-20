"""Benchmark standalone FP32 bias/ReLU epilogues against prepared CPU outputs."""

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
from qpu_xla.ops import bias_activation


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
    """Measure bias-only, ReLU-only, and fused bias+ReLU operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--columns", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("fp32-epilogues-benchmark.json"))
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0 or args.rows % 16 or args.columns % 16:
        parser.error("rows and columns must be positive multiples of 16")

    shape = (args.rows, args.columns)
    rng = np.random.default_rng(args.seed)
    source_value = rng.standard_normal(shape, dtype=np.float32)
    bias_value = rng.standard_normal((args.columns,), dtype=np.float32)
    numpy_output = np.empty_like(source_value)
    torch_source = torch.from_numpy(source_value)
    torch_bias = torch.from_numpy(bias_value)
    torch_output = torch.empty_like(torch_source)
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
        source = device.tensor(shape, np.float32)
        bias = device.tensor((args.columns,), np.float32)
        destination = device.tensor(shape, np.float32)
        source.numpy()[:] = source_value
        bias.numpy()[:] = bias_value

        for operation, use_bias, use_relu in (
            ("bias", True, False),
            ("relu", False, True),
            ("bias_relu", True, True),
        ):
            def numpy_reference() -> None:
                if use_bias:
                    np.add(source_value, bias_value, out=numpy_output)
                if use_relu:
                    np.maximum(numpy_output if use_bias else source_value, 0, out=numpy_output)

            def torch_reference() -> None:
                if use_bias:
                    torch.add(torch_source, torch_bias, out=torch_output)
                if use_relu:
                    if use_bias:
                        torch.relu_(torch_output)
                    else:
                        torch.clamp_min(torch_source, 0, out=torch_output)

            def qpu_candidate() -> None:
                bias_activation(
                    destination,
                    source,
                    bias if use_bias else None,
                    relu=use_relu,
                    queue=queue,
                ).wait()

            numpy_seconds = _samples(numpy_reference, args.warmup, args.repeat)
            torch_seconds = _samples(torch_reference, args.warmup, args.repeat)
            qpu_seconds = _samples(qpu_candidate, args.warmup, args.repeat)
            numpy_reference()
            expected = np.array(numpy_output, copy=True)
            qpu_candidate()
            error = float(np.max(np.abs(destination.numpy() - expected), initial=0.0))
            cpu_seconds = min((numpy_seconds, torch_seconds), key=median)
            speedup = median(cpu_seconds) / median(qpu_seconds)
            results.append(
                {
                    "operation": operation,
                    "shape": list(shape),
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
                f"{operation}: NumPy {median(numpy_seconds) * 1e3:.3f} ms, "
                f"Torch {median(torch_seconds) * 1e3:.3f} ms, QPU {median(qpu_seconds) * 1e3:.3f} ms, "
                f"{speedup:.3f}x, error {error:g}"
            )

        source.buffer.close()
        bias.buffer.close()
        destination.buffer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "case": {"name": f"fp32-epilogue-r{args.rows}-c{args.columns}", "shape": list(shape)},
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
