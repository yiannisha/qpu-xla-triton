"""Benchmark every FP32 Llama projection across CPU, QPU, and hybrid placement."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch

from qpu_xla import Device
from qpu_xla.benchmark import collect_metadata, numpy_backend_label
from qpu_xla.ops import PreparedFP32Linear, hybrid_column_partitions, hybrid_row_partitions
from qpu_xla.scheduler import Placement
from qpu_xla.workloads import LLAMA_DENSE_V1, LlamaWorkload


def _samples(fn: Any, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        values.append(perf_counter() - start)
    return tuple(values)


def _projections(case: LlamaWorkload, selected: str) -> tuple[tuple[str, int, int], ...]:
    values = {
        "q": ("q", case.hidden_size, case.hidden_size),
        "k": ("k", case.hidden_size, case.kv_heads * case.head_dim),
        "v": ("v", case.hidden_size, case.kv_heads * case.head_dim),
        "o": ("o", case.hidden_size, case.hidden_size),
        "gate": ("gate", case.hidden_size, case.intermediate_size),
        "up": ("up", case.hidden_size, case.intermediate_size),
        "down": ("down", case.intermediate_size, case.hidden_size),
        "lm_head": ("lm_head", case.hidden_size, case.vocabulary_size),
    }
    return tuple(values.values()) if selected == "all" else (values[selected],)


def _run(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    projection: tuple[str, int, int],
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    name, inputs, outputs = projection
    source_value = rng.standard_normal((case.tokens, inputs), dtype=np.float32)
    weight_value = rng.standard_normal((outputs, inputs), dtype=np.float32)
    torch_source = torch.from_numpy(source_value)
    torch_weight = torch.from_numpy(weight_value)

    def numpy_cpu() -> np.ndarray:
        return source_value @ weight_value.T

    def torch_cpu() -> torch.Tensor:
        return torch.nn.functional.linear(torch_source, torch_weight)

    expected = numpy_cpu()
    numpy_seconds = _samples(numpy_cpu, warmup, repeat)
    torch_seconds = _samples(torch_cpu, warmup, repeat)
    numpy_name = f"{numpy_backend_label()}.matmul"
    cpu_name, cpu_seconds = min(
        ((numpy_name, numpy_seconds), ("torch.nn.functional.linear", torch_seconds)),
        key=lambda pair: median(pair[1]),
    )
    source = device.tensor(source_value.shape, np.float32)
    destination = device.tensor(expected.shape, np.float32)
    source.numpy()[:] = source_value
    results: list[dict[str, object]] = []
    with PreparedFP32Linear(device, weight_value, max_batch=case.tokens) as plan:
        qpu_seconds = _samples(
            lambda: plan.execute(destination, source, queue=qpu_queue, placement=Placement.QPU).wait(), warmup, repeat
        )
        actual = np.array(destination.numpy(), copy=True)
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0))
        speedup = median(cpu_seconds) / median(qpu_seconds)
        results.append(
            {
                "case": asdict(case),
                "projection": name,
                "shape": [case.tokens, inputs, outputs],
                "placement": "qpu",
                "partition": None,
                "cpu_reference": cpu_name,
                "numpy_cpu_seconds": list(numpy_seconds),
                "numpy_openblas_seconds": list(numpy_seconds),
                "torch_cpu_seconds": list(torch_seconds),
                "candidate_seconds": list(qpu_seconds),
                "speedup": speedup,
                "max_abs_error": error,
                "status": "supported-win" if error <= 1e-3 and speedup >= 1.05 else "experimental-correct-slower",
            }
        )
        print(
            f"{case.name}/{name} QPU: CPU {median(cpu_seconds) * 1e3:.3f} ms ({cpu_name}), "
            f"candidate {median(qpu_seconds) * 1e3:.3f} ms, {speedup:.3f}x, error {error:g}"
        )
        partitions = (
            hybrid_column_partitions(plan.padded_outputs) if case.tokens == 1 else hybrid_row_partitions(case.tokens)
        )
        for units in partitions:
            hybrid_seconds = _samples(
                lambda units=units: plan.execute(
                    destination,
                    source,
                    queue=qpu_queue,
                    cpu_queue=cpu_queue,
                    placement=Placement.HYBRID,
                    qpu_units=units,
                ).wait(),
                warmup,
                repeat,
            )
            actual = np.array(destination.numpy(), copy=True)
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0))
            speedup = median(cpu_seconds) / median(hybrid_seconds)
            axis = "output-columns" if case.tokens == 1 else "rows"
            results.append(
                {
                    "case": asdict(case),
                    "projection": name,
                    "shape": [case.tokens, inputs, outputs],
                    "placement": "hybrid",
                    "partition": {
                        "axis": axis,
                        "qpu_units": units,
                        "total_units": plan.padded_outputs if case.tokens == 1 else case.tokens,
                    },
                    "cpu_reference": cpu_name,
                    "numpy_cpu_seconds": list(numpy_seconds),
                    "numpy_openblas_seconds": list(numpy_seconds),
                    "torch_cpu_seconds": list(torch_seconds),
                    "candidate_seconds": list(hybrid_seconds),
                    "speedup": speedup,
                    "max_abs_error": error,
                    "status": "supported-win" if error <= 1e-3 and speedup >= 1.05 else "experimental-correct-slower",
                }
            )
            print(
                f"{case.name}/{name} hybrid {axis}={units}: "
                f"{median(hybrid_seconds) * 1e3:.3f} ms, {speedup:.3f}x, error {error:g}"
            )
    source.buffer.close()
    destination.buffer.close()
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="decode-h2048-c512",
        choices=[case.name for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)],
    )
    parser.add_argument(
        "--projection", default="all", choices=("q", "k", "v", "o", "gate", "up", "down", "lm_head", "all")
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("fp32-linear-benchmark.json"))
    args = parser.parse_args()
    case = cast(
        LlamaWorkload,
        next(case for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout) if case.name == args.case),
    )
    results: list[dict[str, object]] = []
    with (
        Device.open(data_area_size=args.data_area_mib * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        metadata = collect_metadata(
            device,
            extra={
                "manifest": "llama-dense-v1",
                "semantics": "steady-state-total",
                "cpu_reference": "fastest-of-torch-and-numpy-blas",
                "torch": torch.__version__,
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
            },
        )
        rng = np.random.default_rng(args.seed)
        for projection in _projections(case, args.projection):
            results.extend(_run(device, qpu_queue, cpu_queue, case, projection, args.warmup, args.repeat, rng))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
