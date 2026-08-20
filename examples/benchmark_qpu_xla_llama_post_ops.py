"""Benchmark non-projection Llama operators across CPU, QPU, and hybrid paths."""

from __future__ import annotations

import argparse
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
from qpu_xla.models.tinyllama import KvCacheFp32
from qpu_xla.ops import embedding_lookup_fp32, greedy_sample_fp32, residual_add_fp32
from qpu_xla.scheduler import Placement
from qpu_xla.workloads import LLAMA_DENSE_V1, LlamaWorkload

_NUMPY_BACKEND = numpy_backend_label()


def _samples(fn: Any, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        values.append(perf_counter() - start)
    return tuple(values)


def _partitions(total: int, alignment: int = 1) -> tuple[int, ...]:
    return tuple(
        sorted(
            {int(total * fraction) // alignment * alignment for fraction in (0.125, 0.25, 0.5, 0.75, 0.875)}
            - {0, total}
        )
    )


def _record(
    operation: str,
    placement: str,
    cpu_name: str,
    cpu_seconds: tuple[float, ...],
    candidate_seconds: tuple[float, ...],
    error: float,
    partition: dict[str, int | str] | None = None,
    cpu_backends: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, object]:
    speedup = median(cpu_seconds) / median(candidate_seconds)
    print(
        f"{operation} {placement}{'' if partition is None else ' ' + str(partition)}: "
        f"CPU {median(cpu_seconds) * 1e3:.3f} ms ({cpu_name}), "
        f"candidate {median(candidate_seconds) * 1e3:.3f} ms, {speedup:.3f}x, error {error:g}"
    )
    return {
        "operation": operation,
        "placement": placement,
        "partition": partition,
        "cpu_reference": cpu_name,
        "cpu_seconds": list(cpu_seconds),
        "cpu_backend_seconds": {
            name: list(samples) for name, samples in (cpu_backends or {cpu_name: cpu_seconds}).items()
        },
        "candidate_seconds": list(candidate_seconds),
        "speedup": speedup,
        "max_abs_error": error,
        "status": "supported-win" if error <= 1e-3 and speedup >= 1.05 else "experimental-correct-slower",
    }


def _fastest_cpu(candidates: tuple[tuple[str, tuple[float, ...]], ...]) -> tuple[str, tuple[float, ...]]:
    return min(candidates, key=lambda item: median(item[1]))


def _run(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    # Residual add.
    left_value = rng.standard_normal((case.tokens, case.hidden_size), dtype=np.float32)
    right_value = rng.standard_normal(left_value.shape, dtype=np.float32)
    expected = left_value + right_value
    torch_left, torch_right = torch.from_numpy(left_value), torch.from_numpy(right_value)
    residual_backends = {
        f"{_NUMPY_BACKEND}.add": _samples(lambda: np.add(left_value, right_value), warmup, repeat),
        "torch.add": _samples(lambda: torch.add(torch_left, torch_right), warmup, repeat),
    }
    cpu_name, cpu_seconds = _fastest_cpu(tuple(residual_backends.items()))
    left, right, destination = (
        device.tensor(left_value.shape, np.float32),
        device.tensor(right_value.shape, np.float32),
        device.tensor(expected.shape, np.float32),
    )
    left.numpy()[:], right.numpy()[:] = left_value, right_value
    qpu_seconds = _samples(
        lambda: residual_add_fp32(destination, left, right, queue=qpu_queue, placement=Placement.QPU).wait(),
        warmup,
        repeat,
    )
    results.append(
        _record(
            "residual_add",
            "qpu",
            cpu_name,
            cpu_seconds,
            qpu_seconds,
            float(np.max(np.abs(destination.numpy() - expected))),
            cpu_backends=residual_backends,
        )
    )
    for units in _partitions(case.tokens):
        seconds = _samples(
            lambda units=units: residual_add_fp32(
                destination,
                left,
                right,
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                placement=Placement.HYBRID,
                qpu_rows=units,
            ).wait(),
            warmup,
            repeat,
        )
        results.append(
            _record(
                "residual_add",
                "hybrid",
                cpu_name,
                cpu_seconds,
                seconds,
                float(np.max(np.abs(destination.numpy() - expected))),
                {"axis": "rows", "qpu_units": units, "total_units": case.tokens},
                residual_backends,
            )
        )

    # Embedding gather.
    table_value = rng.standard_normal((case.vocabulary_size, case.hidden_size), dtype=np.float32)
    ids_value = rng.integers(0, case.vocabulary_size, size=(case.tokens,), dtype=np.int32)
    expected_embedding = table_value[ids_value]
    torch_table, torch_ids = torch.from_numpy(table_value), torch.from_numpy(ids_value.astype(np.int64))
    embedding_backends = {
        f"{_NUMPY_BACKEND}.gather": _samples(lambda: table_value[ids_value], warmup, repeat),
        "torch.embedding": _samples(lambda: torch.nn.functional.embedding(torch_ids, torch_table), warmup, repeat),
    }
    cpu_name, cpu_seconds = _fastest_cpu(tuple(embedding_backends.items()))
    table, ids, embedded = (
        device.tensor(table_value.shape, np.float32),
        device.tensor(ids_value.shape, np.int32),
        device.tensor(expected_embedding.shape, np.float32),
    )
    table.numpy()[:], ids.numpy()[:] = table_value, ids_value
    seconds = _samples(
        lambda: embedding_lookup_fp32(embedded, ids, table, queue=qpu_queue, placement=Placement.QPU).wait(),
        warmup,
        repeat,
    )
    results.append(
        _record(
            "embedding",
            "qpu",
            cpu_name,
            cpu_seconds,
            seconds,
            float(np.max(np.abs(embedded.numpy() - expected_embedding))),
            cpu_backends=embedding_backends,
        )
    )
    for units in _partitions(case.tokens):
        seconds = _samples(
            lambda units=units: embedding_lookup_fp32(
                embedded,
                ids,
                table,
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                placement=Placement.HYBRID,
                qpu_tokens=units,
            ).wait(),
            warmup,
            repeat,
        )
        results.append(
            _record(
                "embedding",
                "hybrid",
                cpu_name,
                cpu_seconds,
                seconds,
                float(np.max(np.abs(embedded.numpy() - expected_embedding))),
                {"axis": "tokens", "qpu_units": units, "total_units": case.tokens},
                embedding_backends,
            )
        )

    # Greedy argmax.
    logits_value = rng.standard_normal((case.tokens, case.vocabulary_size), dtype=np.float32)
    expected_tokens = np.argmax(logits_value, axis=1).astype(np.int32)
    torch_logits = torch.from_numpy(logits_value)
    argmax_backends = {
        f"{_NUMPY_BACKEND}.argmax": _samples(lambda: np.argmax(logits_value, axis=1), warmup, repeat),
        "torch.argmax": _samples(lambda: torch.argmax(torch_logits, dim=1), warmup, repeat),
    }
    cpu_name, cpu_seconds = _fastest_cpu(tuple(argmax_backends.items()))
    logits, sampled = device.tensor(logits_value.shape, np.float32), device.tensor((case.tokens,), np.int32)
    logits.numpy()[:] = logits_value
    seconds = _samples(
        lambda: greedy_sample_fp32(sampled, logits, queue=qpu_queue, placement=Placement.QPU).wait(), warmup, repeat
    )
    results.append(
        _record(
            "argmax",
            "qpu",
            cpu_name,
            cpu_seconds,
            seconds,
            float(np.max(np.abs(sampled.numpy().astype(np.int64) - expected_tokens.astype(np.int64)))),
            cpu_backends=argmax_backends,
        )
    )
    for units in _partitions(case.tokens):
        seconds = _samples(
            lambda units=units: greedy_sample_fp32(
                sampled, logits, queue=qpu_queue, cpu_queue=cpu_queue, placement=Placement.HYBRID, qpu_rows=units
            ).wait(),
            warmup,
            repeat,
        )
        results.append(
            _record(
                "argmax",
                "hybrid",
                cpu_name,
                cpu_seconds,
                seconds,
                float(np.max(np.abs(sampled.numpy().astype(np.int64) - expected_tokens.astype(np.int64)))),
                {"axis": "rows", "qpu_units": units, "total_units": case.tokens},
                argmax_backends,
            )
        )

    # KV append copies all KV heads as one contiguous width.
    kv_width = case.kv_heads * case.head_dim
    key_value = rng.standard_normal((case.tokens, kv_width), dtype=np.float32)
    value_value = rng.standard_normal((case.tokens, kv_width), dtype=np.float32)
    key, value = device.tensor(key_value.shape, np.float32), device.tensor(value_value.shape, np.float32)
    key.numpy()[:], value.numpy()[:] = key_value, value_value
    torch_key, torch_value = torch.from_numpy(key_value), torch.from_numpy(value_value)
    kv_backends = {
        f"{_NUMPY_BACKEND}.copy(x2)": _samples(lambda: (np.copy(key_value), np.copy(value_value)), warmup, repeat),
        "torch.clone(x2)": _samples(lambda: (torch_key.clone(), torch_value.clone()), warmup, repeat),
    }
    cpu_name, cpu_seconds = _fastest_cpu(tuple(kv_backends.items()))
    with KvCacheFp32(device, capacity=max(case.tokens, 2), depth=kv_width, value_dim=kv_width) as cache:

        def append(placement: Placement, units: int | None = None) -> None:
            cache.append(
                key,
                value,
                queue=qpu_queue,
                cpu_queue=cpu_queue if placement is Placement.HYBRID else None,
                placement=placement,
                qpu_tokens=units,
            ).wait()
            cache.reset()

        seconds = _samples(lambda: append(Placement.QPU), warmup, repeat)
        results.append(
            _record(
                "kv_append",
                "qpu",
                cpu_name,
                cpu_seconds,
                seconds,
                0.0,
                cpu_backends=kv_backends,
            )
        )
        for units in _partitions(case.tokens):
            seconds = _samples(lambda units=units: append(Placement.HYBRID, units), warmup, repeat)
            results.append(
                _record(
                    "kv_append",
                    "hybrid",
                    cpu_name,
                    cpu_seconds,
                    seconds,
                    0.0,
                    {"axis": "tokens", "qpu_units": units, "total_units": case.tokens},
                    kv_backends,
                )
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="prefill-h512-t16",
        choices=[case.name for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("llama-post-ops-benchmark.json"))
    args = parser.parse_args()
    case = cast(
        LlamaWorkload,
        next(case for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout) if case.name == args.case),
    )
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
                "torch": torch.__version__,
                "cpu_reference": "fastest-of-torch-and-numpy",
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
            },
        )
        results = _run(device, qpu_queue, cpu_queue, case, args.warmup, args.repeat, np.random.default_rng(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "case": asdict(case), "results": results}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
