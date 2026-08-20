from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from qpu_xla import Device
from qpu_xla.benchmark import CandidateRegistry, collect_metadata
from qpu_xla.models.tinyllama import (
    TinyLlamaCheckpoint,
    TinyLlamaConfig,
    TinyLlamaReferenceRuntime,
    TinyLlamaW8A8Runtime,
)
from qpu_xla.scheduler import Placement
from qpu_xla.workloads import LLAMA_DENSE_V1, LlamaWorkload


def _samples(fn: Any, *, warmup: int, repeat: int) -> tuple[tuple[float, ...], Any]:
    result = None
    for _ in range(warmup):
        result = fn()
    durations = []
    for _ in range(repeat):
        start = perf_counter()
        result = fn()
        durations.append(perf_counter() - start)
    return tuple(durations), result


def _synthetic_checkpoint(case: LlamaWorkload, layers: int, seed: int) -> TinyLlamaCheckpoint:
    config = TinyLlamaConfig(
        vocab_size=case.vocabulary_size,
        hidden_size=case.hidden_size,
        intermediate_size=case.intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=case.query_heads,
        num_key_value_heads=case.kv_heads,
        max_position_embeddings=max(case.tokens, case.cache_length + 16, 256),
    )
    rng = np.random.default_rng(seed)
    tensors: dict[str, npt.NDArray[np.float32]] = {}
    shared_embedding = np.ascontiguousarray(
        rng.standard_normal((config.vocab_size, config.hidden_size), dtype=np.float32) * np.float32(0.02)
    )
    for name, shape in config.expected_shapes().items():
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            tensors[name] = shared_embedding
        elif name.endswith("layernorm.weight") or name == "model.norm.weight":
            tensors[name] = np.ones(shape, dtype=np.float32)
        else:
            tensors[name] = np.ascontiguousarray(rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02))
    return TinyLlamaCheckpoint(config, tensors)


def _trace_summary(trace: dict[str, list[dict[str, object]]], invocations: int) -> dict[str, object]:
    durations: dict[str, float] = {}
    categories: dict[str, float] = {}
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or event.get("cat") == "queue":
            continue
        name = str(event["name"])
        category = str(event["cat"])
        raw_duration = event.get("dur", 0.0)
        if not isinstance(raw_duration, int | float):
            continue
        duration = float(raw_duration) / 1_000_000.0
        durations[name] = durations.get(name, 0.0) + duration
        categories[category] = categories.get(category, 0.0) + duration
    return {
        "mean_seconds_per_forward_by_event": {name: value / invocations for name, value in sorted(durations.items())},
        "mean_seconds_per_forward_by_category": {
            name: value / invocations for name, value in sorted(categories.items())
        },
    }


def _difference(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]) -> dict[str, object]:
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    actual64 = np.nan_to_num(actual.astype(np.float64))
    expected64 = np.nan_to_num(expected.astype(np.float64))
    rmse = float(np.sqrt(np.mean(np.square(actual64 - expected64))))
    reference_rms = float(np.sqrt(np.mean(np.square(expected64))))
    denominator = float(np.linalg.norm(actual64.ravel()) * np.linalg.norm(expected64.ravel()))
    cosine = 1.0 if denominator == 0.0 else float(np.dot(actual64.ravel(), expected64.ravel()) / denominator)
    return {
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "mean_abs_error": float(np.mean(absolute)),
        "p99_abs_error": float(np.percentile(absolute, 99)),
        "normalized_rmse": rmse / max(reference_rms, 1e-12),
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "top1_agreement": float(np.mean(np.argmax(actual, axis=1) == np.argmax(expected, axis=1))),
        "nan_count": int(np.count_nonzero(np.isnan(actual))),
        "inf_count": int(np.count_nonzero(np.isinf(actual))),
    }


def _seed_w8a8_session(
    session: Any,
    keys: list[npt.NDArray[np.float32]],
    values: list[npt.NDArray[np.float32]],
    length: int,
) -> None:
    """Install identical synthetic KV history into a runtime benchmark session."""
    for destination, source in zip(session._keys, keys, strict=True):
        destination.numpy()[:length] = source
    for destination, source in zip(session._values, values, strict=True):
        destination.numpy()[:length] = source
    session._length = length


def _run_decode_benchmark(
    case: LlamaWorkload,
    checkpoint: TinyLlamaCheckpoint,
    candidates: CandidateRegistry,
    *,
    warmup: int,
    repeat: int,
    seed: int,
    data_area_mib: int,
) -> tuple[dict[str, object], str]:
    """Measure one-token decode against FP32 and calibrated CPU oracles."""
    rng = np.random.default_rng(seed + 1)
    keys = [
        rng.standard_normal((case.cache_length, checkpoint.config.key_value_size), dtype=np.float32) * np.float32(0.02)
        for _ in range(checkpoint.config.num_hidden_layers)
    ]
    values = [
        rng.standard_normal((case.cache_length, checkpoint.config.key_value_size), dtype=np.float32) * np.float32(0.02)
        for _ in range(checkpoint.config.num_hidden_layers)
    ]
    token = 1

    with (
        Device.fake() as cpu_device,
        TinyLlamaW8A8Runtime(
            cpu_device,
            checkpoint,
            max_batch=1,
            candidates=candidates,
            placement=Placement.CPU,
        ) as cpu_runtime,
        cpu_runtime.session() as cpu_session,
    ):
        _seed_w8a8_session(cpu_session, keys, values, case.cache_length)

        def cpu_decode() -> npt.NDArray[np.float32]:
            cpu_session._length = case.cache_length
            return cast(npt.NDArray[np.float32], cpu_session.decode(token))

        fp32_seconds, fp32_result = _samples(cpu_decode, warmup=warmup, repeat=repeat)

    with (
        Device.fake() as oracle_device,
        TinyLlamaW8A8Runtime(
            oracle_device,
            checkpoint,
            max_batch=1,
            candidates=candidates,
        ) as oracle_runtime,
        oracle_runtime.session() as oracle_session,
    ):
        _seed_w8a8_session(oracle_session, keys, values, case.cache_length)
        oracle_result = oracle_session.decode(token)

    with (
        Device.open(data_area_size=data_area_mib * 1024 * 1024) as device,
        TinyLlamaW8A8Runtime(
            device,
            checkpoint,
            max_batch=1,
            candidates=candidates,
        ) as qpu_runtime,
        qpu_runtime.session() as qpu_session,
    ):
        metadata = collect_metadata(
            device,
            extra={"manifest": f"{LLAMA_DENSE_V1.name}-v{LLAMA_DENSE_V1.version}", "semantics": "steady-state"},
        )
        _seed_w8a8_session(qpu_session, keys, values, case.cache_length)

        def qpu_decode() -> npt.NDArray[np.float32]:
            qpu_session._length = case.cache_length
            return cast(npt.NDArray[np.float32], qpu_session.decode(token))

        qpu_seconds, qpu_result = _samples(qpu_decode, warmup=warmup, repeat=repeat)
        trace = _trace_summary(qpu_runtime.queue.chrome_trace(), warmup + repeat)

    assert fp32_result is not None and qpu_result is not None
    report = {
        "metadata": metadata,
        "case": asdict(case),
        "config": asdict(checkpoint.config),
        "layers": checkpoint.config.num_hidden_layers,
        "weight_bytes": {
            "fp32_contract": checkpoint.config.estimate_weight_bytes(),
            "w8a8_contract": checkpoint.config.estimate_weight_bytes(quantized=True),
        },
        "timings": {
            "fp32_cpu_seconds": list(fp32_seconds),
            "mixed_cpu_qpu_seconds": list(qpu_seconds),
            "speedup_over_fp32_cpu": median(fp32_seconds) / median(qpu_seconds),
        },
        "correctness": {
            "qpu_vs_calibrated_cpu_oracle": _difference(qpu_result[None, :], oracle_result[None, :]),
            "qpu_vs_fp32": _difference(qpu_result[None, :], fp32_result[None, :]),
        },
        "trace": trace,
    }
    message = (
        f"{case.name}, {checkpoint.config.num_hidden_layers} layer(s): "
        f"FP32 CPU {median(fp32_seconds) * 1e3:.3f} ms, "
        f"mixed CPU/QPU {median(qpu_seconds) * 1e3:.3f} ms, "
        f"speedup {median(fp32_seconds) / median(qpu_seconds):.3f}x, "
        f"QPU-vs-oracle max error "
        f"{_difference(qpu_result[None, :], oracle_result[None, :])['max_abs_error']:g}"
    )
    return report, message


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the complete mixed CPU/QPU dense Llama runtime")
    parser.add_argument(
        "--case",
        default="prefill-h1024-t64",
        choices=[case.name for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)],
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=128)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("experiment_logs/20260819-qpu-xla-w8a8/llama-calibrated.candidates.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("tinyllama-runtime-benchmark.json"))
    args = parser.parse_args()
    if args.layers <= 0 or args.warmup < 0 or args.repeat <= 0 or args.data_area_mib <= 0:
        parser.error("layers, repeat, and data-area-mib must be positive; warmup must be non-negative")
    case = next(case for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout) if case.name == args.case)
    assert isinstance(case, LlamaWorkload)
    candidates = CandidateRegistry.load(args.candidates)
    checkpoint = _synthetic_checkpoint(case, args.layers, args.seed)
    if case.phase == "decode":
        report, message = _run_decode_benchmark(
            case,
            checkpoint,
            candidates,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            data_area_mib=args.data_area_mib,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(message)
        print(f"saved {args.output}")
        return
    tokens = np.arange(case.tokens, dtype=np.int32) % checkpoint.config.vocab_size

    fp32_runtime = TinyLlamaReferenceRuntime(checkpoint)
    fp32_seconds, fp32_result = _samples(
        lambda: fp32_runtime.forward(tokens),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    with (
        Device.fake() as cpu_device,
        TinyLlamaW8A8Runtime(
            cpu_device,
            checkpoint,
            max_batch=case.tokens,
            candidates=candidates,
        ) as cpu_runtime,
    ):
        cpu_seconds, cpu_result = _samples(
            lambda: cpu_runtime.forward(tokens),
            warmup=args.warmup,
            repeat=args.repeat,
        )
    with (
        Device.open(data_area_size=args.data_area_mib * 1024 * 1024) as device,
        TinyLlamaW8A8Runtime(
            device,
            checkpoint,
            max_batch=case.tokens,
            candidates=candidates,
        ) as qpu_runtime,
    ):
        metadata = collect_metadata(
            device,
            extra={"manifest": f"{LLAMA_DENSE_V1.name}-v{LLAMA_DENSE_V1.version}", "semantics": "steady-state"},
        )
        qpu_seconds, qpu_result = _samples(
            lambda: qpu_runtime.forward(tokens),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        trace = _trace_summary(qpu_runtime.queue.chrome_trace(), args.warmup + args.repeat)

    assert fp32_result is not None and cpu_result is not None and qpu_result is not None
    cpu_median = median(cpu_seconds)
    qpu_median = median(qpu_seconds)
    report = {
        "metadata": metadata,
        "case": asdict(case),
        "config": asdict(checkpoint.config),
        "layers": args.layers,
        "weight_bytes": {
            "fp32_contract": checkpoint.config.estimate_weight_bytes(),
            "w8a8_contract": checkpoint.config.estimate_weight_bytes(quantized=True),
        },
        "timings": {
            "fp32_numpy_reference_seconds": list(fp32_seconds),
            "calibrated_cpu_oracle_seconds": list(cpu_seconds),
            "mixed_cpu_qpu_seconds": list(qpu_seconds),
            "speedup_over_calibrated_cpu_oracle": cpu_median / qpu_median,
            "speedup_over_fp32_numpy": median(fp32_seconds) / qpu_median,
        },
        "correctness": {
            "qpu_vs_calibrated_cpu_oracle": _difference(qpu_result.logits, cpu_result.logits),
            "calibrated_cpu_oracle_vs_fp32": _difference(cpu_result.logits, fp32_result.logits),
            "qpu_vs_fp32": _difference(qpu_result.logits, fp32_result.logits),
        },
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{case.name}, {args.layers} layer(s): FP32 NumPy {median(fp32_seconds) * 1e3:.3f} ms, "
        f"calibrated CPU oracle {cpu_median * 1e3:.3f} ms, mixed CPU/QPU {qpu_median * 1e3:.3f} ms, "
        f"FP32 speedup {median(fp32_seconds) / qpu_median:.3f}x, "
        f"QPU-vs-oracle max error {_difference(qpu_result.logits, cpu_result.logits)['max_abs_error']:g}"
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
