"""Benchmark the CPU, QPU, and heterogeneous qpu_xla execution paths.

This is intentionally separate from the legacy examples.  It measures the
runtime operators and reports whole-operation wall time, including queue
submission and synchronization.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from qpu_xla import Device
from qpu_xla.ops import attention_fp32, hybrid_matmul, matmul, plan_matmul, scaled_dot_product_attention_fp32
from qpu_xla.scheduler import Placement


def _measure(fn, *, warmup: int, repeat: int) -> tuple[float, object]:
    result: object = None
    for _ in range(warmup):
        result = fn()
    samples: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        result = fn()
        samples.append(perf_counter() - start)
    return float(np.median(np.asarray(samples))), result


def _gops(operations: int, seconds: float) -> float:
    return operations / seconds / 1e9


def _matmul_benchmark(device: Device, *, size: int, warmup: int, repeat: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    left_value = rng.standard_normal((size, size), dtype=np.float32)
    right_value = rng.standard_normal((size, size), dtype=np.float32)
    expected = left_value @ right_value
    left = device.tensor(left_value.shape, np.float32)
    right = device.tensor(right_value.shape, np.float32)
    destination = device.tensor(expected.shape, np.float32)
    left.numpy()[:] = left_value
    right.numpy()[:] = right_value
    operations = 2 * size * size * size

    with device.queue() as cpu_queue, device.queue() as qpu_queue, device.queue() as auto_queue:
        def run_matmul(placement: Placement, queue) -> None:
            matmul(destination, left, right, queue=queue, placement=placement).wait()

        def run_hybrid(qpu_rows: int) -> None:
            hybrid_matmul(
                destination,
                left,
                right,
                qpu_queue=qpu_queue,
                cpu_queue=cpu_queue,
                qpu_rows=qpu_rows,
            ).wait()

        print(f"\n==== qpu_xla matmul matrix ({size}x{size} @ {size}x{size}) ====")
        print("configuration                 median ms    GOP/s   max abs error")

        configurations: list[tuple[str, object]] = [
            ("CPU only", lambda: run_matmul(Placement.CPU, cpu_queue)),
            ("QPU only", lambda: run_matmul(Placement.QPU, qpu_queue)),
            ("AUTO", lambda: run_matmul(Placement.AUTO, auto_queue)),
        ]
        # The useful hybrid region on this hardware is usually a small QPU
        # prefix: enough QPU work to overlap CPU work, without letting the
        # lower-throughput FP32 kernel become the critical path.
        for fraction in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.50, 0.75):
            rows = int(size * fraction) // 16 * 16
            configurations.append((f"CPU+QPU ({rows}/{size} QPU rows)", lambda rows=rows: run_hybrid(rows)))

        timings: dict[str, float] = {}
        for name, fn in configurations:
            elapsed, _ = _measure(fn, warmup=warmup, repeat=repeat)
            error = float(np.max(np.abs(destination.numpy() - expected)))
            timings[name] = elapsed
            print(f"{name:28s} {elapsed * 1e3:10.3f} {_gops(operations, elapsed):8.2f} {error:15.6g}")

        plan = plan_matmul(destination, left, right, placement=Placement.AUTO)
        print(f"AUTO selected implementation: {plan.candidate.name}")
        cpu_time = timings["CPU only"]
        for name, elapsed in timings.items():
            if name != "CPU only":
                print(f"  {name}: {cpu_time / elapsed:.2f}x vs CPU")


def _cpu_sdpa(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    scores = query @ key.T / np.sqrt(np.float32(query.shape[1]))
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ value


def _attention_benchmark(device: Device, *, size: int, warmup: int, repeat: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    query_value = rng.standard_normal((size, size), dtype=np.float32)
    key_value = rng.standard_normal((size, size), dtype=np.float32)
    value_value = rng.standard_normal((size, size), dtype=np.float32)
    query = device.tensor(query_value.shape, np.float32)
    key = device.tensor(key_value.shape, np.float32)
    value = device.tensor(value_value.shape, np.float32)
    destination = device.tensor((size, size), np.float32)
    query.numpy()[:] = query_value
    key.numpy()[:] = key_value
    value.numpy()[:] = value_value
    expected_mixed = _cpu_sdpa(query_value, key_value, value_value)
    expected_core = (query_value @ key_value.T) @ value_value
    operations = 4 * size * size * size

    def run_cpu() -> None:
        destination.numpy()[:] = _cpu_sdpa(query_value, key_value, value_value)

    print(f"\n==== qpu_xla attention matrix ({size}x{size}) ====")
    print("configuration                 median ms    GOP/s   max abs error")
    with device.queue() as qpu_queue:
        def run_core() -> None:
            attention_fp32(destination, query, key, value, queue=qpu_queue).wait()

        def run_mixed() -> None:
            scaled_dot_product_attention_fp32(destination, query, key, value, queue=qpu_queue).wait()

        configurations = (("CPU-only SDPA", run_cpu, expected_mixed), ("QPU GEMM core", run_core, expected_core), ("Mixed QPU+CPU SDPA", run_mixed, expected_mixed))
        timings: dict[str, float] = {}
        for name, fn, expected in configurations:
            elapsed, _ = _measure(fn, warmup=warmup, repeat=repeat)
            error = float(np.max(np.abs(destination.numpy() - expected)))
            timings[name] = elapsed
            print(f"{name:28s} {elapsed * 1e3:10.3f} {_gops(operations, elapsed):8.2f} {error:15.6g}")
        cpu_time = timings["CPU-only SDPA"]
        for name, elapsed in timings.items():
            if name != "CPU-only SDPA":
                print(f"  {name}: {cpu_time / elapsed:.2f}x vs CPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, help="square matmul/attention dimension")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 16:
        parser.error("--size must be a positive multiple of 16")
    # Mixed attention allocates padded intermediates per submission.  Keep
    # enough device arena for the requested repeat count; the v0 allocator
    # reclaims allocations when the device closes rather than immediately.
    data_area_size = max(16 * 1024 * 1024, args.repeat * args.warmup * args.size * args.size * 64)
    with Device.open(data_area_size=data_area_size) as device:
        _matmul_benchmark(device, size=args.size, warmup=args.warmup, repeat=args.repeat, seed=args.seed)
        _attention_benchmark(device, size=args.size, warmup=args.warmup, repeat=args.repeat, seed=args.seed + 1)


if __name__ == "__main__":
    main()
