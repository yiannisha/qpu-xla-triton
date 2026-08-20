from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_functional

from qpu_xla import Device
from qpu_xla.benchmark import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PartitionEvidence,
    PerformanceEvidence,
    collect_metadata,
    numpy_backend_label,
)
from qpu_xla.memory import AccessMode
from qpu_xla.models.tinyllama import rms_norm_fp32
from qpu_xla.ops import apply_rope_tables_fp32, rope_tables_fp32, softmax_fp32, swiglu_fp32
from qpu_xla.scheduler import Placement
from qpu_xla.workloads import LLAMA_DENSE_V1, LlamaWorkload

_NUMPY_BACKEND = numpy_backend_label()


def _samples(fn: Any, *, warmup: int, repeat: int) -> tuple[float, ...]:
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        durations.append(perf_counter() - start)
    return tuple(durations)


def _correctness(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32]) -> CorrectnessEvidence:
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = absolute / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    return CorrectnessEvidence(
        reference="numpy-fp32",
        cases=actual.size,
        exact=bool(np.array_equal(actual, expected)),
        max_abs_error=float(np.max(absolute, initial=0.0)),
        mean_abs_error=float(np.mean(absolute)),
        p99_abs_error=float(np.percentile(absolute, 99)),
        max_relative_error=float(np.max(relative, initial=0.0)),
        nan_count=int(np.count_nonzero(np.isnan(actual))),
        inf_count=int(np.count_nonzero(np.isinf(actual))),
    )


def _source_hash() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in (
        "src/qpu_xla/kernels/swiglu.py",
        "src/qpu_xla/ops/swiglu.py",
        "examples/benchmark_qpu_xla_llama_stages.py",
    ):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _rms_source_hash() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in (
        "src/qpu_xla/kernels/rms_norm.py",
        "src/qpu_xla/models/tinyllama/functional.py",
        "examples/benchmark_qpu_xla_llama_stages.py",
    ):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _rope_source_hash() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in (
        "src/qpu_xla/kernels/rope.py",
        "src/qpu_xla/ops/rope.py",
        "examples/benchmark_qpu_xla_llama_stages.py",
    ):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _softmax_source_hash() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in (
        "src/qpu_xla/kernels/softmax.py",
        "src/qpu_xla/ops/softmax.py",
        "examples/benchmark_qpu_xla_llama_stages.py",
    ):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _run_swiglu(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    *,
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[CandidateRecord]]:
    shape = (case.tokens, case.intermediate_size)
    gate_value = rng.standard_normal(shape, dtype=np.float32) * np.float32(2.0)
    up_value = rng.standard_normal(shape, dtype=np.float32)
    destination_value = np.empty(shape, dtype=np.float32)
    torch_gate = torch.from_numpy(gate_value)
    torch_up = torch.from_numpy(up_value)

    def cpu_reference() -> None:
        destination_value[:] = (gate_value / (np.float32(1.0) + np.exp(-gate_value))) * up_value

    def torch_reference() -> torch.Tensor:
        return torch_functional.silu(torch_gate) * torch_up

    cpu_reference()
    expected = np.array(destination_value, copy=True)
    numpy_seconds = _samples(cpu_reference, warmup=warmup, repeat=repeat)
    torch_seconds = _samples(torch_reference, warmup=warmup, repeat=repeat)
    cpu_name, cpu_seconds = min(
        ((f"{_NUMPY_BACKEND}.silu-mul", numpy_seconds), ("torch-native-silu-mul", torch_seconds)),
        key=lambda item: median(item[1]),
    )
    gate = device.tensor(shape, np.float32)
    up = device.tensor(shape, np.float32)
    destination = device.tensor(shape, np.float32)
    gate.numpy()[:] = gate_value
    up.numpy()[:] = up_value

    def qpu_total() -> None:
        swiglu_fp32(destination, gate, up, queue=qpu_queue, placement=Placement.QPU).wait()

    qpu_seconds = _samples(qpu_total, warmup=warmup, repeat=repeat)
    actual = np.array(destination.numpy(), copy=True)
    correctness = _correctness(actual, expected)
    performance = PerformanceEvidence(cpu_name, cpu_seconds, qpu_seconds)
    numerically_valid = correctness.max_abs_error <= 5e-6 and correctness.max_relative_error <= 5e-5
    status = (
        CandidateStatus.SUPPORTED_WIN
        if numerically_valid and performance.measured_win
        else CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER
    )
    name = f"vc7.swiglu.{case.name}"
    candidate = CandidateRecord(
        name=name,
        operation="swiglu",
        dtype="fp32",
        layout="contiguous-row-major",
        shape_class=f"{case.tokens}x{case.intermediate_size}",
        source_hash=_source_hash(),
        status=status,
        correctness=correctness,
        performance=performance,
        reason="" if status is CandidateStatus.SUPPORTED_WIN else "below the 1.05x gate or numerical tolerance",
        placement="qpu",
        kernels=("vc7.swiglu_fp32",),
    )
    print(
        f"{name}: CPU {median(cpu_seconds) * 1e3:.3f} ms, QPU {median(qpu_seconds) * 1e3:.3f} ms, "
        f"speedup {performance.speedup:.3f}x, max error {correctness.max_abs_error:g}, {status.value}"
    )
    results = [
        {
            "case": asdict(case),
            "operation": "swiglu",
            "shape": list(shape),
            "cpu_seconds": list(cpu_seconds),
            "cpu_reference": performance.cpu_reference,
            "numpy_openblas_seconds": list(numpy_seconds),
            "torch_seconds": list(torch_seconds),
            "qpu_seconds": list(qpu_seconds),
            "speedup_over_cpu": performance.speedup,
            "correctness": asdict(correctness),
            "status": status.value,
            "placement": "qpu",
            "partition": None,
        }
    ]
    candidates = [candidate]
    for qpu_rows in sorted(
        {
            rows
            for fraction in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75, 0.875)
            if 0 < (rows := int(case.tokens * fraction)) < case.tokens
        }
    ):
        qpu_gate = gate.slice((slice(0, qpu_rows), slice(None)))
        qpu_up = up.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        cpu_gate = gate.slice((slice(qpu_rows, case.tokens), slice(None)))
        cpu_up = up.slice((slice(qpu_rows, case.tokens), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, case.tokens), slice(None)))
        torch_cpu_gate = torch.from_numpy(cpu_gate.numpy())
        torch_cpu_up = torch.from_numpy(cpu_up.numpy())
        torch_cpu_destination = torch.from_numpy(cpu_destination.numpy())

        def hybrid_total(qpu_rows: int = qpu_rows) -> None:
            qpu_event = swiglu_fp32(
                qpu_destination,
                qpu_gate,
                qpu_up,
                queue=qpu_queue,
                placement=Placement.QPU,
            )

            def cpu_tail() -> None:
                if cpu_name.startswith("numpy-"):
                    gate_tail = cpu_gate.numpy()
                    cpu_destination.numpy()[:] = (gate_tail / (np.float32(1.0) + np.exp(-gate_tail))) * cpu_up.numpy()
                else:
                    torch_cpu_destination.copy_(torch_functional.silu(torch_cpu_gate) * torch_cpu_up)

            cpu_event = cpu_queue.host_task(
                cpu_tail,
                buffers=(
                    cpu_gate.access(AccessMode.READ),
                    cpu_up.access(AccessMode.READ),
                    cpu_destination.access(AccessMode.WRITE),
                ),
                name="benchmark.swiglu.cpu_tail",
            )
            cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="benchmark.swiglu.hybrid_join",
            ).wait()

        hybrid_seconds = _samples(hybrid_total, warmup=warmup, repeat=repeat)
        hybrid_actual = np.array(destination.numpy(), copy=True)
        hybrid_correctness = _correctness(hybrid_actual, expected)
        hybrid_performance = PerformanceEvidence(cpu_name, cpu_seconds, hybrid_seconds)
        hybrid_valid = hybrid_correctness.max_abs_error <= 5e-6 and hybrid_correctness.max_relative_error <= 5e-5
        hybrid_status = (
            CandidateStatus.SUPPORTED_WIN
            if hybrid_valid and hybrid_performance.measured_win
            else CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER
        )
        partition = PartitionEvidence("rows", qpu_rows, case.tokens, 1)
        hybrid_name = f"vc7.swiglu.hybrid-r{qpu_rows}.{case.name}"
        hybrid_candidate = CandidateRecord(
            name=hybrid_name,
            operation="swiglu",
            dtype="fp32",
            layout="contiguous-row-major",
            shape_class=f"{case.tokens}x{case.intermediate_size}",
            source_hash=_source_hash(),
            status=hybrid_status,
            correctness=hybrid_correctness,
            performance=hybrid_performance,
            reason=(
                "" if hybrid_status is CandidateStatus.SUPPORTED_WIN else "below the 1.05x gate or numerical tolerance"
            ),
            placement="hybrid",
            kernels=("vc7.swiglu_fp32", cpu_name),
            partition=partition,
        )
        candidates.append(hybrid_candidate)
        results.append(
            {
                "case": asdict(case),
                "operation": "swiglu",
                "shape": list(shape),
                "cpu_seconds": list(cpu_seconds),
                "cpu_reference": hybrid_performance.cpu_reference,
                "numpy_openblas_seconds": list(numpy_seconds),
                "torch_seconds": list(torch_seconds),
                "candidate_seconds": list(hybrid_seconds),
                "speedup_over_cpu": hybrid_performance.speedup,
                "correctness": asdict(hybrid_correctness),
                "status": hybrid_status.value,
                "placement": "hybrid",
                "partition": asdict(partition),
            }
        )
        print(
            f"{hybrid_name}: CPU {median(cpu_seconds) * 1e3:.3f} ms, "
            f"candidate {median(hybrid_seconds) * 1e3:.3f} ms, "
            f"speedup {hybrid_performance.speedup:.3f}x, "
            f"max error {hybrid_correctness.max_abs_error:g}, {hybrid_status.value}"
        )
    return results, candidates


def _run_rms_norm(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    *,
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[CandidateRecord]]:
    shape = (case.tokens, case.hidden_size)
    source_value = rng.standard_normal(shape, dtype=np.float32)
    weight_value = rng.standard_normal((case.hidden_size,), dtype=np.float32)
    expected = (
        source_value
        / np.sqrt(np.mean(source_value * source_value, axis=1, keepdims=True, dtype=np.float32) + np.float32(1e-5))
        * weight_value
    )
    torch_source = torch.from_numpy(source_value)
    torch_weight = torch.from_numpy(weight_value)

    def numpy_reference() -> npt.NDArray[np.float32]:
        return (
            source_value
            / np.sqrt(np.mean(source_value * source_value, axis=1, keepdims=True, dtype=np.float32) + np.float32(1e-5))
            * weight_value
        )

    def torch_reference() -> torch.Tensor:
        return torch_functional.rms_norm(
            torch_source,
            (case.hidden_size,),
            torch_weight,
            eps=1e-5,
        )

    source = device.tensor(shape, np.float32)
    weight = device.tensor(weight_value.shape, np.float32)
    destination = device.tensor(shape, np.float32)
    source.numpy()[:] = source_value
    weight.numpy()[:] = weight_value

    def qpu_total() -> None:
        rms_norm_fp32(
            destination,
            source,
            weight,
            queue=qpu_queue,
            placement=Placement.QPU,
        ).wait()

    numpy_seconds = _samples(numpy_reference, warmup=warmup, repeat=repeat)
    torch_seconds = _samples(torch_reference, warmup=warmup, repeat=repeat)
    cpu_name, cpu_seconds = min(
        ((f"{_NUMPY_BACKEND}.rms-norm", numpy_seconds), ("torch-native-rms-norm", torch_seconds)),
        key=lambda item: median(item[1]),
    )
    qpu_seconds = _samples(qpu_total, warmup=warmup, repeat=repeat)
    qpu_actual = np.array(destination.numpy(), copy=True)
    configurations: list[tuple[str, str, int | None, tuple[float, ...], npt.NDArray[np.float32]]] = [
        ("qpu", "vc7.rms_norm_fp32", None, qpu_seconds, qpu_actual)
    ]
    for qpu_rows in sorted(
        {
            rows
            for fraction in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75, 0.875)
            if 0 < (rows := int(case.tokens * fraction)) < case.tokens
        }
    ):
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        cpu_source = source.slice((slice(qpu_rows, case.tokens), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, case.tokens), slice(None)))
        torch_cpu_source = torch.from_numpy(cpu_source.numpy())
        torch_cpu_destination = torch.from_numpy(cpu_destination.numpy())

        def hybrid_total(qpu_rows: int = qpu_rows) -> None:
            qpu_event = rms_norm_fp32(
                qpu_destination,
                qpu_source,
                weight,
                queue=qpu_queue,
                placement=Placement.QPU,
            )

            def cpu_tail() -> None:
                if cpu_name.startswith("numpy-"):
                    values = cpu_source.numpy()
                    mean_square = np.mean(values * values, axis=1, keepdims=True, dtype=np.float32)
                    cpu_destination.numpy()[:] = (
                        values * np.reciprocal(np.sqrt(mean_square + np.float32(1e-5))) * weight_value
                    )
                else:
                    torch_cpu_destination.copy_(
                        torch_functional.rms_norm(
                            torch_cpu_source,
                            (case.hidden_size,),
                            torch_weight,
                            eps=1e-5,
                        )
                    )

            cpu_event = cpu_queue.host_task(
                cpu_tail,
                buffers=(
                    cpu_source.access(AccessMode.READ),
                    weight.access(AccessMode.READ),
                    cpu_destination.access(AccessMode.WRITE),
                ),
                name="benchmark.rms_norm.cpu_tail",
            )
            cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="benchmark.rms_norm.hybrid_join",
            ).wait()

        seconds = _samples(hybrid_total, warmup=warmup, repeat=repeat)
        configurations.append(
            (
                "hybrid",
                f"vc7.rms_norm_fp32.hybrid-r{qpu_rows}",
                qpu_rows,
                seconds,
                np.array(destination.numpy(), copy=True),
            )
        )

    results: list[dict[str, object]] = []
    candidates: list[CandidateRecord] = []
    for placement, name, partition_rows, seconds, actual in configurations:
        correctness = _correctness(actual, expected)
        performance = PerformanceEvidence(cpu_name, cpu_seconds, seconds)
        valid = np.allclose(actual, expected, atol=2e-5, rtol=2e-5)
        status = (
            CandidateStatus.SUPPORTED_WIN
            if valid and performance.measured_win
            else CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER
        )
        partition = None if partition_rows is None else PartitionEvidence("rows", partition_rows, case.tokens, 1)
        candidate = CandidateRecord(
            name=f"{name}.{case.name}",
            operation="rms-norm",
            dtype="fp32",
            layout="contiguous-row-major",
            shape_class=f"{case.tokens}x{case.hidden_size}",
            source_hash=_rms_source_hash(),
            status=status,
            correctness=correctness,
            performance=performance,
            reason="" if status is CandidateStatus.SUPPORTED_WIN else "below the 1.05x gate or numerical tolerance",
            placement=placement,
            kernels=("vc7.rms_norm_fp32",) if placement == "qpu" else ("vc7.rms_norm_fp32", cpu_name),
            partition=partition,
        )
        candidates.append(candidate)
        results.append(
            {
                "case": asdict(case),
                "operation": "rms-norm",
                "shape": list(shape),
                "placement": placement,
                "partition": None if partition is None else asdict(partition),
                "cpu_seconds": list(cpu_seconds),
                "cpu_reference": performance.cpu_reference,
                "numpy_openblas_seconds": list(numpy_seconds),
                "torch_seconds": list(torch_seconds),
                "candidate_seconds": list(seconds),
                "speedup_over_cpu": performance.speedup,
                "correctness": asdict(correctness),
                "status": status.value,
            }
        )
        print(
            f"{candidate.name}: CPU {median(cpu_seconds) * 1e3:.3f} ms, "
            f"candidate {median(seconds) * 1e3:.3f} ms, speedup {performance.speedup:.3f}x, "
            f"max error {correctness.max_abs_error:g}, {status.value}"
        )
    return results, candidates


def _run_rope(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    *,
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[CandidateRecord]]:
    rows = case.tokens * case.query_heads
    shape = (rows, case.head_dim)
    source_value = rng.standard_normal(shape, dtype=np.float32)
    positions = np.repeat(np.arange(case.tokens, dtype=np.int32), case.query_heads)
    cosine_value, signed_sine_value = rope_tables_fp32(positions, case.head_dim)
    adjacent = source_value.reshape(-1, 2)[:, ::-1].reshape(shape)
    expected = source_value * cosine_value + adjacent * signed_sine_value
    torch_source = torch.from_numpy(source_value)
    torch_cosine_pairs = torch.from_numpy(np.ascontiguousarray(cosine_value[:, 0::2]))
    torch_sine_pairs = torch.from_numpy(np.ascontiguousarray(signed_sine_value[:, 1::2]))
    torch_rotor = torch.complex(torch_cosine_pairs, torch_sine_pairs)

    def numpy_reference() -> npt.NDArray[np.float32]:
        reversed_pairs = source_value.reshape(-1, 2)[:, ::-1].reshape(shape)
        return source_value * cosine_value + reversed_pairs * signed_sine_value

    def torch_complex_reference() -> torch.Tensor:
        source_pairs = torch.view_as_complex(torch_source.reshape(rows, case.head_dim // 2, 2))
        return torch.view_as_real(source_pairs * torch_rotor).reshape(shape)

    def torch_pairwise_reference() -> torch.Tensor:
        even = torch_source[:, 0::2]
        odd = torch_source[:, 1::2]
        return torch.stack(
            (
                even * torch_cosine_pairs - odd * torch_sine_pairs,
                even * torch_sine_pairs + odd * torch_cosine_pairs,
            ),
            dim=-1,
        ).reshape(shape)

    source = device.tensor(shape, np.float32)
    cosine = device.tensor(shape, np.float32)
    signed_sine = device.tensor(shape, np.float32)
    destination = device.tensor(shape, np.float32)
    source.numpy()[:] = source_value
    cosine.numpy()[:] = cosine_value
    signed_sine.numpy()[:] = signed_sine_value

    def run(placement: Placement, qpu_rows: int | None = None) -> None:
        apply_rope_tables_fp32(
            destination,
            source,
            cosine,
            signed_sine,
            queue=qpu_queue if placement is not Placement.CPU else cpu_queue,
            cpu_queue=cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()

    numpy_seconds = _samples(numpy_reference, warmup=warmup, repeat=repeat)
    torch_complex_seconds = _samples(torch_complex_reference, warmup=warmup, repeat=repeat)
    torch_pairwise_seconds = _samples(torch_pairwise_reference, warmup=warmup, repeat=repeat)
    cpu_name, cpu_seconds = min(
        (
            (f"{_NUMPY_BACKEND}.rope", numpy_seconds),
            ("torch-complex-rope", torch_complex_seconds),
            ("torch-pairwise-rope", torch_pairwise_seconds),
        ),
        key=lambda item: median(item[1]),
    )
    qpu_seconds = _samples(lambda: run(Placement.QPU), warmup=warmup, repeat=repeat)
    configurations: list[tuple[str, int | None, tuple[float, ...], npt.NDArray[np.float32]]] = [
        ("qpu", None, qpu_seconds, np.array(destination.numpy(), copy=True))
    ]
    for split_rows in sorted(
        {
            count
            for fraction in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75, 0.875)
            if 0 < (count := int(rows * fraction)) < rows
        }
    ):
        qpu_source = source.slice((slice(0, split_rows), slice(None)))
        qpu_cosine = cosine.slice((slice(0, split_rows), slice(None)))
        qpu_signed_sine = signed_sine.slice((slice(0, split_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, split_rows), slice(None)))
        cpu_source = source.slice((slice(split_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(split_rows, rows), slice(None)))
        torch_cpu_source = torch.from_numpy(cpu_source.numpy())
        torch_cpu_destination = torch.from_numpy(cpu_destination.numpy())
        torch_cpu_cosine = torch_cosine_pairs[split_rows:]
        torch_cpu_sine = torch_sine_pairs[split_rows:]
        torch_cpu_rotor = torch_rotor[split_rows:]

        def cpu_rope_tail() -> None:
            if cpu_name.startswith("numpy-"):
                values = cpu_source.numpy()
                reversed_pairs = values.reshape(-1, 2)[:, ::-1].reshape(values.shape)
                cpu_destination.numpy()[:] = (
                    values * cosine.numpy()[split_rows:] + reversed_pairs * signed_sine.numpy()[split_rows:]
                )
            elif cpu_name == "torch-complex-rope":
                source_pairs = torch.view_as_complex(torch_cpu_source.reshape(-1, case.head_dim // 2, 2))
                torch_cpu_destination.copy_(
                    torch.view_as_real(source_pairs * torch_cpu_rotor).reshape(-1, case.head_dim)
                )
            else:
                even = torch_cpu_source[:, 0::2]
                odd = torch_cpu_source[:, 1::2]
                torch_cpu_destination.copy_(
                    torch.stack(
                        (
                            even * torch_cpu_cosine - odd * torch_cpu_sine,
                            even * torch_cpu_sine + odd * torch_cpu_cosine,
                        ),
                        dim=-1,
                    ).reshape(-1, case.head_dim)
                )

        def hybrid_total() -> None:
            qpu_event = apply_rope_tables_fp32(
                qpu_destination,
                qpu_source,
                qpu_cosine,
                qpu_signed_sine,
                queue=qpu_queue,
                placement=Placement.QPU,
            )
            cpu_event = cpu_queue.host_task(
                cpu_rope_tail,
                buffers=(
                    cpu_source.access(AccessMode.READ),
                    cpu_destination.access(AccessMode.WRITE),
                ),
                name="benchmark.rope.cpu_tail",
            )
            cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="benchmark.rope.hybrid_join",
            ).wait()

        seconds = _samples(hybrid_total, warmup=warmup, repeat=repeat)
        configurations.append(("hybrid", split_rows, seconds, np.array(destination.numpy(), copy=True)))

    results: list[dict[str, object]] = []
    candidates: list[CandidateRecord] = []
    for placement, partition_rows, seconds, actual in configurations:
        correctness = _correctness(actual, expected)
        performance = PerformanceEvidence(cpu_name, cpu_seconds, seconds)
        valid = np.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        status = (
            CandidateStatus.SUPPORTED_WIN
            if valid and performance.measured_win
            else CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER
        )
        partition = None if partition_rows is None else PartitionEvidence("rows", partition_rows, rows, 1)
        suffix = "qpu" if partition is None else f"hybrid-r{partition_rows}"
        candidate = CandidateRecord(
            name=f"vc7.rope_fp32.{suffix}.{case.name}",
            operation="rope",
            dtype="fp32",
            layout="cached-full-tables",
            shape_class=f"{rows}x{case.head_dim}",
            source_hash=_rope_source_hash(),
            status=status,
            correctness=correctness,
            performance=performance,
            reason="" if status is CandidateStatus.SUPPORTED_WIN else "below the 1.05x gate or numerical tolerance",
            placement=placement,
            kernels=("vc7.rope_fp32",) if placement == "qpu" else ("vc7.rope_fp32", cpu_name),
            partition=partition,
        )
        candidates.append(candidate)
        results.append(
            {
                "case": asdict(case),
                "operation": "rope",
                "shape": list(shape),
                "placement": placement,
                "partition": None if partition is None else asdict(partition),
                "cpu_seconds": list(cpu_seconds),
                "cpu_reference": performance.cpu_reference,
                "numpy_openblas_seconds": list(numpy_seconds),
                "torch_complex_seconds": list(torch_complex_seconds),
                "torch_pairwise_seconds": list(torch_pairwise_seconds),
                "candidate_seconds": list(seconds),
                "speedup_over_cpu": performance.speedup,
                "correctness": asdict(correctness),
                "status": status.value,
            }
        )
        print(
            f"{candidate.name}: CPU {median(cpu_seconds) * 1e3:.3f} ms, "
            f"candidate {median(seconds) * 1e3:.3f} ms, speedup {performance.speedup:.3f}x, "
            f"max error {correctness.max_abs_error:g}, {status.value}"
        )
    return results, candidates


def _run_softmax(
    device: Device,
    qpu_queue: Any,
    cpu_queue: Any,
    case: LlamaWorkload,
    *,
    warmup: int,
    repeat: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[CandidateRecord]]:
    active_columns = case.cache_length if case.phase == "decode" else case.tokens
    columns = (active_columns + 15) // 16 * 16
    rows = case.tokens * case.query_heads
    shape = (rows, columns)
    source_value = rng.standard_normal(shape, dtype=np.float32) * np.float32(2.0)
    expected_shifted = source_value - np.max(source_value, axis=1, keepdims=True)
    expected = np.exp(expected_shifted)
    expected /= np.sum(expected, axis=1, keepdims=True)
    torch_source = torch.from_numpy(source_value)

    def numpy_reference() -> npt.NDArray[np.float32]:
        shifted = source_value - np.max(source_value, axis=1, keepdims=True)
        output = np.exp(shifted)
        output /= np.sum(output, axis=1, keepdims=True)
        return output

    def torch_reference() -> torch.Tensor:
        return torch.softmax(torch_source, dim=-1)

    source = device.tensor(shape, np.float32)
    destination = device.tensor(shape, np.float32)
    source.numpy()[:] = source_value

    def run(placement: Placement, qpu_rows: int | None = None) -> None:
        softmax_fp32(
            destination,
            source,
            queue=qpu_queue if placement is not Placement.CPU else cpu_queue,
            cpu_queue=cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()

    numpy_seconds = _samples(numpy_reference, warmup=warmup, repeat=repeat)
    torch_seconds = _samples(torch_reference, warmup=warmup, repeat=repeat)
    cpu_name, cpu_seconds = min(
        ((f"{_NUMPY_BACKEND}.softmax", numpy_seconds), ("torch-native-softmax", torch_seconds)),
        key=lambda item: median(item[1]),
    )
    qpu_seconds = _samples(lambda: run(Placement.QPU), warmup=warmup, repeat=repeat)
    configurations: list[tuple[str, int | None, tuple[float, ...], npt.NDArray[np.float32]]] = [
        ("qpu", None, qpu_seconds, np.array(destination.numpy(), copy=True))
    ]
    for split_rows in sorted(
        {
            count
            for fraction in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75, 0.875)
            if 0 < (count := int(rows * fraction)) < rows
        }
    ):
        qpu_source = source.slice((slice(0, split_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, split_rows), slice(None)))
        cpu_source = source.slice((slice(split_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(split_rows, rows), slice(None)))
        torch_cpu_source = torch.from_numpy(cpu_source.numpy())
        torch_cpu_destination = torch.from_numpy(cpu_destination.numpy())

        def hybrid_total() -> None:
            qpu_event = softmax_fp32(
                qpu_destination,
                qpu_source,
                queue=qpu_queue,
                placement=Placement.QPU,
            )

            def cpu_tail() -> None:
                if cpu_name.startswith("numpy-"):
                    values = cpu_source.numpy()
                    output = cpu_destination.numpy()
                    np.subtract(values, np.max(values, axis=1, keepdims=True), out=output)
                    np.exp(output, out=output)
                    output /= np.sum(output, axis=1, keepdims=True)
                else:
                    torch_cpu_destination.copy_(torch.softmax(torch_cpu_source, dim=-1))

            cpu_event = cpu_queue.host_task(
                cpu_tail,
                buffers=(
                    cpu_source.access(AccessMode.READ),
                    cpu_destination.access(AccessMode.WRITE),
                ),
                name="benchmark.softmax.cpu_tail",
            )
            cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="benchmark.softmax.hybrid_join",
            ).wait()

        seconds = _samples(hybrid_total, warmup=warmup, repeat=repeat)
        configurations.append(("hybrid", split_rows, seconds, np.array(destination.numpy(), copy=True)))

    results: list[dict[str, object]] = []
    candidates: list[CandidateRecord] = []
    for placement, partition_rows, seconds, actual in configurations:
        correctness = _correctness(actual, expected)
        performance = PerformanceEvidence(cpu_name, cpu_seconds, seconds)
        valid = np.allclose(actual, expected, atol=2e-6, rtol=2e-5)
        status = (
            CandidateStatus.SUPPORTED_WIN
            if valid and performance.measured_win
            else CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER
        )
        partition = None if partition_rows is None else PartitionEvidence("rows", partition_rows, rows, 1)
        suffix = "qpu" if partition is None else f"hybrid-r{partition_rows}"
        candidate = CandidateRecord(
            name=f"vc7.softmax_fp32.{suffix}.{case.name}",
            operation="softmax",
            dtype="fp32",
            layout="contiguous-row-major",
            shape_class=f"{rows}x{columns}",
            source_hash=_softmax_source_hash(),
            status=status,
            correctness=correctness,
            performance=performance,
            reason="" if status is CandidateStatus.SUPPORTED_WIN else "below the 1.05x gate or numerical tolerance",
            placement=placement,
            kernels=("vc7.softmax_fp32",) if placement == "qpu" else ("vc7.softmax_fp32", cpu_name),
            partition=partition,
        )
        candidates.append(candidate)
        results.append(
            {
                "case": asdict(case),
                "operation": "softmax",
                "shape": list(shape),
                "placement": placement,
                "partition": None if partition is None else asdict(partition),
                "cpu_seconds": list(cpu_seconds),
                "cpu_reference": performance.cpu_reference,
                "numpy_openblas_seconds": list(numpy_seconds),
                "torch_seconds": list(torch_seconds),
                "candidate_seconds": list(seconds),
                "speedup_over_cpu": performance.speedup,
                "correctness": asdict(correctness),
                "status": status.value,
            }
        )
        print(
            f"{candidate.name}: CPU {median(cpu_seconds) * 1e3:.3f} ms, "
            f"candidate {median(seconds) * 1e3:.3f} ms, speedup {performance.speedup:.3f}x, "
            f"max error {correctness.max_abs_error:g}, {status.value}"
        )
    return results, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark non-projection dense Llama QPU stages")
    parser.add_argument(
        "--case",
        default="prefill-h512-t16",
        choices=[case.name for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--data-area-mib", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("llama-stages-benchmark.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.data_area_mib <= 0:
        parser.error("warmup must be non-negative; repeat and data-area-mib must be positive")
    case = next(case for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout) if case.name == args.case)
    assert isinstance(case, LlamaWorkload)
    registry = CandidateRegistry()
    with (
        Device.open(data_area_size=args.data_area_mib * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        metadata = collect_metadata(
            device,
            extra={
                "manifest": f"{LLAMA_DENSE_V1.name}-v{LLAMA_DENSE_V1.version}",
                "semantics": "steady-state",
                "cpu_reference": "fastest-of-torch-and-numpy-blas",
                "torch": torch.__version__,
                "torch_threads": str(torch.get_num_threads()),
                "torch_interop_threads": str(torch.get_num_interop_threads()),
            },
        )
        swiglu_results, swiglu_candidates = _run_swiglu(
            device,
            qpu_queue,
            cpu_queue,
            case,
            warmup=args.warmup,
            repeat=args.repeat,
            rng=np.random.default_rng(args.seed),
        )
        for swiglu_candidate in swiglu_candidates:
            registry.register(swiglu_candidate)
        rms_results, rms_candidates = _run_rms_norm(
            device,
            qpu_queue,
            cpu_queue,
            case,
            warmup=args.warmup,
            repeat=args.repeat,
            rng=np.random.default_rng(args.seed + 1),
        )
        for rms_candidate in rms_candidates:
            registry.register(rms_candidate)
        rope_results, rope_candidates = _run_rope(
            device,
            qpu_queue,
            cpu_queue,
            case,
            warmup=args.warmup,
            repeat=args.repeat,
            rng=np.random.default_rng(args.seed + 2),
        )
        for rope_candidate in rope_candidates:
            registry.register(rope_candidate)
        softmax_results, softmax_candidates = _run_softmax(
            device,
            qpu_queue,
            cpu_queue,
            case,
            warmup=args.warmup,
            repeat=args.repeat,
            rng=np.random.default_rng(args.seed + 3),
        )
        for softmax_candidate in softmax_candidates:
            registry.register(softmax_candidate)
        results = [*swiglu_results, *rms_results, *rope_results, *softmax_results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2, sort_keys=True))
    registry_path = args.output.with_suffix(".candidates.json")
    registry.save(registry_path)
    print(f"saved {args.output} and {registry_path}")


if __name__ == "__main__":
    main()
