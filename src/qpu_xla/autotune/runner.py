"""Bounded proposal-to-validation pipeline for generated DSL candidates."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.compiler import compile_source, lower_vc7
from qpu_xla.device import Device
from qpu_xla.errors import DslCompileError
from qpu_xla.kernel import Kernel


class CandidateStatus(Enum):
    """Terminal outcomes recorded by the constrained candidate pipeline."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CandidateLimits:
    """Static resource limits that keep candidate parsing deterministic and bounded."""

    max_source_bytes: int = 65_536
    max_ast_nodes: int = 4_096
    max_constants: int = 32
    compile_timeout_seconds: float = 1.0

    def __post_init__(self: Self) -> None:
        """Reject nonsensical limits before any candidate is inspected."""
        if (
            self.max_source_bytes <= 0
            or self.max_ast_nodes <= 0
            or self.max_constants < 0
            or self.compile_timeout_seconds <= 0
        ):
            raise ValueError("candidate resource limits must be positive")


@dataclass(frozen=True, slots=True)
class CandidateSource:
    """Inert source text and compile-time constants for one generated kernel."""

    source: str
    filename: str = "<qpu-xla-candidate>"
    constants: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self: Self) -> None:
        """Preserve a stable immutable constant mapping for reproducible hashing."""
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("candidate source must be a non-empty string")
        if not self.filename:
            raise ValueError("candidate filename must be non-empty")
        object.__setattr__(self, "constants", dict(self.constants))


@dataclass(frozen=True, slots=True)
class CandidateCase:
    """Small deterministic inputs for a canonical copy/minimum/maximum candidate."""

    inputs: tuple[npt.NDArray[np.generic], ...]

    def __post_init__(self: Self) -> None:
        """Require contiguous four-byte input tensors accepted by VC7 vector kernels."""
        if not self.inputs:
            raise ValueError("candidate case requires at least one input")
        for value in self.inputs:
            if value.dtype not in {np.dtype(np.int32), np.dtype(np.float32)}:
                raise ValueError("candidate inputs must have int32 or float32 dtype")
            if not value.flags.c_contiguous or value.nbytes == 0 or value.nbytes % (16 * 4):
                raise ValueError("candidate inputs must be contiguous non-empty multiples of 16 words")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Machine-readable outcome of parse, verify, reference, and optional QPU stages."""

    status: CandidateStatus
    source_hash: str | None
    kernel_name: str | None
    compile_seconds: float
    cpu_reference_passed: bool
    qpu_differential_passed: bool | None
    qpu_median_seconds: float | None
    diagnostic: str | None = None

    def to_dict(self: Self) -> dict[str, object]:
        """Return a stable JSON-safe representation for a result database."""
        return {
            "status": self.status.value,
            "source_hash": self.source_hash,
            "kernel_name": self.kernel_name,
            "compile_seconds": self.compile_seconds,
            "cpu_reference_passed": self.cpu_reference_passed,
            "qpu_differential_passed": self.qpu_differential_passed,
            "qpu_median_seconds": self.qpu_median_seconds,
            "diagnostic": self.diagnostic,
        }


class CandidateResultStore:
    """Append explicitly reviewed runner results to a JSON-lines result database."""

    def __init__(self: Self, path: str | PathLike[str]) -> None:
        """Select a caller-owned result file without granting candidates file access."""
        self._path = Path(path)

    def append(self: Self, result: CandidateEvaluation) -> None:
        """Append one completed evaluation in a durable line-oriented format."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result.to_dict(), sort_keys=True))
            output.write("\n")


class CandidateRunner:
    """Validate canonical generated kernels without evaluating candidate Python.

    Candidate source is parsed solely by :func:`compile_source`, which rejects
    imports and non-DSL calls. Only the host-owned runner allocates tensors,
    opens queues, persists results, or accesses a supplied hardware device.
    The initial candidate surface intentionally matches the three currently
    hardware-validated canonical VC7 lowerings: copy, minimum, and maximum.
    """

    def __init__(self: Self, *, limits: CandidateLimits = CandidateLimits()) -> None:
        """Use explicit static limits rather than executing arbitrary candidates."""
        self._limits = limits

    def evaluate(
        self: Self,
        candidate: CandidateSource,
        case: CandidateCase,
        *,
        device: Device | None = None,
        timeout_seconds: float = 5.0,
        benchmark_repetitions: int = 3,
        store: CandidateResultStore | None = None,
    ) -> CandidateEvaluation:
        """Parse, verify, lower, differentially test, time, and optionally persist one candidate.

        CPU validation uses the exact canonical operation selected by VC7
        lowering. A supplied hardware-backed device adds a QPU differential
        stage; without one, the outcome explicitly records that it was not run.
        """
        if timeout_seconds <= 0 or benchmark_repetitions <= 0:
            raise ValueError("timeout_seconds and benchmark_repetitions must be positive")
        evaluation = self._compile_and_validate(candidate, case)
        if evaluation.status is CandidateStatus.ACCEPTED and device is not None:
            evaluation = self._run_qpu(evaluation, case, device, timeout_seconds, benchmark_repetitions)
        if store is not None:
            store.append(evaluation)
        return evaluation

    def _compile_and_validate(self: Self, candidate: CandidateSource, case: CandidateCase) -> CandidateEvaluation:
        """Run all source-only stages under fixed input-size and elapsed-time limits."""
        source_bytes = len(candidate.source.encode("utf-8"))
        if source_bytes > self._limits.max_source_bytes:
            return self._rejected(f"candidate source exceeds {self._limits.max_source_bytes} byte limit")
        if len(candidate.constants) > self._limits.max_constants:
            return self._rejected(f"candidate constants exceed {self._limits.max_constants} item limit")
        try:
            node_count = sum(1 for _ in ast.walk(ast.parse(candidate.source, filename=candidate.filename)))
        except SyntaxError:
            node_count = 0
        if node_count > self._limits.max_ast_nodes:
            return self._rejected(f"candidate AST exceeds {self._limits.max_ast_nodes} node limit")
        started = perf_counter()
        try:
            program = compile_source(candidate.source, filename=candidate.filename, constants=candidate.constants)
            elapsed = perf_counter() - started
            if elapsed > self._limits.compile_timeout_seconds:
                return self._rejected(
                    f"candidate compilation exceeded {self._limits.compile_timeout_seconds} seconds",
                    compile_seconds=elapsed,
                )
            kernel = lower_vc7(program)
            self._reference(kernel, case)
        except (DslCompileError, ValueError) as exc:
            return self._rejected(str(exc), compile_seconds=perf_counter() - started)
        return CandidateEvaluation(
            CandidateStatus.ACCEPTED,
            program.source_hash,
            kernel.name,
            elapsed,
            cpu_reference_passed=True,
            qpu_differential_passed=None,
            qpu_median_seconds=None,
        )

    @staticmethod
    def _rejected(diagnostic: str, *, compile_seconds: float = 0.0) -> CandidateEvaluation:
        """Form one explicit rejected result without pretending a later stage ran."""
        return CandidateEvaluation(
            CandidateStatus.REJECTED,
            None,
            None,
            compile_seconds,
            cpu_reference_passed=False,
            qpu_differential_passed=None,
            qpu_median_seconds=None,
            diagnostic=diagnostic,
        )

    @staticmethod
    def _reference(kernel: Kernel, case: CandidateCase) -> npt.NDArray[np.generic]:
        """Compute the exact NumPy oracle and validate a canonical kernel's inputs."""
        inputs = case.inputs
        if kernel.name == "vc7.copy_words":
            if len(inputs) != 1:
                raise ValueError("copy candidate cases require one input")
            return np.array(inputs[0], copy=True)
        if kernel.name in {"vc7.minimum_words", "vc7.maximum_words"}:
            if len(inputs) != 2:
                raise ValueError("minimum/maximum candidate cases require two inputs")
            left, right = inputs
            if left.dtype != right.dtype or left.shape != right.shape:
                raise ValueError("minimum/maximum candidate inputs must have equal shapes and dtypes")
            result = np.minimum(left, right) if kernel.name == "vc7.minimum_words" else np.maximum(left, right)
            return cast(npt.NDArray[np.generic], result)
        raise ValueError(f"no CPU oracle is registered for kernel {kernel.name!r}")

    def _run_qpu(
        self: Self,
        evaluation: CandidateEvaluation,
        case: CandidateCase,
        device: Device,
        timeout_seconds: float,
        benchmark_repetitions: int,
    ) -> CandidateEvaluation:
        """Execute the selected canonical kernel and compare every output value to NumPy."""
        assert evaluation.kernel_name is not None
        expected_kernel = self._kernel_from_name(evaluation.kernel_name)
        expected = self._reference(expected_kernel, case)
        tensors = tuple(device.tensor(value.shape, value.dtype) for value in case.inputs)
        destination = device.tensor(expected.shape, expected.dtype)
        for tensor, value in zip(tensors, case.inputs, strict=True):
            tensor.numpy()[:] = value
        try:
            with device.queue() as queue:
                event = queue.submit(expected_kernel, (*tensors, destination))
                event.wait(timeout_seconds)
                np.testing.assert_array_equal(destination.numpy(), expected)
                durations: list[float] = []
                for _ in range(benchmark_repetitions):
                    started = perf_counter()
                    queue.submit(expected_kernel, (*tensors, destination)).wait(timeout_seconds)
                    durations.append(perf_counter() - started)
        except BaseException as exc:
            return CandidateEvaluation(
                CandidateStatus.REJECTED,
                evaluation.source_hash,
                evaluation.kernel_name,
                evaluation.compile_seconds,
                cpu_reference_passed=True,
                qpu_differential_passed=False,
                qpu_median_seconds=None,
                diagnostic=f"QPU differential failed: {exc}",
            )
        return CandidateEvaluation(
            CandidateStatus.ACCEPTED,
            evaluation.source_hash,
            evaluation.kernel_name,
            evaluation.compile_seconds,
            cpu_reference_passed=True,
            qpu_differential_passed=True,
            qpu_median_seconds=float(np.median(durations)),
        )

    @staticmethod
    def _kernel_from_name(name: str) -> Kernel:
        """Recover the already verified lowering selected during source validation."""
        from qpu_xla.kernels import MAXIMUM_WORD_KERNEL, MINIMUM_WORD_KERNEL, WORD_COPY_KERNEL

        kernels = {kernel.name: kernel for kernel in (WORD_COPY_KERNEL, MINIMUM_WORD_KERNEL, MAXIMUM_WORD_KERNEL)}
        return kernels[name]
