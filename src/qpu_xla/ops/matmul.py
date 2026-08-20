"""Matrix multiplication contract with CPU reference and tiled QPU specialization."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernel import Kernel
from qpu_xla.kernels.gemm import TILED_INT32_GEMM_KERNEL, supports_tiled_int32_gemm
from qpu_xla.kernels.gemm_fp32 import TILED_FP32_GEMM_KERNEL, supports_tiled_fp32_gemm
from qpu_xla.kernels.gemv_fp32 import FP32_GEMV_KERNEL, supports_fp32_gemv
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import (
    CapabilityRegistry,
    CostModel,
    ExecutionCandidate,
    OperationSpec,
    PartitionSpec,
    Placement,
    PlannedExecution,
)

_HYBRID_FRACTIONS = (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75, 0.875)


def _matmul_specification(destination: Tensor, left: Tensor, right: Tensor) -> OperationSpec:
    """Build the normalized placement/cost-model key for a validated matmul."""
    p, q = left.shape
    return OperationSpec("matmul", destination.dtype.name, (p, q, right.shape[1]), "row-major")


def _qpu_kernel(left: Tensor, right: Tensor, destination: Tensor) -> tuple[Kernel, str] | None:
    """Return the exact tiled QPU specialization applicable to these tensors."""
    backend = destination.buffer.device.backend
    if supports_tiled_int32_gemm(left, right, destination, backend):
        return TILED_INT32_GEMM_KERNEL, "vc7.tiled_int32_gemm"
    if supports_tiled_fp32_gemm(left, right, destination, backend):
        return TILED_FP32_GEMM_KERNEL, "vc7.tiled_fp32_gemm"
    if supports_fp32_gemv(left, right, destination, backend):
        return FP32_GEMV_KERNEL, "vc7.fp32_gemv"
    return None


def hybrid_row_partitions(rows: int) -> tuple[int, ...]:
    """Return stable aligned QPU-row candidates that leave non-empty CPU work."""
    if rows <= 16:
        return ()
    return tuple(
        sorted(
            {qpu_rows for fraction in _HYBRID_FRACTIONS if 0 < (qpu_rows := int(rows * fraction) // 16 * 16) < rows}
        )
    )


def hybrid_column_partitions(columns: int) -> tuple[int, ...]:
    """Return 16-aligned QPU-column candidates for single-row decode."""
    return hybrid_row_partitions(columns)


def _hybrid_kernel(
    left: Tensor,
    right: Tensor,
    destination: Tensor,
    qpu_rows: int,
) -> tuple[Kernel, str] | None:
    """Return the QPU kernel for one aligned row prefix of a larger operation."""
    if qpu_rows <= 0 or qpu_rows >= left.shape[0] or qpu_rows % 16:
        return None
    qpu_left = left.slice((slice(0, qpu_rows), slice(None)))
    qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
    return _qpu_kernel(qpu_left, right, qpu_destination)


def _hybrid_column_kernel(
    left: Tensor,
    right: Tensor,
    destination: Tensor,
    qpu_columns: int,
) -> tuple[Kernel, str] | None:
    if left.shape[0] != 1 or qpu_columns <= 0 or qpu_columns >= right.shape[1] or qpu_columns % 16:
        return None
    qpu_right = right.slice((slice(None), slice(0, qpu_columns)))
    qpu_destination = destination.slice((slice(None), slice(0, qpu_columns)))
    return _qpu_kernel(left, qpu_right, qpu_destination)


def plan_matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    placement: Placement = Placement.AUTO,
    cost_model: CostModel | None = None,
    allow_hybrid: bool = False,
    qpu_rows: int | None = None,
    qpu_columns: int | None = None,
) -> PlannedExecution:
    """Select a supported CPU or tiled-QPU matrix multiplication implementation.

    The uncalibrated estimates retain the validated tiled-QPU preference for
    compatible INT32 matrices; an exact shape/dtype sample in ``cost_model``
    overrides that heuristic.  Requesting ``Placement.QPU`` never silently
    falls back to CPU when the tiled contract is unavailable.
    """
    p, q = left.shape
    r = right.shape[1]
    specification = _matmul_specification(destination, left, right)
    operations = 2 * p * q * r
    qpu_specialization = _qpu_kernel(left, right, destination)
    qpu_name = (
        qpu_specialization[1]
        if qpu_specialization is not None
        else ("vc7.tiled_fp32_gemm" if destination.dtype == np.dtype(np.float32) else "vc7.tiled_int32_gemm")
    )
    candidates = [
        ExecutionCandidate(
            "numpy.matmul",
            Placement.CPU,
            lambda _: True,
            lambda _: 2e-6 + operations / 2e9,
        ),
        ExecutionCandidate(
            qpu_name,
            Placement.QPU,
            lambda _: qpu_specialization is not None,
            lambda _: 2e-6 + operations / 10e9,
            requires_calibration=destination.dtype == np.dtype(np.float32),
        ),
    ]
    if allow_hybrid or placement is Placement.HYBRID:
        use_columns = p == 1
        partitions = (
            (qpu_columns,)
            if use_columns and qpu_columns is not None
            else hybrid_column_partitions(r)
            if use_columns
            else (qpu_rows,)
            if qpu_rows is not None
            else hybrid_row_partitions(p)
        )
        for partition_rows in partitions:
            hybrid_specialization = (
                _hybrid_column_kernel(left, right, destination, partition_rows)
                if use_columns
                else _hybrid_kernel(left, right, destination, partition_rows)
            )
            total_units = r if use_columns else p
            qpu_operations = operations * partition_rows / total_units
            cpu_operations = operations - qpu_operations

            def supports_hybrid(
                _: OperationSpec,
                *,
                available: bool = hybrid_specialization is not None,
            ) -> bool:
                return available

            def estimate_hybrid(
                _: OperationSpec,
                *,
                cpu_ops: float = cpu_operations,
                qpu_ops: float = qpu_operations,
            ) -> float:
                return 4e-6 + max(cpu_ops / 2e9, qpu_ops / 10e9)

            candidates.append(
                ExecutionCandidate(
                    f"numpy+{qpu_name}.{'columns' if use_columns else 'rows'}.{partition_rows}",
                    Placement.HYBRID,
                    supports_hybrid,
                    estimate_hybrid,
                    PartitionSpec("columns" if use_columns else "rows", partition_rows, total_units, 16),
                    requires_calibration=True,
                )
            )
    registry = CapabilityRegistry(candidates)
    return registry.choose(specification, preference=placement, cost_model=cost_model)


def calibrate_matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue,
    cpu_queue: Queue | None = None,
    warmup: int = 1,
    repeat: int = 5,
    cost_model: CostModel | None = None,
) -> CostModel:
    """Measure supported CPU/QPU variants and seed exact-shape placement costs.

    Measurements include queue submission and completion, so they model
    whole-operator placement rather than raw kernel time. The supplied
    destination is overwritten by every iteration. Unsupported QPU shapes
    record CPU measurements only and are never mislabeled as QPU samples.
    """
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if queue.device is not destination.buffer.device:
        raise DependencyError("calibration queue and destination tensor must belong to the same device")
    plan_matmul(destination, left, right, placement=Placement.CPU)
    model = CostModel() if cost_model is None else cost_model
    specification = _matmul_specification(destination, left, right)

    def measure(placement: Placement, candidate_name: str) -> None:
        for _ in range(warmup):
            matmul(destination, left, right, queue=queue, placement=placement).wait()
        for _ in range(repeat):
            start = perf_counter()
            matmul(destination, left, right, queue=queue, placement=placement).wait()
            model.record(specification, candidate_name, perf_counter() - start)

    measure(Placement.CPU, "numpy.matmul")
    try:
        qpu_plan = plan_matmul(destination, left, right, placement=Placement.QPU)
    except ValueError:
        qpu_plan = None
    if qpu_plan is not None:
        measure(Placement.QPU, qpu_plan.candidate.name)
    if cpu_queue is not None:
        if cpu_queue is queue or cpu_queue.device is not queue.device:
            raise DependencyError("hybrid calibration requires a distinct CPU queue on the same device")
        use_columns = left.shape[0] == 1
        partitions = hybrid_column_partitions(right.shape[1]) if use_columns else hybrid_row_partitions(left.shape[0])
        for partition_units in partitions:
            try:
                hybrid_plan = plan_matmul(
                    destination,
                    left,
                    right,
                    placement=Placement.HYBRID,
                    qpu_rows=None if use_columns else partition_units,
                    qpu_columns=partition_units if use_columns else None,
                )
            except ValueError:
                continue
            for _ in range(warmup):
                matmul(
                    destination,
                    left,
                    right,
                    queue=queue,
                    cpu_queue=cpu_queue,
                    placement=Placement.HYBRID,
                    qpu_rows=None if use_columns else partition_units,
                    qpu_columns=partition_units if use_columns else None,
                ).wait()
            for _ in range(repeat):
                start = perf_counter()
                matmul(
                    destination,
                    left,
                    right,
                    queue=queue,
                    cpu_queue=cpu_queue,
                    placement=Placement.HYBRID,
                    qpu_rows=None if use_columns else partition_units,
                    qpu_columns=partition_units if use_columns else None,
                ).wait()
                model.record(specification, hybrid_plan.candidate.name, perf_counter() - start)
    return model


def hybrid_matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    qpu_queue: Queue,
    cpu_queue: Queue,
    qpu_rows: int | None = None,
    qpu_columns: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Split a compatible matrix multiplication into disjoint QPU and CPU work.

    Prefill uses a 16-aligned output-row prefix. Single-row decode uses a
    16-aligned output-column prefix instead, because no row split exists.
    The remaining region runs through NumPy on a separate queue and the
    returned event joins both submissions.
    """
    if qpu_queue is cpu_queue:
        raise DependencyError("hybrid matmul requires separate CPU and QPU queues")
    if qpu_queue.device is not destination.buffer.device or cpu_queue.device is not destination.buffer.device:
        raise DependencyError("hybrid matmul queues and tensors must belong to the same device")
    rows, _ = left.shape
    if rows == 1:
        columns = right.shape[1]
        selected_qpu_columns = max(16, columns // 4 // 16 * 16) if qpu_columns is None else qpu_columns
        if selected_qpu_columns < 0 or selected_qpu_columns > columns or selected_qpu_columns % 16:
            raise ValueError("hybrid decode matmul qpu_columns must be a multiple of 16 within the output range")
        if selected_qpu_columns == 0:
            return matmul(destination, left, right, queue=cpu_queue, wait_for=wait_for, placement=Placement.CPU)
        if selected_qpu_columns == columns:
            return matmul(destination, left, right, queue=qpu_queue, wait_for=wait_for, placement=Placement.QPU)
        return matmul(
            destination,
            left,
            right,
            queue=qpu_queue,
            cpu_queue=cpu_queue,
            wait_for=wait_for,
            placement=Placement.HYBRID,
            qpu_columns=selected_qpu_columns,
        )
    selected_qpu_rows = max(16, rows // 4 // 16 * 16) if qpu_rows is None else qpu_rows
    if selected_qpu_rows < 0 or selected_qpu_rows > rows or selected_qpu_rows % 16:
        raise ValueError("hybrid matmul qpu_rows must be a multiple of 16 within the output row range")
    if selected_qpu_rows == 0:
        return matmul(destination, left, right, queue=cpu_queue, wait_for=wait_for, placement=Placement.CPU)
    if selected_qpu_rows == rows:
        return matmul(destination, left, right, queue=qpu_queue, wait_for=wait_for, placement=Placement.QPU)
    return matmul(
        destination,
        left,
        right,
        queue=qpu_queue,
        cpu_queue=cpu_queue,
        wait_for=wait_for,
        placement=Placement.HYBRID,
        qpu_rows=selected_qpu_rows,
    )


def matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
    placement: Placement = Placement.AUTO,
    cost_model: CostModel | None = None,
    qpu_rows: int | None = None,
    qpu_columns: int | None = None,
) -> Event:
    """Compute ``destination = left @ right`` under the v0 no-broadcast contract."""
    if left.buffer.device is not destination.buffer.device or right.buffer.device is not destination.buffer.device:
        raise DependencyError("matmul tensors must belong to the same device")
    if len(left.shape) != 2 or len(right.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("matmul v0 requires rank-2 tensors")
    p, q = left.shape
    q_right, r = right.shape
    if q != q_right or destination.shape != (p, r):
        raise ValueError("matmul tensor shapes do not align")
    if left.dtype != right.dtype or left.dtype != destination.dtype:
        raise ValueError("matmul v0 requires equal tensor dtypes")

    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and destination tensor must belong to the same device")
    accesses = (
        destination.access(AccessMode.WRITE),
        left.access(AccessMode.READ),
        right.access(AccessMode.READ),
    )

    def cpu_reference() -> None:
        np.matmul(left.numpy(), right.numpy(), out=destination.numpy())

    if cpu_queue is not None and (cpu_queue is selected_queue or cpu_queue.device is not selected_queue.device):
        raise DependencyError("hybrid matmul requires distinct CPU and QPU queues on the same device")
    plan = plan_matmul(
        destination,
        left,
        right,
        placement=placement,
        cost_model=cost_model,
        allow_hybrid=cpu_queue is not None,
        qpu_rows=qpu_rows,
        qpu_columns=qpu_columns,
    )
    if plan.candidate.placement is Placement.HYBRID:
        if cpu_queue is None or plan.candidate.partition is None:
            raise DependencyError("hybrid matmul execution requires a distinct CPU queue")
        partition_units = plan.candidate.partition.qpu_units
        if plan.candidate.partition.axis == "columns":
            qpu_left = left
            qpu_right = right.slice((slice(None), slice(0, partition_units)))
            qpu_destination = destination.slice((slice(None), slice(0, partition_units)))
        else:
            qpu_left = left.slice((slice(0, partition_units), slice(None)))
            qpu_right = right
            qpu_destination = destination.slice((slice(0, partition_units), slice(None)))
        specialization = _qpu_kernel(qpu_left, qpu_right, qpu_destination)
        assert specialization is not None
        kernel, _ = specialization
        qpu_event = selected_queue.submit(
            kernel,
            (qpu_left, qpu_right, qpu_destination),
            grid=(qpu_destination.shape[1] // 16, max(1, qpu_destination.shape[0] // 16), 1),
            wait_for=wait_for,
            buffers=(
                qpu_destination.access(AccessMode.WRITE),
                qpu_left.access(AccessMode.READ),
                qpu_right.access(AccessMode.READ),
            ),
        )
        if plan.candidate.partition.axis == "columns":
            cpu_left = left
            cpu_right = right.slice((slice(None), slice(partition_units, r)))
            cpu_destination = destination.slice((slice(None), slice(partition_units, r)))
        else:
            cpu_left = left.slice((slice(partition_units, p), slice(None)))
            cpu_right = right
            cpu_destination = destination.slice((slice(partition_units, p), slice(None)))

        def cpu_tail() -> None:
            np.matmul(cpu_left.numpy(), cpu_right.numpy(), out=cpu_destination.numpy())

        cpu_event = cpu_queue.host_task(
            cpu_tail,
            wait_for=wait_for,
            buffers=(
                cpu_destination.access(AccessMode.WRITE),
                cpu_left.access(AccessMode.READ),
                cpu_right.access(AccessMode.READ),
            ),
            name="hybrid_matmul.cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_event),
            name="hybrid_matmul.join",
        )
    elif plan.candidate.placement is Placement.QPU:
        specialization = _qpu_kernel(left, right, destination)
        assert specialization is not None
        kernel, _ = specialization
        event = selected_queue.submit(
            kernel,
            (left, right, destination),
            grid=(r // 16, max(1, p // 16), 1),
            wait_for=wait_for,
            buffers=accesses,
        )
    else:
        event = selected_queue.host_task(cpu_reference, wait_for=wait_for, buffers=accesses)
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


__all__ = [
    "calibrate_matmul",
    "hybrid_column_partitions",
    "hybrid_matmul",
    "hybrid_row_partitions",
    "matmul",
    "plan_matmul",
]
