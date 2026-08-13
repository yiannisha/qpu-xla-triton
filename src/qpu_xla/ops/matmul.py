"""Matrix multiplication contract with CPU reference and tiled QPU specialization."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernel import Kernel
from qpu_xla.kernels.gemm import TILED_INT32_GEMM_KERNEL, supports_tiled_int32_gemm
from qpu_xla.kernels.gemm_fp32 import TILED_FP32_GEMM_KERNEL, supports_tiled_fp32_gemm
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import (
    CapabilityRegistry,
    CostModel,
    ExecutionCandidate,
    OperationSpec,
    Placement,
    PlannedExecution,
)


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
    return None


def plan_matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    placement: Placement = Placement.AUTO,
    cost_model: CostModel | None = None,
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
    qpu_name = "vc7.tiled_fp32_gemm" if destination.dtype == np.dtype(np.float32) else "vc7.tiled_int32_gemm"
    registry = CapabilityRegistry(
        (
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
            ),
        )
    )
    return registry.choose(specification, preference=placement, cost_model=cost_model)


def calibrate_matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue,
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
        return model
    measure(Placement.QPU, qpu_plan.candidate.name)
    return model


def hybrid_matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    qpu_queue: Queue,
    cpu_queue: Queue,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Split a compatible matrix multiplication into disjoint QPU and CPU rows.

    A tiled QPU prefix uses a multiple of 16 output rows while the residual
    rows run through NumPy on a separate queue. Both queues share the same
    device allocation and only read the common right matrix, so the output
    regions are disjoint. The returned event joins both submissions.
    """
    if qpu_queue is cpu_queue:
        raise DependencyError("hybrid matmul requires separate CPU and QPU queues")
    if qpu_queue.device is not destination.buffer.device or cpu_queue.device is not destination.buffer.device:
        raise DependencyError("hybrid matmul queues and tensors must belong to the same device")
    if len(left.shape) != 2 or len(right.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("hybrid matmul requires rank-2 tensors")
    rows, reduction = left.shape
    right_rows, columns = right.shape
    if reduction != right_rows or destination.shape != (rows, columns):
        raise ValueError("hybrid matmul tensor shapes do not align")
    dependencies = tuple(wait_for)
    selected_qpu_rows = rows - rows % 16 if qpu_rows is None else qpu_rows
    if selected_qpu_rows < 0 or selected_qpu_rows > rows or selected_qpu_rows % 16:
        raise ValueError("hybrid matmul qpu_rows must be a multiple of 16 within the output row range")
    if selected_qpu_rows == 0:
        return matmul(destination, left, right, queue=cpu_queue, wait_for=dependencies, placement=Placement.CPU)

    qpu_left = left.slice((slice(0, selected_qpu_rows), slice(None)))
    qpu_destination = destination.slice((slice(0, selected_qpu_rows), slice(None)))
    plan_matmul(qpu_destination, qpu_left, right, placement=Placement.QPU)
    qpu_event = matmul(
        qpu_destination,
        qpu_left,
        right,
        queue=qpu_queue,
        wait_for=dependencies,
        placement=Placement.QPU,
    )
    if selected_qpu_rows == rows:
        return qpu_event

    cpu_left = left.slice((slice(selected_qpu_rows, rows), slice(None)))
    cpu_destination = destination.slice((slice(selected_qpu_rows, rows), slice(None)))
    cpu_event = matmul(
        cpu_destination,
        cpu_left,
        right,
        queue=cpu_queue,
        wait_for=dependencies,
        placement=Placement.CPU,
    )
    return cpu_queue.host_task(lambda: None, wait_for=(cpu_event, qpu_event), name="hybrid_matmul.join")


def matmul(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
    placement: Placement = Placement.AUTO,
    cost_model: CostModel | None = None,
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

    plan = plan_matmul(destination, left, right, placement=placement, cost_model=cost_model)
    if plan.candidate.placement is Placement.QPU:
        specialization = _qpu_kernel(left, right, destination)
        assert specialization is not None
        kernel, _ = specialization
        event = selected_queue.submit(
            kernel,
            (left, right, destination),
            grid=(r // 16, p // 16, 1),
            wait_for=wait_for,
            buffers=accesses,
        )
    else:
        event = selected_queue.host_task(cpu_reference, wait_for=wait_for, buffers=accesses)
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
