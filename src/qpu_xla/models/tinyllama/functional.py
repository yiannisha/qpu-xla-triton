"""Numerically sensitive TinyLlama FP32 primitives kept on the CPU queue path."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.rms_norm import RMS_NORM_FP32_KERNEL, supports_rms_norm_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def rms_norm_fp32(
    destination: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    epsilon: float = 1e-5,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply row-wise FP32 RMSNorm with a learned hidden-dimension weight."""
    tensors = (source, weight, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("RMSNorm tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("rms_norm_fp32 requires float32 tensors")
    if len(source.shape) != 2 or len(destination.shape) != 2 or len(weight.shape) != 1:
        raise ValueError("rms_norm_fp32 requires rank-2 source/destination and rank-1 weight")
    if destination.shape != source.shape or weight.shape != (source.shape[1],):
        raise ValueError("RMSNorm tensor shapes do not align")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and RMSNorm tensors must belong to the same device")

    qpu_supported = supports_rms_norm_fp32(source, weight, destination, selected_queue.device.backend)
    if placement is Placement.QPU and not qpu_supported:
        raise ValueError("RMSNorm QPU placement requires contiguous FP32 rows with a 16-aligned width")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected_queue or cpu_queue.device is not selected_queue.device:
            raise DependencyError("hybrid RMSNorm requires distinct CPU and QPU queues on the same device")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid RMSNorm qpu_rows must leave non-empty CPU and QPU row partitions")
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_rms_norm_fp32(qpu_source, weight, qpu_destination, selected_queue.device.backend):
            raise ValueError("RMSNorm QPU partition does not satisfy the kernel contract")
        qpu_event = selected_queue.submit(
            RMS_NORM_FP32_KERNEL,
            (qpu_source, weight, qpu_destination, float(epsilon)),
            grid=(qpu_rows, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                qpu_destination.access(AccessMode.WRITE),
            ),
        )
        cpu_source = source.slice((slice(qpu_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, rows), slice(None)))

        def apply_tail() -> None:
            values = cast(npt.NDArray[np.float32], cpu_source.numpy())
            scale = cast(npt.NDArray[np.float32], weight.numpy())
            mean_square = np.mean(values * values, axis=1, keepdims=True, dtype=np.float32)
            cpu_destination.numpy()[:] = values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon))) * scale

        cpu_event = cpu_queue.host_task(
            apply_tail,
            wait_for=wait_for,
            buffers=(
                cpu_source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="rms_norm_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_event),
            name="rms_norm_fp32.hybrid_join",
        )
        if close_queue:
            event.wait()
            selected_queue.close()
        return event

    if placement is Placement.QPU:
        event = selected_queue.submit(
            RMS_NORM_FP32_KERNEL,
            (source, weight, destination, float(epsilon)),
            grid=(source.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
        if close_queue:
            event.wait()
            selected_queue.close()
        return event

    def apply() -> None:
        values = cast(npt.NDArray[np.float32], source.numpy())
        scale = cast(npt.NDArray[np.float32], weight.numpy())
        mean_square = np.mean(values * values, axis=1, keepdims=True, dtype=np.float32)
        destination.numpy()[:] = values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon))) * scale

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            weight.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.rms_norm_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def rope_fp32(
    destination: Tensor,
    source: Tensor,
    positions: Tensor,
    *,
    base: float = 10_000.0,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply standard FP32 rotary position embedding to even/odd feature pairs."""
    tensors = (source, positions, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("RoPE tensors must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("rope_fp32 requires float32 source and destination tensors")
    if positions.dtype != np.dtype(np.int32):
        raise ValueError("rope_fp32 requires int32 positions")
    if len(source.shape) != 2 or destination.shape != source.shape or positions.shape != (source.shape[0],):
        raise ValueError("RoPE tensor shapes do not align")
    if source.shape[1] % 2:
        raise ValueError("rope_fp32 requires an even feature dimension")
    if not np.isfinite(base) or base <= 1:
        raise ValueError("RoPE base must be finite and greater than one")
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and RoPE tensors must belong to the same device")

    def apply() -> None:
        values = np.array(source.numpy(), copy=True)
        feature_dim = values.shape[1]
        frequencies = np.power(np.float32(base), -np.arange(0, feature_dim, 2, dtype=np.float32) / feature_dim)
        angles = positions.numpy().astype(np.float32, copy=False)[:, None] * frequencies[None, :]
        cosine, sine = np.cos(angles), np.sin(angles)
        destination.numpy()[:, 0::2] = values[:, 0::2] * cosine - values[:, 1::2] * sine
        destination.numpy()[:, 1::2] = values[:, 0::2] * sine + values[:, 1::2] * cosine

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            positions.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        ),
        name="tinyllama.rope_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def embedding_lookup_fp32(
    destination: Tensor,
    token_ids: Tensor,
    table: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_tokens: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Look up FP32 token embeddings using the shared placement-aware operator."""
    from qpu_xla.ops.embedding import embedding_lookup_fp32 as execute

    return execute(
        destination,
        token_ids,
        table,
        queue=queue,
        cpu_queue=cpu_queue,
        placement=placement,
        qpu_tokens=qpu_tokens,
        wait_for=wait_for,
    )


def silu_gated_fp32(
    destination: Tensor,
    gate: Tensor,
    up: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute ``silu(gate) * up`` using the shared placement-aware operator."""
    from qpu_xla.ops.swiglu import swiglu_fp32

    return swiglu_fp32(
        destination,
        gate,
        up,
        queue=queue,
        cpu_queue=cpu_queue,
        placement=placement,
        qpu_rows=qpu_rows,
        wait_for=wait_for,
    )


def residual_add_fp32(
    destination: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Add residual activations using the shared placement-aware operator."""
    from qpu_xla.ops.residual import residual_add_fp32 as execute

    return execute(
        destination,
        left,
        right,
        queue=queue,
        cpu_queue=cpu_queue,
        placement=placement,
        qpu_rows=qpu_rows,
        wait_for=wait_for,
    )


def greedy_sample_fp32(
    destination: Tensor,
    logits: Tensor,
    *,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Select highest-logit tokens using the shared placement-aware operator."""
    from qpu_xla.ops.sampling import greedy_sample_fp32 as execute

    return execute(
        destination,
        logits,
        queue=queue,
        cpu_queue=cpu_queue,
        placement=placement,
        qpu_rows=qpu_rows,
        wait_for=wait_for,
    )
