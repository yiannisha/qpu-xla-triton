"""Affine FP32 LayerNorm with CPU, QPU, and concurrent row partitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.layer_norm import LAYER_NORM_FP32_KERNEL, supports_layer_norm_fp32
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def layer_norm_fp32(
    destination: Tensor,
    source: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    epsilon: float,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply row-wise ``(x-mean)/sqrt(var+eps) * weight + bias``."""
    tensors = (source, weight, bias, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("LayerNorm tensors must belong to one device")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("LayerNorm epsilon must be finite and positive")
    if (
        any(tensor.dtype != np.dtype(np.float32) for tensor in tensors)
        or len(source.shape) != 2
        or destination.shape != source.shape
        or weight.shape != (source.shape[1],)
        or bias.shape != weight.shape
    ):
        raise ValueError("LayerNorm requires FP32 rows and matching affine vectors")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected.device is not destination.buffer.device:
        raise DependencyError("LayerNorm queue belongs to a different device")

    def reference(src: Tensor = source, dst: Tensor = destination) -> None:
        values = cast(npt.NDArray[np.float32], src.numpy())
        destination_values = cast(npt.NDArray[np.float32], dst.numpy())
        weight_values = cast(npt.NDArray[np.float32], weight.numpy())
        bias_values = cast(npt.NDArray[np.float32], bias.numpy())
        mean = np.mean(values, axis=1, keepdims=True, dtype=np.float32)
        centered = values - mean
        variance = np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float32)
        destination_values[:] = centered * np.reciprocal(np.sqrt(variance + np.float32(epsilon))) * weight_values
        np.add(destination_values, bias_values, out=destination_values)

    supported = supports_layer_norm_fp32(source, weight, bias, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("LayerNorm QPU placement requires contiguous FP32 rows with a 16-aligned width")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid LayerNorm requires distinct CPU and QPU queues on one device")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid LayerNorm qpu_rows must leave non-empty CPU and QPU partitions")
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_layer_norm_fp32(qpu_source, weight, bias, qpu_destination, selected.device.backend):
            raise ValueError("LayerNorm QPU partition does not satisfy the row kernel contract")
        qpu_event = selected.submit(
            LAYER_NORM_FP32_KERNEL,
            (qpu_source, weight, bias, qpu_destination, float(epsilon)),
            grid=(qpu_rows, 1, 1),
            wait_for=wait_for,
            buffers=(
                qpu_source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                bias.access(AccessMode.READ),
                qpu_destination.access(AccessMode.WRITE),
            ),
        )
        cpu_source = source.slice((slice(qpu_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, rows), slice(None)))

        def cpu_tail() -> None:
            values = cast(npt.NDArray[np.float32], cpu_source.numpy())
            destination_values = cast(npt.NDArray[np.float32], cpu_destination.numpy())
            weight_values = cast(npt.NDArray[np.float32], weight.numpy())
            bias_values = cast(npt.NDArray[np.float32], bias.numpy())
            mean = np.mean(values, axis=1, keepdims=True, dtype=np.float32)
            centered = values - mean
            variance = np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float32)
            destination_values[:] = (
                centered * np.reciprocal(np.sqrt(variance + np.float32(epsilon))) * weight_values + bias_values
            )

        cpu_event = cpu_queue.host_task(
            cpu_tail,
            wait_for=wait_for,
            buffers=(
                cpu_source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                bias.access(AccessMode.READ),
                cpu_destination.access(AccessMode.WRITE),
            ),
            name="layer_norm_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(qpu_event, cpu_event),
            name="layer_norm_fp32.hybrid_join",
        )
    elif placement is Placement.QPU:
        event = selected.submit(
            LAYER_NORM_FP32_KERNEL,
            (source, weight, bias, destination, float(epsilon)),
            grid=(source.shape[0], 1, 1),
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                bias.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
        )
    else:
        event = selected.host_task(
            reference,
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                weight.access(AccessMode.READ),
                bias.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="layer_norm_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


__all__ = ["layer_norm_fp32"]
