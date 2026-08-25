"""Standalone FP32 activations with true CPU/QPU row-split execution."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypedDict, Unpack, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.activation import (
    GELU_TANH_FP32_KERNEL,
    SILU_FP32_KERNEL,
    supports_activation_fp32,
)
from qpu_xla.kernels.swiglu import _workgroup_count
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement

Activation = Literal["silu", "gelu_tanh"]


class _ActivationOptions(TypedDict, total=False):
    queue: Queue | None
    cpu_queue: Queue | None
    placement: Placement
    qpu_rows: int | None
    wait_for: Iterable[Event]


def activation_fp32(
    destination: Tensor,
    source: Tensor,
    *,
    activation: Activation,
    queue: Queue | None = None,
    cpu_queue: Queue | None = None,
    placement: Placement = Placement.AUTO,
    qpu_rows: int | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply SiLU or PyTorch's tanh-approximate GELU."""
    if activation not in {"silu", "gelu_tanh"}:
        raise ValueError("activation must be 'silu' or 'gelu_tanh'")
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("activation tensors must belong to one device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != source.dtype or destination.shape != source.shape:
        raise ValueError("activation requires equal FP32 tensors")
    selected = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected.device is not destination.buffer.device:
        raise DependencyError("activation queue belongs to a different device")
    kernel = SILU_FP32_KERNEL if activation == "silu" else GELU_TANH_FP32_KERNEL

    def apply(
        source_values: npt.NDArray[np.float32],
        destination_values: npt.NDArray[np.float32],
    ) -> None:
        if activation == "silu":
            destination_values[:] = source_values / (np.float32(1.0) + np.exp(-source_values))
        else:
            inner = np.float32(np.sqrt(2.0 / np.pi)) * (
                source_values + np.float32(0.044715) * source_values * source_values * source_values
            )
            destination_values[:] = np.float32(0.5) * source_values * (np.float32(1.0) + np.tanh(inner))

    supported = supports_activation_fp32(source, destination, selected.device.backend)
    if placement is Placement.QPU and not supported:
        raise ValueError("activation QPU placement requires contiguous tensors aligned to 16 FP32 values")
    if placement is Placement.HYBRID:
        if cpu_queue is None or cpu_queue is selected or cpu_queue.device is not selected.device:
            raise DependencyError("hybrid activation requires distinct CPU and QPU queues on one device")
        if len(source.shape) != 2:
            raise ValueError("hybrid activation partitions rank-2 tensors by rows")
        rows = source.shape[0]
        if qpu_rows is None or qpu_rows <= 0 or qpu_rows >= rows:
            raise ValueError("hybrid activation qpu_rows must leave non-empty CPU and QPU partitions")
        qpu_source = source.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        if not supports_activation_fp32(qpu_source, qpu_destination, selected.device.backend):
            raise ValueError("activation QPU partition does not satisfy vector alignment")
        vectors = qpu_source.nbytes // 64
        qpu_event = selected.submit(
            kernel,
            (qpu_source, qpu_destination),
            grid=(_workgroup_count(vectors), 1, 1),
            wait_for=wait_for,
            buffers=(qpu_source.access(AccessMode.READ), qpu_destination.access(AccessMode.WRITE)),
        )
        cpu_source = source.slice((slice(qpu_rows, rows), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, rows), slice(None)))
        cpu_event = cpu_queue.host_task(
            lambda: apply(
                cast(npt.NDArray[np.float32], cpu_source.numpy()),
                cast(npt.NDArray[np.float32], cpu_destination.numpy()),
            ),
            wait_for=wait_for,
            buffers=(cpu_source.access(AccessMode.READ), cpu_destination.access(AccessMode.WRITE)),
            name=f"{activation}_fp32.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(qpu_event, cpu_event),
            name=f"{activation}_fp32.hybrid_join",
        )
    elif placement is Placement.QPU:
        vectors = source.nbytes // 64
        event = selected.submit(
            kernel,
            (source, destination),
            grid=(_workgroup_count(vectors), 1, 1),
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        )
    else:
        event = selected.host_task(
            lambda: apply(
                cast(npt.NDArray[np.float32], source.numpy()),
                cast(npt.NDArray[np.float32], destination.numpy()),
            ),
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name=f"{activation}_fp32.cpu",
        )
    if close_queue:
        event.wait()
        selected.close()
    return event


def silu_fp32(destination: Tensor, source: Tensor, **kwargs: Unpack[_ActivationOptions]) -> Event:
    """Apply standalone SiLU with the selected placement."""
    return activation_fp32(destination, source, activation="silu", **kwargs)


def gelu_tanh_fp32(destination: Tensor, source: Tensor, **kwargs: Unpack[_ActivationOptions]) -> Event:
    """Apply PyTorch's tanh-approximate GELU with the selected placement."""
    return activation_fp32(destination, source, activation="gelu_tanh", **kwargs)


__all__ = ["activation_fp32", "gelu_tanh_fp32", "silu_fp32"]
