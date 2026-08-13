"""Deterministic per-output-channel INT8 packing for TinyLlama weight preprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue


@dataclass(frozen=True, slots=True)
class QuantizedMatrixInt8:
    """Row-wise symmetric INT8 matrix plus one FP32 dequantization scale per row."""

    values: npt.NDArray[np.int8]
    scales: npt.NDArray[np.float32]

    def __post_init__(self: Self) -> None:
        """Require compact, self-consistent storage before model code consumes it."""
        if self.values.ndim != 2 or self.scales.ndim != 1 or self.scales.shape != (self.values.shape[0],):
            raise ValueError("quantized matrix values must be rank-2 with one scale per output row")
        if not self.values.flags.c_contiguous or not self.scales.flags.c_contiguous:
            raise ValueError("quantized matrix values and scales must be contiguous")
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("quantized matrix scales must be finite and positive")

    @property
    def shape(self: Self) -> tuple[int, int]:
        """Return the logical ``(output_features, input_features)`` matrix shape."""
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    def dequantize(self: Self) -> npt.NDArray[np.float32]:
        """Return a contiguous FP32 reconstruction for reference execution."""
        return np.ascontiguousarray(self.values.astype(np.float32) * self.scales[:, None])


def quantize_per_output_channel_int8(weight: npt.NDArray[np.float32]) -> QuantizedMatrixInt8:
    """Symmetrically quantize FP32 ``(output, input)`` weights to INT8 rows."""
    if weight.ndim != 2 or weight.dtype != np.dtype(np.float32):
        raise ValueError("quantize_per_output_channel_int8 requires a rank-2 float32 matrix")
    max_abs = np.max(np.abs(weight), axis=1)
    scales = np.maximum(max_abs / np.float32(127.0), np.float32(1.0 / 127.0)).astype(np.float32)
    values = np.rint(weight / scales[:, None]).clip(-127, 127).astype(np.int8)
    return QuantizedMatrixInt8(np.ascontiguousarray(values), np.ascontiguousarray(scales))


def quantized_linear_fp32(
    destination: Tensor,
    source: Tensor,
    weight: QuantizedMatrixInt8,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply a packed INT8 weight matrix through the CPU FP32 reference path.

    This establishes model-loader and numerical contracts while a dedicated
    quantized QPU projection kernel is still pending. Weights are dequantized
    once at submission time into a host closure, not per output element.
    """
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("quantized linear source and destination must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("quantized_linear_fp32 requires float32 source and destination tensors")
    if len(source.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("quantized_linear_fp32 requires rank-2 source and destination tensors")
    batch, in_features = source.shape
    out_features, weight_features = weight.shape
    if in_features != weight_features or destination.shape != (batch, out_features):
        raise ValueError("quantized linear tensor shapes do not align")
    dequantized = weight.dequantize()
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and quantized linear tensors must belong to the same device")

    def apply() -> None:
        np.matmul(source.numpy(), dequantized.T, out=destination.numpy())

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="tinyllama.quantized_linear_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def quantized_linear_int8_gemm_fp32(
    destination: Tensor,
    source: Tensor,
    weight: QuantizedMatrixInt8,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Project FP32 activations through padded INT32 GEMM with INT8-range operands.

    Source rows are dynamically symmetrically quantized and weight INT8 rows
    are widened to INT32. The padded matrix multiplication is eligible for the
    existing ``smul24`` QPU kernel, while fake/CPU backends use the same NumPy
    fallback through :func:`qpu_xla.ops.matmul.matmul`.
    """
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("quantized linear source and destination must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("quantized_linear_int8_gemm_fp32 requires float32 source and destination tensors")
    if len(source.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("quantized linear requires rank-2 source and destination tensors")
    batch, in_features = source.shape
    out_features, weight_features = weight.shape
    if in_features != weight_features or destination.shape != (batch, out_features):
        raise ValueError("quantized linear tensor shapes do not align")
    if in_features * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError("quantized linear INT32 accumulation can overflow")
    padded_batch = (batch + 15) // 16 * 16
    padded_features = (in_features + 3) // 4 * 4
    padded_outputs = (out_features + 15) // 16 * 16
    device = destination.buffer.device
    quantized_source = device.tensor((padded_batch, padded_features), np.int32)
    quantized_weight = device.tensor((padded_features, padded_outputs), np.int32)
    accumulator = device.tensor((padded_batch, padded_outputs), np.int32)
    source_scales = np.empty((batch,), dtype=np.float32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and quantized linear tensors must belong to the same device")

    def prepare() -> None:
        values = source.numpy().astype(np.float32, copy=False)
        row_max = np.max(np.abs(values), axis=1)
        np.maximum(row_max / np.float32(127.0), np.float32(1.0 / 127.0), out=source_scales)
        quantized_source.numpy().fill(0)
        quantized_weight.numpy().fill(0)
        quantized_source.numpy()[:batch, :in_features] = np.rint(values / source_scales[:, None]).clip(-127, 127)
        quantized_weight.numpy()[:in_features, :out_features] = weight.values.T

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            quantized_source.access(AccessMode.WRITE),
            quantized_weight.access(AccessMode.WRITE),
        ),
        name="tinyllama.quantized_linear_int8.prepare",
    )
    gemm_event = matmul(
        accumulator,
        quantized_source,
        quantized_weight,
        queue=selected_queue,
        wait_for=(prepare_event,),
    )

    def dequantize() -> None:
        active = accumulator.numpy()[:batch, :out_features].astype(np.float32)
        destination.numpy()[:] = active * source_scales[:, None] * weight.scales[None, :]

    event = selected_queue.host_task(
        dequantize,
        wait_for=(gemm_event,),
        buffers=(accumulator.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="tinyllama.quantized_linear_int8.dequantize",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
