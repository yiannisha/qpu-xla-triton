"""Two-layer FP32 MLP assembled from queued QPU GEMM and epilogue stages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.bias_activation import bias_activation
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue


def _round_up(value: int, tile: int) -> int:
    return (value + tile - 1) // tile * tile


def mlp_fp32(
    destination: Tensor,
    source: Tensor,
    weight1: Tensor,
    bias1: Tensor,
    weight2: Tensor,
    bias2: Tensor,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute ``relu(source @ weight1 + bias1) @ weight2 + bias2``.

    Matrix stages use the packaged FP32 tiled GEMM when their padded shapes
    satisfy its contract. Bias and ReLU use the packaged tiled epilogue under
    the same conditions, with a queue-integrated NumPy fallback otherwise.
    """
    tensors = (source, weight1, bias1, weight2, bias2, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("MLP tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
        raise ValueError("mlp_fp32 requires float32 tensors")
    if len(source.shape) != 2 or len(weight1.shape) != 2 or len(weight2.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("mlp_fp32 requires rank-2 source, weights, and destination tensors")
    if len(bias1.shape) != 1 or len(bias2.shape) != 1:
        raise ValueError("mlp_fp32 requires rank-1 bias tensors")

    batch, in_features = source.shape
    weight1_rows, hidden_features = weight1.shape
    weight2_rows, out_features = weight2.shape
    if weight1_rows != in_features or weight2_rows != hidden_features:
        raise ValueError("MLP weight dimensions do not align")
    if bias1.shape != (hidden_features,) or bias2.shape != (out_features,):
        raise ValueError("MLP bias dimensions do not align")
    if destination.shape != (batch, out_features):
        raise ValueError("MLP destination shape does not align")

    padded_batch = _round_up(batch, 16)
    padded_input = _round_up(in_features, 4)
    padded_hidden = _round_up(hidden_features, 16)
    padded_output = _round_up(out_features, 16)
    device = destination.buffer.device
    padded_source = device.tensor((padded_batch, padded_input), np.float32)
    padded_weight1 = device.tensor((padded_input, padded_hidden), np.float32)
    hidden = device.tensor((padded_batch, padded_hidden), np.float32)
    padded_bias1 = device.tensor((padded_hidden,), np.float32)
    padded_weight2 = device.tensor((padded_hidden, padded_output), np.float32)
    padded_result = device.tensor((padded_batch, padded_output), np.float32)
    padded_bias2 = device.tensor((padded_output,), np.float32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def prepare() -> None:
        for tensor in (padded_source, padded_weight1, padded_bias1, padded_weight2, padded_bias2):
            tensor.numpy().fill(0.0)
        padded_source.numpy()[:batch, :in_features] = source.numpy()
        padded_weight1.numpy()[:in_features, :hidden_features] = weight1.numpy()
        padded_bias1.numpy()[:hidden_features] = bias1.numpy()
        padded_weight2.numpy()[:hidden_features, :out_features] = weight2.numpy()
        padded_bias2.numpy()[:out_features] = bias2.numpy()

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            weight1.access(AccessMode.READ),
            bias1.access(AccessMode.READ),
            weight2.access(AccessMode.READ),
            bias2.access(AccessMode.READ),
            padded_source.access(AccessMode.WRITE),
            padded_weight1.access(AccessMode.WRITE),
            padded_bias1.access(AccessMode.WRITE),
            padded_weight2.access(AccessMode.WRITE),
            padded_bias2.access(AccessMode.WRITE),
        ),
        name="mlp_fp32.prepare",
    )
    hidden_gemm = matmul(hidden, padded_source, padded_weight1, queue=selected_queue, wait_for=(prepare_event,))
    activation_event = bias_activation(
        hidden,
        hidden,
        padded_bias1,
        relu=True,
        queue=selected_queue,
        wait_for=(hidden_gemm,),
    )
    output_gemm = matmul(padded_result, hidden, padded_weight2, queue=selected_queue, wait_for=(activation_event,))
    output_bias_event = bias_activation(
        padded_result,
        padded_result,
        padded_bias2,
        queue=selected_queue,
        wait_for=(output_gemm,),
    )

    def finish() -> None:
        result = cast(npt.NDArray[np.float32], padded_result.numpy()[:batch, :out_features])
        np.copyto(destination.numpy(), result, casting="no")

    result_event = selected_queue.host_task(
        finish,
        wait_for=(output_bias_event,),
        buffers=(padded_result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="mlp_fp32.finish",
    )
    if close_queue:
        result_event.wait()
        selected_queue.close()
    return result_event


__all__ = ["mlp_fp32"]
