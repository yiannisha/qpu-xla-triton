"""Fused bias and ReLU operator with a QPU epilogue specialization."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qpu_xla.errors import DependencyError
from qpu_xla.kernels.bias_activation import bias_activation_kernel, supports_bias_activation
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.queue import Event, Queue


def bias_activation(
    destination: Tensor,
    source: Tensor,
    bias: Tensor | None = None,
    *,
    relu: bool = False,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply ``destination = relu(source + bias)`` through the runtime.

    Bias is a one-dimensional per-column vector. The QPU specialization uses
    16x16 tiles and supports contiguous FP32 and INT32 matrices. Unsupported
    shapes retain a queue-integrated NumPy reference implementation.
    """
    tensors = (source, destination) if bias is None else (source, bias, destination)
    if any(tensor.buffer.device is not destination.buffer.device for tensor in tensors):
        raise DependencyError("bias activation tensors must belong to the same device")
    if source.dtype != destination.dtype:
        raise ValueError("bias activation source and destination dtypes must match")
    if source.shape != destination.shape or len(source.shape) != 2:
        raise ValueError("bias activation requires equal rank-2 source and destination shapes")
    if bias is not None and (bias.dtype != source.dtype or bias.shape != (source.shape[1],)):
        raise ValueError("bias activation bias must match the source dtype and column dimension")
    if source.dtype not in {np.dtype(np.float32), np.dtype(np.int32)}:
        raise ValueError("bias activation supports only float32 and int32")

    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and bias activation tensors must belong to the same device")

    use_bias = bias is not None
    accesses = [destination.access(AccessMode.WRITE), source.access(AccessMode.READ)]
    if bias is not None:
        accesses.append(bias.access(AccessMode.READ))

    def reference() -> None:
        if bias is None:
            np.copyto(destination.numpy(), source.numpy(), casting="no")
        else:
            np.add(source.numpy(), bias.numpy(), out=destination.numpy())
        if relu:
            np.maximum(destination.numpy(), 0, out=destination.numpy())

    if supports_bias_activation(source, bias, destination, selected_queue.device.backend):
        kernel = bias_activation_kernel(source.dtype, use_bias=use_bias, apply_relu=relu)
        event = selected_queue.submit(
            kernel,
            (source, bias, destination),
            grid=(source.shape[1] // 16, source.shape[0] // 16, 1),
            wait_for=wait_for,
            buffers=accesses,
        )
    else:
        event = selected_queue.host_task(reference, wait_for=wait_for, buffers=accesses)
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


__all__ = ["bias_activation"]
