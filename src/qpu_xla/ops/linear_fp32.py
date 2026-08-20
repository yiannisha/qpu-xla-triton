"""Persistent FP32 linear projection plans for prefill and decode."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class PreparedFP32Linear:
    """Own a transposed weight matrix and reusable QPU projection scratch.

    Public weights use the conventional ``(output_features, input_features)``
    layout. The plan transposes and pads them once into the GEMM ``(K, N)``
    layout. Decode uses the dedicated one-row GEMV; prefill uses tiled GEMM.
    Explicit hybrid execution divides decode output columns or prefill rows
    into disjoint QPU and NumPy regions.
    """

    def __init__(self: Self, device: Device, weight: np.ndarray, *, max_batch: int) -> None:
        """Prepare one immutable output-by-input weight matrix."""
        if weight.ndim != 2 or weight.dtype != np.dtype(np.float32):
            raise ValueError("PreparedFP32Linear weight must be a rank-2 float32 array")
        if max_batch <= 0:
            raise ValueError("PreparedFP32Linear max_batch must be positive")
        self.device = device
        self.output_features, self.input_features = map(int, weight.shape)
        self.max_batch = max_batch
        self.padded_batch = _round_up(max_batch, 16)
        self.padded_features = _round_up(self.input_features, 4)
        self.padded_outputs = _round_up(self.output_features, 16)
        self._logical_weight = np.ascontiguousarray(weight)
        self._weight = device.tensor((self.padded_features, self.padded_outputs), np.float32)
        self._source = device.tensor((self.padded_batch, self.padded_features), np.float32)
        self._result = device.tensor((self.padded_batch, self.padded_outputs), np.float32)
        self._weight.numpy().fill(0.0)
        self._weight.numpy()[: self.input_features, : self.output_features] = self._logical_weight.T
        self._last_event: Event | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter the plan's context-managed lifetime."""
        if self._closed:
            raise RuntimeError("prepared FP32 linear plan is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release plan-owned buffers at context exit."""
        self.close()

    def close(self: Self) -> None:
        """Wait for outstanding work and release plan-owned buffers."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.wait()
        for tensor in (self._weight, self._source, self._result):
            tensor.buffer.close()
        self._closed = True

    def _validate(self: Self, destination: Tensor, source: Tensor, queue: Queue) -> int:
        if self._closed:
            raise RuntimeError("prepared FP32 linear plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared FP32 linear plan already has an in-flight invocation")
        if (
            queue.device is not self.device
            or source.buffer.device is not self.device
            or destination.buffer.device is not self.device
        ):
            raise DependencyError("prepared FP32 linear queue and tensors must belong to the plan device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("prepared FP32 linear requires float32 source and destination")
        if len(source.shape) != 2 or source.shape[1] != self.input_features:
            raise ValueError("prepared FP32 linear source shape does not align")
        batch = source.shape[0]
        if batch > self.max_batch or destination.shape != (batch, self.output_features):
            raise ValueError("prepared FP32 linear destination shape does not align or batch exceeds the plan")
        return batch

    def execute(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        queue: Queue,
        placement: Placement = Placement.QPU,
        cpu_queue: Queue | None = None,
        qpu_units: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Execute CPU-only, QPU-only, or disjoint CPU/QPU projection work."""
        batch = self._validate(destination, source, queue)
        if placement is Placement.CPU:
            event = queue.host_task(
                lambda: np.matmul(source.numpy(), self._logical_weight.T, out=destination.numpy()),
                wait_for=wait_for,
                buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
                name="fp32_linear.cpu",
            )
            self._last_event = event
            return event
        if placement is Placement.AUTO:
            raise ValueError(
                "PreparedFP32Linear AUTO requires external benchmark evidence; request CPU, QPU, or HYBRID"
            )
        if placement is Placement.HYBRID and (cpu_queue is None or cpu_queue is queue):
            raise DependencyError("hybrid FP32 linear requires a distinct CPU queue")
        if cpu_queue is not None and cpu_queue.device is not self.device:
            raise DependencyError("hybrid FP32 linear queues must belong to the plan device")

        active_batch = 1 if batch == 1 else _round_up(batch, 16)
        kwargs: dict[str, object] = {}
        if placement is Placement.HYBRID:
            assert cpu_queue is not None
            if qpu_units is None:
                qpu_units = (self.padded_outputs if batch == 1 else active_batch) // 4 // 16 * 16
            kwargs["cpu_queue"] = cpu_queue
            if batch == 1:
                kwargs["qpu_columns"] = qpu_units
            else:
                kwargs["qpu_rows"] = qpu_units

        direct = (
            self.padded_features == self.input_features
            and self.padded_outputs == self.output_features
            and active_batch == batch
            and source.numpy().flags.c_contiguous
            and destination.numpy().flags.c_contiguous
        )
        if direct:
            event = matmul(
                destination,
                source,
                self._weight,
                queue=queue,
                wait_for=wait_for,
                placement=placement,
                **kwargs,
            )
            self._last_event = event
            return event

        def prepare() -> None:
            active = self._source.numpy()[:active_batch]
            active.fill(0.0)
            active[:batch, : self.input_features] = source.numpy()

        prepare_event = queue.host_task(
            prepare,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), self._source.access(AccessMode.WRITE)),
            name="fp32_linear.prepare",
        )
        kernel_source = self._source.slice((slice(0, active_batch), slice(None)))
        kernel_result = self._result.slice((slice(0, active_batch), slice(None)))
        compute_event = matmul(
            kernel_result,
            kernel_source,
            self._weight,
            queue=queue,
            wait_for=(prepare_event,),
            placement=placement,
            **kwargs,
        )

        def finish() -> None:
            destination.numpy()[:] = self._result.numpy()[:batch, : self.output_features]

        event = queue.host_task(
            finish,
            wait_for=(compute_event,),
            buffers=(self._result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="fp32_linear.finish",
        )
        self._last_event = event
        return event


__all__ = ["PreparedFP32Linear"]
