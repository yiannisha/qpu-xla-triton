"""Persistent INT32 MLP execution plan with reusable padded device buffers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError, DeviceClosedError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue


def _round_up(value: int, tile: int) -> int:
    """Round one positive matrix dimension to a tiled GEMM alignment."""
    return (value + tile - 1) // tile * tile


class MlpInt32Plan:
    """Prepare fixed-shape INT32 MLP weights once and execute many inputs.

    The plan implements ``relu(source @ weight1 + bias1) @ weight2 + bias2``
    for fixed ``(batch, input, hidden, output)`` dimensions.  It owns all
    padded staging and intermediate tensors, so subsequent :meth:`execute`
    calls only upload the new source and write the supplied destination.
    Calls are explicitly serialized through the plan's previous event; this
    makes reuse safe even when a caller submits from multiple same-device
    queues.
    """

    def __init__(
        self: Self,
        device: Device,
        *,
        batch: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        """Allocate the persistent padded tensors for one fixed MLP topology."""
        if min(batch, in_features, hidden_features, out_features) <= 0:
            raise ValueError("MLP plan dimensions must be positive")
        self._device = device
        self.batch = batch
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self._closed = False
        self._weight_event: Event | None = None
        self._tail: Event | None = None

        padded_batch = _round_up(batch, 16)
        padded_input = _round_up(in_features, 4)
        padded_hidden = _round_up(hidden_features, 16)
        padded_output = _round_up(out_features, 16)
        self._source = device.tensor((padded_batch, padded_input), np.int32)
        self._weight1 = device.tensor((padded_input, padded_hidden), np.int32)
        self._bias1 = device.tensor((hidden_features,), np.int32)
        self._hidden = device.tensor((padded_batch, padded_hidden), np.int32)
        self._weight2 = device.tensor((padded_hidden, padded_output), np.int32)
        self._bias2 = device.tensor((out_features,), np.int32)
        self._result = device.tensor((padded_batch, padded_output), np.int32)

    @property
    def device(self: Self) -> Device:
        """Return the device that owns this plan and all reusable buffers."""
        return self._device

    @property
    def closed(self: Self) -> bool:
        """Whether this plan has released its owned buffer views."""
        return self._closed

    def _require_open(self: Self) -> None:
        """Reject work after the plan's persistent tensors have been closed."""
        if self._closed:
            raise DeviceClosedError("MLP plan is closed")

    def _queue(self: Self, queue: Queue) -> None:
        """Validate that a submission queue belongs to the plan's device."""
        self._require_open()
        if queue.device is not self._device:
            raise DependencyError("MLP plan queue belongs to a different device")

    def _dependencies(self: Self, wait_for: Iterable[Event]) -> tuple[Event, ...]:
        """Append the previous plan event once to serialize shared intermediates."""
        dependencies = list(wait_for)
        if self._tail is not None and self._tail not in dependencies:
            dependencies.append(self._tail)
        return tuple(dependencies)

    def load_weights(
        self: Self,
        weight1: Tensor,
        bias1: Tensor,
        weight2: Tensor,
        bias2: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Pad and retain one fixed set of INT32 weights and biases."""
        self._queue(queue)
        tensors = (weight1, bias1, weight2, bias2)
        if any(tensor.buffer.device is not self._device for tensor in tensors):
            raise DependencyError("MLP plan weights must belong to the plan device")
        if any(tensor.dtype != np.dtype(np.int32) for tensor in tensors):
            raise ValueError("MLP plan weights and biases must use int32")
        if weight1.shape != (self.in_features, self.hidden_features) or weight2.shape != (
            self.hidden_features,
            self.out_features,
        ):
            raise ValueError("MLP plan weight shapes do not match its topology")
        if bias1.shape != (self.hidden_features,) or bias2.shape != (self.out_features,):
            raise ValueError("MLP plan bias shapes do not match its topology")

        def prepare_weights() -> None:
            self._weight1.numpy().fill(0)
            self._weight2.numpy().fill(0)
            self._weight1.numpy()[: self.in_features, : self.hidden_features] = weight1.numpy()
            self._weight2.numpy()[: self.hidden_features, : self.out_features] = weight2.numpy()
            self._bias1.numpy()[:] = bias1.numpy()
            self._bias2.numpy()[:] = bias2.numpy()

        event = queue.host_task(
            prepare_weights,
            wait_for=self._dependencies(wait_for),
            buffers=(
                weight1.access(AccessMode.READ),
                bias1.access(AccessMode.READ),
                weight2.access(AccessMode.READ),
                bias2.access(AccessMode.READ),
                self._weight1.access(AccessMode.WRITE),
                self._weight2.access(AccessMode.WRITE),
                self._bias1.access(AccessMode.WRITE),
                self._bias2.access(AccessMode.WRITE),
            ),
            name="mlp_int32_plan.load_weights",
        )
        self._weight_event = event
        self._tail = event
        return event

    def execute(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Execute the loaded MLP weights for one source batch into ``destination``."""
        self._queue(queue)
        if self._weight_event is None:
            raise DependencyError("MLP plan weights must be loaded before execution")
        if source.buffer.device is not self._device or destination.buffer.device is not self._device:
            raise DependencyError("MLP plan source and destination must belong to the plan device")
        if source.dtype != np.dtype(np.int32) or destination.dtype != np.dtype(np.int32):
            raise ValueError("MLP plan source and destination must use int32")
        if source.shape != (self.batch, self.in_features) or destination.shape != (self.batch, self.out_features):
            raise ValueError("MLP plan source or destination shape does not match its topology")

        def prepare_source() -> None:
            self._source.numpy().fill(0)
            self._source.numpy()[: self.batch, : self.in_features] = source.numpy()

        source_event = queue.host_task(
            prepare_source,
            wait_for=self._dependencies(wait_for),
            buffers=(source.access(AccessMode.READ), self._source.access(AccessMode.WRITE)),
            name="mlp_int32_plan.prepare_source",
        )
        hidden_event = matmul(self._hidden, self._source, self._weight1, queue=queue, wait_for=(source_event,))

        def bias_relu() -> None:
            active = cast(npt.NDArray[np.int32], self._hidden.numpy()[: self.batch, : self.hidden_features])
            np.add(active, self._bias1.numpy(), out=active)
            np.maximum(active, 0, out=active)
            self._hidden.numpy()[self.batch :, :].fill(0)
            self._hidden.numpy()[:, self.hidden_features :].fill(0)

        activation_event = queue.host_task(
            bias_relu,
            wait_for=(hidden_event,),
            buffers=(self._hidden.access(AccessMode.READ_WRITE), self._bias1.access(AccessMode.READ)),
            name="mlp_int32_plan.bias_relu",
        )
        result_event = matmul(self._result, self._hidden, self._weight2, queue=queue, wait_for=(activation_event,))

        def finish() -> None:
            result = self._result.numpy()[: self.batch, : self.out_features]
            np.add(result, self._bias2.numpy(), out=destination.numpy())

        event = queue.host_task(
            finish,
            wait_for=(result_event,),
            buffers=(
                self._result.access(AccessMode.READ),
                self._bias2.access(AccessMode.READ),
                destination.access(AccessMode.WRITE),
            ),
            name="mlp_int32_plan.finish",
        )
        self._tail = event
        return event

    def close(self: Self) -> None:
        """Invalidate all persistent plan buffers while leaving caller tensors alone."""
        if self._closed:
            return
        for tensor in (
            self._source,
            self._weight1,
            self._bias1,
            self._hidden,
            self._weight2,
            self._bias2,
            self._result,
        ):
            tensor.buffer.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter a context-managed persistent plan lifetime."""
        self._require_open()
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close persistent plan buffers at context exit."""
        self.close()
