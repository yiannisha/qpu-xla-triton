"""Persistent GEMM-backed INT32 NCHW convolution execution plan."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError, DeviceClosedError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.conv2d import _im2col_nchw, _output_hw, _pair, _round_up
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue


class Conv2dInt32Plan:
    """Reuse fixed-shape GEMM buffers and prepared weights across NCHW calls.

    This remains the documented im2col-plus-GEMM convolution implementation;
    it is not a direct spatial-convolution kernel.  The plan persists the
    padded weight matrix and all GEMM workspaces, avoiding repeat allocation
    and weight transformation for stable inference topologies.
    """

    def __init__(
        self: Self,
        device: Device,
        *,
        source_shape: tuple[int, int, int, int],
        weight_shape: tuple[int, int, int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        """Allocate persistent lowering/GEMM storage for one convolution shape."""
        if any(dimension <= 0 for dimension in source_shape + weight_shape):
            raise ValueError("convolution plan dimensions must be positive")
        self._device = device
        self.source_shape = source_shape
        self.weight_shape = weight_shape
        self.stride = _pair(stride, name="stride")
        self.padding = _pair(padding, name="padding")
        self.dilation = _pair(dilation, name="dilation")
        if any(component == 0 for component in self.stride + self.dilation):
            raise ValueError("convolution plan stride and dilation must be positive")
        batch, in_channels, height, width = source_shape
        out_channels, weight_channels, kernel_height, kernel_width = weight_shape
        if in_channels != weight_channels:
            raise ValueError("convolution plan input and weight channels do not align")
        output_height, output_width = _output_hw(
            height, width, kernel_height, kernel_width, self.stride, self.padding, self.dilation
        )
        self.destination_shape = (batch, out_channels, output_height, output_width)
        self._rows = batch * output_height * output_width
        self._reduction = in_channels * kernel_height * kernel_width
        self._padded_rows = _round_up(self._rows, 16)
        self._padded_reduction = _round_up(self._reduction, 4)
        self._padded_columns = _round_up(out_channels, 16)
        self._columns = device.tensor((self._padded_rows, self._padded_reduction), np.int32)
        self._matrix_weight = device.tensor((self._padded_reduction, self._padded_columns), np.int32)
        self._matrix_output = device.tensor((self._padded_rows, self._padded_columns), np.int32)
        self._closed = False
        self._weight_event: Event | None = None
        self._tail: Event | None = None

    @property
    def device(self: Self) -> Device:
        """Return the plan's owning device."""
        return self._device

    @property
    def closed(self: Self) -> bool:
        """Whether the plan's persistent buffers have been invalidated."""
        return self._closed

    def _require_open(self: Self) -> None:
        """Reject use after closing plan-owned tensors."""
        if self._closed:
            raise DeviceClosedError("convolution plan is closed")

    def _queue(self: Self, queue: Queue) -> None:
        """Require one queue for this plan's device."""
        self._require_open()
        if queue.device is not self._device:
            raise DependencyError("convolution plan queue belongs to a different device")

    def _dependencies(self: Self, wait_for: Iterable[Event]) -> tuple[Event, ...]:
        """Serialize plan workspace reuse after any caller dependencies."""
        dependencies = list(wait_for)
        if self._tail is not None and self._tail not in dependencies:
            dependencies.append(self._tail)
        return tuple(dependencies)

    def load_weight(self: Self, weight: Tensor, *, queue: Queue, wait_for: Iterable[Event] = ()) -> Event:
        """Transform and pad OIHW weights once for repeated plan execution."""
        self._queue(queue)
        if weight.buffer.device is not self._device:
            raise DependencyError("convolution plan weight must belong to the plan device")
        if weight.dtype != np.dtype(np.int32) or weight.shape != self.weight_shape:
            raise ValueError("convolution plan weight must be int32 with the configured OIHW shape")
        out_channels = self.weight_shape[0]

        def prepare_weight() -> None:
            self._matrix_weight.numpy().fill(0)
            values = cast(npt.NDArray[np.int32], weight.numpy())
            lowered = values.transpose(1, 2, 3, 0).reshape(self._reduction, out_channels)
            self._matrix_weight.numpy()[: self._reduction, :out_channels] = lowered

        event = queue.host_task(
            prepare_weight,
            wait_for=self._dependencies(wait_for),
            buffers=(weight.access(AccessMode.READ), self._matrix_weight.access(AccessMode.WRITE)),
            name="conv2d_int32_plan.load_weight",
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
        """Lower one source tensor and execute with the retained padded weight matrix."""
        self._queue(queue)
        if self._weight_event is None:
            raise DependencyError("convolution plan weight must be loaded before execution")
        if source.buffer.device is not self._device or destination.buffer.device is not self._device:
            raise DependencyError("convolution plan source and destination must belong to the plan device")
        if source.dtype != np.dtype(np.int32) or destination.dtype != np.dtype(np.int32):
            raise ValueError("convolution plan source and destination must use int32")
        if source.shape != self.source_shape or destination.shape != self.destination_shape:
            raise ValueError("convolution plan source or destination shape does not match its topology")

        def prepare_columns() -> None:
            self._columns.numpy().fill(0)
            values = cast(npt.NDArray[np.int32], source.numpy())
            lowered = _im2col_nchw(
                values,
                self.weight_shape[2],
                self.weight_shape[3],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            self._columns.numpy()[: self._rows, : self._reduction] = lowered

        columns_event = queue.host_task(
            prepare_columns,
            wait_for=self._dependencies(wait_for),
            buffers=(source.access(AccessMode.READ), self._columns.access(AccessMode.WRITE)),
            name="conv2d_int32_plan.prepare_columns",
        )
        gemm_event = matmul(
            self._matrix_output,
            self._columns,
            self._matrix_weight,
            queue=queue,
            wait_for=(columns_event,),
        )
        batch, out_channels, output_height, output_width = self.destination_shape

        def reshape_output() -> None:
            lowered = self._matrix_output.numpy()[: self._rows, :out_channels]
            reshaped = lowered.reshape(batch, output_height, output_width, out_channels)
            destination.numpy()[:] = reshaped.transpose(0, 3, 1, 2)

        event = queue.host_task(
            reshape_output,
            wait_for=(gemm_event,),
            buffers=(self._matrix_output.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="conv2d_int32_plan.reshape_output",
        )
        self._tail = event
        return event

    def close(self: Self) -> None:
        """Invalidate only plan-owned persistent tensors."""
        if self._closed:
            return
        for tensor in (self._columns, self._matrix_weight, self._matrix_output):
            tensor.buffer.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter a context-managed persistent convolution plan lifetime."""
        self._require_open()
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close plan-owned buffers when leaving a context manager."""
        self.close()
