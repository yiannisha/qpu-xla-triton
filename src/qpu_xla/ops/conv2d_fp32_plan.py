"""Persistent grouped FP32 convolution plan backed by prepared QPU GEMM."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError, DeviceClosedError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.conv2d import _im2col_nchw, _is_pointwise_1x1, _output_hw, _pair, _pointwise_nchw_matrix
from qpu_xla.ops.linear_fp32 import PreparedFP32Linear
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


class Conv2dFP32Plan:
    """Reuse lowered activation storage and immutable OIHW weights.

    Pointwise 1x1 calls avoid window expansion. Spatial kernels use the
    vectorized lowering shared with the established convolution operator. The
    expensive padded GEMM weight transformation and device buffers persist for
    the plan lifetime.
    """

    def __init__(
        self: Self,
        device: Device,
        *,
        source_shape: tuple[int, int, int, int],
        weight: npt.NDArray[np.float32],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> None:
        """Prepare one fixed convolution topology and its per-group weights."""
        if len(source_shape) != 4 or any(value <= 0 for value in source_shape):
            raise ValueError("FP32 convolution source shape must be positive NCHW")
        if weight.ndim != 4 or weight.dtype != np.dtype(np.float32) or not weight.flags.c_contiguous:
            raise ValueError("FP32 convolution weight must be contiguous FP32 OIHW")
        if groups <= 0:
            raise ValueError("FP32 convolution groups must be positive")
        batch, input_channels, height, width = source_shape
        output_channels, weight_channels, kernel_height, kernel_width = weight.shape
        if input_channels % groups or output_channels % groups or weight_channels != input_channels // groups:
            raise ValueError("FP32 grouped convolution channels do not align")
        self.device = device
        self.source_shape = source_shape
        self.weight_shape = cast(tuple[int, int, int, int], weight.shape)
        self.stride = _pair(stride, name="stride")
        self.padding = _pair(padding, name="padding")
        self.dilation = _pair(dilation, name="dilation")
        if any(value == 0 for value in self.stride + self.dilation):
            raise ValueError("FP32 convolution stride and dilation must be positive")
        self.groups = groups
        output_height, output_width = _output_hw(
            height, width, kernel_height, kernel_width, self.stride, self.padding, self.dilation
        )
        self.destination_shape = (batch, output_channels, output_height, output_width)
        self.rows = batch * output_height * output_width
        self.input_channels_per_group = input_channels // groups
        self.output_channels_per_group = output_channels // groups
        self.reduction_per_group = self.input_channels_per_group * kernel_height * kernel_width
        self._columns = [device.tensor((self.rows, self.reduction_per_group), np.float32) for _ in range(groups)]
        self._results = [device.tensor((self.rows, self.output_channels_per_group), np.float32) for _ in range(groups)]
        self._linear = []
        for group in range(groups):
            start = group * self.output_channels_per_group
            stop = start + self.output_channels_per_group
            matrix = np.ascontiguousarray(weight[start:stop].reshape(self.output_channels_per_group, -1))
            self._linear.append(PreparedFP32Linear(device, matrix, max_batch=self.rows))
        self._qpu_queue: Queue | None = None
        self._cpu_queue: Queue | None = None
        self._tail: Event | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter the plan lifetime."""
        if self._closed:
            raise DeviceClosedError("FP32 convolution plan is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close persistent allocations."""
        self.close()

    def close(self: Self) -> None:
        """Wait for outstanding work and release all plan-owned buffers."""
        if self._closed:
            return
        if self._tail is not None:
            self._tail.wait()
        for plan in self._linear:
            plan.close()
        for tensor in (*self._columns, *self._results):
            tensor.buffer.close()
        self._closed = True

    def execute(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        queue: Queue,
        placement: Placement = Placement.QPU,
        cpu_queue: Queue | None = None,
        qpu_rows: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Execute a CPU, QPU, or row-partitioned hybrid convolution."""
        if self._closed:
            raise DeviceClosedError("FP32 convolution plan is closed")
        if self._tail is not None and not self._tail.done:
            raise RuntimeError("FP32 convolution plan already has an in-flight invocation")
        if (
            queue.device is not self.device
            or source.buffer.device is not self.device
            or destination.buffer.device is not self.device
        ):
            raise DependencyError("FP32 convolution queue and tensors must belong to the plan device")
        if source.shape != self.source_shape or destination.shape != self.destination_shape:
            raise ValueError("FP32 convolution tensors do not match the planned topology")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("FP32 convolution tensors must use float32")
        if placement is Placement.HYBRID:
            if cpu_queue is None or cpu_queue is queue or cpu_queue.device is not self.device:
                raise DependencyError("hybrid FP32 convolution requires a distinct CPU queue")
            if qpu_rows is None:
                qpu_rows = max(16, self.rows // 2 // 16 * 16)
            if qpu_rows <= 0 or qpu_rows >= self.rows or qpu_rows % 16:
                raise ValueError("hybrid FP32 convolution qpu_rows must be aligned and leave a CPU tail")
        elif qpu_rows is not None:
            raise ValueError("qpu_rows is only valid for hybrid FP32 convolution")
        if self._qpu_queue is None:
            self._qpu_queue = queue
        elif self._qpu_queue is not queue:
            raise DependencyError("FP32 convolution plan is bound to its first queue")
        if cpu_queue is not None:
            if self._cpu_queue is None:
                self._cpu_queue = cpu_queue
            elif self._cpu_queue is not cpu_queue:
                raise DependencyError("FP32 convolution plan is bound to its first CPU queue")

        kernel_height, kernel_width = self.weight_shape[2:]

        def lower() -> None:
            values = cast(npt.NDArray[np.float32], source.numpy())
            if _is_pointwise_1x1(kernel_height, kernel_width, self.stride, self.padding, self.dilation):
                lowered = _pointwise_nchw_matrix(values).reshape(self.rows, -1, 1, 1)
            else:
                lowered = _im2col_nchw(
                    values,
                    kernel_height,
                    kernel_width,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                ).reshape(self.rows, -1, kernel_height, kernel_width)
            for group, columns in enumerate(self._columns):
                start = group * self.input_channels_per_group
                stop = start + self.input_channels_per_group
                columns.numpy()[:] = lowered[:, start:stop].reshape(self.rows, self.reduction_per_group)

        host_queue = cpu_queue if placement is Placement.HYBRID else queue
        assert host_queue is not None
        lower_event = host_queue.host_task(
            lower,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), *(item.access(AccessMode.WRITE) for item in self._columns)),
            name="conv2d_fp32_plan.lower",
        )
        events = tuple(
            plan.execute(
                result,
                columns,
                queue=queue,
                placement=placement,
                cpu_queue=cpu_queue,
                qpu_units=qpu_rows,
                wait_for=(lower_event,),
            )
            for plan, columns, result in zip(self._linear, self._columns, self._results, strict=True)
        )
        batch, outputs, output_height, output_width = self.destination_shape

        def restore() -> None:
            matrix = np.concatenate([item.numpy() for item in self._results], axis=1)
            destination.numpy()[:] = matrix.reshape(batch, output_height, output_width, outputs).transpose(0, 3, 1, 2)

        event = host_queue.host_task(
            restore,
            wait_for=events,
            buffers=(*(item.access(AccessMode.READ) for item in self._results), destination.access(AccessMode.WRITE)),
            name="conv2d_fp32_plan.restore",
        )
        self._tail = event
        return event


__all__ = ["Conv2dFP32Plan"]
