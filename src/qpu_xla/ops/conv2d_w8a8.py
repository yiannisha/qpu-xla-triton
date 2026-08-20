"""Persistent grouped W8A8 convolution plan backed by packed QPU GEMM."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError, DeviceClosedError
from qpu_xla.kernels.depthwise_w8a8 import DEPTHWISE_W8A8_3X3_KERNEL
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.conv2d import _im2col_nchw, _is_pointwise_1x1, _output_hw, _pair, _pointwise_nchw_matrix
from qpu_xla.queue import Event, Queue

if TYPE_CHECKING:
    from qpu_xla.models.tinyllama.quantization import PreparedW8A8Linear


class Conv2dW8A8Plan:
    """Reuse packed weights and workspaces for FP32-in/FP32-out W8A8 convolution.

    Dense and grouped convolutions are lowered into one packed W8A8 GEMM per
    group. Pointwise 1x1 convolutions use the direct NCHW-pixel matrix view and
    avoid spatial-window expansion. General 3x3 convolutions still use
    vectorized im2col and remain candidates for future direct microkernels.
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
        """Quantize fixed OIHW weights and allocate one persistent plan per group."""
        from qpu_xla.models.tinyllama.quantization import PreparedW8A8Linear, quantize_per_output_channel_int8

        if len(source_shape) != 4 or any(dimension <= 0 for dimension in source_shape):
            raise ValueError("W8A8 convolution source shape must be positive NCHW")
        if weight.ndim != 4 or weight.dtype != np.dtype(np.float32) or not weight.flags.c_contiguous:
            raise ValueError("W8A8 convolution weight must be contiguous FP32 OIHW")
        if groups <= 0:
            raise ValueError("W8A8 convolution groups must be positive")
        batch, in_channels, height, width = source_shape
        out_channels, weight_channels, kernel_height, kernel_width = weight.shape
        if in_channels % groups or out_channels % groups or weight_channels != in_channels // groups:
            raise ValueError("W8A8 grouped convolution channels do not align")
        self.device = device
        self.source_shape = source_shape
        self.weight_shape = cast(tuple[int, int, int, int], weight.shape)
        self.stride = _pair(stride, name="stride")
        self.padding = _pair(padding, name="padding")
        self.dilation = _pair(dilation, name="dilation")
        if any(component == 0 for component in self.stride + self.dilation):
            raise ValueError("W8A8 convolution stride and dilation must be positive")
        self.groups = groups
        output_height, output_width = _output_hw(
            height,
            width,
            kernel_height,
            kernel_width,
            self.stride,
            self.padding,
            self.dilation,
        )
        self.destination_shape = (batch, out_channels, output_height, output_width)
        self.rows = batch * output_height * output_width
        self.input_channels_per_group = in_channels // groups
        self.output_channels_per_group = out_channels // groups
        self.reduction_per_group = self.input_channels_per_group * kernel_height * kernel_width
        self._results: list[Tensor] = []
        self._linear_plans: list[PreparedW8A8Linear] = []
        self._depthwise_packed_source: Tensor | None = None
        self._depthwise_packed_weight: Tensor | None = None
        self._depthwise_row_scales: Tensor | None = None
        self._depthwise_column_scales: Tensor | None = None
        self._depthwise_result: Tensor | None = None
        self._depthwise_flat_result: Tensor | None = None
        self._depthwise_source_scratch: npt.NDArray[np.int8] | None = None
        self._direct_depthwise = (
            (kernel_height, kernel_width) == (3, 3)
            and self.stride == (1, 1)
            and self.padding == (1, 1)
            and self.dilation == (1, 1)
            and groups == in_channels == out_channels
            and weight_channels == 1
            and groups >= 16
            and not groups & (groups - 1)
        )
        if self._direct_depthwise:
            quantized_weight = quantize_per_output_channel_int8(
                np.ascontiguousarray(weight.reshape(out_channels, kernel_height * kernel_width))
            )
            elements = self.rows * groups
            self._depthwise_packed_source = device.tensor((elements, 4), np.uint32)
            self._depthwise_packed_weight = device.tensor((groups, 4), np.uint32)
            self._depthwise_row_scales = device.tensor((elements,), np.float32)
            self._depthwise_column_scales = device.tensor((groups,), np.float32)
            self._depthwise_result = device.tensor((self.rows, groups), np.float32)
            self._depthwise_flat_result = self._depthwise_result.buffer.tensor((elements,), np.float32)
            packed_weight_bytes = self._depthwise_packed_weight.numpy().view(np.int8).reshape(groups, 16)
            packed_weight_bytes.fill(0)
            packed_weight_bytes[:, :9] = quantized_weight.values
            self._depthwise_column_scales.numpy()[:] = quantized_weight.scales
            self._depthwise_source_scratch = self._depthwise_packed_source.numpy().view(np.int8).reshape(elements, 16)
        else:
            for group in range(groups):
                output_start = group * self.output_channels_per_group
                output_stop = output_start + self.output_channels_per_group
                matrix_weight = np.ascontiguousarray(
                    weight[output_start:output_stop].reshape(
                        self.output_channels_per_group,
                        self.reduction_per_group,
                    )
                )
                quantized_weight = quantize_per_output_channel_int8(matrix_weight)
                self._results.append(device.tensor((self.rows, self.output_channels_per_group), np.float32))
                self._linear_plans.append(
                    PreparedW8A8Linear(
                        device,
                        quantized_weight,
                        max_batch=self.rows,
                        qpu_dequantize=True,
                    )
                )
        self._tail: Event | None = None
        self._queue: Queue | None = None
        self._cpu_queue: Queue | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter a context-managed plan lifetime."""
        if self._closed:
            raise DeviceClosedError("W8A8 convolution plan is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release plan-owned buffers."""
        self.close()

    def close(self: Self) -> None:
        """Wait for outstanding work and release persistent plans and matrices."""
        if self._closed:
            return
        if self._tail is not None:
            self._tail.wait()
        for plan in self._linear_plans:
            plan.close()
        direct_tensors = tuple(
            tensor
            for tensor in (
                self._depthwise_packed_source,
                self._depthwise_packed_weight,
                self._depthwise_row_scales,
                self._depthwise_column_scales,
                self._depthwise_result,
            )
            if tensor is not None
        )
        for tensor in (*self._results, *direct_tensors):
            tensor.buffer.close()
        self._closed = True

    def execute(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Lower one activation tensor, run packed group GEMMs, and restore NCHW."""
        if self._closed:
            raise DeviceClosedError("W8A8 convolution plan is closed")
        if self._tail is not None and not self._tail.done:
            raise RuntimeError("W8A8 convolution plan already has an in-flight invocation")
        if self._queue is None:
            self._queue = queue
        elif self._queue is not queue:
            raise DependencyError("W8A8 convolution plan is bound to its first submission queue")
        if queue.device is not self.device:
            raise DependencyError("W8A8 convolution queue belongs to a different device")
        if source.buffer.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("W8A8 convolution tensors must belong to the plan device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("W8A8 convolution requires FP32 source and destination")
        if source.shape != self.source_shape or destination.shape != self.destination_shape:
            raise ValueError("W8A8 convolution tensors do not match the planned topology")

        kernel_height, kernel_width = self.weight_shape[2:]

        def prepare_columns() -> None:
            source_values = cast(npt.NDArray[np.float32], source.numpy())
            if _is_pointwise_1x1(
                kernel_height,
                kernel_width,
                self.stride,
                self.padding,
                self.dilation,
            ):
                lowered = _pointwise_nchw_matrix(source_values).reshape(self.rows, -1, 1, 1)
            else:
                lowered = _im2col_nchw(
                    source_values,
                    kernel_height,
                    kernel_width,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                ).reshape(self.rows, -1, kernel_height, kernel_width)
            if self._direct_depthwise:
                assert self._depthwise_source_scratch is not None
                assert self._depthwise_row_scales is not None
                values = np.ascontiguousarray(lowered.reshape(self.rows * self.groups, 9))
                scales = cast(npt.NDArray[np.float32], self._depthwise_row_scales.numpy())
                np.maximum(
                    np.max(np.abs(values), axis=1) / np.float32(127.0),
                    np.float32(1.0 / 127.0),
                    out=scales,
                )
                self._depthwise_source_scratch.fill(0)
                self._depthwise_source_scratch[:, :9] = np.rint(values / scales[:, None]).clip(-127, 127)
                return
            for group, plan in enumerate(self._linear_plans):
                channel_start = group * self.input_channels_per_group
                channel_stop = channel_start + self.input_channels_per_group
                plan._prepare_values(
                    np.ascontiguousarray(
                        lowered[:, channel_start:channel_stop].reshape(
                            self.rows,
                            self.reduction_per_group,
                        )
                    )
                )

        if self._direct_depthwise:
            assert self._depthwise_packed_source is not None
            assert self._depthwise_row_scales is not None
            prepare_writes = (
                self._depthwise_packed_source.access(AccessMode.WRITE),
                self._depthwise_row_scales.access(AccessMode.WRITE),
            )
        else:
            prepare_writes = (
                *(plan._packed_source.access(AccessMode.WRITE) for plan in self._linear_plans),
                *(plan._row_scales.access(AccessMode.WRITE) for plan in self._linear_plans),
            )
        prepare_event = queue.host_task(
            prepare_columns,
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                *prepare_writes,
            ),
            name="conv2d_w8a8.prepare_columns",
        )
        if self._direct_depthwise:
            assert self._depthwise_packed_source is not None
            assert self._depthwise_packed_weight is not None
            assert self._depthwise_row_scales is not None
            assert self._depthwise_column_scales is not None
            assert self._depthwise_flat_result is not None
            direct_event = queue.submit(
                DEPTHWISE_W8A8_3X3_KERNEL,
                (
                    self._depthwise_packed_source,
                    self._depthwise_packed_weight,
                    self._depthwise_row_scales,
                    self._depthwise_column_scales,
                    self._depthwise_flat_result,
                ),
                grid=(self.rows * self.groups // 16, 1, 1),
                wait_for=(prepare_event,),
                buffers=(
                    self._depthwise_packed_source.access(AccessMode.READ),
                    self._depthwise_packed_weight.access(AccessMode.READ),
                    self._depthwise_row_scales.access(AccessMode.READ),
                    self._depthwise_column_scales.access(AccessMode.READ),
                    self._depthwise_flat_result.access(AccessMode.WRITE),
                ),
            )
            group_events = (direct_event,)
        else:
            group_events = tuple(
                plan.execute_prepared(result, batch=self.rows, queue=queue, wait_for=(prepare_event,))
                for plan, result in zip(self._linear_plans, self._results, strict=True)
            )
        batch, out_channels, output_height, output_width = self.destination_shape

        def restore_nchw() -> None:
            if self._direct_depthwise:
                assert self._depthwise_result is not None
                matrix = self._depthwise_result.numpy()
            else:
                matrix = np.concatenate([result.numpy() for result in self._results], axis=1)
            destination.numpy()[:] = matrix.reshape(
                batch,
                output_height,
                output_width,
                out_channels,
            ).transpose(0, 3, 1, 2)

        event = queue.host_task(
            restore_nchw,
            wait_for=group_events,
            buffers=(
                *(
                    (self._depthwise_result.access(AccessMode.READ),)
                    if self._depthwise_result is not None
                    else tuple(result.access(AccessMode.READ) for result in self._results)
                ),
                destination.access(AccessMode.WRITE),
            ),
            name="conv2d_w8a8.restore_nchw",
        )
        self._tail = event
        return event

    @property
    def output_split_alignment(self: Self) -> int | None:
        """Return the total-channel alignment for output hybrids, if supported."""
        if self._direct_depthwise or self.output_channels_per_group <= 16:
            return None
        return self.groups * 16

    def execute_hybrid(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        qpu_queue: Queue,
        cpu_queue: Queue,
        qpu_rows: int | None = None,
        qpu_outputs: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Lower once and split disjoint rows or per-group outputs across CPU/QPU."""
        if self._closed:
            raise DeviceClosedError("W8A8 convolution plan is closed")
        if self._tail is not None and not self._tail.done:
            raise RuntimeError("W8A8 convolution plan already has an in-flight invocation")
        if qpu_rows is not None and qpu_outputs is not None:
            raise ValueError("qpu_rows and qpu_outputs are mutually exclusive")
        if qpu_rows is None and qpu_outputs is None:
            raise ValueError("hybrid W8A8 convolution requires a row or output partition")
        if qpu_queue is cpu_queue:
            raise DependencyError("hybrid W8A8 convolution requires distinct CPU and QPU queues")
        if qpu_queue.device is not self.device or cpu_queue.device is not self.device:
            raise DependencyError("hybrid W8A8 convolution queues belong to a different device")
        if self._queue is None:
            self._queue = qpu_queue
        elif self._queue is not qpu_queue:
            raise DependencyError("W8A8 convolution plan is bound to its first QPU queue")
        if self._cpu_queue is None:
            self._cpu_queue = cpu_queue
        elif self._cpu_queue is not cpu_queue:
            raise DependencyError("W8A8 convolution plan is bound to its first CPU queue")
        if source.buffer.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("W8A8 convolution tensors must belong to the plan device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("W8A8 convolution requires FP32 source and destination")
        if source.shape != self.source_shape or destination.shape != self.destination_shape:
            raise ValueError("W8A8 convolution tensors do not match the planned topology")
        if qpu_rows is not None and (qpu_rows <= 0 or qpu_rows >= self.rows or qpu_rows % 16):
            raise ValueError("hybrid convolution qpu_rows must be aligned and leave a non-empty CPU tail")
        if qpu_outputs is not None:
            alignment = self.output_split_alignment
            if alignment is None:
                raise ValueError("hybrid convolution output splitting is unsupported for this group topology")
            if qpu_outputs <= 0 or qpu_outputs >= self.weight_shape[0] or qpu_outputs % alignment:
                raise ValueError("hybrid convolution qpu_outputs violates per-group output alignment")

        kernel_height, kernel_width = self.weight_shape[2:]

        def prepare_columns() -> None:
            source_values = cast(npt.NDArray[np.float32], source.numpy())
            if _is_pointwise_1x1(
                kernel_height,
                kernel_width,
                self.stride,
                self.padding,
                self.dilation,
            ):
                lowered = _pointwise_nchw_matrix(source_values).reshape(self.rows, -1, 1, 1)
            else:
                lowered = _im2col_nchw(
                    source_values,
                    kernel_height,
                    kernel_width,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                ).reshape(self.rows, -1, kernel_height, kernel_width)
            if self._direct_depthwise:
                assert self._depthwise_source_scratch is not None
                assert self._depthwise_row_scales is not None
                values = np.ascontiguousarray(lowered.reshape(self.rows * self.groups, 9))
                scales = cast(npt.NDArray[np.float32], self._depthwise_row_scales.numpy())
                np.maximum(
                    np.max(np.abs(values), axis=1) / np.float32(127.0),
                    np.float32(1.0 / 127.0),
                    out=scales,
                )
                self._depthwise_source_scratch.fill(0)
                self._depthwise_source_scratch[:, :9] = np.rint(values / scales[:, None]).clip(-127, 127)
                return
            for group, plan in enumerate(self._linear_plans):
                channel_start = group * self.input_channels_per_group
                channel_stop = channel_start + self.input_channels_per_group
                plan._prepare_values(
                    np.ascontiguousarray(
                        lowered[:, channel_start:channel_stop].reshape(
                            self.rows,
                            self.reduction_per_group,
                        )
                    )
                )

        if self._direct_depthwise:
            assert self._depthwise_packed_source is not None
            assert self._depthwise_row_scales is not None
            prepare_writes = (
                self._depthwise_packed_source.access(AccessMode.WRITE),
                self._depthwise_row_scales.access(AccessMode.WRITE),
            )
        else:
            prepare_writes = (
                *(plan._packed_source.access(AccessMode.WRITE) for plan in self._linear_plans),
                *(plan._row_scales.access(AccessMode.WRITE) for plan in self._linear_plans),
            )
        prepare_event = cpu_queue.host_task(
            prepare_columns,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), *prepare_writes),
            name="conv2d_w8a8.prepare_columns",
        )

        if self._direct_depthwise:
            assert qpu_rows is not None
            assert self._depthwise_packed_source is not None
            assert self._depthwise_packed_weight is not None
            assert self._depthwise_row_scales is not None
            assert self._depthwise_column_scales is not None
            assert self._depthwise_flat_result is not None
            assert self._depthwise_result is not None
            assert self._depthwise_source_scratch is not None
            qpu_elements = qpu_rows * self.groups
            qpu_source = self._depthwise_packed_source.slice((slice(0, qpu_elements), slice(None)))
            qpu_scales = self._depthwise_row_scales.slice((slice(0, qpu_elements),))
            qpu_result = self._depthwise_flat_result.slice((slice(0, qpu_elements),))
            qpu_event = qpu_queue.submit(
                DEPTHWISE_W8A8_3X3_KERNEL,
                (
                    qpu_source,
                    self._depthwise_packed_weight,
                    qpu_scales,
                    self._depthwise_column_scales,
                    qpu_result,
                ),
                grid=(qpu_elements // 16, 1, 1),
                wait_for=(prepare_event,),
                buffers=(
                    qpu_source.access(AccessMode.READ),
                    self._depthwise_packed_weight.access(AccessMode.READ),
                    qpu_scales.access(AccessMode.READ),
                    self._depthwise_column_scales.access(AccessMode.READ),
                    qpu_result.access(AccessMode.WRITE),
                ),
            )

            def depthwise_cpu_tail() -> None:
                packed_weight = self._depthwise_packed_weight.numpy().view(np.int8).reshape(self.groups, 16)
                quantized_source = self._depthwise_source_scratch[qpu_elements : self.rows * self.groups, :9].reshape(
                    self.rows - qpu_rows, self.groups, 9
                )
                accumulation = np.sum(
                    quantized_source.astype(np.int32) * packed_weight[None, :, :9].astype(np.int32),
                    axis=2,
                    dtype=np.int32,
                )
                row_scales = self._depthwise_row_scales.numpy()[qpu_elements:].reshape(
                    self.rows - qpu_rows,
                    self.groups,
                )
                self._depthwise_result.numpy()[qpu_rows:] = (
                    accumulation.astype(np.float32) * row_scales * self._depthwise_column_scales.numpy()[None, :]
                )

            cpu_event = cpu_queue.host_task(
                depthwise_cpu_tail,
                wait_for=(prepare_event,),
                buffers=(
                    self._depthwise_result.slice((slice(qpu_rows, self.rows), slice(None))).access(AccessMode.WRITE),
                ),
                name="conv2d_w8a8.depthwise_cpu_tail",
            )
            group_events = (qpu_event, cpu_event)
        elif qpu_rows is not None:
            group_events = tuple(
                plan.execute_prepared_hybrid_rows(
                    result,
                    batch=self.rows,
                    qpu_queue=qpu_queue,
                    cpu_queue=cpu_queue,
                    qpu_rows=qpu_rows,
                    wait_for=(prepare_event,),
                )
                for plan, result in zip(self._linear_plans, self._results, strict=True)
            )
        else:
            assert qpu_outputs is not None
            outputs_per_group = qpu_outputs // self.groups
            group_events = tuple(
                plan.execute_prepared_hybrid_outputs(
                    result,
                    batch=self.rows,
                    qpu_queue=qpu_queue,
                    cpu_queue=cpu_queue,
                    qpu_outputs=outputs_per_group,
                    wait_for=(prepare_event,),
                )
                for plan, result in zip(self._linear_plans, self._results, strict=True)
            )

        batch, out_channels, output_height, output_width = self.destination_shape

        def restore_nchw() -> None:
            if self._direct_depthwise:
                assert self._depthwise_result is not None
                matrix = self._depthwise_result.numpy()
            else:
                matrix = np.concatenate([result.numpy() for result in self._results], axis=1)
            destination.numpy()[:] = matrix.reshape(
                batch,
                output_height,
                output_width,
                out_channels,
            ).transpose(0, 3, 1, 2)

        event = cpu_queue.host_task(
            restore_nchw,
            wait_for=group_events,
            buffers=(
                *(
                    (self._depthwise_result.access(AccessMode.READ),)
                    if self._depthwise_result is not None
                    else tuple(result.access(AccessMode.READ) for result in self._results)
                ),
                destination.access(AccessMode.WRITE),
            ),
            name="conv2d_w8a8.restore_nchw",
        )
        self._tail = event
        return event


__all__ = ["Conv2dW8A8Plan"]
