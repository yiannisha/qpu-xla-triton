"""GEMM-backed NCHW INT32 convolution built from the packaged matmul substrate."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue


def _pair(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    """Normalize one positive scalar/pair convolution parameter."""
    pair = (value, value) if isinstance(value, int) else value
    if len(pair) != 2 or any(component < 0 for component in pair):
        raise ValueError(f"{name} must contain two non-negative integers")
    return pair


def _output_hw(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Return the valid NCHW convolution output spatial dimensions."""
    output_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("convolution parameters produce an empty output")
    return output_height, output_width


def _round_up(value: int, tile: int) -> int:
    """Round a positive dimension up to one microkernel tile."""
    return (value + tile - 1) // tile * tile


def _im2col_nchw(
    source: npt.NDArray[np.generic],
    kernel_height: int,
    kernel_width: int,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> npt.NDArray[np.generic]:
    """Lower an NCHW tensor to the row-major matrix consumed by tiled GEMM."""
    batch, channels, height, width = source.shape
    output_height, output_width = _output_hw(height, width, kernel_height, kernel_width, stride, padding, dilation)
    padded = np.pad(source, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    columns = np.empty(
        (batch * output_height * output_width, channels * kernel_height * kernel_width), dtype=source.dtype
    )
    row = 0
    for batch_index in range(batch):
        for output_y in range(output_height):
            input_y = output_y * stride[0]
            for output_x in range(output_width):
                input_x = output_x * stride[1]
                column = 0
                for channel in range(channels):
                    for kernel_y in range(kernel_height):
                        for kernel_x in range(kernel_width):
                            columns[row, column] = padded[
                                batch_index,
                                channel,
                                input_y + kernel_y * dilation[0],
                                input_x + kernel_x * dilation[1],
                            ]
                            column += 1
                row += 1
    return columns


def _is_pointwise_1x1(
    kernel_height: int,
    kernel_width: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> bool:
    """Return whether NCHW convolution can avoid general im2col expansion."""
    return (kernel_height, kernel_width) == (1, 1) and stride == (1, 1) and padding == (0, 0) and dilation == (1, 1)


def _pointwise_nchw_matrix(source: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Pack NCHW 1x1 inputs to GEMM rows without spatial-window duplication."""
    batch, channels, height, width = source.shape
    return source.transpose(0, 2, 3, 1).reshape(batch * height * width, channels)


def conv2d_int32(
    destination: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute an exact INT32 NCHW convolution using padded tiled GEMM.

    Inputs have shape ``(N, C_in, H, W)`` and OIHW weights have shape
    ``(C_out, C_in, K_h, K_w)``. The result is INT32 and is written into the
    supplied NCHW destination tensor. This is deliberately a GEMM-backed path,
    not a direct-convolution QPU kernel.
    """
    if any(tensor.buffer.device is not destination.buffer.device for tensor in (source, weight, destination)):
        raise DependencyError("convolution tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.int32) for tensor in (source, weight, destination)):
        raise ValueError("conv2d_int32 requires int32 source, weight, and destination tensors")
    if len(source.shape) != 4 or len(weight.shape) != 4 or len(destination.shape) != 4:
        raise ValueError("conv2d_int32 requires NCHW source/destination and OIHW weight tensors")

    stride_pair = _pair(stride, name="stride")
    padding_pair = _pair(padding, name="padding")
    dilation_pair = _pair(dilation, name="dilation")
    if any(component == 0 for component in stride_pair + dilation_pair):
        raise ValueError("stride and dilation must be positive")
    batch, in_channels, height, width = source.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if in_channels != weight_channels:
        raise ValueError("input and weight channel dimensions do not align")
    output_height, output_width = _output_hw(
        height, width, kernel_height, kernel_width, stride_pair, padding_pair, dilation_pair
    )
    if destination.shape != (batch, out_channels, output_height, output_width):
        raise ValueError("destination shape does not match the requested convolution")

    rows = batch * output_height * output_width
    reduction = in_channels * kernel_height * kernel_width
    padded_rows = _round_up(rows, 16)
    padded_reduction = _round_up(reduction, 4)
    padded_columns = _round_up(out_channels, 16)
    device = destination.buffer.device
    columns = device.tensor((padded_rows, padded_reduction), np.int32)
    matrix_weight = device.tensor((padded_reduction, padded_columns), np.int32)
    matrix_output = device.tensor((padded_rows, padded_columns), np.int32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def prepare() -> None:
        columns.numpy().fill(0)
        matrix_weight.numpy().fill(0)
        source_values = cast(npt.NDArray[np.int32], source.numpy())
        weight_values = cast(npt.NDArray[np.int32], weight.numpy())
        if _is_pointwise_1x1(kernel_height, kernel_width, stride_pair, padding_pair, dilation_pair):
            lowered = _pointwise_nchw_matrix(source_values)
            lowered_weight = weight_values[:, :, 0, 0].T
        else:
            lowered = _im2col_nchw(
                source_values,
                kernel_height,
                kernel_width,
                stride=stride_pair,
                padding=padding_pair,
                dilation=dilation_pair,
            )
            lowered_weight = weight_values.transpose(1, 2, 3, 0).reshape(reduction, out_channels)
        columns.numpy()[:rows, :reduction] = lowered
        matrix_weight.numpy()[:reduction, :out_channels] = lowered_weight

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            weight.access(AccessMode.READ),
            columns.access(AccessMode.WRITE),
            matrix_weight.access(AccessMode.WRITE),
        ),
    )
    gemm_event = matmul(
        matrix_output,
        columns,
        matrix_weight,
        queue=selected_queue,
        wait_for=(prepare_event,),
    )

    def reshape_output() -> None:
        lowered = matrix_output.numpy()[:rows, :out_channels]
        reshaped = lowered.reshape(batch, output_height, output_width, out_channels).transpose(0, 3, 1, 2)
        destination.numpy()[:] = reshaped

    result_event = selected_queue.host_task(
        reshape_output,
        wait_for=(gemm_event,),
        buffers=(matrix_output.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
    )
    if close_queue:
        result_event.wait()
        selected_queue.close()
    return result_event


def conv2d_fp32(
    destination: Tensor,
    source: Tensor,
    weight: Tensor,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Compute FP32 NCHW convolution through the packaged tiled GEMM substrate.

    The contract mirrors :func:`conv2d_int32` but uses FP32 arithmetic and
    tolerance-based validation. It remains an im2col lowering, not a native
    direct-convolution QPU kernel.
    """
    if any(tensor.buffer.device is not destination.buffer.device for tensor in (source, weight, destination)):
        raise DependencyError("convolution tensors must belong to the same device")
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (source, weight, destination)):
        raise ValueError("conv2d_fp32 requires float32 source, weight, and destination tensors")
    if len(source.shape) != 4 or len(weight.shape) != 4 or len(destination.shape) != 4:
        raise ValueError("conv2d_fp32 requires NCHW source/destination and OIHW weight tensors")

    stride_pair = _pair(stride, name="stride")
    padding_pair = _pair(padding, name="padding")
    dilation_pair = _pair(dilation, name="dilation")
    if any(component == 0 for component in stride_pair + dilation_pair):
        raise ValueError("stride and dilation must be positive")
    batch, in_channels, height, width = source.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if in_channels != weight_channels:
        raise ValueError("input and weight channel dimensions do not align")
    output_height, output_width = _output_hw(
        height, width, kernel_height, kernel_width, stride_pair, padding_pair, dilation_pair
    )
    if destination.shape != (batch, out_channels, output_height, output_width):
        raise ValueError("destination shape does not match the requested convolution")

    rows = batch * output_height * output_width
    reduction = in_channels * kernel_height * kernel_width
    padded_rows = _round_up(rows, 16)
    padded_reduction = _round_up(reduction, 4)
    padded_columns = _round_up(out_channels, 16)
    device = destination.buffer.device
    columns = device.tensor((padded_rows, padded_reduction), np.float32)
    matrix_weight = device.tensor((padded_reduction, padded_columns), np.float32)
    matrix_output = device.tensor((padded_rows, padded_columns), np.float32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and destination tensor must belong to the same device")

    def prepare() -> None:
        columns.numpy().fill(0.0)
        matrix_weight.numpy().fill(0.0)
        source_values = cast(npt.NDArray[np.float32], source.numpy())
        weight_values = cast(npt.NDArray[np.float32], weight.numpy())
        if _is_pointwise_1x1(kernel_height, kernel_width, stride_pair, padding_pair, dilation_pair):
            lowered = _pointwise_nchw_matrix(source_values)
            lowered_weight = weight_values[:, :, 0, 0].T
        else:
            lowered = _im2col_nchw(
                source_values,
                kernel_height,
                kernel_width,
                stride=stride_pair,
                padding=padding_pair,
                dilation=dilation_pair,
            )
            lowered_weight = weight_values.transpose(1, 2, 3, 0).reshape(reduction, out_channels)
        columns.numpy()[:rows, :reduction] = lowered
        matrix_weight.numpy()[:reduction, :out_channels] = lowered_weight

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            weight.access(AccessMode.READ),
            columns.access(AccessMode.WRITE),
            matrix_weight.access(AccessMode.WRITE),
        ),
        name="conv2d_fp32.prepare",
    )
    gemm_event = matmul(matrix_output, columns, matrix_weight, queue=selected_queue, wait_for=(prepare_event,))

    def reshape_output() -> None:
        lowered = matrix_output.numpy()[:rows, :out_channels]
        reshaped = lowered.reshape(batch, output_height, output_width, out_channels).transpose(0, 3, 1, 2)
        destination.numpy()[:] = reshaped

    result_event = selected_queue.host_task(
        reshape_output,
        wait_for=(gemm_event,),
        buffers=(matrix_output.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="conv2d_fp32.reshape_output",
    )
    if close_queue:
        result_event.wait()
        selected_queue.close()
    return result_event
