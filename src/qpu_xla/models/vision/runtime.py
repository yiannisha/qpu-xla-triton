"""Torch graph adapters that execute learned convolutions on VideoCore VII."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_functional

from qpu_xla.device import Device
from qpu_xla.memory import Tensor
from qpu_xla.ops import Conv2dFP32Plan, Conv2dW8A8Plan
from qpu_xla.ops.conv2d import _im2col_nchw, _is_pointwise_1x1, _output_hw, _pointwise_nchw_matrix
from qpu_xla.scheduler import Placement


class VisionMode(Enum):
    """Explicit numerical and CPU/QPU placement combinations."""

    CPU_FP32 = "cpu_fp32"
    QPU_FP32 = "qpu_fp32"
    HYBRID_FP32 = "hybrid_fp32"
    CPU_W8A8 = "cpu_w8a8"
    QPU_W8A8 = "qpu_w8a8"
    HYBRID_W8A8 = "hybrid_w8a8"

    @property
    def uses_qpu(self: Self) -> bool:
        """Return whether this mode must dispatch QPU work."""
        return self in {
            VisionMode.QPU_FP32,
            VisionMode.HYBRID_FP32,
            VisionMode.QPU_W8A8,
            VisionMode.HYBRID_W8A8,
        }

    @property
    def hybrid(self: Self) -> bool:
        """Return whether work is deliberately split across CPU and QPU."""
        return self in {VisionMode.HYBRID_FP32, VisionMode.HYBRID_W8A8}

    @property
    def w8a8(self: Self) -> bool:
        """Return whether convolution operands use dynamic signed W8A8."""
        return self in {VisionMode.CPU_W8A8, VisionMode.QPU_W8A8, VisionMode.HYBRID_W8A8}


@dataclass(slots=True)
class VisionTelemetry:
    """Auditable placement coverage for a full model invocation."""

    eligible_layers: int = 0
    cpu_calls: int = 0
    qpu_dispatches: int = 0
    hybrid_calls: int = 0
    expected_fallbacks: int = 0
    unexpected_fallbacks: int = 0
    qpu_rows: int = 0
    cpu_rows: int = 0

    def to_dict(self: Self) -> dict[str, int]:
        """Return a JSON-compatible snapshot."""
        return asdict(self)


def _dynamic_w8a8_conv(
    source: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> npt.NDArray[np.float32]:
    """Calculate the exact dynamic-row/per-output-channel W8A8 contract."""
    from qpu_xla.models.tinyllama.quantization import quantize_per_output_channel_int8

    batch, channels, height, width = source.shape
    outputs, _, kernel_height, kernel_width = weight.shape
    output_height, output_width = _output_hw(height, width, kernel_height, kernel_width, stride, padding, dilation)
    rows = batch * output_height * output_width
    if _is_pointwise_1x1(kernel_height, kernel_width, stride, padding, dilation):
        lowered = _pointwise_nchw_matrix(source).reshape(rows, channels, 1, 1)
    else:
        lowered = _im2col_nchw(
            source,
            kernel_height,
            kernel_width,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ).reshape(rows, channels, kernel_height, kernel_width)
    input_group = channels // groups
    output_group = outputs // groups
    results: list[npt.NDArray[np.float32]] = []
    for group in range(groups):
        input_start = group * input_group
        output_start = group * output_group
        source_matrix = np.ascontiguousarray(lowered[:, input_start : input_start + input_group].reshape(rows, -1))
        weight_matrix = np.ascontiguousarray(
            weight[output_start : output_start + output_group].reshape(output_group, -1)
        )
        quantized_weight = quantize_per_output_channel_int8(weight_matrix)
        row_scales = np.maximum(np.max(np.abs(source_matrix), axis=1) / np.float32(127.0), np.float32(1.0 / 127.0))
        quantized_source = np.rint(source_matrix / row_scales[:, None]).clip(-127, 127).astype(np.int32)
        accumulation = quantized_source @ quantized_weight.values.astype(np.int32).T
        results.append(accumulation.astype(np.float32) * row_scales[:, None] * quantized_weight.scales[None, :])
    matrix = np.concatenate(results, axis=1)
    return np.ascontiguousarray(matrix.reshape(batch, output_height, output_width, outputs).transpose(0, 3, 1, 2))


class _PreparedVisionConv:
    """One lazily shaped convolution plan used by a Torch module adapter."""

    def __init__(
        self: Self,
        context: VisionExecutionContext,
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32] | None,
        *,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ) -> None:
        self.context = context
        self.weight = np.ascontiguousarray(weight)
        self.bias = None if bias is None else np.ascontiguousarray(bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.source_shape: tuple[int, int, int, int] | None = None
        self.source: Tensor | None = None
        self.destination: Tensor | None = None
        self.plan: Conv2dFP32Plan | Conv2dW8A8Plan | None = None

    def _prepare(self: Self, shape: tuple[int, int, int, int]) -> None:
        if self.source_shape is not None:
            if shape != self.source_shape:
                raise ValueError(f"vision convolution was prepared for {self.source_shape}, received {shape}")
            return
        device = self.context.device
        if device is None:
            self.source_shape = shape
            return
        self.source_shape = shape
        plan_type = Conv2dW8A8Plan if self.context.mode.w8a8 else Conv2dFP32Plan
        self.plan = plan_type(
            device,
            source_shape=shape,
            weight=self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        self.source = device.tensor(shape, np.float32)
        self.destination = device.tensor(self.plan.destination_shape, np.float32)

    def close(self: Self) -> None:
        """Release this layer's prepared QPU buffers."""
        if self.plan is not None:
            self.plan.close()
        for tensor in (self.source, self.destination):
            if tensor is not None:
                tensor.buffer.close()

    def execute(self: Self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run one fixed-shape convolution and update placement telemetry."""
        shape = cast(tuple[int, int, int, int], values.shape)
        self._prepare(shape)
        mode = self.context.mode
        telemetry = self.context.telemetry
        if mode is VisionMode.CPU_FP32:
            result = torch_functional.conv2d(
                torch.from_numpy(values),
                torch.from_numpy(self.weight),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            ).numpy()
            telemetry.cpu_calls += 1
        elif mode is VisionMode.CPU_W8A8:
            result = _dynamic_w8a8_conv(
                values,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            telemetry.cpu_calls += 1
        else:
            assert self.source is not None and self.destination is not None and self.plan is not None
            assert self.context.qpu_queue is not None
            self.source.numpy()[:] = values
            if mode is VisionMode.QPU_FP32:
                assert isinstance(self.plan, Conv2dFP32Plan)
                event = self.plan.execute(
                    self.destination, self.source, queue=self.context.qpu_queue, placement=Placement.QPU
                )
                telemetry.qpu_dispatches += self.groups
                telemetry.qpu_rows += self.plan.rows
            elif mode is VisionMode.HYBRID_FP32:
                assert isinstance(self.plan, Conv2dFP32Plan) and self.context.cpu_queue is not None
                qpu_rows = max(16, self.plan.rows // 2 // 16 * 16)
                if qpu_rows >= self.plan.rows:
                    raise ValueError("vision hybrid convolution requires at least 17 output rows")
                event = self.plan.execute(
                    self.destination,
                    self.source,
                    queue=self.context.qpu_queue,
                    placement=Placement.HYBRID,
                    cpu_queue=self.context.cpu_queue,
                    qpu_rows=qpu_rows,
                )
                telemetry.qpu_dispatches += self.groups
                telemetry.hybrid_calls += 1
                telemetry.qpu_rows += qpu_rows
                telemetry.cpu_rows += self.plan.rows - qpu_rows
            elif mode is VisionMode.QPU_W8A8:
                assert isinstance(self.plan, Conv2dW8A8Plan)
                event = self.plan.execute(self.destination, self.source, queue=self.context.qpu_queue)
                telemetry.qpu_dispatches += self.groups
                telemetry.qpu_rows += self.plan.rows
            else:
                assert mode is VisionMode.HYBRID_W8A8
                assert isinstance(self.plan, Conv2dW8A8Plan) and self.context.cpu_queue is not None
                qpu_rows = max(16, self.plan.rows // 2 // 16 * 16)
                if qpu_rows >= self.plan.rows:
                    raise ValueError("vision hybrid convolution requires at least 17 output rows")
                event = self.plan.execute_hybrid(
                    self.destination,
                    self.source,
                    qpu_queue=self.context.qpu_queue,
                    cpu_queue=self.context.cpu_queue,
                    qpu_rows=qpu_rows,
                )
                telemetry.qpu_dispatches += self.groups
                telemetry.hybrid_calls += 1
                telemetry.qpu_rows += qpu_rows
                telemetry.cpu_rows += self.plan.rows - qpu_rows
            event.wait()
            result = np.array(self.destination.numpy(), copy=True)
        if self.bias is not None:
            result += self.bias[None, :, None, None]
        return np.ascontiguousarray(result, dtype=np.float32)


class QPUConv2dModule(torch.nn.Module):
    """Drop-in inference-only ``torch.nn.Conv2d`` backed by QPU-XLA."""

    def __init__(self: Self, original: torch.nn.Conv2d, context: VisionExecutionContext, name: str) -> None:
        """Capture immutable convolution parameters and defer shape allocation."""
        super().__init__()
        if original.padding_mode != "zeros":
            raise ValueError("QPU vision convolutions require zero padding")
        weight = np.ascontiguousarray(original.weight.detach().cpu().numpy(), dtype=np.float32)
        bias = (
            None
            if original.bias is None
            else np.ascontiguousarray(original.bias.detach().cpu().numpy(), dtype=np.float32)
        )
        self.name = name
        self.context = context
        self._prepared = _PreparedVisionConv(
            context,
            weight,
            bias,
            stride=cast(tuple[int, int], original.stride),
            padding=cast(tuple[int, int], original.padding),
            dilation=cast(tuple[int, int], original.dilation),
            groups=original.groups,
        )
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        context.telemetry.eligible_layers += 1
        context._layers.append(self)

    def forward(self: Self, source: torch.Tensor) -> torch.Tensor:
        """Execute one CPU-resident, batch-one FP32 inference tensor."""
        if source.device.type != "cpu" or source.dtype != torch.float32 or source.ndim != 4 or source.shape[0] != 1:
            raise ValueError("QPU vision modules require CPU float32 NCHW input with batch size one")
        if torch.is_grad_enabled():
            raise RuntimeError("QPU vision modules are inference-only; use torch.inference_mode()")
        values = np.ascontiguousarray(source.detach().numpy(), dtype=np.float32)
        return torch.from_numpy(self._prepared.execute(values))

    def close(self: Self) -> None:
        """Release prepared device state."""
        self._prepared.close()


class VisionExecutionContext:
    """Own shared queues, model placement, and full-graph dispatch telemetry."""

    def __init__(self: Self, mode: VisionMode, device: Device | None = None) -> None:
        """Create queues and telemetry for one explicit model placement."""
        if mode.uses_qpu and device is None:
            raise ValueError("QPU vision modes require an open Device")
        self.mode = mode
        self.device = device
        self.qpu_queue = None if device is None else device.queue()
        self.cpu_queue = None if device is None or not mode.hybrid else device.queue()
        self.telemetry = VisionTelemetry()
        self._layers: list[QPUConv2dModule] = []
        self._closed = False

    def reset_counts(self: Self) -> None:
        """Reset invocation counters while retaining eligible-layer topology."""
        eligible = self.telemetry.eligible_layers
        self.telemetry = VisionTelemetry(eligible_layers=eligible)

    def close(self: Self) -> None:
        """Release layer plans and context-owned queues."""
        if self._closed:
            return
        for layer in self._layers:
            layer.close()
        for queue in (self.cpu_queue, self.qpu_queue):
            if queue is not None:
                queue.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter the context lifetime."""
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release queues and layer plans at context exit."""
        self.close()


def instrument_torch_convolutions(
    module: torch.nn.Module,
    context: VisionExecutionContext,
    *,
    prefix: str = "",
) -> torch.nn.Module:
    """Replace every Conv2d in an inference graph with an auditable adapter."""
    for child_name, child in tuple(module.named_children()):
        qualified = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, torch.nn.Conv2d):
            setattr(module, child_name, QPUConv2dModule(child, context, qualified))
        else:
            instrument_torch_convolutions(child, context, prefix=qualified)
    return module


__all__ = [
    "QPUConv2dModule",
    "VisionExecutionContext",
    "VisionMode",
    "VisionTelemetry",
    "instrument_torch_convolutions",
]
