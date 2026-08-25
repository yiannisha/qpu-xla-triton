"""Persistent native SmolVLA runtime with real QPU and CPU/QPU paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.kernels.gemm_int8 import TILED_W8A8_GEMM_KERNEL, pack_int8_quads
from qpu_xla.kernels.gemv_int8 import W8A8_GEMV_KERNEL
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.models.smolvla.checkpoint import (
    SmolVLAArtifact,
    SmolVLACheckpoint,
    SmolVLANumerics,
    W8A8Weight,
)
from qpu_xla.models.smolvla.placement import SmolVLAMemoryPlan, SmolVLAPlacementPolicy
from qpu_xla.models.smolvla.reference import (
    SmolVLAReferenceRuntime,
    resize_with_top_left_padding_rgb,
)
from qpu_xla.ops.activation import gelu_tanh_fp32, silu_fp32
from qpu_xla.ops.affine import affine_fp32
from qpu_xla.ops.embedding import embedding_lookup_fp32
from qpu_xla.ops.gather import PreparedPatchifyFP32, PreparedPixelShuffleFP32
from qpu_xla.ops.layer_norm import layer_norm_fp32
from qpu_xla.ops.matmul import matmul
from qpu_xla.ops.multihead_sdpa import PreparedMultiHeadSDPAFP32
from qpu_xla.ops.residual import residual_add_fp32
from qpu_xla.ops.rgb_resize import PreparedRGBResizeNormFP32
from qpu_xla.ops.rms_norm import rms_norm_fp32
from qpu_xla.ops.rope_split_half import apply_split_half_rope_tables_fp32, split_half_rope_tables_fp32
from qpu_xla.ops.swiglu import swiglu_fp32
from qpu_xla.scheduler import Placement


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _linear_batch(config: object, name: str) -> int:
    # Kept name-driven to mirror the strict checkpoint contract and avoid a
    # second hand-maintained list of all 500 tensor shapes.
    cfg = cast("SmolVLAConfigLike", config)
    if "patch_embedding.weight" in name or "vision_model.encoder.layers" in name:
        return cfg.vision_token_count
    if "connector.modality_projection" in name:
        return cfg.image_token_count
    if ".text_model.layers." in name:
        return cfg.prefix_length
    if ".lm_expert.layers." in name:
        parts = name.split(".layers.", 1)[1]
        layer = int(parts.split(".", 1)[0])
        if layer % cfg.self_attn_every_n_layers and name.endswith(
            ("self_attn.k_proj.weight", "self_attn.v_proj.weight")
        ):
            return cfg.prefix_length
        return cfg.chunk_size
    if name.endswith("state_proj.weight"):
        return 1
    return cfg.chunk_size


class SmolVLAConfigLike:
    vision_token_count: int
    image_token_count: int
    prefix_length: int
    self_attn_every_n_layers: int
    chunk_size: int


def _linear_capacities(checkpoint: SmolVLACheckpoint, *, reduction_alignment: int) -> tuple[int, int, int, int]:
    weight_capacity = source_capacity = result_capacity = bias_capacity = 0
    for name, shape in checkpoint.config.expected_shapes().items():
        if len(shape) == 4:
            outputs, channels, height, width = shape
            inputs = channels * height * width
        elif len(shape) == 2 and not name.endswith(
            ("embed_tokens.weight", "position_embedding.weight", "lm_head.weight")
        ):
            outputs, inputs = shape
        else:
            continue
        batch = _linear_batch(checkpoint.config, name)
        padded_batch = 1 if batch == 1 else _round_up(batch, 16)
        padded_inputs = _round_up(inputs, reduction_alignment)
        padded_outputs = _round_up(outputs, 16)
        weight_capacity = max(weight_capacity, padded_inputs * padded_outputs)
        source_capacity = max(source_capacity, padded_batch * padded_inputs)
        result_capacity = max(result_capacity, padded_batch * padded_outputs)
        bias_capacity = max(bias_capacity, padded_outputs)
    return weight_capacity, source_capacity, result_capacity, bias_capacity


class _StagedFP32Linear:
    """One persistent set of padded buffers reused by every FP32 projection."""

    def __init__(self: Self, runtime: SmolVLARuntime) -> None:
        self.runtime = runtime
        self.device = runtime.device
        capacities = _linear_capacities(runtime.checkpoint, reduction_alignment=4)
        weight, source, result, bias = capacities
        self._weight_buffer = self.device.allocate(weight * 4)
        self._source_buffer = self.device.allocate(source * 4)
        self._result_buffer = self.device.allocate(result * 4)
        self._bias_buffer = self.device.allocate(bias * 4)

    def close(self: Self) -> None:
        for buffer in (self._weight_buffer, self._source_buffer, self._result_buffer, self._bias_buffer):
            buffer.close()

    def execute(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32] | None,
        placement: Placement,
    ) -> npt.NDArray[np.float32]:
        batch, inputs = values.shape
        outputs = weight.shape[0]
        if placement is Placement.CPU:
            cpu_result = cast(npt.NDArray[np.float32], np.matmul(values, weight.T, dtype=np.float32))
            if bias is not None:
                cpu_result += bias
            return np.ascontiguousarray(cpu_result)
        padded_batch = 1 if batch == 1 else _round_up(batch, 16)
        padded_inputs = _round_up(inputs, 4)
        padded_outputs = _round_up(outputs, 16)
        source = self._source_buffer.tensor((padded_batch, padded_inputs), np.float32)
        staged_weight = self._weight_buffer.tensor((padded_inputs, padded_outputs), np.float32)
        result_tensor = self._result_buffer.tensor((padded_batch, padded_outputs), np.float32)
        staged_bias = self._bias_buffer.tensor((padded_outputs,), np.float32)

        def prepare() -> None:
            source.numpy().fill(0.0)
            staged_weight.numpy().fill(0.0)
            staged_bias.numpy().fill(0.0)
            source.numpy()[:batch, :inputs] = values
            staged_weight.numpy()[:inputs, :outputs] = weight.T
            if bias is not None:
                staged_bias.numpy()[:outputs] = bias

        prepared = self.runtime.queue.host_task(
            prepare,
            buffers=(
                source.access(AccessMode.WRITE),
                staged_weight.access(AccessMode.WRITE),
                staged_bias.access(AccessMode.WRITE),
            ),
            name="smolvla.fp32_linear.stage",
        )
        qpu_rows: int | None = None
        qpu_columns: int | None = None
        cpu_queue = None
        if placement is Placement.HYBRID:
            try:
                if padded_batch == 1:
                    qpu_columns = self.runtime.policy.row_partition(padded_outputs, alignment=16)
                else:
                    qpu_rows = self.runtime.policy.row_partition(padded_batch, alignment=16)
                cpu_queue = self.runtime.cpu_queue
            except ValueError:
                # A single 16-wide tile cannot be divided into two non-empty
                # aligned partitions. It still uses the explicit QPU path.
                placement = Placement.QPU
        compute = matmul(
            result_tensor,
            source,
            staged_weight,
            queue=self.runtime.queue,
            cpu_queue=cpu_queue,
            placement=placement,
            qpu_rows=qpu_rows,
            qpu_columns=qpu_columns,
            wait_for=(prepared,),
        )
        affine_placement = placement
        affine_rows: int | None = None
        if placement is Placement.HYBRID:
            if padded_batch > 1:
                affine_rows = qpu_rows
            else:
                # The projection split columns; the affine row kernel runs on
                # the QPU after the join and preserves the projection's true
                # concurrent CPU/QPU implementation.
                affine_placement = Placement.QPU
        adjusted = affine_fp32(
            result_tensor,
            result_tensor,
            staged_bias,
            queue=self.runtime.queue,
            cpu_queue=self.runtime.cpu_queue if affine_placement is Placement.HYBRID else None,
            placement=affine_placement,
            qpu_rows=affine_rows,
            wait_for=(compute,),
        )
        adjusted.wait()
        # The staged result buffer is reused by the very next projection
        # (notably Q/K/V). Return an owning copy so earlier graph values cannot
        # be silently overwritten by that reuse.
        result_values = cast(npt.NDArray[np.float32], result_tensor.numpy())
        return np.array(result_values[:batch, :outputs], dtype=np.float32, copy=True, order="C")


class _StagedW8A8Linear:
    """Persistent packed operands with a newly staged artifact weight per call."""

    def __init__(self: Self, runtime: SmolVLARuntime) -> None:
        self.runtime = runtime
        self.device = runtime.device
        weight, source, result, bias = _linear_capacities(runtime.checkpoint, reduction_alignment=16)
        self._weight_buffer = self.device.allocate(weight)
        self._source_buffer = self.device.allocate(source)
        self._accumulator_buffer = self.device.allocate(result * 4)
        self._result_buffer = self.device.allocate(result * 4)
        max_rows = _round_up(
            max(
                runtime.config.vision_token_count,
                runtime.config.image_token_count,
                runtime.config.prefix_length,
                runtime.config.chunk_size,
                16,
            ),
            16,
        )
        self._row_scale_buffer = self.device.allocate(max_rows * 4)
        self._column_scale_buffer = self.device.allocate(bias * 4)
        self._bias_buffer = self.device.allocate(bias * 4)
        self._host_weight = np.empty((weight,), dtype=np.int8)

    def close(self: Self) -> None:
        for buffer in (
            self._weight_buffer,
            self._source_buffer,
            self._accumulator_buffer,
            self._result_buffer,
            self._row_scale_buffer,
            self._column_scale_buffer,
            self._bias_buffer,
        ):
            buffer.close()

    @staticmethod
    def _cpu(
        values: npt.NDArray[np.float32],
        weight: W8A8Weight,
        bias: npt.NDArray[np.float32] | None,
    ) -> npt.NDArray[np.float32]:
        scales = np.maximum(
            np.max(np.abs(values), axis=1) / np.float32(127.0),
            np.float32(1.0 / 127.0),
        )
        quantized = np.rint(values / scales[:, None]).clip(-127, 127).astype(np.int32)
        accumulation = np.matmul(quantized, weight.values.reshape(weight.values.shape[0], -1).astype(np.int32).T)
        result = accumulation.astype(np.float32) * scales[:, None] * weight.scales[None, :]
        if bias is not None:
            result += bias
        return np.ascontiguousarray(result)

    def execute(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: W8A8Weight,
        bias: npt.NDArray[np.float32] | None,
        placement: Placement,
    ) -> npt.NDArray[np.float32]:
        batch, inputs = values.shape
        outputs = weight.values.shape[0]
        if placement is Placement.CPU:
            return self._cpu(values, weight, bias)
        padded_batch = 1 if batch == 1 else _round_up(batch, 16)
        padded_inputs = _round_up(inputs, 16)
        padded_outputs = _round_up(outputs, 16)
        packed_source = self._source_buffer.tensor((padded_batch, padded_inputs // 4), np.uint32)
        packed_weight = self._weight_buffer.tensor((padded_inputs // 4, padded_outputs), np.uint32)
        accumulator = self._accumulator_buffer.tensor((padded_batch, padded_outputs), np.int32)
        result = self._result_buffer.tensor((padded_batch, padded_outputs), np.float32)
        row_scales = self._row_scale_buffer.tensor((padded_batch,), np.float32)
        column_scales = self._column_scale_buffer.tensor((padded_outputs,), np.float32)
        staged_bias = self._bias_buffer.tensor((padded_outputs,), np.float32)
        source_i8 = packed_source.numpy().view(np.int8).reshape(padded_batch, padded_inputs)
        host_weight = self._host_weight[: padded_outputs * padded_inputs].reshape(padded_outputs, padded_inputs)

        def prepare() -> None:
            row_scales.numpy().fill(np.float32(1.0 / 127.0))
            active_scales = np.maximum(
                np.max(np.abs(values), axis=1) / np.float32(127.0),
                np.float32(1.0 / 127.0),
            )
            row_scales.numpy()[:batch] = active_scales
            source_i8.fill(0)
            source_i8[:batch, :inputs] = np.rint(values / active_scales[:, None]).clip(-127, 127)
            host_weight.fill(0)
            host_weight[:outputs, :inputs] = weight.values.reshape(outputs, inputs)
            packed_weight.numpy()[:] = pack_int8_quads(host_weight).T
            column_scales.numpy().fill(0.0)
            column_scales.numpy()[:outputs] = weight.scales
            staged_bias.numpy().fill(0.0)
            if bias is not None:
                staged_bias.numpy()[:outputs] = bias

        prepared = self.runtime.queue.host_task(
            prepare,
            buffers=(
                packed_source.access(AccessMode.WRITE),
                packed_weight.access(AccessMode.WRITE),
                row_scales.access(AccessMode.WRITE),
                column_scales.access(AccessMode.WRITE),
                staged_bias.access(AccessMode.WRITE),
            ),
            name="smolvla.w8a8_linear.stage_pack",
        )
        if placement is Placement.HYBRID:
            try:
                if batch == 1:
                    qpu_outputs = self.runtime.policy.row_partition(outputs, alignment=16)
                    qpu_rows = 1
                else:
                    qpu_rows = self.runtime.policy.row_partition(batch, alignment=16)
                    qpu_outputs = outputs
            except ValueError:
                placement = Placement.QPU
        if placement is Placement.QPU:
            qpu_rows = 1 if batch == 1 else padded_batch
            qpu_outputs = outputs
        kernel_source = packed_source.slice((slice(0, qpu_rows), slice(None)))
        kernel_weight = packed_weight.slice((slice(None), slice(0, _round_up(qpu_outputs, 16))))
        kernel_accumulator = accumulator.slice((slice(0, qpu_rows), slice(0, _round_up(qpu_outputs, 16))))
        kernel = W8A8_GEMV_KERNEL if qpu_rows == 1 else TILED_W8A8_GEMM_KERNEL
        qpu_event = self.runtime.queue.submit(
            kernel,
            (kernel_source, kernel_weight, kernel_accumulator),
            grid=(kernel_weight.shape[1] // 16, 1 if qpu_rows == 1 else qpu_rows // 16, 1),
            wait_for=(prepared,),
            buffers=(
                kernel_source.access(AccessMode.READ),
                kernel_weight.access(AccessMode.READ),
                kernel_accumulator.access(AccessMode.WRITE),
            ),
        )

        def qpu_dequantize() -> None:
            row_scale_values = cast(npt.NDArray[np.float32], row_scales.numpy())
            column_scale_values = cast(npt.NDArray[np.float32], column_scales.numpy())
            result.numpy()[:qpu_rows, :qpu_outputs] = (
                accumulator.numpy()[:qpu_rows, :qpu_outputs].astype(np.float32)
                * row_scale_values[:qpu_rows, None]
                * column_scale_values[None, :qpu_outputs]
            )

        qpu_finish = self.runtime.queue.host_task(
            qpu_dequantize,
            wait_for=(qpu_event,),
            buffers=(accumulator.access(AccessMode.READ), result.access(AccessMode.WRITE)),
            name="smolvla.w8a8_linear.dequantize_qpu",
        )
        dependency = qpu_finish
        if placement is Placement.HYBRID:
            if batch == 1:

                def cpu_outputs() -> None:
                    result.numpy()[:, qpu_outputs:outputs] = self._cpu(values, weight, None)[:, qpu_outputs:]

                cpu_event = self.runtime.cpu_queue.host_task(
                    cpu_outputs,
                    wait_for=(prepared,),
                    buffers=(result.access(AccessMode.WRITE),),
                    name="smolvla.w8a8_linear.hybrid_cpu_outputs",
                )
            else:

                def cpu_rows() -> None:
                    result.numpy()[qpu_rows:batch, :outputs] = self._cpu(values[qpu_rows:], weight, None)

                cpu_event = self.runtime.cpu_queue.host_task(
                    cpu_rows,
                    wait_for=(prepared,),
                    buffers=(result.access(AccessMode.WRITE),),
                    name="smolvla.w8a8_linear.hybrid_cpu_rows",
                )
            dependency = self.runtime.cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_finish, cpu_event),
                name="smolvla.w8a8_linear.hybrid_join",
            )
        affine_placement = Placement.QPU if placement is Placement.HYBRID and batch == 1 else placement
        adjusted = affine_fp32(
            result.slice((slice(0, batch), slice(0, padded_outputs))),
            result.slice((slice(0, batch), slice(0, padded_outputs))),
            staged_bias,
            queue=self.runtime.queue,
            cpu_queue=self.runtime.cpu_queue if affine_placement is Placement.HYBRID else None,
            placement=affine_placement,
            qpu_rows=qpu_rows if affine_placement is Placement.HYBRID else None,
            wait_for=(dependency,),
        )
        adjusted.wait()
        return np.array(result.numpy()[:batch, :outputs], dtype=np.float32, copy=True, order="C")


@dataclass(slots=True)
class _ElementwiseWorkspace:
    source: Tensor
    destination: Tensor
    auxiliary: Tensor | None = None
    weight: Tensor | None = None
    bias: Tensor | None = None
    secondary: Tensor | None = None

    def close(self: Self) -> None:
        tensors = (
            self.source,
            self.destination,
            self.auxiliary,
            self.weight,
            self.bias,
            self.secondary,
        )
        for tensor in tensors:
            if tensor is not None:
                tensor.buffer.close()


@dataclass(slots=True)
class _AttentionWorkspace:
    query: Tensor
    key: Tensor
    value: Tensor
    destination: Tensor
    plan: PreparedMultiHeadSDPAFP32

    def close(self: Self) -> None:
        self.plan.close()
        for tensor in (self.query, self.key, self.value, self.destination):
            tensor.buffer.close()


class SmolVLARuntime(SmolVLAReferenceRuntime):
    """End-to-end runtime whose forced and hybrid modes execute real QPU work."""

    def __init__(
        self: Self,
        device: Device,
        checkpoint: SmolVLACheckpoint,
        *,
        policy: SmolVLAPlacementPolicy,
        memory_plan: SmolVLAMemoryPlan | None = None,
        owns_device: bool = False,
    ) -> None:
        """Create persistent staged operators around an already-open device."""
        super().__init__(checkpoint)
        self.device = device
        self.policy = policy
        self.memory_plan = (
            SmolVLAMemoryPlan.build(checkpoint.config, checkpoint.numerics) if memory_plan is None else memory_plan
        )
        if self.memory_plan.numerics is not checkpoint.numerics:
            raise ValueError("SmolVLA memory plan numerics do not match the checkpoint")
        self.queue = device.queue()
        self.cpu_queue = device.queue()
        self._owns_device = owns_device
        self._closed = False
        self._elementwise: dict[tuple[str, tuple[int, int]], _ElementwiseWorkspace] = {}
        self._attention_workspaces: dict[tuple[int, int, int, int, int], _AttentionWorkspace] = {}
        self._embedding_table: Tensor | None = None
        self._embedding_ids: Tensor | None = None
        self._embedding_output: Tensor | None = None
        self._linear_engine: _StagedFP32Linear | _StagedW8A8Linear
        if checkpoint.numerics is SmolVLANumerics.FP32:
            self._linear_engine = _StagedFP32Linear(self)
        else:
            self._linear_engine = _StagedW8A8Linear(self)

        cfg = checkpoint.config
        self._rgb_source = device.tensor((cfg.input_image_height, cfg.input_image_width, 3), np.float32)
        self._rgb_destination = device.tensor((3, cfg.image_size, cfg.image_size), np.float32)
        self._rgb_plan = PreparedRGBResizeNormFP32(self._rgb_source, self._rgb_destination)
        self._patches = device.tensor((cfg.vision_token_count, 3 * cfg.patch_size**2), np.float32)
        self._patch_plan = PreparedPatchifyFP32(
            self._rgb_destination,
            self._patches,
            patch_size=cfg.patch_size,
        )
        self._pixel_source = device.tensor((cfg.vision_token_count, cfg.vision_hidden_size), np.float32)
        self._pixel_destination = device.tensor((cfg.image_token_count, cfg.connector_input_size), np.float32)
        self._pixel_plan = PreparedPixelShuffleFP32(
            self._pixel_source,
            self._pixel_destination,
            scale_factor=cfg.pixel_shuffle_factor,
        )

    @classmethod
    def open(
        cls: type[SmolVLARuntime],
        artifact: SmolVLAArtifact,
        *,
        policy: SmolVLAPlacementPolicy,
        memory_limit_bytes: int = 6 * 1024**3,
        **driver_kwargs: object,
    ) -> SmolVLARuntime:
        """Budget the model first, then create a correctly sized VideoCore BO."""
        plan = SmolVLAMemoryPlan.build(
            artifact.checkpoint.config,
            artifact.checkpoint.numerics,
            limit_bytes=memory_limit_bytes,
        )
        if "data_area_size" in driver_kwargs:
            raise ValueError("SmolVLARuntime.open owns data_area_size; use the calculated memory plan")
        device = Device.open(data_area_size=plan.qpu_arena_bytes, **driver_kwargs)
        try:
            return cls(device, artifact.checkpoint, policy=policy, memory_plan=plan, owns_device=True)
        except BaseException:
            device.close()
            raise

    def __enter__(self: Self) -> Self:
        """Enter the persistent runtime lifetime."""
        if self._closed:
            raise RuntimeError("SmolVLA runtime is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the runtime and any owned device."""
        self.close()

    def close(self: Self) -> None:
        """Release all persistent plans, buffers, queues, and an owned device."""
        if self._closed:
            return
        self.queue.close()
        self.cpu_queue.close()
        self._rgb_plan.close()
        self._patch_plan.close()
        self._pixel_plan.close()
        self._linear_engine.close()
        for workspace in self._elementwise.values():
            workspace.close()
        for workspace in self._attention_workspaces.values():
            workspace.close()
        for tensor in (
            self._embedding_table,
            self._embedding_ids,
            self._embedding_output,
            self._rgb_source,
            self._rgb_destination,
            self._patches,
            self._pixel_source,
            self._pixel_destination,
        ):
            if tensor is not None:
                tensor.buffer.close()
        self._closed = True
        if self._owns_device:
            self.device.close()

    def _placement(self: Self, stage: str, shape: tuple[int, ...]) -> Placement:
        model_dtype = self.checkpoint.numerics.value
        dtype = "w8a8-i32-fp32" if stage == "linear" and self.checkpoint.numerics is SmolVLANumerics.W8A8 else "fp32"
        return self.policy.choose(
            stage,
            dtype=dtype,
            shape_class="x".join(map(str, shape)),
            model_dtype=model_dtype,
            model_shape_class=self.config.model_shape_class,
        )

    def _row_placement(
        self: Self,
        placement: Placement,
        rows: int,
        *,
        alignment: int = 1,
    ) -> tuple[Placement, int | None]:
        """Resolve a hybrid split, using QPU when a single tile cannot split."""
        if placement is not Placement.HYBRID:
            return placement, None
        try:
            return placement, self.policy.row_partition(rows, alignment=alignment)
        except ValueError:
            return Placement.QPU, None

    def _workspace(
        self: Self,
        operation: str,
        shape: tuple[int, int],
        *,
        auxiliary: bool = False,
        affine: bool = False,
        secondary: bool = False,
    ) -> _ElementwiseWorkspace:
        key = (operation, shape)
        workspace = self._elementwise.get(key)
        if workspace is None:
            workspace = _ElementwiseWorkspace(
                source=self.device.tensor(shape, np.float32),
                destination=self.device.tensor(shape, np.float32),
                auxiliary=self.device.tensor(shape, np.float32) if auxiliary else None,
                weight=self.device.tensor((shape[1],), np.float32) if affine else None,
                bias=self.device.tensor((shape[1],), np.float32) if affine else None,
                secondary=self.device.tensor(shape, np.float32) if secondary else None,
            )
            self._elementwise[key] = workspace
        return workspace

    def _linear(
        self: Self,
        values: npt.NDArray[np.float32],
        weight_name: str,
        bias_name: str | None = None,
    ) -> npt.NDArray[np.float32]:
        original_shape = values.shape
        flat = np.ascontiguousarray(values.reshape(-1, values.shape[-1]), dtype=np.float32)
        weight = self.checkpoint.linear_weight(weight_name)
        bias = None if bias_name is None else self.checkpoint.fp32(bias_name)
        placement = self._placement("linear", (flat.shape[0], flat.shape[1], weight.shape[0]))
        result = self._execute_linear(flat, weight, bias, placement)
        return result.reshape(*original_shape[:-1], result.shape[-1])

    def _execute_linear(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32] | W8A8Weight,
        bias: npt.NDArray[np.float32] | None,
        placement: Placement,
    ) -> npt.NDArray[np.float32]:
        """Dispatch a validated artifact weight to its matching staged engine."""
        if isinstance(weight, W8A8Weight):
            if not isinstance(self._linear_engine, _StagedW8A8Linear):
                raise TypeError("quantized SmolVLA weight requires the W8A8 staged engine")
            return self._linear_engine.execute(values, weight, bias, placement)
        if not isinstance(self._linear_engine, _StagedFP32Linear):
            raise TypeError("FP32 SmolVLA weight requires the FP32 staged engine")
        return self._linear_engine.execute(values, weight, bias, placement)

    def _linear_weight(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32] | W8A8Weight,
        bias: npt.NDArray[np.float32] | None,
    ) -> npt.NDArray[np.float32]:
        flat = np.ascontiguousarray(values.reshape(-1, values.shape[-1]), dtype=np.float32)
        placement = self._placement("linear", (flat.shape[0], flat.shape[1], weight.shape[0]))
        return self._execute_linear(flat, weight, bias, placement)

    def _layer_norm(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32],
        epsilon: float,
    ) -> npt.NDArray[np.float32]:
        shape = cast(tuple[int, int], values.shape)
        placement = self._placement("layer_norm", shape)
        if placement is Placement.CPU:
            return super()._layer_norm(values, weight, bias, epsilon)
        ws = self._workspace("layer_norm", shape, affine=True)
        assert ws.weight is not None and ws.bias is not None
        ws.source.numpy()[:] = values
        ws.weight.numpy()[:] = weight
        ws.bias.numpy()[:] = bias
        placement, qpu_rows = self._row_placement(placement, shape[0])
        layer_norm_fp32(
            ws.destination,
            ws.source,
            ws.weight,
            ws.bias,
            epsilon=epsilon,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy(), copy=True)

    def _rms_norm(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
        epsilon: float,
    ) -> npt.NDArray[np.float32]:
        shape = cast(tuple[int, int], values.shape)
        placement = self._placement("rms_norm", shape)
        if placement is Placement.CPU:
            return super()._rms_norm(values, weight, epsilon)
        ws = self._workspace("rms_norm", shape, affine=True)
        assert ws.weight is not None
        ws.source.numpy()[:] = values
        ws.weight.numpy()[:] = weight
        placement, qpu_rows = self._row_placement(placement, shape[0])
        rms_norm_fp32(
            ws.destination,
            ws.source,
            ws.weight,
            epsilon=epsilon,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy(), copy=True)

    def _activation(
        self: Self,
        values: npt.NDArray[np.float32],
        *,
        gelu: bool,
    ) -> npt.NDArray[np.float32]:
        shape = cast(tuple[int, int], values.shape)
        placement = self._placement("activation", shape)
        if placement is Placement.CPU:
            return super()._gelu_tanh(values) if gelu else super()._silu(values)
        ws = self._workspace("gelu" if gelu else "silu", shape)
        ws.source.numpy()[:] = values
        placement, qpu_rows = self._row_placement(placement, shape[0])
        operation = gelu_tanh_fp32 if gelu else silu_fp32
        operation(
            ws.destination,
            ws.source,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy(), copy=True)

    def _gelu_tanh(self: Self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self._activation(values, gelu=True)

    def _silu(self: Self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self._activation(values, gelu=False)

    def _swiglu(
        self: Self,
        gate: npt.NDArray[np.float32],
        up: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        shape = cast(tuple[int, int], gate.shape)
        placement = self._placement("swiglu", shape)
        if placement is Placement.CPU:
            return super()._swiglu(gate, up)
        ws = self._workspace("swiglu", shape, auxiliary=True)
        assert ws.auxiliary is not None
        ws.source.numpy()[:] = gate
        ws.auxiliary.numpy()[:] = up
        placement, qpu_rows = self._row_placement(placement, shape[0])
        swiglu_fp32(
            ws.destination,
            ws.source,
            ws.auxiliary,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy(), copy=True)

    def _residual(
        self: Self,
        left: npt.NDArray[np.float32],
        right: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        shape = cast(tuple[int, int], left.shape)
        placement = self._placement("residual", shape)
        if placement is Placement.CPU:
            return super()._residual(left, right)
        ws = self._workspace("residual", shape, auxiliary=True)
        assert ws.auxiliary is not None
        ws.source.numpy()[:] = left
        ws.auxiliary.numpy()[:] = right
        placement, qpu_rows = self._row_placement(placement, shape[0])
        residual_add_fp32(
            ws.destination,
            ws.source,
            ws.auxiliary,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy(), copy=True)

    def _scale(self: Self, values: npt.NDArray[np.float32], scale: np.float32) -> npt.NDArray[np.float32]:
        shape = cast(tuple[int, int], values.shape)
        placement = self._placement("affine", shape)
        if placement is Placement.CPU:
            return super()._scale(values, scale)
        ws = self._workspace("scale", shape, affine=True)
        assert ws.bias is not None
        ws.source.numpy()[:] = values
        ws.bias.numpy().fill(0.0)
        placement, qpu_rows = self._row_placement(placement, shape[0])
        affine_fp32(
            ws.destination,
            ws.source,
            ws.bias,
            scale=float(scale),
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy(), copy=True)

    def _embedding(
        self: Self,
        token_ids: npt.NDArray[np.int64],
        table: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        placement = self._placement("embedding", (token_ids.size, table.shape[1]))
        if placement is Placement.CPU:
            return super()._embedding(token_ids, table)
        if self._embedding_table is None:
            self._embedding_table = self.device.tensor(table.shape, np.float32)
            self._embedding_table.numpy()[:] = table
            self._embedding_ids = self.device.tensor((self.config.tokenizer_max_length,), np.int32)
            self._embedding_output = self.device.tensor(
                (self.config.tokenizer_max_length, self.config.vlm_hidden_size), np.float32
            )
        assert self._embedding_ids is not None and self._embedding_output is not None
        self._embedding_ids.numpy()[:] = token_ids.astype(np.int32)
        placement, qpu_tokens = self._row_placement(placement, token_ids.size)
        embedding_lookup_fp32(
            self._embedding_output,
            self._embedding_ids,
            self._embedding_table,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_tokens=qpu_tokens,
        ).wait()
        return np.array(self._embedding_output.numpy(), copy=True)

    def _split_half_rope(
        self: Self,
        values: npt.NDArray[np.float32],
        positions: npt.NDArray[np.int32],
        theta: float,
    ) -> npt.NDArray[np.float32]:
        tokens, heads, head_dim = values.shape
        rows = tokens * heads
        placement = self._placement("rope", (rows, head_dim))
        if placement is Placement.CPU:
            return super()._split_half_rope(values, positions, theta)
        shape = (rows, head_dim)
        ws = self._workspace("rope", shape, auxiliary=True, secondary=True)
        assert ws.auxiliary is not None and ws.secondary is not None
        cosine_values, sine_values = split_half_rope_tables_fp32(
            positions,
            heads=heads,
            head_dim=head_dim,
            base=theta,
        )
        cosine = ws.auxiliary.buffer.tensor(cosine_values.shape, np.float32)
        sine = ws.secondary.buffer.tensor(sine_values.shape, np.float32)
        ws.source.numpy()[:] = values.reshape(shape)
        cosine.numpy()[:] = cosine_values
        sine.numpy()[:] = sine_values
        placement, qpu_rows = self._row_placement(placement, rows)
        apply_split_half_rope_tables_fp32(
            ws.destination,
            ws.source,
            cosine,
            sine,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_rows=qpu_rows,
        ).wait()
        return np.array(ws.destination.numpy().reshape(values.shape), copy=True)

    def _attention(
        self: Self,
        query: npt.NDArray[np.float32],
        key: npt.NDArray[np.float32],
        value: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_] | None,
    ) -> npt.NDArray[np.float32]:
        topology = (query.shape[0], key.shape[0], query.shape[1], key.shape[1], query.shape[2])
        placement = self._placement("attention", topology)
        if placement is Placement.CPU:
            return super()._attention(query, key, value, mask)
        workspace = self._attention_workspaces.get(topology)
        if workspace is None:
            q = self.device.tensor(query.shape, np.float32)
            k = self.device.tensor(key.shape, np.float32)
            v = self.device.tensor(value.shape, np.float32)
            d = self.device.tensor(query.shape, np.float32)
            plan = PreparedMultiHeadSDPAFP32(
                self.device,
                query_length=query.shape[0],
                key_length=key.shape[0],
                query_heads=query.shape[1],
                key_value_heads=key.shape[1],
                head_dim=query.shape[2],
            )
            workspace = _AttentionWorkspace(q, k, v, d, plan)
            self._attention_workspaces[topology] = workspace
        workspace.query.numpy()[:] = query
        workspace.key.numpy()[:] = key
        workspace.value.numpy()[:] = value
        placement, qpu_heads = self._row_placement(placement, query.shape[1])
        workspace.plan.execute(
            workspace.destination,
            workspace.query,
            workspace.key,
            workspace.value,
            mask=mask,
            queue=self.queue,
            cpu_queue=self.cpu_queue if placement is Placement.HYBRID else None,
            placement=placement,
            qpu_heads=qpu_heads,
        ).wait()
        return np.array(workspace.destination.numpy().reshape(query.shape[0], -1), copy=True)

    def _vision_image(self: Self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        config = self.config
        preprocess = self._placement(
            "rgb_preprocess", (image.shape[0], image.shape[1], config.image_size, config.image_size)
        )
        if preprocess is Placement.CPU:
            self._rgb_destination.numpy()[:] = resize_with_top_left_padding_rgb(
                image, config.image_size, config.image_size
            )
        else:
            uploaded = self._rgb_plan.upload(image, queue=self.queue)
            preprocess, resize_rows = self._row_placement(preprocess, 3 * config.image_size, alignment=1)
            self._rgb_plan.execute(
                queue=self.queue,
                cpu_queue=self.cpu_queue if preprocess is Placement.HYBRID else None,
                placement=preprocess,
                qpu_rows=resize_rows,
                wait_for=(uploaded,),
            ).wait()
        patch_placement = self._placement("patchify", self._patches.shape)
        patch_placement, patch_rows = self._row_placement(patch_placement, config.vision_token_count)
        self._patch_plan.execute(
            queue=self.queue,
            cpu_queue=self.cpu_queue if patch_placement is Placement.HYBRID else None,
            placement=patch_placement,
            qpu_rows=patch_rows,
        ).wait()
        vision = f"{self._root}vlm_with_expert.vlm.model.vision_model"
        patch_weight = self.checkpoint.linear_weight(f"{vision}.embeddings.patch_embedding.weight")
        if isinstance(patch_weight, W8A8Weight):
            flat_patch_weight: npt.NDArray[np.float32] | W8A8Weight = W8A8Weight(
                patch_weight.values.reshape(config.vision_hidden_size, -1),
                patch_weight.scales,
            )
        else:
            flat_patch_weight = patch_weight.reshape(config.vision_hidden_size, -1)
        hidden = self._linear_weight(
            np.ascontiguousarray(self._patches.numpy()),
            flat_patch_weight,
            self.checkpoint.fp32(f"{vision}.embeddings.patch_embedding.bias"),
        )
        hidden = self._residual(
            hidden,
            self.checkpoint.fp32(f"{vision}.embeddings.position_embedding.weight"),
        )
        for layer in range(config.vision_num_layers):
            prefix = f"{vision}.encoder.layers.{layer}"
            normalized = self._layer_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.layer_norm1.weight"),
                self.checkpoint.fp32(f"{prefix}.layer_norm1.bias"),
                config.vision_layer_norm_eps,
            )
            query = self._linear(
                normalized, f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.q_proj.bias"
            ).reshape(config.vision_token_count, config.vision_num_heads, -1)
            key = self._linear(
                normalized, f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.k_proj.bias"
            ).reshape(config.vision_token_count, config.vision_num_heads, -1)
            value = self._linear(
                normalized, f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.v_proj.bias"
            ).reshape(config.vision_token_count, config.vision_num_heads, -1)
            attended = self._attention(query, key, value, None)
            hidden = self._residual(
                hidden,
                self._linear(
                    attended,
                    f"{prefix}.self_attn.out_proj.weight",
                    f"{prefix}.self_attn.out_proj.bias",
                ),
            )
            residual = hidden
            normalized = self._layer_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.layer_norm2.weight"),
                self.checkpoint.fp32(f"{prefix}.layer_norm2.bias"),
                config.vision_layer_norm_eps,
            )
            intermediate = self._linear(
                normalized,
                f"{prefix}.mlp.fc1.weight",
                f"{prefix}.mlp.fc1.bias",
            )
            hidden = self._residual(
                residual,
                self._linear(
                    self._gelu_tanh(intermediate),
                    f"{prefix}.mlp.fc2.weight",
                    f"{prefix}.mlp.fc2.bias",
                ),
            )
        hidden = self._layer_norm(
            hidden,
            self.checkpoint.fp32(f"{vision}.post_layernorm.weight"),
            self.checkpoint.fp32(f"{vision}.post_layernorm.bias"),
            config.vision_layer_norm_eps,
        )
        self._pixel_source.numpy()[:] = hidden
        pixel_placement = self._placement("pixel_shuffle", self._pixel_destination.shape)
        pixel_placement, pixel_rows = self._row_placement(pixel_placement, config.image_token_count)
        self._pixel_plan.execute(
            queue=self.queue,
            cpu_queue=self.cpu_queue if pixel_placement is Placement.HYBRID else None,
            placement=pixel_placement,
            qpu_rows=pixel_rows,
        ).wait()
        return self._linear(
            np.ascontiguousarray(self._pixel_destination.numpy()),
            f"{self._root}vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
        )


__all__ = ["SmolVLARuntime"]
