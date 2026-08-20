"""Deterministic per-output-channel INT8 packing for TinyLlama weight preprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.backend import PyVideoCore7Backend
from qpu_xla.benchmark import CandidateRecord, CandidateRegistry
from qpu_xla.device import Device
from qpu_xla.errors import DependencyError
from qpu_xla.kernels.gemm_int8 import (
    TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
    TILED_W8A8_GEMM_KERNEL,
    pack_int8_quads,
    supports_tiled_w8a8_gemm,
    supports_tiled_w8a8_gemm_dequantize,
)
from qpu_xla.kernels.gemv_int8 import W8A8_GEMV_KERNEL
from qpu_xla.kernels.w8a8_epilogue import W8A8_DEQUANTIZE_KERNEL, supports_w8a8_dequantize
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


@dataclass(frozen=True, slots=True)
class QuantizedMatrixInt8:
    """Row-wise symmetric INT8 matrix plus one FP32 dequantization scale per row."""

    values: npt.NDArray[np.int8]
    scales: npt.NDArray[np.float32]

    def __post_init__(self: Self) -> None:
        """Require compact, self-consistent storage before model code consumes it."""
        if self.values.ndim != 2 or self.scales.ndim != 1 or self.scales.shape != (self.values.shape[0],):
            raise ValueError("quantized matrix values must be rank-2 with one scale per output row")
        if not self.values.flags.c_contiguous or not self.scales.flags.c_contiguous:
            raise ValueError("quantized matrix values and scales must be contiguous")
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("quantized matrix scales must be finite and positive")

    @property
    def shape(self: Self) -> tuple[int, int]:
        """Return the logical ``(output_features, input_features)`` matrix shape."""
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    def dequantize(self: Self) -> npt.NDArray[np.float32]:
        """Return a contiguous FP32 reconstruction for reference execution."""
        return np.ascontiguousarray(self.values.astype(np.float32) * self.scales[:, None])


class PreparedW8A8Linear:
    """Persistent packed-weight W8A8 projection candidate for QPU evaluation.

    The plan owns padded packed weights and reusable activation/accumulator
    storage. Activations are quantized per row on the CPU, the QPU performs an
    exact INT8-by-INT8 GEMM with INT32 accumulation, and the CPU applies the
    activation and per-output weight scales. A plan is intentionally bound to
    one queue and permits one in-flight invocation so its scratch storage
    cannot be overwritten by concurrent submissions.
    """

    def __init__(
        self: Self,
        device: Device,
        weight: QuantizedMatrixInt8,
        *,
        max_batch: int,
        qpu_dequantize: bool = False,
        fuse_dequantize: bool = True,
    ) -> None:
        """Pack one immutable weight matrix and allocate reusable scratch tensors."""
        if max_batch <= 0:
            raise ValueError("PreparedW8A8Linear max_batch must be positive")
        self.device = device
        self.weight = weight
        self.max_batch = max_batch
        self.qpu_dequantize = qpu_dequantize
        self.fuse_dequantize = fuse_dequantize
        if fuse_dequantize and not qpu_dequantize:
            self.fuse_dequantize = False
        self.padded_batch = (max_batch + 15) // 16 * 16
        self.padded_features = (weight.shape[1] + 15) // 16 * 16
        self.padded_outputs = (weight.shape[0] + 15) // 16 * 16
        if self.padded_features * 127 * 127 > np.iinfo(np.int32).max:
            raise ValueError("prepared W8A8 linear accumulation can overflow int32")

        padded_weight = np.zeros((self.padded_outputs, self.padded_features), dtype=np.int8)
        padded_weight[: weight.shape[0], : weight.shape[1]] = weight.values
        packed_weight = np.ascontiguousarray(pack_int8_quads(padded_weight).T)
        self._packed_source = device.tensor((self.padded_batch, self.padded_features // 4), np.uint32)
        self._packed_weight = device.tensor(packed_weight.shape, np.uint32)
        self._accumulator = device.tensor((self.padded_batch, self.padded_outputs), np.int32)
        self._row_scales = device.tensor((self.padded_batch,), np.float32)
        self._column_scales = device.tensor((self.padded_outputs,), np.float32)
        self._gemv_source = self._packed_source.slice((slice(0, 1), slice(None)))
        self._gemv_accumulator = self._accumulator.slice((slice(0, 1), slice(None)))
        self._packed_weight.numpy()[:] = packed_weight
        self._column_scales.numpy().fill(0)
        self._column_scales.numpy()[: weight.shape[0]] = weight.scales
        self._source_scales = cast(npt.NDArray[np.float32], self._row_scales.numpy())
        # Four adjacent INT8 values have exactly the packed uint32 layout that
        # the QPU consumes.  Quantize directly into the mapped packed buffer so
        # every invocation avoids a second full activation pass and temporary.
        self._source_scratch = cast(
            npt.NDArray[np.int8],
            self._packed_source.numpy().view(np.int8).reshape(self.padded_batch, self.padded_features),
        )
        self._hybrid_weight_key: int | None = None
        self._hybrid_weight_tail: npt.NDArray[np.int32] | None = None
        self._hybrid_weight_full: npt.NDArray[np.int32] | None = None
        self._queue: Queue | None = None
        self._cpu_queue: Queue | None = None
        self._last_event: Event | None = None
        self._closed = False
        if not supports_tiled_w8a8_gemm(
            self._packed_source,
            self._packed_weight,
            self._accumulator,
            device.backend,
        ):
            self.close()
            raise ValueError("PreparedW8A8Linear requires the VideoCore VII packed W8A8 backend")

    def __enter__(self: Self) -> Self:
        """Enter a context-managed plan lifetime."""
        if self._closed:
            raise RuntimeError("prepared W8A8 linear plan is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release plan-owned scratch buffers."""
        self.close()

    def close(self: Self) -> None:
        """Wait for outstanding work and release all plan-owned allocations."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.wait()
        for tensor in (
            self._packed_source,
            self._packed_weight,
            self._accumulator,
            self._row_scales,
            self._column_scales,
        ):
            tensor.buffer.close()
        self._closed = True

    def _prepare_values(self: Self, values: npt.NDArray[np.float32]) -> None:
        """Quantize one logical FP32 batch directly into persistent QPU input."""
        if values.ndim != 2 or values.dtype != np.dtype(np.float32):
            raise ValueError("prepared W8A8 values must be a rank-2 float32 array")
        batch, features = values.shape
        if batch > self.max_batch or features != self.weight.shape[1]:
            raise ValueError("prepared W8A8 values exceed the plan shape")
        row_max = np.max(np.abs(values), axis=1)
        scales = self._source_scales[:batch]
        np.maximum(row_max / np.float32(127.0), np.float32(1.0 / 127.0), out=scales)
        if batch == 1:
            self._source_scratch[0].fill(0)
            self._source_scratch[0, :features] = np.rint(values[0] / scales[0]).clip(-127, 127)
        else:
            self._source_scratch.fill(0)
            self._source_scratch[:batch, :features] = np.rint(values / scales[:, None]).clip(-127, 127)

    def _submit_prepared(
        self: Self,
        destination: Tensor,
        *,
        batch: int,
        queue: Queue,
        wait_for: Iterable[Event],
    ) -> Event:
        """Submit the QPU and epilogue stages after activation packing."""
        dependencies = tuple(wait_for)
        active_padded_batch = (batch + 15) // 16 * 16
        kernel = W8A8_GEMV_KERNEL if batch == 1 else TILED_W8A8_GEMM_KERNEL
        kernel_source = (
            self._gemv_source
            if batch == 1
            else self._packed_source.slice((slice(0, active_padded_batch), slice(None)))
        )
        kernel_accumulator = (
            self._gemv_accumulator
            if batch == 1
            else self._accumulator.slice((slice(0, active_padded_batch), slice(None)))
        )
        active_row_scales = self._row_scales.slice((slice(0, active_padded_batch),))
        if (
            self.qpu_dequantize
            and self.fuse_dequantize
            and supports_tiled_w8a8_gemm_dequantize(
                kernel_source,
                self._packed_weight,
                active_row_scales,
                self._column_scales,
                destination,
                self.device.backend,
            )
        ):
            event = queue.submit(
                TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
                (
                    kernel_source,
                    self._packed_weight,
                    active_row_scales,
                    self._column_scales,
                    destination,
                ),
                grid=(self.padded_outputs // 16, active_padded_batch // 16, 1),
                wait_for=dependencies,
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    self._packed_weight.access(AccessMode.READ),
                    active_row_scales.access(AccessMode.READ),
                    self._column_scales.access(AccessMode.READ),
                    destination.access(AccessMode.WRITE),
                ),
            )
            self._last_event = event
            return event

        gemm_event = queue.submit(
            kernel,
            (kernel_source, self._packed_weight, kernel_accumulator),
            grid=(self.padded_outputs // 16, 1 if batch == 1 else active_padded_batch // 16, 1),
            wait_for=dependencies,
            buffers=(
                kernel_source.access(AccessMode.READ),
                self._packed_weight.access(AccessMode.READ),
                kernel_accumulator.access(AccessMode.WRITE),
            ),
        )

        tiled_accumulator = self._accumulator.slice((slice(0, batch), slice(None)))
        active_row_scales = self._row_scales.slice((slice(0, batch),))
        if self.qpu_dequantize and supports_w8a8_dequantize(
            tiled_accumulator,
            active_row_scales,
            self._column_scales,
            destination,
            self.device.backend,
        ):
            event = queue.submit(
                W8A8_DEQUANTIZE_KERNEL,
                (tiled_accumulator, active_row_scales, self._column_scales, destination),
                grid=(self.padded_outputs // 16, batch // 16, 1),
                wait_for=(gemm_event,),
                buffers=(
                    tiled_accumulator.access(AccessMode.READ),
                    active_row_scales.access(AccessMode.READ),
                    self._column_scales.access(AccessMode.READ),
                    destination.access(AccessMode.WRITE),
                ),
            )
            self._last_event = event
            return event

        def dequantize() -> None:
            active = self._accumulator.numpy()[:batch, : self.weight.shape[0]].astype(np.float32)
            destination.numpy()[:] = active * self._source_scales[:batch, None] * self.weight.scales[None, :]

        event = queue.host_task(
            dequantize,
            wait_for=(gemm_event,),
            buffers=(self._accumulator.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="tinyllama.w8a8_linear.dequantize",
        )
        self._last_event = event
        return event

    def execute(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Submit host quantization, packed QPU GEMM, and FP32 dequantization."""
        if self._closed:
            raise RuntimeError("prepared W8A8 linear plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared W8A8 linear plan already has an in-flight invocation")
        if self._queue is None:
            self._queue = queue
        elif self._queue is not queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first submission queue")
        if queue.device is not self.device:
            raise DependencyError("prepared W8A8 linear queue and tensors must belong to the plan device")
        if source.buffer.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("prepared W8A8 linear source and destination must belong to the plan device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("prepared W8A8 linear requires float32 source and destination")
        if len(source.shape) != 2 or len(destination.shape) != 2:
            raise ValueError("prepared W8A8 linear requires rank-2 source and destination")
        batch, features = source.shape
        if batch > self.max_batch or features != self.weight.shape[1]:
            raise ValueError("prepared W8A8 linear source exceeds the plan shape")
        if destination.shape != (batch, self.weight.shape[0]):
            raise ValueError("prepared W8A8 linear destination shape does not align")

        def prepare() -> None:
            self._prepare_values(source.numpy().astype(np.float32, copy=False))

        prepare_event = queue.host_task(
            prepare,
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                self._packed_source.access(AccessMode.WRITE),
                self._row_scales.access(AccessMode.WRITE),
            ),
            name="tinyllama.w8a8_linear.prepare",
        )
        return self._submit_prepared(destination, batch=batch, queue=queue, wait_for=(prepare_event,))

    def execute_prepared(
        self: Self,
        destination: Tensor,
        *,
        batch: int,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Submit an input already packed by an enclosing fused host stage."""
        if self._closed:
            raise RuntimeError("prepared W8A8 linear plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared W8A8 linear plan already has an in-flight invocation")
        if self._queue is None:
            self._queue = queue
        elif self._queue is not queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first submission queue")
        if queue.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("prepared W8A8 linear queue and destination must belong to the plan device")
        if destination.dtype != np.dtype(np.float32) or len(destination.shape) != 2:
            raise ValueError("prepared W8A8 linear destination must be rank-2 float32")
        if batch <= 0 or batch > self.max_batch or destination.shape != (batch, self.weight.shape[0]):
            raise ValueError("prepared W8A8 linear destination shape does not align")
        return self._submit_prepared(destination, batch=batch, queue=queue, wait_for=wait_for)

    def execute_hybrid(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        qpu_queue: Queue,
        cpu_queue: Queue,
        qpu_outputs: int,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Run an aligned output prefix on QPU and the disjoint tail on CPU."""
        if self._closed:
            raise RuntimeError("prepared W8A8 linear plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared W8A8 linear plan already has an in-flight invocation")
        if qpu_queue is cpu_queue:
            raise DependencyError("hybrid W8A8 linear requires distinct CPU and QPU queues")
        if qpu_queue.device is not self.device or cpu_queue.device is not self.device:
            raise DependencyError("hybrid W8A8 queues must belong to the plan device")
        if self._queue is None:
            self._queue = qpu_queue
        elif self._queue is not qpu_queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first QPU queue")
        if self._cpu_queue is None:
            self._cpu_queue = cpu_queue
        elif self._cpu_queue is not cpu_queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first CPU queue")
        if source.buffer.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("prepared W8A8 linear tensors must belong to the plan device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("prepared W8A8 linear requires float32 source and destination")
        if len(source.shape) != 2 or len(destination.shape) != 2:
            raise ValueError("prepared W8A8 linear requires rank-2 source and destination")
        batch, features = source.shape
        outputs = self.weight.shape[0]
        if batch > self.max_batch or features != self.weight.shape[1]:
            raise ValueError("prepared W8A8 linear source exceeds the plan shape")
        if destination.shape != (batch, outputs):
            raise ValueError("prepared W8A8 linear destination shape does not align")
        if qpu_outputs <= 0 or qpu_outputs >= outputs or qpu_outputs % 16:
            raise ValueError("hybrid W8A8 qpu_outputs must be aligned and leave a non-empty CPU tail")
        if self._hybrid_weight_key != qpu_outputs:
            self._hybrid_weight_tail = self.weight.values[qpu_outputs:].astype(np.int32)
            self._hybrid_weight_key = qpu_outputs

        def prepare() -> None:
            values = source.numpy().astype(np.float32, copy=False)
            row_max = np.max(np.abs(values), axis=1)
            scales = self._source_scales[:batch]
            np.maximum(row_max / np.float32(127.0), np.float32(1.0 / 127.0), out=scales)
            self._source_scratch.fill(0)
            self._source_scratch[:batch, :features] = np.rint(values / scales[:, None]).clip(-127, 127)

        prepare_event = cpu_queue.host_task(
            prepare,
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                self._packed_source.access(AccessMode.WRITE),
                self._row_scales.access(AccessMode.WRITE),
            ),
            name="w8a8_linear.hybrid_prepare",
        )
        active_padded_batch = (batch + 15) // 16 * 16
        kernel = W8A8_GEMV_KERNEL if batch == 1 else TILED_W8A8_GEMM_KERNEL
        kernel_source = (
            self._gemv_source
            if batch == 1
            else self._packed_source.slice((slice(0, active_padded_batch), slice(None)))
        )
        kernel_weight = self._packed_weight.slice((slice(None), slice(0, qpu_outputs)))
        kernel_accumulator = (
            self._gemv_accumulator.slice((slice(None), slice(0, qpu_outputs)))
            if batch == 1
            else self._accumulator.slice((slice(0, active_padded_batch), slice(0, qpu_outputs)))
        )
        qpu_destination = destination.slice((slice(None), slice(0, qpu_outputs)))
        cpu_destination = destination.slice((slice(None), slice(qpu_outputs, outputs)))
        active_row_scales = self._row_scales.slice((slice(0, active_padded_batch),))
        active_column_scales = self._column_scales.slice((slice(0, qpu_outputs),))
        fused_qpu = (
            self.qpu_dequantize
            and self.fuse_dequantize
            and supports_tiled_w8a8_gemm_dequantize(
                kernel_source,
                kernel_weight,
                active_row_scales,
                active_column_scales,
                qpu_destination,
                self.device.backend,
            )
        )
        if fused_qpu:
            qpu_finish = qpu_queue.submit(
                TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
                (
                    kernel_source,
                    kernel_weight,
                    active_row_scales,
                    active_column_scales,
                    qpu_destination,
                ),
                grid=(qpu_outputs // 16, active_padded_batch // 16, 1),
                wait_for=(prepare_event,),
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    kernel_weight.access(AccessMode.READ),
                    active_row_scales.access(AccessMode.READ),
                    active_column_scales.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
            )
        else:
            qpu_event = qpu_queue.submit(
                kernel,
                (kernel_source, kernel_weight, kernel_accumulator),
                grid=(qpu_outputs // 16, 1 if batch == 1 else active_padded_batch // 16, 1),
                wait_for=(prepare_event,),
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    kernel_weight.access(AccessMode.READ),
                    kernel_accumulator.access(AccessMode.WRITE),
                ),
            )

            def dequantize_qpu_prefix() -> None:
                active = self._accumulator.numpy()[:batch, :qpu_outputs].astype(np.float32)
                qpu_destination.numpy()[:] = (
                    active * self._source_scales[:batch, None] * self.weight.scales[None, :qpu_outputs]
                )

            qpu_finish = qpu_queue.host_task(
                dequantize_qpu_prefix,
                wait_for=(qpu_event,),
                buffers=(
                    kernel_accumulator.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
                name="w8a8_linear.hybrid_qpu_dequantize",
            )

        def cpu_tail() -> None:
            assert self._hybrid_weight_tail is not None
            quantized_source = self._source_scratch[:batch, :features].astype(np.int32)
            cpu_values = cast(npt.NDArray[np.float32], cpu_destination.numpy())
            accumulation = quantized_source @ self._hybrid_weight_tail.T
            cpu_values[:] = (
                accumulation.astype(np.float32)
                * self._source_scales[:batch, None]
                * self.weight.scales[None, qpu_outputs:]
            )

        cpu_event = cpu_queue.host_task(
            cpu_tail,
            wait_for=(prepare_event,),
            buffers=(cpu_destination.access(AccessMode.WRITE),),
            name="w8a8_linear.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_finish),
            name="w8a8_linear.hybrid_join",
        )
        self._last_event = event
        return event

    def execute_hybrid_rows(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        qpu_queue: Queue,
        cpu_queue: Queue,
        qpu_rows: int,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Run an aligned row prefix on QPU and the disjoint tail on CPU."""
        if self._closed:
            raise RuntimeError("prepared W8A8 linear plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared W8A8 linear plan already has an in-flight invocation")
        if qpu_queue is cpu_queue:
            raise DependencyError("hybrid W8A8 linear requires distinct CPU and QPU queues")
        if qpu_queue.device is not self.device or cpu_queue.device is not self.device:
            raise DependencyError("hybrid W8A8 queues must belong to the plan device")
        if self._queue is None:
            self._queue = qpu_queue
        elif self._queue is not qpu_queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first QPU queue")
        if self._cpu_queue is None:
            self._cpu_queue = cpu_queue
        elif self._cpu_queue is not cpu_queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first CPU queue")
        if source.buffer.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("prepared W8A8 linear tensors must belong to the plan device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("prepared W8A8 linear requires float32 source and destination")
        if len(source.shape) != 2 or len(destination.shape) != 2:
            raise ValueError("prepared W8A8 linear requires rank-2 source and destination")
        batch, features = source.shape
        outputs = self.weight.shape[0]
        if batch > self.max_batch or features != self.weight.shape[1]:
            raise ValueError("prepared W8A8 linear source exceeds the plan shape")
        if destination.shape != (batch, outputs):
            raise ValueError("prepared W8A8 linear destination shape does not align")
        if qpu_rows <= 0 or qpu_rows >= batch or qpu_rows % 16:
            raise ValueError("hybrid W8A8 qpu_rows must be aligned and leave a non-empty CPU tail")
        if self._hybrid_weight_full is None:
            self._hybrid_weight_full = self.weight.values.astype(np.int32)

        def prepare() -> None:
            self._prepare_values(source.numpy().astype(np.float32, copy=False))

        prepare_event = cpu_queue.host_task(
            prepare,
            wait_for=wait_for,
            buffers=(
                source.access(AccessMode.READ),
                self._packed_source.access(AccessMode.WRITE),
                self._row_scales.access(AccessMode.WRITE),
            ),
            name="w8a8_linear.hybrid_prepare",
        )
        kernel_source = self._packed_source.slice((slice(0, qpu_rows), slice(None)))
        kernel_accumulator = self._accumulator.slice((slice(0, qpu_rows), slice(None)))
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(None)))
        cpu_destination = destination.slice((slice(qpu_rows, batch), slice(None)))
        active_row_scales = self._row_scales.slice((slice(0, qpu_rows),))
        fused_qpu = (
            self.qpu_dequantize
            and self.fuse_dequantize
            and supports_tiled_w8a8_gemm_dequantize(
                kernel_source,
                self._packed_weight,
                active_row_scales,
                self._column_scales,
                qpu_destination,
                self.device.backend,
            )
        )
        if fused_qpu:
            qpu_finish = qpu_queue.submit(
                TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
                (
                    kernel_source,
                    self._packed_weight,
                    active_row_scales,
                    self._column_scales,
                    qpu_destination,
                ),
                grid=(self.padded_outputs // 16, qpu_rows // 16, 1),
                wait_for=(prepare_event,),
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    self._packed_weight.access(AccessMode.READ),
                    active_row_scales.access(AccessMode.READ),
                    self._column_scales.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
            )
        else:
            qpu_event = qpu_queue.submit(
                TILED_W8A8_GEMM_KERNEL,
                (kernel_source, self._packed_weight, kernel_accumulator),
                grid=(self.padded_outputs // 16, qpu_rows // 16, 1),
                wait_for=(prepare_event,),
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    self._packed_weight.access(AccessMode.READ),
                    kernel_accumulator.access(AccessMode.WRITE),
                ),
            )

            def dequantize_qpu_prefix() -> None:
                active = self._accumulator.numpy()[:qpu_rows, :outputs].astype(np.float32)
                qpu_destination.numpy()[:] = (
                    active * self._source_scales[:qpu_rows, None] * self.weight.scales[None, :]
                )

            qpu_finish = qpu_queue.host_task(
                dequantize_qpu_prefix,
                wait_for=(qpu_event,),
                buffers=(
                    kernel_accumulator.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
                name="w8a8_linear.hybrid_qpu_dequantize",
            )

        def cpu_tail() -> None:
            assert self._hybrid_weight_full is not None
            quantized_source = self._source_scratch[qpu_rows:batch, :features].astype(np.int32)
            cpu_values = cast(npt.NDArray[np.float32], cpu_destination.numpy())
            accumulation = quantized_source @ self._hybrid_weight_full.T
            cpu_values[:] = (
                accumulation.astype(np.float32)
                * self._source_scales[qpu_rows:batch, None]
                * self.weight.scales[None, :]
            )

        cpu_event = cpu_queue.host_task(
            cpu_tail,
            wait_for=(prepare_event,),
            buffers=(cpu_destination.access(AccessMode.WRITE),),
            name="w8a8_linear.hybrid_cpu_tail",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_finish),
            name="w8a8_linear.hybrid_join",
        )
        self._last_event = event
        return event

    def execute_prepared_hybrid_rows(
        self: Self,
        destination: Tensor,
        *,
        batch: int,
        qpu_queue: Queue,
        cpu_queue: Queue,
        qpu_rows: int,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Partition rows after an enclosing operation has packed activations."""
        return self._execute_prepared_hybrid(
            destination,
            batch=batch,
            qpu_queue=qpu_queue,
            cpu_queue=cpu_queue,
            axis="rows",
            qpu_units=qpu_rows,
            wait_for=wait_for,
        )

    def execute_prepared_hybrid_outputs(
        self: Self,
        destination: Tensor,
        *,
        batch: int,
        qpu_queue: Queue,
        cpu_queue: Queue,
        qpu_outputs: int,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Partition outputs after an enclosing operation has packed activations."""
        return self._execute_prepared_hybrid(
            destination,
            batch=batch,
            qpu_queue=qpu_queue,
            cpu_queue=cpu_queue,
            axis="outputs",
            qpu_units=qpu_outputs,
            wait_for=wait_for,
        )

    def _execute_prepared_hybrid(
        self: Self,
        destination: Tensor,
        *,
        batch: int,
        qpu_queue: Queue,
        cpu_queue: Queue,
        axis: str,
        qpu_units: int,
        wait_for: Iterable[Event],
    ) -> Event:
        """Submit one hybrid split using activation bytes already in plan scratch."""
        if self._closed:
            raise RuntimeError("prepared W8A8 linear plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("prepared W8A8 linear plan already has an in-flight invocation")
        if qpu_queue is cpu_queue:
            raise DependencyError("hybrid W8A8 linear requires distinct CPU and QPU queues")
        if qpu_queue.device is not self.device or cpu_queue.device is not self.device:
            raise DependencyError("hybrid W8A8 queues must belong to the plan device")
        if self._queue is None:
            self._queue = qpu_queue
        elif self._queue is not qpu_queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first QPU queue")
        if self._cpu_queue is None:
            self._cpu_queue = cpu_queue
        elif self._cpu_queue is not cpu_queue:
            raise DependencyError("prepared W8A8 linear plan is bound to its first CPU queue")
        outputs, features = self.weight.shape
        if destination.buffer.device is not self.device:
            raise DependencyError("prepared W8A8 linear destination must belong to the plan device")
        if destination.dtype != np.dtype(np.float32) or destination.shape != (batch, outputs):
            raise ValueError("prepared W8A8 linear destination shape does not align")
        if batch <= 0 or batch > self.max_batch:
            raise ValueError("prepared W8A8 linear batch exceeds the plan shape")
        if axis == "rows":
            if qpu_units <= 0 or qpu_units >= batch or qpu_units % 16:
                raise ValueError("hybrid W8A8 qpu_rows must be aligned and leave a non-empty CPU tail")
            qpu_rows = qpu_units
            qpu_outputs = outputs
        elif axis == "outputs":
            if qpu_units <= 0 or qpu_units >= outputs or qpu_units % 16:
                raise ValueError("hybrid W8A8 qpu_outputs must be aligned and leave a non-empty CPU tail")
            qpu_rows = batch
            qpu_outputs = qpu_units
        else:
            raise ValueError("prepared W8A8 hybrid axis must be rows or outputs")

        active_padded_batch = (qpu_rows + 15) // 16 * 16
        kernel = W8A8_GEMV_KERNEL if qpu_rows == 1 else TILED_W8A8_GEMM_KERNEL
        kernel_source = (
            self._gemv_source
            if qpu_rows == 1
            else self._packed_source.slice((slice(0, active_padded_batch), slice(None)))
        )
        kernel_weight = self._packed_weight.slice((slice(None), slice(0, qpu_outputs)))
        kernel_accumulator = (
            self._gemv_accumulator.slice((slice(None), slice(0, qpu_outputs)))
            if qpu_rows == 1
            else self._accumulator.slice((slice(0, active_padded_batch), slice(0, qpu_outputs)))
        )
        qpu_destination = destination.slice((slice(0, qpu_rows), slice(0, qpu_outputs)))
        active_row_scales = self._row_scales.slice((slice(0, active_padded_batch),))
        active_column_scales = self._column_scales.slice((slice(0, qpu_outputs),))
        fused_qpu = (
            self.qpu_dequantize
            and self.fuse_dequantize
            and supports_tiled_w8a8_gemm_dequantize(
                kernel_source,
                kernel_weight,
                active_row_scales,
                active_column_scales,
                qpu_destination,
                self.device.backend,
            )
        )
        if fused_qpu:
            qpu_finish = qpu_queue.submit(
                TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
                (
                    kernel_source,
                    kernel_weight,
                    active_row_scales,
                    active_column_scales,
                    qpu_destination,
                ),
                grid=(qpu_outputs // 16, active_padded_batch // 16, 1),
                wait_for=wait_for,
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    kernel_weight.access(AccessMode.READ),
                    active_row_scales.access(AccessMode.READ),
                    active_column_scales.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
            )
        else:
            qpu_event = qpu_queue.submit(
                kernel,
                (kernel_source, kernel_weight, kernel_accumulator),
                grid=(qpu_outputs // 16, 1 if qpu_rows == 1 else active_padded_batch // 16, 1),
                wait_for=wait_for,
                buffers=(
                    kernel_source.access(AccessMode.READ),
                    kernel_weight.access(AccessMode.READ),
                    kernel_accumulator.access(AccessMode.WRITE),
                ),
            )

            def dequantize_qpu_partition() -> None:
                active = self._accumulator.numpy()[:qpu_rows, :qpu_outputs].astype(np.float32)
                qpu_destination.numpy()[:] = (
                    active * self._source_scales[:qpu_rows, None] * self.weight.scales[None, :qpu_outputs]
                )

            qpu_finish = qpu_queue.host_task(
                dequantize_qpu_partition,
                wait_for=(qpu_event,),
                buffers=(
                    kernel_accumulator.access(AccessMode.READ),
                    qpu_destination.access(AccessMode.WRITE),
                ),
                name="w8a8_linear.hybrid_qpu_dequantize",
            )

        if axis == "rows":
            if self._hybrid_weight_full is None:
                self._hybrid_weight_full = self.weight.values.astype(np.int32)
            cpu_destination = destination.slice((slice(qpu_rows, batch), slice(None)))

            def cpu_partition() -> None:
                assert self._hybrid_weight_full is not None
                quantized_source = self._source_scratch[qpu_rows:batch, :features].astype(np.int32)
                cpu_values = cast(npt.NDArray[np.float32], cpu_destination.numpy())
                accumulation = quantized_source @ self._hybrid_weight_full.T
                cpu_values[:] = (
                    accumulation.astype(np.float32)
                    * self._source_scales[qpu_rows:batch, None]
                    * self.weight.scales[None, :]
                )

        else:
            if self._hybrid_weight_key != qpu_outputs:
                self._hybrid_weight_tail = self.weight.values[qpu_outputs:].astype(np.int32)
                self._hybrid_weight_key = qpu_outputs
            cpu_destination = destination.slice((slice(None), slice(qpu_outputs, outputs)))

            def cpu_partition() -> None:
                assert self._hybrid_weight_tail is not None
                quantized_source = self._source_scratch[:batch, :features].astype(np.int32)
                cpu_values = cast(npt.NDArray[np.float32], cpu_destination.numpy())
                accumulation = quantized_source @ self._hybrid_weight_tail.T
                cpu_values[:] = (
                    accumulation.astype(np.float32)
                    * self._source_scales[:batch, None]
                    * self.weight.scales[None, qpu_outputs:]
                )

        cpu_event = cpu_queue.host_task(
            cpu_partition,
            wait_for=wait_for,
            buffers=(cpu_destination.access(AccessMode.WRITE),),
            name="w8a8_linear.hybrid_cpu_partition",
        )
        event = cpu_queue.host_task(
            lambda: None,
            wait_for=(cpu_event, qpu_finish),
            name="w8a8_linear.hybrid_join",
        )
        self._last_event = event
        return event


class CalibratedW8A8Linear:
    """Dispatch only measured QPU winners; otherwise use the CPU reference."""

    def __init__(
        self: Self,
        device: Device,
        weight: QuantizedMatrixInt8,
        *,
        max_batch: int,
        candidates: CandidateRegistry,
    ) -> None:
        """Retain quantized weights and prepare optional hardware state."""
        if max_batch <= 0:
            raise ValueError("calibrated W8A8 linear max_batch must be positive")
        self.device = device
        self.weight = weight
        self.max_batch = max_batch
        self.candidates = candidates
        has_supported_shape = any(
            record.placement in {"qpu", "hybrid"}
            for batch in range(1, max_batch + 1)
            for record in self.candidates.supported_for(
                operation="linear",
                dtype="w8a8-i32-fp32",
                layout="row-major-packed-k4",
                shape_class=f"{batch}x{weight.shape[1]}x{weight.shape[0]}",
            )
        )
        self._qpu = (
            PreparedW8A8Linear(device, weight, max_batch=max_batch, qpu_dequantize=True)
            if isinstance(device.backend, PyVideoCore7Backend) and has_supported_shape
            else None
        )
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter a context-managed dispatcher lifetime."""
        if self._closed:
            raise RuntimeError("calibrated W8A8 linear is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release optional packed QPU state."""
        self.close()

    def close(self: Self) -> None:
        """Release the optional packed implementation."""
        if self._closed:
            return
        if self._qpu is not None:
            self._qpu.close()
        self._closed = True

    def _supported_records(
        self: Self,
        batch: int,
        placement: Placement | None = None,
    ) -> tuple[CandidateRecord, ...]:
        """Return exact-shape winners, optionally restricted by placement."""
        shape_class = f"{batch}x{self.weight.shape[1]}x{self.weight.shape[0]}"
        records = self.candidates.supported_for(
            operation="linear",
            dtype="w8a8-i32-fp32",
            layout="row-major-packed-k4",
            shape_class=shape_class,
        )
        if placement is None:
            return records
        return tuple(record for record in records if record.placement == placement.value)

    def execute(
        self: Self,
        destination: Tensor,
        source: Tensor,
        *,
        queue: Queue,
        cpu_queue: Queue | None = None,
        placement: Placement = Placement.AUTO,
        qpu_rows: int | None = None,
        qpu_outputs: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Run an exact-shape supported winner or the dynamic INT8 CPU reference."""
        if self._closed:
            raise RuntimeError("calibrated W8A8 linear is closed")
        if queue.device is not self.device:
            raise DependencyError("calibrated W8A8 queue belongs to a different device")
        if source.buffer.device is not self.device or destination.buffer.device is not self.device:
            raise DependencyError("calibrated W8A8 tensors belong to a different device")
        if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("calibrated W8A8 linear requires FP32 source and destination")
        if len(source.shape) != 2 or len(destination.shape) != 2:
            raise ValueError("calibrated W8A8 linear requires rank-2 tensors")
        batch, features = source.shape
        if batch > self.max_batch or features != self.weight.shape[1]:
            raise ValueError("calibrated W8A8 source exceeds its planned shape")
        if destination.shape != (batch, self.weight.shape[0]):
            raise ValueError("calibrated W8A8 destination shape does not align")
        if qpu_rows is not None and qpu_outputs is not None:
            raise ValueError("qpu_rows and qpu_outputs are mutually exclusive")
        if placement in {Placement.CPU, Placement.QPU} and (qpu_rows is not None or qpu_outputs is not None):
            raise ValueError("hybrid partitions require HYBRID or AUTO placement")
        qpu_records = self._supported_records(batch, Placement.QPU)
        hybrid_records = self._supported_records(batch, Placement.HYBRID)
        matching_hybrid_records = tuple(
            record
            for record in hybrid_records
            if record.partition is not None
            and (
                (qpu_rows is None and qpu_outputs is None)
                or (
                    qpu_rows is not None and record.partition.axis == "rows" and record.partition.qpu_units == qpu_rows
                )
                or (
                    qpu_outputs is not None
                    and record.partition.axis == "outputs"
                    and record.partition.qpu_units == qpu_outputs
                )
            )
        )
        qpu_supported = bool(qpu_records) and self._qpu is not None
        hybrid_supported = bool(matching_hybrid_records) and self._qpu is not None and cpu_queue is not None
        if placement is Placement.QPU and not qpu_supported:
            raise ValueError("no measured supported-win W8A8 QPU candidate exists for this exact shape")
        if placement is Placement.HYBRID and not hybrid_supported:
            raise ValueError("no executable supported-win W8A8 hybrid candidate exists for this exact shape")
        if placement is Placement.AUTO and (qpu_rows is not None or qpu_outputs is not None) and not hybrid_supported:
            raise ValueError("no supported W8A8 hybrid partition matches the requested axis and size")

        selected = placement
        selected_record: CandidateRecord | None = None
        if placement is Placement.AUTO:
            auto_qpu_records = () if qpu_rows is not None or qpu_outputs is not None else qpu_records
            executable = [
                record
                for record in (*auto_qpu_records, *matching_hybrid_records)
                if record.performance is not None
                and (record.placement != "hybrid" or hybrid_supported)
                and self._qpu is not None
            ]
            if executable:
                selected_record = min(
                    executable,
                    key=lambda record: (
                        record.performance.candidate_median_seconds if record.performance is not None else float("inf")
                    ),
                )
                selected = Placement(selected_record.placement)
            else:
                selected = Placement.CPU
        if selected is Placement.QPU:
            assert self._qpu is not None
            return self._qpu.execute(destination, source, queue=queue, wait_for=wait_for)
        if selected is Placement.HYBRID:
            assert self._qpu is not None and cpu_queue is not None
            if not matching_hybrid_records:
                raise ValueError("no supported W8A8 hybrid partition matches the requested axis and size")
            record = selected_record or min(
                matching_hybrid_records,
                key=lambda candidate: (
                    candidate.performance.candidate_median_seconds
                    if candidate.performance is not None
                    else float("inf")
                ),
            )
            assert record.partition is not None
            if record.partition.axis == "rows":
                return self._qpu.execute_hybrid_rows(
                    destination,
                    source,
                    qpu_queue=queue,
                    cpu_queue=cpu_queue,
                    qpu_rows=record.partition.qpu_units,
                    wait_for=wait_for,
                )
            return self._qpu.execute_hybrid(
                destination,
                source,
                qpu_queue=queue,
                cpu_queue=cpu_queue,
                qpu_outputs=record.partition.qpu_units,
                wait_for=wait_for,
            )

        def cpu_reference() -> None:
            values = cast(npt.NDArray[np.float32], source.numpy())
            source_scales = np.maximum(
                np.max(np.abs(values), axis=1) / np.float32(127.0),
                np.float32(1.0 / 127.0),
            )
            quantized_source = np.rint(values / source_scales[:, None]).clip(-127, 127).astype(np.int32)
            accumulation = quantized_source @ self.weight.values.astype(np.int32).T
            destination.numpy()[:] = (
                accumulation.astype(np.float32) * source_scales[:, None] * self.weight.scales[None, :]
            )

        return queue.host_task(
            cpu_reference,
            wait_for=wait_for,
            buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="w8a8_linear.cpu_reference",
        )


def quantize_per_output_channel_int8(weight: npt.NDArray[np.float32]) -> QuantizedMatrixInt8:
    """Symmetrically quantize FP32 ``(output, input)`` weights to INT8 rows."""
    if weight.ndim != 2 or weight.dtype != np.dtype(np.float32):
        raise ValueError("quantize_per_output_channel_int8 requires a rank-2 float32 matrix")
    max_abs = np.max(np.abs(weight), axis=1)
    scales = np.maximum(max_abs / np.float32(127.0), np.float32(1.0 / 127.0)).astype(np.float32)
    values = np.rint(weight / scales[:, None]).clip(-127, 127).astype(np.int8)
    return QuantizedMatrixInt8(np.ascontiguousarray(values), np.ascontiguousarray(scales))


def quantized_linear_fp32(
    destination: Tensor,
    source: Tensor,
    weight: QuantizedMatrixInt8,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Apply a packed INT8 weight matrix through the CPU FP32 reference path.

    This establishes model-loader and numerical contracts while a dedicated
    quantized QPU projection kernel is still pending. Weights are dequantized
    once at submission time into a host closure, not per output element.
    """
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("quantized linear source and destination must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("quantized_linear_fp32 requires float32 source and destination tensors")
    if len(source.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("quantized_linear_fp32 requires rank-2 source and destination tensors")
    batch, in_features = source.shape
    out_features, weight_features = weight.shape
    if in_features != weight_features or destination.shape != (batch, out_features):
        raise ValueError("quantized linear tensor shapes do not align")
    dequantized = weight.dequantize()
    selected_queue = destination.buffer.device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not destination.buffer.device:
        raise DependencyError("queue and quantized linear tensors must belong to the same device")

    def apply() -> None:
        np.matmul(source.numpy(), dequantized.T, out=destination.numpy())

    event = selected_queue.host_task(
        apply,
        wait_for=wait_for,
        buffers=(source.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="tinyllama.quantized_linear_fp32",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event


def quantized_linear_int8_gemm_fp32(
    destination: Tensor,
    source: Tensor,
    weight: QuantizedMatrixInt8,
    *,
    queue: Queue | None = None,
    wait_for: Iterable[Event] = (),
) -> Event:
    """Project FP32 activations through padded INT32 GEMM with INT8-range operands.

    Source rows are dynamically symmetrically quantized and weight INT8 rows
    are widened to INT32. The padded matrix multiplication is eligible for the
    existing ``smul24`` QPU kernel, while fake/CPU backends use the same NumPy
    fallback through :func:`qpu_xla.ops.matmul.matmul`.
    """
    if source.buffer.device is not destination.buffer.device:
        raise DependencyError("quantized linear source and destination must belong to the same device")
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise ValueError("quantized_linear_int8_gemm_fp32 requires float32 source and destination tensors")
    if len(source.shape) != 2 or len(destination.shape) != 2:
        raise ValueError("quantized linear requires rank-2 source and destination tensors")
    batch, in_features = source.shape
    out_features, weight_features = weight.shape
    if in_features != weight_features or destination.shape != (batch, out_features):
        raise ValueError("quantized linear tensor shapes do not align")
    if in_features * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError("quantized linear INT32 accumulation can overflow")
    padded_batch = (batch + 15) // 16 * 16
    padded_features = (in_features + 3) // 4 * 4
    padded_outputs = (out_features + 15) // 16 * 16
    device = destination.buffer.device
    quantized_source = device.tensor((padded_batch, padded_features), np.int32)
    quantized_weight = device.tensor((padded_features, padded_outputs), np.int32)
    accumulator = device.tensor((padded_batch, padded_outputs), np.int32)
    source_scales = np.empty((batch,), dtype=np.float32)
    selected_queue = device.queue() if queue is None else queue
    close_queue = queue is None
    if selected_queue.device is not device:
        raise DependencyError("queue and quantized linear tensors must belong to the same device")

    def prepare() -> None:
        values = source.numpy().astype(np.float32, copy=False)
        row_max = np.max(np.abs(values), axis=1)
        np.maximum(row_max / np.float32(127.0), np.float32(1.0 / 127.0), out=source_scales)
        quantized_source.numpy().fill(0)
        quantized_weight.numpy().fill(0)
        quantized_source.numpy()[:batch, :in_features] = np.rint(values / source_scales[:, None]).clip(-127, 127)
        quantized_weight.numpy()[:in_features, :out_features] = weight.values.T

    prepare_event = selected_queue.host_task(
        prepare,
        wait_for=wait_for,
        buffers=(
            source.access(AccessMode.READ),
            quantized_source.access(AccessMode.WRITE),
            quantized_weight.access(AccessMode.WRITE),
        ),
        name="tinyllama.quantized_linear_int8.prepare",
    )
    gemm_event = matmul(
        accumulator,
        quantized_source,
        quantized_weight,
        queue=selected_queue,
        wait_for=(prepare_event,),
    )

    def dequantize() -> None:
        active = accumulator.numpy()[:batch, :out_features].astype(np.float32)
        destination.numpy()[:] = active * source_scales[:, None] * weight.scales[None, :]

    event = selected_queue.host_task(
        dequantize,
        wait_for=(gemm_event,),
        buffers=(accumulator.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
        name="tinyllama.quantized_linear_int8.dequantize",
    )
    if close_queue:
        event.wait()
        selected_queue.close()
    return event
