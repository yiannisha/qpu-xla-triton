"""Persistent grouped-query multi-head FP32 attention plan."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np
import numpy.typing as npt

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.attention import _round_up
from qpu_xla.ops.matmul import matmul
from qpu_xla.ops.softmax import softmax_fp32
from qpu_xla.queue import Event, Queue
from qpu_xla.scheduler import Placement


class PreparedMultiHeadSDPAFP32:
    """Reuse one padded QPU head workspace across a fixed GQA topology.

    Explicit hybrid placement assigns a non-empty prefix of query heads to the
    QPU queue while a separate CPU queue computes the remaining heads.  Both
    partitions start from the same dependencies and write disjoint head slices,
    so this is concurrent heterogeneous execution rather than sequential
    fallback.
    """

    def __init__(
        self: Self,
        device: Device,
        *,
        query_length: int,
        key_length: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
    ) -> None:
        """Allocate reusable per-head SDPA scratch for a fixed topology."""
        sizes = (query_length, key_length, query_heads, key_value_heads, head_dim)
        if any(value <= 0 for value in sizes):
            raise ValueError("multi-head SDPA dimensions must be positive")
        if query_heads % key_value_heads:
            raise ValueError("multi-head SDPA query heads must be divisible by key/value heads")
        if head_dim % 16:
            raise ValueError("multi-head SDPA head_dim must be 16-aligned")
        self.device = device
        self.query_length = query_length
        self.key_length = key_length
        self.query_heads = query_heads
        self.key_value_heads = key_value_heads
        self.head_dim = head_dim
        self._groups = query_heads // key_value_heads
        self._padded_query = _round_up(query_length, 16)
        self._padded_key = _round_up(key_length, 16)
        self._query = device.tensor((self._padded_query, head_dim), np.float32)
        self._key_t = device.tensor((head_dim, self._padded_key), np.float32)
        self._value = device.tensor((self._padded_key, head_dim), np.float32)
        self._scores = device.tensor((self._padded_query, self._padded_key), np.float32)
        self._result = device.tensor((self._padded_query, head_dim), np.float32)
        self._last_event: Event | None = None
        self._closed = False

    def __enter__(self: Self) -> Self:
        """Enter the prepared attention-plan lifetime."""
        if self._closed:
            raise RuntimeError("multi-head SDPA plan is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release all plan-owned scratch."""
        self.close()

    def close(self: Self) -> None:
        """Wait for work and release persistent padded head buffers."""
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.wait()
        for tensor in (self._query, self._key_t, self._value, self._scores, self._result):
            tensor.buffer.close()
        self._closed = True

    def _validate(
        self: Self,
        destination: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: npt.NDArray[np.bool_] | None,
        queue: Queue,
    ) -> None:
        if self._closed:
            raise RuntimeError("multi-head SDPA plan is closed")
        if self._last_event is not None and not self._last_event.done:
            raise RuntimeError("multi-head SDPA plan already has an in-flight invocation")
        tensors = (query, key, value, destination)
        if queue.device is not self.device or any(tensor.buffer.device is not self.device for tensor in tensors):
            raise DependencyError("multi-head SDPA tensors and queue must belong to its device")
        if any(tensor.dtype != np.dtype(np.float32) for tensor in tensors):
            raise ValueError("multi-head SDPA requires FP32 tensors")
        if query.shape != (self.query_length, self.query_heads, self.head_dim):
            raise ValueError("multi-head SDPA query shape does not match its plan")
        if key.shape != (self.key_length, self.key_value_heads, self.head_dim) or value.shape != key.shape:
            raise ValueError("multi-head SDPA key/value shape does not match its plan")
        if destination.shape != query.shape:
            raise ValueError("multi-head SDPA destination must match the query shape")
        if mask is not None and (
            mask.dtype != np.dtype(np.bool_) or mask.shape != (self.query_length, self.key_length)
        ):
            raise ValueError("multi-head SDPA mask must be boolean query_length by key_length")

    def _cpu_heads(
        self: Self,
        destination: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: npt.NDArray[np.bool_] | None,
        start_head: int,
        stop_head: int,
    ) -> None:
        output = destination.numpy()
        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        for head in range(start_head, stop_head):
            key_head = head // self._groups
            scores = np.matmul(query.numpy()[:, head], key.numpy()[:, key_head].T, dtype=np.float32)
            scores *= scale
            if mask is not None:
                scores[:] = np.where(mask, scores, np.finfo(np.float32).min)
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True, dtype=np.float32)
            output[:, head] = np.matmul(scores, value.numpy()[:, key_head], dtype=np.float32)

    def _qpu_heads(
        self: Self,
        destination: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: npt.NDArray[np.bool_] | None,
        stop_head: int,
        queue: Queue,
        wait_for: tuple[Event, ...],
    ) -> Event:
        dependency = wait_for
        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        for head in range(stop_head):
            key_head = head // self._groups

            def prepare(head: int = head, key_head: int = key_head) -> None:
                self._query.numpy().fill(0.0)
                self._key_t.numpy().fill(0.0)
                self._value.numpy().fill(0.0)
                self._query.numpy()[: self.query_length] = query.numpy()[:, head]
                self._key_t.numpy()[:, : self.key_length] = key.numpy()[:, key_head].T
                self._value.numpy()[: self.key_length] = value.numpy()[:, key_head]

            prepare_event = queue.host_task(
                prepare,
                wait_for=dependency,
                buffers=(
                    query.access(AccessMode.READ),
                    key.access(AccessMode.READ),
                    value.access(AccessMode.READ),
                    self._query.access(AccessMode.WRITE),
                    self._key_t.access(AccessMode.WRITE),
                    self._value.access(AccessMode.WRITE),
                ),
                name="multihead_sdpa_fp32.prepare_head",
            )
            score_event = matmul(
                self._scores,
                self._query,
                self._key_t,
                queue=queue,
                placement=Placement.QPU,
                wait_for=(prepare_event,),
            )

            def prepare_softmax() -> None:
                active = self._scores.numpy()[: self.query_length, : self.key_length]
                np.multiply(active, scale, out=active)
                if mask is not None:
                    active[:] = np.where(mask, active, np.finfo(np.float32).min)
                self._scores.numpy()[: self.query_length, self.key_length :].fill(np.finfo(np.float32).min)
                self._scores.numpy()[self.query_length :, :].fill(0.0)

            mask_event = queue.host_task(
                prepare_softmax,
                wait_for=(score_event,),
                buffers=(self._scores.access(AccessMode.READ_WRITE),),
                name="multihead_sdpa_fp32.scale_mask",
            )
            softmax_event = softmax_fp32(
                self._scores,
                self._scores,
                queue=queue,
                placement=Placement.QPU,
                wait_for=(mask_event,),
            )
            value_event = matmul(
                self._result,
                self._scores,
                self._value,
                queue=queue,
                placement=Placement.QPU,
                wait_for=(softmax_event,),
            )

            def finish(head: int = head) -> None:
                destination.numpy()[:, head] = self._result.numpy()[: self.query_length]

            finish_event = queue.host_task(
                finish,
                wait_for=(value_event,),
                buffers=(self._result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
                name="multihead_sdpa_fp32.finish_head",
            )
            dependency = (finish_event,)
        return dependency[0]

    def execute(
        self: Self,
        destination: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        mask: npt.NDArray[np.bool_] | None,
        queue: Queue,
        cpu_queue: Queue | None = None,
        placement: Placement = Placement.CPU,
        qpu_heads: int | None = None,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Run fixed-shape GQA with explicit CPU, QPU, or head-split placement."""
        self._validate(destination, query, key, value, mask, queue)
        dependencies = tuple(wait_for)
        accesses = (
            query.access(AccessMode.READ),
            key.access(AccessMode.READ),
            value.access(AccessMode.READ),
            destination.access(AccessMode.WRITE),
        )
        if placement is Placement.CPU:
            event = queue.host_task(
                lambda: self._cpu_heads(destination, query, key, value, mask, 0, self.query_heads),
                wait_for=dependencies,
                buffers=accesses,
                name="multihead_sdpa_fp32.cpu",
            )
        elif placement is Placement.QPU:
            event = self._qpu_heads(
                destination,
                query,
                key,
                value,
                mask,
                self.query_heads,
                queue,
                dependencies,
            )
        elif placement is Placement.HYBRID:
            if cpu_queue is None or cpu_queue is queue or cpu_queue.device is not self.device:
                raise DependencyError("hybrid multi-head SDPA requires distinct CPU and QPU queues")
            if qpu_heads is None or qpu_heads <= 0 or qpu_heads >= self.query_heads:
                raise ValueError("hybrid multi-head SDPA qpu_heads must leave two non-empty head partitions")
            qpu_event = self._qpu_heads(
                destination,
                query,
                key,
                value,
                mask,
                qpu_heads,
                queue,
                dependencies,
            )
            cpu_event = cpu_queue.host_task(
                lambda: self._cpu_heads(destination, query, key, value, mask, qpu_heads, self.query_heads),
                wait_for=dependencies,
                buffers=accesses,
                name="multihead_sdpa_fp32.hybrid_cpu_heads",
            )
            event = cpu_queue.host_task(
                lambda: None,
                wait_for=(qpu_event, cpu_event),
                name="multihead_sdpa_fp32.hybrid_join",
            )
        else:
            raise ValueError("multi-head SDPA AUTO requires exact-shape runtime benchmark evidence")
        self._last_event = event
        return event


__all__ = ["PreparedMultiHeadSDPAFP32"]
