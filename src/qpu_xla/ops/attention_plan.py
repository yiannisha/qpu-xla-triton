"""Persistent unnormalized INT32 attention plan built from packaged GEMM stages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError, DeviceClosedError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.attention import _round_up
from qpu_xla.ops.matmul import matmul
from qpu_xla.queue import Event, Queue

_SMUL24_LIMIT = 1 << 23


class AttentionInt32Plan:
    """Reuse fixed key/value preparation and GEMM workspaces across queries.

    The fixed topology evaluates unnormalized ``(query @ key.T) @ value``.
    It deliberately does not add scale, masking, or softmax.  Key transposition
    and padded value preparation happen only in :meth:`load_key_value`; each
    execution only uploads a new query, launches two GEMM stages, and copies
    the unpadded result to the caller's destination.
    """

    def __init__(
        self: Self,
        device: Device,
        *,
        query_shape: tuple[int, int],
        key_shape: tuple[int, int],
        value_shape: tuple[int, int],
    ) -> None:
        """Allocate persistent tensors for one fixed attention topology."""
        if any(dimension <= 0 for dimension in query_shape + key_shape + value_shape):
            raise ValueError("attention plan dimensions must be positive")
        query_len, depth = query_shape
        key_len, key_depth = key_shape
        value_rows, value_dim = value_shape
        if key_depth != depth or value_rows != key_len:
            raise ValueError("attention plan query, key, and value dimensions do not align")
        self._device = device
        self.query_shape = query_shape
        self.key_shape = key_shape
        self.value_shape = value_shape
        self.destination_shape = (query_len, value_dim)
        self._query_len = query_len
        self._key_len = key_len
        self._depth = depth
        self._value_dim = value_dim
        padded_query = _round_up(query_len, 16)
        padded_key = _round_up(key_len, 16)
        padded_depth = _round_up(depth, 4)
        padded_value = _round_up(value_dim, 16)
        self._query = device.tensor((padded_query, padded_depth), np.int32)
        self._key_t = device.tensor((padded_depth, padded_key), np.int32)
        self._value = device.tensor((padded_key, padded_value), np.int32)
        self._scores = device.tensor((padded_query, padded_key), np.int32)
        self._result = device.tensor((padded_query, padded_value), np.int32)
        self._closed = False
        self._key_value_event: Event | None = None
        self._tail: Event | None = None
        self._max_key: int | None = None
        self._max_value: int | None = None

    @property
    def device(self: Self) -> Device:
        """Return the device that owns all persistent attention tensors."""
        return self._device

    def _require_open(self: Self) -> None:
        """Reject plan use after its reusable buffers have been closed."""
        if self._closed:
            raise DeviceClosedError("attention plan is closed")

    def _queue(self: Self, queue: Queue) -> None:
        """Require a queue bound to the plan's device."""
        self._require_open()
        if queue.device is not self._device:
            raise DependencyError("attention plan queue belongs to a different device")

    def _dependencies(self: Self, wait_for: Iterable[Event]) -> tuple[Event, ...]:
        """Serialize shared workspace reuse after caller-specified dependencies."""
        dependencies = list(wait_for)
        if self._tail is not None and self._tail not in dependencies:
            dependencies.append(self._tail)
        return tuple(dependencies)

    def load_key_value(
        self: Self,
        key: Tensor,
        value: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Transpose/pad and retain the fixed INT32 key and value matrices."""
        self._queue(queue)
        if key.buffer.device is not self._device or value.buffer.device is not self._device:
            raise DependencyError("attention plan key and value must belong to the plan device")
        if key.dtype != np.dtype(np.int32) or value.dtype != np.dtype(np.int32):
            raise ValueError("attention plan key and value must use int32")
        if key.shape != self.key_shape or value.shape != self.value_shape:
            raise ValueError("attention plan key or value shape does not match its topology")

        def prepare_key_value() -> None:
            self._key_t.numpy().fill(0)
            self._value.numpy().fill(0)
            self._key_t.numpy()[: self._depth, : self._key_len] = key.numpy().T
            self._value.numpy()[: self._key_len, : self._value_dim] = value.numpy()
            key_values = key.numpy().astype(np.int64, copy=False)
            value_values = value.numpy().astype(np.int64, copy=False)
            self._max_key = int(np.max(np.abs(key_values), initial=0))
            self._max_value = int(np.max(np.abs(value_values), initial=0))

        event = queue.host_task(
            prepare_key_value,
            wait_for=self._dependencies(wait_for),
            buffers=(
                key.access(AccessMode.READ),
                value.access(AccessMode.READ),
                self._key_t.access(AccessMode.WRITE),
                self._value.access(AccessMode.WRITE),
            ),
            name="attention_int32_plan.load_key_value",
        )
        self._key_value_event = event
        self._tail = event
        return event

    def execute(
        self: Self,
        destination: Tensor,
        query: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Run attention for one query matrix against the retained key/value pair."""
        self._queue(queue)
        if self._key_value_event is None:
            raise DependencyError("attention plan key and value must be loaded before execution")
        if query.buffer.device is not self._device or destination.buffer.device is not self._device:
            raise DependencyError("attention plan query and destination must belong to the plan device")
        if query.dtype != np.dtype(np.int32) or destination.dtype != np.dtype(np.int32):
            raise ValueError("attention plan query and destination must use int32")
        if query.shape != self.query_shape or destination.shape != self.destination_shape:
            raise ValueError("attention plan query or destination shape does not match its topology")

        def validate_query_range() -> None:
            assert self._max_key is not None
            assert self._max_value is not None
            query_values = query.numpy().astype(np.int64, copy=False)
            max_query = int(np.max(np.abs(query_values), initial=0))
            if max(max_query, self._max_key, self._max_value) >= _SMUL24_LIMIT:
                raise ValueError("attention_int32 uses smul24; query, key, and value must fit the signed 24-bit range")
            score_bound = self._depth * max_query * self._max_key
            if score_bound >= _SMUL24_LIMIT:
                raise ValueError(
                    "attention score values may exceed the signed 24-bit range required by the value stage"
                )
            output_bound = self._key_len * score_bound * self._max_value
            if output_bound > np.iinfo(np.int32).max:
                raise ValueError("attention output may exceed the signed int32 accumulation range")

        validation_event = queue.host_task(
            validate_query_range,
            wait_for=self._dependencies(wait_for),
            buffers=(query.access(AccessMode.READ),),
            name="attention_int32_plan.validate_range",
        )

        def prepare_query() -> None:
            self._query.numpy().fill(0)
            self._query.numpy()[: self._query_len, : self._depth] = query.numpy()

        query_event = queue.host_task(
            prepare_query,
            wait_for=(validation_event,),
            buffers=(query.access(AccessMode.READ), self._query.access(AccessMode.WRITE)),
            name="attention_int32_plan.prepare_query",
        )
        score_event = matmul(self._scores, self._query, self._key_t, queue=queue, wait_for=(query_event,))
        value_event = matmul(self._result, self._scores, self._value, queue=queue, wait_for=(score_event,))

        def finish() -> None:
            destination.numpy()[:] = self._result.numpy()[: self._query_len, : self._value_dim]

        event = queue.host_task(
            finish,
            wait_for=(value_event,),
            buffers=(self._result.access(AccessMode.READ), destination.access(AccessMode.WRITE)),
            name="attention_int32_plan.finish",
        )
        self._tail = event
        return event

    def close(self: Self) -> None:
        """Invalidate plan-owned padded matrices and workspaces."""
        if self._closed:
            return
        for tensor in (self._query, self._key_t, self._value, self._scores, self._result):
            tensor.buffer.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter a context-managed persistent attention plan lifetime."""
        self._require_open()
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close plan-owned buffers at context exit."""
        self.close()
