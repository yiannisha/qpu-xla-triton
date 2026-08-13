"""Fixed-capacity FP32 key/value cache for staged transformer attention."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np

from qpu_xla.device import Device
from qpu_xla.errors import DependencyError, DeviceClosedError
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.ops.sdpa import scaled_dot_product_attention_fp32
from qpu_xla.queue import Event, Queue


class KvCacheFp32:
    """Append-only device-resident FP32 key/value cache for one attention head.

    This is the model-runtime building block, not a complete TinyLlama loader.
    Keys and values retain a fixed capacity on the device. Appends and reads
    are event-chained on the supplied queue, allowing a decode query to attend
    to all committed cache positions with the correct causal offset.
    """

    def __init__(self: Self, device: Device, *, capacity: int, depth: int, value_dim: int) -> None:
        """Allocate a fixed-capacity cache for one key/value stream."""
        if min(capacity, depth, value_dim) <= 0:
            raise ValueError("KV cache capacity, depth, and value_dim must be positive")
        self._device = device
        self.capacity = capacity
        self.depth = depth
        self.value_dim = value_dim
        self._key = device.tensor((capacity, depth), np.float32)
        self._value = device.tensor((capacity, value_dim), np.float32)
        self._length = 0
        self._tail: Event | None = None
        self._closed = False

    @property
    def length(self: Self) -> int:
        """Return the number of scheduled cache positions."""
        return self._length

    def _require_open(self: Self) -> None:
        """Reject operations after cache-owned tensors were closed."""
        if self._closed:
            raise DeviceClosedError("KV cache is closed")

    def _dependencies(self: Self, wait_for: Iterable[Event]) -> tuple[Event, ...]:
        """Append the prior cache operation to serialize cache mutation/use."""
        dependencies = list(wait_for)
        if self._tail is not None and self._tail not in dependencies:
            dependencies.append(self._tail)
        return tuple(dependencies)

    def append(
        self: Self,
        key: Tensor,
        value: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Append one or more token positions to the cache without copying devices."""
        self._require_open()
        if (
            queue.device is not self._device
            or key.buffer.device is not self._device
            or value.buffer.device is not self._device
        ):
            raise DependencyError("KV cache queue, key, and value must belong to the cache device")
        if key.dtype != np.dtype(np.float32) or value.dtype != np.dtype(np.float32):
            raise ValueError("KV cache key and value must use float32")
        if len(key.shape) != 2 or len(value.shape) != 2 or key.shape[0] != value.shape[0]:
            raise ValueError("KV cache key and value must be equal-length rank-2 tensors")
        token_count = key.shape[0]
        if key.shape[1] != self.depth or value.shape[1] != self.value_dim:
            raise ValueError("KV cache key or value width does not match its topology")
        if self._length + token_count > self.capacity:
            raise ValueError("KV cache capacity would be exceeded")
        start = self._length
        stop = start + token_count

        def copy_into_cache() -> None:
            self._key.numpy()[start:stop, :] = key.numpy()
            self._value.numpy()[start:stop, :] = value.numpy()

        event = queue.host_task(
            copy_into_cache,
            wait_for=self._dependencies(wait_for),
            buffers=(
                key.access(AccessMode.READ),
                value.access(AccessMode.READ),
                self._key.access(AccessMode.WRITE),
                self._value.access(AccessMode.WRITE),
            ),
            name="tinyllama.kv_cache.append",
        )
        self._length = stop
        self._tail = event
        return event

    def attend(
        self: Self,
        destination: Tensor,
        query: Tensor,
        *,
        queue: Queue,
        wait_for: Iterable[Event] = (),
    ) -> Event:
        """Attend query rows to cached keys/values with position-aware causality."""
        self._require_open()
        if self._length == 0:
            raise DependencyError("KV cache must contain at least one key/value position")
        if (
            queue.device is not self._device
            or query.buffer.device is not self._device
            or destination.buffer.device is not self._device
        ):
            raise DependencyError("KV cache attend tensors and queue must belong to the cache device")
        if query.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
            raise ValueError("KV cache attend query and destination must use float32")
        if query.shape[1] != self.depth or destination.shape != (query.shape[0], self.value_dim):
            raise ValueError("KV cache attend shapes do not match its topology")
        if query.shape[0] > self._length:
            raise ValueError("KV cache query length cannot exceed the cached sequence length")
        key = self._key.slice((slice(0, self._length), slice(None)))
        value = self._value.slice((slice(0, self._length), slice(None)))
        event = scaled_dot_product_attention_fp32(
            destination,
            query,
            key,
            value,
            causal=True,
            causal_offset=self._length - query.shape[0],
            queue=queue,
            wait_for=self._dependencies(wait_for),
        )
        self._tail = event
        return event

    def close(self: Self) -> None:
        """Invalidate the fixed cache tensors without closing the caller's device."""
        if self._closed:
            return
        self._key.buffer.close()
        self._value.buffer.close()
        self._closed = True

    def __enter__(self: Self) -> Self:
        """Enter a context-managed cache lifetime."""
        self._require_open()
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close cache-owned tensors at context exit."""
        self.close()
