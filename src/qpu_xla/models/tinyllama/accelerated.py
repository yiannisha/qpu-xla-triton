"""Persistent mixed CPU/QPU W8A8 runtime for dense Llama inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.benchmark import CandidateRegistry
from qpu_xla.device import Device
from qpu_xla.memory import AccessMode, Tensor
from qpu_xla.models.tinyllama.checkpoint import TinyLlamaCheckpoint
from qpu_xla.models.tinyllama.quantization import (
    CalibratedW8A8Linear,
    quantize_per_output_channel_int8,
)
from qpu_xla.models.tinyllama.reference import TinyLlamaForwardResult
from qpu_xla.ops.swiglu import swiglu_fp32
from qpu_xla.queue import Event
from qpu_xla.scheduler import Placement


def _rms_norm_values(
    values: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    epsilon: float,
) -> npt.NDArray[np.float32]:
    mean_square = np.mean(values * values, axis=1, keepdims=True, dtype=np.float32)
    return cast(
        npt.NDArray[np.float32],
        values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon))) * weight,
    )


def _rope_heads_inplace(
    values: npt.NDArray[np.float32],
    positions: npt.NDArray[np.int32],
    *,
    heads: int,
    head_dim: int,
    theta: float,
) -> None:
    shaped = values.reshape(values.shape[0], heads, head_dim)
    frequencies = np.power(
        np.float32(theta),
        -np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim),
    )
    angles = positions.astype(np.float32, copy=False)[:, None] * frequencies[None, :]
    cosine = np.cos(angles)[:, None, :]
    sine = np.sin(angles)[:, None, :]
    even = np.array(shaped[..., 0::2], copy=True)
    odd = np.array(shaped[..., 1::2], copy=True)
    shaped[..., 0::2] = even * cosine - odd * sine
    shaped[..., 1::2] = even * sine + odd * cosine


def _causal_gqa(
    query: npt.NDArray[np.float32],
    key: npt.NDArray[np.float32],
    value: npt.NDArray[np.float32],
    *,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> npt.NDArray[np.float32]:
    tokens = query.shape[0]
    query_heads_values = query.reshape(tokens, query_heads, head_dim)
    key_heads_values = key.reshape(tokens, key_value_heads, head_dim)
    value_heads_values = value.reshape(tokens, key_value_heads, head_dim)
    repeat = query_heads // key_value_heads
    expanded_key = np.repeat(key_heads_values, repeat, axis=1)
    expanded_value = np.repeat(value_heads_values, repeat, axis=1)
    query_by_head = query_heads_values.transpose(1, 0, 2)
    key_by_head = expanded_key.transpose(1, 0, 2)
    value_by_head = expanded_value.transpose(1, 0, 2)
    scores = np.matmul(query_by_head, np.swapaxes(key_by_head, 1, 2), dtype=np.float32)
    scores *= np.float32(1.0 / np.sqrt(head_dim))
    upper = np.triu_indices(tokens, k=1)
    scores[:, upper[0], upper[1]] = -np.inf
    scores -= np.max(scores, axis=2, keepdims=True)
    np.exp(scores, out=scores)
    scores /= np.sum(scores, axis=2, keepdims=True, dtype=np.float32)
    attended = np.matmul(scores, value_by_head, dtype=np.float32)
    return np.ascontiguousarray(attended.transpose(1, 0, 2).reshape(tokens, query_heads * head_dim))


class TinyLlamaW8A8Runtime:
    """Dense Llama prefill runtime with persistent calibrated W8A8 projections.

    Projection weights are quantized once. Exact-shape benchmark winners use
    packed QPU GEMM; every other projection uses the same dynamic-W8A8 CPU
    reference. Numerically sensitive normalization, RoPE, GQA softmax,
    residual, and SwiGLU stages remain explicit CPU tasks in the same queue.
    """

    def __init__(
        self: Self,
        device: Device,
        checkpoint: TinyLlamaCheckpoint,
        *,
        max_batch: int,
        candidates: CandidateRegistry | None = None,
        placement: Placement = Placement.AUTO,
    ) -> None:
        """Quantize projection weights and allocate fixed-capacity activations."""
        if max_batch <= 0 or max_batch > checkpoint.config.max_position_embeddings:
            raise ValueError("W8A8 runtime max_batch must fit max_position_embeddings")
        self.device = device
        self.checkpoint = checkpoint
        self.max_batch = max_batch
        self.candidates = CandidateRegistry() if candidates is None else candidates
        self.placement = placement
        self.queue = device.queue()
        self.cpu_queue = device.queue()
        self._closed = False
        self._plans: dict[str, CalibratedW8A8Linear] = {}
        tensors = checkpoint.tensors
        config = checkpoint.config

        def prepare_plan(name: str) -> None:
            weight = tensors[name]
            output_features, input_features = weight.shape
            if not any(
                self.candidates.supported_for(
                    operation="linear",
                    dtype="w8a8-i32-fp32",
                    layout="row-major-packed-k4",
                    shape_class=f"{batch}x{input_features}x{output_features}",
                )
                for batch in range(1, max_batch + 1)
            ):
                return
            self._plans[name] = CalibratedW8A8Linear(
                device,
                quantize_per_output_channel_int8(weight),
                max_batch=max_batch,
                candidates=self.candidates,
            )

        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            for projection in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ):
                name = f"{prefix}.{projection}.weight"
                prepare_plan(name)
        prepare_plan("lm_head.weight")

        hidden = config.hidden_size
        key_value = config.key_value_size
        intermediate = config.intermediate_size
        vocabulary = config.vocab_size
        self._owned: list[Tensor] = []

        def allocate(shape: tuple[int, ...], dtype: npt.DTypeLike = np.float32) -> Tensor:
            tensor = device.tensor(shape, dtype)
            self._owned.append(tensor)
            return tensor

        self._hidden = allocate((max_batch, hidden))
        self._normalized = allocate((max_batch, hidden))
        self._query = allocate((max_batch, hidden))
        self._key = allocate((max_batch, key_value))
        self._value = allocate((max_batch, key_value))
        self._attention = allocate((max_batch, hidden))
        self._projected = allocate((max_batch, hidden))
        self._gate = allocate((max_batch, intermediate))
        self._up = allocate((max_batch, intermediate))
        self._activated = allocate((max_batch, intermediate))
        self._logits = allocate((max_batch, vocabulary))
        self._positions = allocate((max_batch,), np.int32)

    def __enter__(self: Self) -> Self:
        """Enter a context-managed runtime lifetime."""
        if self._closed:
            raise RuntimeError("W8A8 runtime is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release all persistent model state."""
        self.close()

    def close(self: Self) -> None:
        """Close projection plans, workspaces, and the owned queue."""
        if self._closed:
            return
        self.queue.close()
        self.cpu_queue.close()
        for plan in self._plans.values():
            plan.close()
        for tensor in self._owned:
            tensor.buffer.close()
        self._closed = True

    @staticmethod
    def _rows(tensor: Tensor, count: int) -> Tensor:
        return tensor.slice((slice(0, count), *tuple(slice(None) for _ in tensor.shape[1:])))

    def _host(
        self: Self,
        name: str,
        fn: Callable[[], None],
        *,
        reads: tuple[Tensor, ...] = (),
        writes: tuple[Tensor, ...] = (),
        read_writes: tuple[Tensor, ...] = (),
        wait_for: tuple[Event, ...] = (),
    ) -> Event:
        return self.queue.host_task(
            fn,
            wait_for=wait_for,
            buffers=(
                *(tensor.access(AccessMode.READ) for tensor in reads),
                *(tensor.access(AccessMode.WRITE) for tensor in writes),
                *(tensor.access(AccessMode.READ_WRITE) for tensor in read_writes),
            ),
            name=name,
        )

    def _linear(
        self: Self,
        name: str,
        destination: Tensor,
        source: Tensor,
        dependency: Event,
    ) -> Event:
        shape_class = f"{source.shape[0]}x{source.shape[1]}x{destination.shape[1]}"
        supported = bool(
            self.candidates.supported_for(
                operation="linear",
                dtype="w8a8-i32-fp32",
                layout="row-major-packed-k4",
                shape_class=shape_class,
            )
        )
        if self.placement is Placement.CPU or (self.placement is Placement.AUTO and not supported):
            weight = self.checkpoint.tensors[name]

            def reference() -> None:
                np.matmul(source.numpy(), weight.T, out=destination.numpy())

            return self._host(
                "llama.linear_fp32_reference",
                reference,
                reads=(source,),
                writes=(destination,),
                wait_for=(dependency,),
            )
        if not supported:
            raise ValueError("no measured supported-win W8A8 QPU candidate exists for this exact shape")
        return self._plans[name].execute(
            destination,
            source,
            queue=self.queue,
            cpu_queue=self.cpu_queue,
            placement=self.placement,
            wait_for=(dependency,),
        )

    def _swiglu(
        self: Self,
        destination: Tensor,
        gate: Tensor,
        up: Tensor,
        dependency: tuple[Event, Event],
    ) -> Event:
        shape_class = f"{gate.shape[0]}x{gate.shape[1]}"
        supported = bool(
            self.candidates.supported_for(
                operation="swiglu",
                dtype="fp32",
                layout="contiguous-row-major",
                shape_class=shape_class,
            )
        )
        if self.placement is Placement.QPU and not supported:
            raise ValueError("no measured supported-win SwiGLU QPU candidate exists for this exact shape")
        if self.placement is Placement.QPU or (self.placement is Placement.AUTO and supported):
            return swiglu_fp32(
                destination,
                gate,
                up,
                queue=self.queue,
                wait_for=dependency,
            )

        def reference() -> None:
            gate_values = cast(npt.NDArray[np.float32], gate.numpy())
            up_values = cast(npt.NDArray[np.float32], up.numpy())
            destination.numpy()[:] = (gate_values / (np.float32(1.0) + np.exp(-gate_values))) * up_values

        return self._host(
            "llama.swiglu",
            reference,
            reads=(gate, up),
            writes=(destination,),
            wait_for=dependency,
        )

    def forward(
        self: Self,
        token_ids: npt.NDArray[np.integer],
        *,
        position_offset: int = 0,
    ) -> TinyLlamaForwardResult:
        """Run quantized full-sequence causal prefill and return FP32 logits."""
        return self._forward(token_ids, position_offset=position_offset)

    def _forward(
        self: Self,
        token_ids: npt.NDArray[np.integer],
        *,
        position_offset: int = 0,
        key_caches: list[Tensor] | None = None,
        value_caches: list[Tensor] | None = None,
    ) -> TinyLlamaForwardResult:
        """Run prefill, optionally committing each layer's rotated KV tensors."""
        if self._closed:
            raise RuntimeError("W8A8 runtime is closed")
        config = self.checkpoint.config
        if token_ids.ndim != 1 or token_ids.size == 0 or not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must be a non-empty rank-1 integer array")
        count = int(token_ids.size)
        if count > self.max_batch or position_offset < 0 or position_offset + count > config.max_position_embeddings:
            raise ValueError("tokens exceed the W8A8 runtime batch or position capacity")
        ids = token_ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= config.vocab_size):
            raise ValueError("token ids are outside the vocabulary range")
        tensors = self.checkpoint.tensors
        if (key_caches is None) != (value_caches is None):
            raise ValueError("prefill key and value caches must be supplied together")
        if key_caches is not None and (
            len(key_caches) != config.num_hidden_layers
            or value_caches is None
            or len(value_caches) != config.num_hidden_layers
        ):
            raise ValueError("prefill caches must contain one key/value tensor per layer")
        hidden = self._rows(self._hidden, count)
        normalized = self._rows(self._normalized, count)
        query = self._rows(self._query, count)
        key = self._rows(self._key, count)
        value = self._rows(self._value, count)
        attention = self._rows(self._attention, count)
        projected = self._rows(self._projected, count)
        gate = self._rows(self._gate, count)
        up = self._rows(self._up, count)
        activated = self._rows(self._activated, count)
        logits = self._rows(self._logits, count)
        positions = self._rows(self._positions, count)

        def embed() -> None:
            hidden.numpy()[:] = tensors["model.embed_tokens.weight"][ids]
            positions.numpy()[:] = np.arange(position_offset, position_offset + count, dtype=np.int32)

        event = self._host("llama.embed", embed, writes=(hidden, positions))
        fixtures: list[npt.NDArray[np.float32]] = []
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"

            def input_norm(prefix: str = prefix) -> None:
                normalized.numpy()[:] = _rms_norm_values(
                    cast(npt.NDArray[np.float32], hidden.numpy()),
                    tensors[f"{prefix}.input_layernorm.weight"],
                    config.rms_norm_eps,
                )

            norm_event = self._host(
                "llama.input_rms_norm",
                input_norm,
                reads=(hidden,),
                writes=(normalized,),
                wait_for=(event,),
            )
            query_event = self._linear(f"{prefix}.self_attn.q_proj.weight", query, normalized, norm_event)
            key_event = self._linear(f"{prefix}.self_attn.k_proj.weight", key, normalized, norm_event)
            value_event = self._linear(f"{prefix}.self_attn.v_proj.weight", value, normalized, norm_event)

            def rope() -> None:
                position_values = cast(npt.NDArray[np.int32], positions.numpy())
                _rope_heads_inplace(
                    cast(npt.NDArray[np.float32], query.numpy()),
                    position_values,
                    heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    theta=config.rope_theta,
                )
                _rope_heads_inplace(
                    cast(npt.NDArray[np.float32], key.numpy()),
                    position_values,
                    heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    theta=config.rope_theta,
                )

            rope_event = self._host(
                "llama.rope",
                rope,
                reads=(positions,),
                read_writes=(query, key),
                wait_for=(query_event, key_event),
            )

            key_cache = None if key_caches is None else key_caches[layer]
            value_cache = None if value_caches is None else value_caches[layer]

            def attend(key_cache: Tensor | None = key_cache, value_cache: Tensor | None = value_cache) -> None:
                if key_cache is not None and value_cache is not None:
                    key_cache.numpy()[position_offset : position_offset + count] = key.numpy()
                    value_cache.numpy()[position_offset : position_offset + count] = value.numpy()
                attention.numpy()[:] = _causal_gqa(
                    cast(npt.NDArray[np.float32], query.numpy()),
                    cast(npt.NDArray[np.float32], key.numpy()),
                    cast(npt.NDArray[np.float32], value.numpy()),
                    query_heads=config.num_attention_heads,
                    key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                )

            cache_writes = (key_cache, value_cache) if key_cache is not None and value_cache is not None else ()
            attention_event = self._host(
                "llama.causal_gqa",
                attend,
                reads=(query, key, value),
                writes=(attention, *cache_writes),
                wait_for=(rope_event, value_event),
            )
            output_event = self._linear(
                f"{prefix}.self_attn.o_proj.weight",
                projected,
                attention,
                attention_event,
            )

            def attention_residual() -> None:
                np.add(hidden.numpy(), projected.numpy(), out=hidden.numpy())

            residual_event = self._host(
                "llama.attention_residual",
                attention_residual,
                reads=(projected,),
                read_writes=(hidden,),
                wait_for=(output_event,),
            )

            def post_norm(prefix: str = prefix) -> None:
                normalized.numpy()[:] = _rms_norm_values(
                    cast(npt.NDArray[np.float32], hidden.numpy()),
                    tensors[f"{prefix}.post_attention_layernorm.weight"],
                    config.rms_norm_eps,
                )

            post_norm_event = self._host(
                "llama.post_attention_rms_norm",
                post_norm,
                reads=(hidden,),
                writes=(normalized,),
                wait_for=(residual_event,),
            )
            gate_event = self._linear(f"{prefix}.mlp.gate_proj.weight", gate, normalized, post_norm_event)
            up_event = self._linear(f"{prefix}.mlp.up_proj.weight", up, normalized, post_norm_event)

            activation_event = self._swiglu(activated, gate, up, (gate_event, up_event))
            down_event = self._linear(
                f"{prefix}.mlp.down_proj.weight",
                projected,
                activated,
                activation_event,
            )

            def mlp_residual() -> None:
                np.add(hidden.numpy(), projected.numpy(), out=hidden.numpy())

            residual_event = self._host(
                "llama.mlp_residual",
                mlp_residual,
                reads=(projected,),
                read_writes=(hidden,),
                wait_for=(down_event,),
            )
            fixture = np.empty((count, config.hidden_size), dtype=np.float32)

            def snapshot(fixture: npt.NDArray[np.float32] = fixture) -> None:
                fixture[:] = hidden.numpy()

            event = self._host(
                "llama.snapshot",
                snapshot,
                reads=(hidden,),
                wait_for=(residual_event,),
            )
            fixtures.append(fixture)

        def final_norm() -> None:
            normalized.numpy()[:] = _rms_norm_values(
                cast(npt.NDArray[np.float32], hidden.numpy()),
                tensors["model.norm.weight"],
                config.rms_norm_eps,
            )

        norm_event = self._host(
            "llama.final_rms_norm",
            final_norm,
            reads=(hidden,),
            writes=(normalized,),
            wait_for=(event,),
        )
        logits_event = self._linear("lm_head.weight", logits, normalized, norm_event)
        logits_event.wait()
        return TinyLlamaForwardResult(np.array(logits.numpy(), dtype=np.float32, copy=True), tuple(fixtures))

    def session(self: Self) -> TinyLlamaW8A8Session:
        """Create an incremental decoder with persistent per-layer KV storage."""
        if self._closed:
            raise RuntimeError("W8A8 runtime is closed")
        return TinyLlamaW8A8Session(self)


class TinyLlamaW8A8Session:
    """Incremental W8A8 decoder using the runtime's plans and device queue."""

    def __init__(self: Self, runtime: TinyLlamaW8A8Runtime) -> None:
        """Allocate fixed-capacity key/value tensors for every decoder layer."""
        self.runtime = runtime
        config = runtime.checkpoint.config
        self._keys = [
            runtime.device.tensor((config.max_position_embeddings, config.key_value_size), np.float32)
            for _ in range(config.num_hidden_layers)
        ]
        self._values = [
            runtime.device.tensor((config.max_position_embeddings, config.key_value_size), np.float32)
            for _ in range(config.num_hidden_layers)
        ]
        self._length = 0
        self._closed = False

    @property
    def length(self: Self) -> int:
        """Return the number of committed cache positions."""
        return self._length

    def __enter__(self: Self) -> Self:
        """Enter a context-managed session lifetime."""
        if self._closed:
            raise RuntimeError("W8A8 session is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release session-owned cache tensors."""
        self.close()

    def close(self: Self) -> None:
        """Release key/value cache allocations without closing the runtime."""
        if self._closed:
            return
        for tensor in (*self._keys, *self._values):
            tensor.buffer.close()
        self._closed = True

    def reset(self: Self) -> None:
        """Logically clear all caches without rewriting their storage."""
        if self._closed:
            raise RuntimeError("W8A8 session is closed")
        self._length = 0

    def decode(self: Self, token_id: int) -> npt.NDArray[np.float32]:
        """Append one token to every layer cache and return its logits row."""
        if self._closed or self.runtime._closed:
            raise RuntimeError("W8A8 session or runtime is closed")
        config = self.runtime.checkpoint.config
        if not isinstance(token_id, int) or not 0 <= token_id < config.vocab_size:
            raise ValueError("token_id is outside the vocabulary range")
        if self._length >= config.max_position_embeddings:
            raise ValueError("decode would exceed max_position_embeddings")
        runtime = self.runtime
        tensors = runtime.checkpoint.tensors
        hidden = runtime._rows(runtime._hidden, 1)
        normalized = runtime._rows(runtime._normalized, 1)
        query = runtime._rows(runtime._query, 1)
        key = runtime._rows(runtime._key, 1)
        value = runtime._rows(runtime._value, 1)
        attention = runtime._rows(runtime._attention, 1)
        projected = runtime._rows(runtime._projected, 1)
        gate = runtime._rows(runtime._gate, 1)
        up = runtime._rows(runtime._up, 1)
        activated = runtime._rows(runtime._activated, 1)
        logits = runtime._rows(runtime._logits, 1)
        positions = runtime._rows(runtime._positions, 1)
        position = self._length

        def embed() -> None:
            hidden.numpy()[0] = tensors["model.embed_tokens.weight"][token_id]
            positions.numpy()[0] = position

        event = runtime._host("llama.decode.embed", embed, writes=(hidden, positions))
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"

            def input_norm(prefix: str = prefix) -> None:
                normalized.numpy()[:] = _rms_norm_values(
                    cast(npt.NDArray[np.float32], hidden.numpy()),
                    tensors[f"{prefix}.input_layernorm.weight"],
                    config.rms_norm_eps,
                )

            norm_event = runtime._host(
                "llama.decode.input_rms_norm",
                input_norm,
                reads=(hidden,),
                writes=(normalized,),
                wait_for=(event,),
            )
            query_event = runtime._linear(f"{prefix}.self_attn.q_proj.weight", query, normalized, norm_event)
            key_event = runtime._linear(f"{prefix}.self_attn.k_proj.weight", key, normalized, norm_event)
            value_event = runtime._linear(f"{prefix}.self_attn.v_proj.weight", value, normalized, norm_event)

            def rope() -> None:
                position_values = cast(npt.NDArray[np.int32], positions.numpy())
                _rope_heads_inplace(
                    cast(npt.NDArray[np.float32], query.numpy()),
                    position_values,
                    heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    theta=config.rope_theta,
                )
                _rope_heads_inplace(
                    cast(npt.NDArray[np.float32], key.numpy()),
                    position_values,
                    heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    theta=config.rope_theta,
                )

            rope_event = runtime._host(
                "llama.decode.rope",
                rope,
                reads=(positions,),
                read_writes=(query, key),
                wait_for=(query_event, key_event),
            )
            key_cache = self._keys[layer]
            value_cache = self._values[layer]

            def cache_attention(key_cache: Tensor = key_cache, value_cache: Tensor = value_cache) -> None:
                key_cache.numpy()[position] = key.numpy()[0]
                value_cache.numpy()[position] = value.numpy()[0]
                length = position + 1
                query_heads = cast(npt.NDArray[np.float32], query.numpy()).reshape(
                    config.num_attention_heads,
                    config.head_dim,
                )
                cached_keys = cast(npt.NDArray[np.float32], key_cache.numpy())[:length].reshape(
                    length,
                    config.num_key_value_heads,
                    config.head_dim,
                )
                cached_values = cast(npt.NDArray[np.float32], value_cache.numpy())[:length].reshape(
                    length,
                    config.num_key_value_heads,
                    config.head_dim,
                )
                repeat = config.num_attention_heads // config.num_key_value_heads
                expanded_keys = np.repeat(cached_keys, repeat, axis=1)
                expanded_values = np.repeat(cached_values, repeat, axis=1)
                scores = np.einsum("hd,thd->ht", query_heads, expanded_keys, dtype=np.float32)
                scores *= np.float32(1.0 / np.sqrt(config.head_dim))
                scores -= np.max(scores, axis=1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= np.sum(scores, axis=1, keepdims=True, dtype=np.float32)
                attended = np.einsum("ht,thd->hd", scores, expanded_values, dtype=np.float32)
                attention.numpy()[0] = attended.reshape(config.hidden_size)

            attention_event = runtime._host(
                "llama.decode.cached_gqa",
                cache_attention,
                reads=(query, key, value),
                writes=(attention,),
                read_writes=(key_cache, value_cache),
                wait_for=(rope_event, value_event),
            )
            output_event = runtime._linear(
                f"{prefix}.self_attn.o_proj.weight",
                projected,
                attention,
                attention_event,
            )

            def attention_residual() -> None:
                np.add(hidden.numpy(), projected.numpy(), out=hidden.numpy())

            residual_event = runtime._host(
                "llama.decode.attention_residual",
                attention_residual,
                reads=(projected,),
                read_writes=(hidden,),
                wait_for=(output_event,),
            )

            def post_norm(prefix: str = prefix) -> None:
                normalized.numpy()[:] = _rms_norm_values(
                    cast(npt.NDArray[np.float32], hidden.numpy()),
                    tensors[f"{prefix}.post_attention_layernorm.weight"],
                    config.rms_norm_eps,
                )

            post_norm_event = runtime._host(
                "llama.decode.post_attention_rms_norm",
                post_norm,
                reads=(hidden,),
                writes=(normalized,),
                wait_for=(residual_event,),
            )
            gate_event = runtime._linear(f"{prefix}.mlp.gate_proj.weight", gate, normalized, post_norm_event)
            up_event = runtime._linear(f"{prefix}.mlp.up_proj.weight", up, normalized, post_norm_event)

            activation_event = runtime._swiglu(activated, gate, up, (gate_event, up_event))
            down_event = runtime._linear(
                f"{prefix}.mlp.down_proj.weight",
                projected,
                activated,
                activation_event,
            )

            def mlp_residual() -> None:
                np.add(hidden.numpy(), projected.numpy(), out=hidden.numpy())

            event = runtime._host(
                "llama.decode.mlp_residual",
                mlp_residual,
                reads=(projected,),
                read_writes=(hidden,),
                wait_for=(down_event,),
            )

        def final_norm() -> None:
            normalized.numpy()[:] = _rms_norm_values(
                cast(npt.NDArray[np.float32], hidden.numpy()),
                tensors["model.norm.weight"],
                config.rms_norm_eps,
            )

        norm_event = runtime._host(
            "llama.decode.final_rms_norm",
            final_norm,
            reads=(hidden,),
            writes=(normalized,),
            wait_for=(event,),
        )
        logits_event = runtime._linear("lm_head.weight", logits, normalized, norm_event)
        logits_event.wait()
        self._length += 1
        return np.array(logits.numpy()[0], dtype=np.float32, copy=True)

    def prefill(self: Self, token_ids: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
        """Populate caches by incremental prefill and return each logits row."""
        if token_ids.ndim != 1 or token_ids.size == 0 or not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must be a non-empty rank-1 integer array")
        if self._length == 0 and token_ids.size <= self.runtime.max_batch:
            result = self.runtime._forward(
                token_ids,
                key_caches=self._keys,
                value_caches=self._values,
            )
            self._length = int(token_ids.size)
            return result.logits
        return np.stack([self.decode(int(token)) for token in token_ids], axis=0)


__all__ = ["TinyLlamaW8A8Runtime", "TinyLlamaW8A8Session"]
