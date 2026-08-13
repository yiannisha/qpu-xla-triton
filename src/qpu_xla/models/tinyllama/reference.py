"""CPU-correct causal TinyLlama reference execution over a validated checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.models.tinyllama.checkpoint import TinyLlamaCheckpoint


@dataclass(frozen=True, slots=True)
class TinyLlamaForwardResult:
    """Causal logits plus post-block hidden-state fixtures for differential testing."""

    logits: npt.NDArray[np.float32]
    hidden_states: tuple[npt.NDArray[np.float32], ...]


def _rms_norm(
    values: npt.NDArray[np.float32], weight: npt.NDArray[np.float32], epsilon: float
) -> npt.NDArray[np.float32]:
    """Apply the row-wise RMSNorm used before each TinyLlama sublayer."""
    mean_square = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    normalized = values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon)))
    return cast(npt.NDArray[np.float32], normalized * weight)


def _linear(values: npt.NDArray[np.float32], weight: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Apply one bias-free checkpoint projection stored as ``(output, input)``."""
    return cast(npt.NDArray[np.float32], np.matmul(values, weight.T, dtype=np.float32))


def _rope(values: npt.NDArray[np.float32], positions: npt.NDArray[np.int32], theta: float) -> npt.NDArray[np.float32]:
    """Apply rotary positions to ``(tokens, heads, head_dim)`` FP32 vectors."""
    _, _, head_dim = values.shape
    frequencies = np.power(np.float32(theta), -np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    angles = positions.astype(np.float32)[:, None] * frequencies[None, :]
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    result = np.array(values, copy=True)
    result[..., 0::2] = values[..., 0::2] * cosine - values[..., 1::2] * sine
    result[..., 1::2] = values[..., 0::2] * sine + values[..., 1::2] * cosine
    return result


def _causal_attention(
    query: npt.NDArray[np.float32], key: npt.NDArray[np.float32], value: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Compute grouped-query causal attention for one full prefill sequence."""
    token_count, query_heads, head_dim = query.shape
    key_value_heads = key.shape[1]
    repeat = query_heads // key_value_heads
    key_expanded = np.repeat(key, repeat, axis=1)
    value_expanded = np.repeat(value, repeat, axis=1)
    query_by_head = np.transpose(query, (1, 0, 2))
    key_by_head = np.transpose(key_expanded, (1, 0, 2))
    value_by_head = np.transpose(value_expanded, (1, 0, 2))
    scores = np.matmul(query_by_head, np.swapaxes(key_by_head, -1, -2), dtype=np.float32)
    scores *= np.float32(1.0 / np.sqrt(head_dim))
    scores[:, np.triu_indices(token_count, k=1)[0], np.triu_indices(token_count, k=1)[1]] = -np.inf
    row_max = np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores - row_max)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True, dtype=np.float32)
    attended = np.matmul(probabilities, value_by_head, dtype=np.float32)
    return cast(npt.NDArray[np.float32], np.transpose(attended, (1, 0, 2)))


class TinyLlamaReferenceRuntime:
    """Numerically transparent CPU reference for the validated decoder-only topology.

    It is intentionally a prefill-oriented oracle, not a performance path. It
    supplies layer-by-layer hidden-state fixtures and causal logits while QPU
    projection/KV-cache execution is integrated separately.
    """

    def __init__(self: Self, checkpoint: TinyLlamaCheckpoint) -> None:
        """Retain one immutable checkpoint and its architecture contract."""
        self.checkpoint = checkpoint

    def forward(self: Self, token_ids: npt.NDArray[np.integer], *, position_offset: int = 0) -> TinyLlamaForwardResult:
        """Compute full-sequence causal logits and post-block hidden states in FP32."""
        config = self.checkpoint.config
        if token_ids.ndim != 1 or token_ids.size == 0:
            raise ValueError("TinyLlama token_ids must be a non-empty rank-1 array")
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("TinyLlama token_ids must have an integer dtype")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        token_values = token_ids.astype(np.int64, copy=False)
        if np.any(token_values < 0) or np.any(token_values >= config.vocab_size):
            raise ValueError("TinyLlama token ids are outside the vocabulary range")
        if position_offset + token_values.size > config.max_position_embeddings:
            raise ValueError("TinyLlama positions exceed max_position_embeddings")
        tensors = self.checkpoint.tensors
        hidden = np.array(tensors["model.embed_tokens.weight"][token_values], dtype=np.float32, copy=True)
        positions = np.arange(position_offset, position_offset + token_values.size, dtype=np.int32)
        fixtures: list[npt.NDArray[np.float32]] = []
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            input_normalized = _rms_norm(hidden, tensors[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
            query = _linear(input_normalized, tensors[f"{prefix}.self_attn.q_proj.weight"])
            key = _linear(input_normalized, tensors[f"{prefix}.self_attn.k_proj.weight"])
            value = _linear(input_normalized, tensors[f"{prefix}.self_attn.v_proj.weight"])
            query = query.reshape(token_values.size, config.num_attention_heads, config.head_dim)
            key = key.reshape(token_values.size, config.num_key_value_heads, config.head_dim)
            value = value.reshape(token_values.size, config.num_key_value_heads, config.head_dim)
            rotated_query = _rope(query, positions, config.rope_theta)
            rotated_key = _rope(key, positions, config.rope_theta)
            attention = _causal_attention(rotated_query, rotated_key, value)
            attention_output = _linear(
                attention.reshape(token_values.size, config.hidden_size), tensors[f"{prefix}.self_attn.o_proj.weight"]
            )
            residual = cast(npt.NDArray[np.float32], hidden + attention_output)
            mlp_normalized = _rms_norm(
                residual, tensors[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
            )
            gate = _linear(mlp_normalized, tensors[f"{prefix}.mlp.gate_proj.weight"])
            up = _linear(mlp_normalized, tensors[f"{prefix}.mlp.up_proj.weight"])
            activated = cast(npt.NDArray[np.float32], (gate / (np.float32(1.0) + np.exp(-gate))) * up)
            hidden = np.empty_like(residual)
            np.add(residual, _linear(activated, tensors[f"{prefix}.mlp.down_proj.weight"]), out=hidden)
            fixtures.append(np.array(hidden, copy=True))
        final_hidden = _rms_norm(hidden, tensors["model.norm.weight"], config.rms_norm_eps)
        logits = _linear(final_hidden, tensors["lm_head.weight"])
        return TinyLlamaForwardResult(logits, tuple(fixtures))

    def session(self: Self) -> TinyLlamaReferenceSession:
        """Create an empty CPU incremental-decode session over this checkpoint."""
        return TinyLlamaReferenceSession(self)


class TinyLlamaReferenceSession:
    """Cache-backed single-token decoder that matches :class:`TinyLlamaReferenceRuntime`.

    This retains per-layer key/value tensors on the CPU reference path. It is
    the correctness baseline for a future device-resident multi-layer KV plan.
    """

    def __init__(self: Self, runtime: TinyLlamaReferenceRuntime) -> None:
        """Allocate logical empty caches without materializing model-sized buffers."""
        self.runtime = runtime
        self._keys: list[list[npt.NDArray[np.float32]]] = [
            [] for _ in range(runtime.checkpoint.config.num_hidden_layers)
        ]
        self._values: list[list[npt.NDArray[np.float32]]] = [
            [] for _ in range(runtime.checkpoint.config.num_hidden_layers)
        ]
        self._length = 0

    @property
    def length(self: Self) -> int:
        """Return the number of appended decode positions."""
        return self._length

    def decode(self: Self, token_id: int) -> npt.NDArray[np.float32]:
        """Append one token to every layer cache and return its causal FP32 logits."""
        config = self.runtime.checkpoint.config
        if not isinstance(token_id, int) or not 0 <= token_id < config.vocab_size:
            raise ValueError("token_id is outside the TinyLlama vocabulary range")
        if self._length >= config.max_position_embeddings:
            raise ValueError("decode would exceed max_position_embeddings")
        tensors = self.runtime.checkpoint.tensors
        hidden = np.array(tensors["model.embed_tokens.weight"][token_id : token_id + 1], copy=True)
        positions = np.asarray([self._length], dtype=np.int32)
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            input_normalized = _rms_norm(hidden, tensors[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
            query = _linear(input_normalized, tensors[f"{prefix}.self_attn.q_proj.weight"])
            key = _linear(input_normalized, tensors[f"{prefix}.self_attn.k_proj.weight"])
            value = _linear(input_normalized, tensors[f"{prefix}.self_attn.v_proj.weight"])
            query = _rope(query.reshape(1, config.num_attention_heads, config.head_dim), positions, config.rope_theta)[
                0
            ]
            key = _rope(key.reshape(1, config.num_key_value_heads, config.head_dim), positions, config.rope_theta)[0]
            value = value.reshape(config.num_key_value_heads, config.head_dim)
            self._keys[layer].append(key)
            self._values[layer].append(value)
            keys = np.stack(self._keys[layer], axis=0)
            values = np.stack(self._values[layer], axis=0)
            repeat = config.num_attention_heads // config.num_key_value_heads
            expanded_keys = np.repeat(keys, repeat, axis=1)
            expanded_values = np.repeat(values, repeat, axis=1)
            scores = np.einsum("hd,thd->ht", query, expanded_keys, dtype=np.float32)
            scores *= np.float32(1.0 / np.sqrt(config.head_dim))
            probabilities = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=1, keepdims=True, dtype=np.float32)
            attention = np.einsum("ht,thd->hd", probabilities, expanded_values, dtype=np.float32)
            attention_output = _linear(
                attention.reshape(1, config.hidden_size), tensors[f"{prefix}.self_attn.o_proj.weight"]
            )
            residual = cast(npt.NDArray[np.float32], hidden + attention_output)
            mlp_normalized = _rms_norm(
                residual, tensors[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
            )
            gate = _linear(mlp_normalized, tensors[f"{prefix}.mlp.gate_proj.weight"])
            up = _linear(mlp_normalized, tensors[f"{prefix}.mlp.up_proj.weight"])
            activated = cast(npt.NDArray[np.float32], (gate / (np.float32(1.0) + np.exp(-gate))) * up)
            hidden = np.empty_like(residual)
            np.add(residual, _linear(activated, tensors[f"{prefix}.mlp.down_proj.weight"]), out=hidden)
        self._length += 1
        final_hidden = _rms_norm(hidden, tensors["model.norm.weight"], config.rms_norm_eps)
        return cast(npt.NDArray[np.float32], _linear(final_hidden, tensors["lm_head.weight"])[0])

    def prefill(self: Self, token_ids: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
        """Append a token sequence through incremental decode and return one logit row each."""
        if token_ids.ndim != 1 or token_ids.size == 0 or not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must be a non-empty rank-1 integer array")
        return np.stack([self.decode(int(token_id)) for token_id in token_ids], axis=0)
