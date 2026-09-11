"""Numerically transparent end-to-end SmolVLA inference reference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.models.smolvla.checkpoint import SmolVLACheckpoint, W8A8Weight
from qpu_xla.models.smolvla.config import SmolVLAConfig


class _CheckpointLike(Protocol):
    """Minimal topology provider accepted by observation validation."""

    @property
    def config(self) -> SmolVLAConfig:
        """Return the inference topology."""
        ...


@dataclass(frozen=True, slots=True)
class SmolVLAObservation:
    """One recorded inference input at the native RGB tensor boundary.

    Images are unprocessed ``uint8[H,W,3]`` RGB tensors.  ``state`` is the
    normalized policy state, and language tokens/masks are the output of the
    pinned upstream tokenizer.  Supplying noise makes the flow trajectory
    deterministic and directly comparable with upstream PyTorch.
    """

    images: tuple[npt.NDArray[np.uint8], ...]
    image_masks: npt.NDArray[np.bool_]
    language_tokens: npt.NDArray[np.int64]
    language_mask: npt.NDArray[np.bool_]
    state: npt.NDArray[np.float32]
    noise: npt.NDArray[np.float32]

    def validate(self: Self, checkpoint: _CheckpointLike) -> None:
        """Validate shapes and values before a model-sized invocation begins."""
        config = checkpoint.config
        if len(self.images) != len(config.image_keys):
            raise ValueError("SmolVLA observation must provide one RGB tensor per configured camera")
        for image in self.images:
            if image.dtype != np.dtype(np.uint8) or image.shape != (
                config.input_image_height,
                config.input_image_width,
                3,
            ):
                raise ValueError("SmolVLA RGB images do not match the configured uint8 HWC boundary")
            if not image.flags.c_contiguous:
                raise ValueError("SmolVLA RGB images must be contiguous")
        if self.image_masks.dtype != np.dtype(np.bool_) or self.image_masks.shape != (len(self.images),):
            raise ValueError("SmolVLA image mask must contain one boolean per camera")
        if self.language_tokens.dtype != np.dtype(np.int64) or self.language_tokens.shape != (
            config.tokenizer_max_length,
        ):
            raise ValueError("SmolVLA language tokens must be fixed-length int64")
        if self.language_mask.dtype != np.dtype(np.bool_) or self.language_mask.shape != (
            config.tokenizer_max_length,
        ):
            raise ValueError("SmolVLA language mask must be fixed-length boolean")
        if np.any(self.language_tokens < 0) or np.any(self.language_tokens >= config.vocab_size):
            raise ValueError("SmolVLA language token is outside the checkpoint vocabulary")
        if self.state.dtype != np.dtype(np.float32) or self.state.shape != (config.state_dim,):
            raise ValueError("SmolVLA state does not match the logical normalized state contract")
        if self.noise.dtype != np.dtype(np.float32) or self.noise.shape != (
            config.chunk_size,
            config.max_action_dim,
        ):
            raise ValueError("SmolVLA noise must cover the padded action chunk")
        if not np.all(np.isfinite(self.state)) or not np.all(np.isfinite(self.noise)):
            raise ValueError("SmolVLA state and noise must be finite")


@dataclass(frozen=True, slots=True)
class SmolVLAActionChunk:
    """Logical normalized action chunk and optional inference timing details."""

    actions: npt.NDArray[np.float32]
    padded_actions: npt.NDArray[np.float32]
    stage_ns: dict[str, int]


@dataclass(frozen=True, slots=True)
class _PrefixCache:
    keys: tuple[npt.NDArray[np.float32], ...]
    values: tuple[npt.NDArray[np.float32], ...]
    padding_mask: npt.NDArray[np.bool_]


def resize_with_top_left_padding_rgb(
    image: npt.NDArray[np.uint8], target_height: int, target_width: int
) -> npt.NDArray[np.float32]:
    """Match LeRobot's ``align_corners=False`` bilinear resize and top/left pad."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.dtype(np.uint8):
        raise ValueError("RGB resize requires one uint8 HWC image")
    source_height, source_width = image.shape[:2]
    ratio = max(source_width / target_width, source_height / target_height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    source = image.astype(np.float32) / np.float32(255.0)
    resized: npt.NDArray[np.float32]
    if (resized_height, resized_width) != (source_height, source_width):
        y = (np.arange(resized_height, dtype=np.float32) + np.float32(0.5)) * (
            np.float32(source_height) / np.float32(resized_height)
        ) - np.float32(0.5)
        x = (np.arange(resized_width, dtype=np.float32) + np.float32(0.5)) * (
            np.float32(source_width) / np.float32(resized_width)
        ) - np.float32(0.5)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        y1, x1 = y0 + 1, x0 + 1
        wy, wx = y - y0, x - x0
        np.clip(y0, 0, source_height - 1, out=y0)
        np.clip(y1, 0, source_height - 1, out=y1)
        np.clip(x0, 0, source_width - 1, out=x0)
        np.clip(x1, 0, source_width - 1, out=x1)
        top = (
            source[y0[:, None], x0[None, :]] * (np.float32(1.0) - wx[None, :, None])
            + source[y0[:, None], x1[None, :]] * wx[None, :, None]
        )
        bottom = (
            source[y1[:, None], x0[None, :]] * (np.float32(1.0) - wx[None, :, None])
            + source[y1[:, None], x1[None, :]] * wx[None, :, None]
        )
        resized = cast(
            npt.NDArray[np.float32],
            top * (np.float32(1.0) - wy[:, None, None]) + bottom * wy[:, None, None],
        )
    else:
        resized = cast(npt.NDArray[np.float32], source)
    output = np.zeros((target_height, target_width, 3), dtype=np.float32)
    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    output[pad_height:, pad_width:] = resized
    np.multiply(output, np.float32(2.0), out=output)
    np.subtract(output, np.float32(1.0), out=output)
    return np.ascontiguousarray(output.transpose(2, 0, 1))


def sinusoidal_time_embedding(
    timestep: np.float32,
    dimension: int,
    min_period: float,
    max_period: float,
) -> npt.NDArray[np.float32]:
    """Match the upstream float64 frequency construction and FP32 output cast."""
    if dimension <= 0 or dimension % 2:
        raise ValueError("time embedding dimension must be positive and even")
    fraction = np.linspace(0.0, 1.0, dimension // 2, dtype=np.float64)
    period = min_period * np.power(max_period / min_period, fraction)
    angle = np.float64(timestep) * (np.float64(2.0 * np.pi) / period)
    return np.ascontiguousarray(np.concatenate((np.sin(angle), np.cos(angle))).astype(np.float32))


def _dynamic_w8a8_linear(
    values: npt.NDArray[np.float32],
    weight: W8A8Weight,
) -> npt.NDArray[np.float32]:
    flattened = np.ascontiguousarray(values.reshape(-1, values.shape[-1]), dtype=np.float32)
    maxima = np.max(np.abs(flattened), axis=1)
    row_scales = np.maximum(maxima / np.float32(127.0), np.float32(1.0 / 127.0)).astype(np.float32)
    quantized = np.rint(flattened / row_scales[:, None])
    np.clip(quantized, -127, 127, out=quantized)
    left = np.ascontiguousarray(quantized, dtype=np.int8).astype(np.int32)
    right = weight.values.reshape(weight.values.shape[0], -1).astype(np.int32)
    accumulated = np.matmul(left, right.T, dtype=np.int32)
    result = accumulated.astype(np.float32)
    result *= row_scales[:, None]
    result *= weight.scales[None, :]
    return np.ascontiguousarray(result.reshape(*values.shape[:-1], weight.values.shape[0]))


class SmolVLAReferenceRuntime:
    """Complete NumPy oracle for FP32 and the dynamic-W8A8 inference contract."""

    def __init__(self: Self, checkpoint: SmolVLACheckpoint) -> None:
        """Bind the strict checkpoint used by the reference graph."""
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        self._root = self.config.source_prefix

    def _weight(self: Self, name: str) -> npt.NDArray[np.float32] | W8A8Weight:
        return self.checkpoint.linear_weight(name)

    def _linear(
        self: Self,
        values: npt.NDArray[np.float32],
        weight_name: str,
        bias_name: str | None = None,
    ) -> npt.NDArray[np.float32]:
        weight = self._weight(weight_name)
        if isinstance(weight, W8A8Weight):
            result = _dynamic_w8a8_linear(values, weight)
        else:
            result = cast(npt.NDArray[np.float32], np.matmul(values, weight.T, dtype=np.float32))
        if bias_name is not None:
            np.add(result, self.checkpoint.fp32(bias_name), out=result)
        return np.ascontiguousarray(result, dtype=np.float32)

    def _layer_norm(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32],
        epsilon: float,
    ) -> npt.NDArray[np.float32]:
        mean = np.mean(values, axis=-1, keepdims=True, dtype=np.float32)
        centered = values - mean
        variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
        normalized = centered * np.reciprocal(np.sqrt(variance + np.float32(epsilon)))
        return np.ascontiguousarray(normalized * weight + bias, dtype=np.float32)

    def _rms_norm(
        self: Self,
        values: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
        epsilon: float,
    ) -> npt.NDArray[np.float32]:
        mean_square = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        return np.ascontiguousarray(
            values * np.reciprocal(np.sqrt(mean_square + np.float32(epsilon))) * weight,
            dtype=np.float32,
        )

    def _gelu_tanh(self: Self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        cubic = values * values * values
        inner = np.float32(np.sqrt(2.0 / np.pi)) * (values + np.float32(0.044715) * cubic)
        return np.ascontiguousarray(np.float32(0.5) * values * (np.float32(1.0) + np.tanh(inner)))

    def _silu(self: Self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.ascontiguousarray(values / (np.float32(1.0) + np.exp(-values)))

    def _swiglu(
        self: Self,
        gate: npt.NDArray[np.float32],
        up: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return np.ascontiguousarray(self._silu(gate) * up)

    def _residual(
        self: Self,
        left: npt.NDArray[np.float32],
        right: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return np.ascontiguousarray(left + right)

    def _scale(
        self: Self, values: npt.NDArray[np.float32], scale: np.float32
    ) -> npt.NDArray[np.float32]:
        return np.ascontiguousarray(values * scale)

    def _embedding(
        self: Self,
        token_ids: npt.NDArray[np.int64],
        table: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return np.array(table[token_ids], dtype=np.float32, copy=True)

    def _split_half_rope(
        self: Self,
        values: npt.NDArray[np.float32],
        positions: npt.NDArray[np.int32],
        theta: float,
    ) -> npt.NDArray[np.float32]:
        head_dim = values.shape[-1]
        half = head_dim // 2
        exponents = np.float32(2.0 / head_dim) * np.arange(half, dtype=np.float32)
        timescale = np.power(np.float32(theta), exponents)
        radians = positions.astype(np.float32)[:, None] / timescale[None, :]
        cosine, sine = np.cos(radians)[:, None, :], np.sin(radians)[:, None, :]
        first, second = values[..., :half], values[..., half:]
        result = np.empty_like(values)
        result[..., :half] = first * cosine - second * sine
        result[..., half:] = second * cosine + first * sine
        return result

    def _attention(
        self: Self,
        query: npt.NDArray[np.float32],
        key: npt.NDArray[np.float32],
        value: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_] | None,
    ) -> npt.NDArray[np.float32]:
        query_heads = query.shape[1]
        key_value_heads = key.shape[1]
        repeat = query_heads // key_value_heads
        expanded_key = np.repeat(key, repeat, axis=1)
        expanded_value = np.repeat(value, repeat, axis=1)
        query_by_head = query.transpose(1, 0, 2)
        key_by_head = expanded_key.transpose(1, 0, 2)
        value_by_head = expanded_value.transpose(1, 0, 2)
        scores = np.matmul(query_by_head, key_by_head.transpose(0, 2, 1), dtype=np.float32)
        scores *= np.float32(1.0 / np.sqrt(query.shape[-1]))
        if mask is not None:
            scores[:] = np.where(mask[None, :, :], scores, np.finfo(np.float32).min)
        scores -= np.max(scores, axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=-1, keepdims=True, dtype=np.float32)
        attended = np.matmul(scores, value_by_head, dtype=np.float32)
        return np.ascontiguousarray(attended.transpose(1, 0, 2).reshape(query.shape[0], -1))

    def _vision_image(self: Self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        config = self.config
        vision = f"{self._root}vlm_with_expert.vlm.model.vision_model"
        pixels = resize_with_top_left_padding_rgb(image, config.image_size, config.image_size)
        patches = (
            pixels.reshape(
                3,
                config.patches_per_side,
                config.patch_size,
                config.patches_per_side,
                config.patch_size,
            )
            .transpose(1, 3, 0, 2, 4)
            .reshape(config.vision_token_count, -1)
        )
        patch_weight = self._weight(f"{vision}.embeddings.patch_embedding.weight")
        if isinstance(patch_weight, W8A8Weight):
            flattened_weight = W8A8Weight(
                patch_weight.values.reshape(config.vision_hidden_size, -1), patch_weight.scales
            )
            hidden = _dynamic_w8a8_linear(np.ascontiguousarray(patches), flattened_weight)
        else:
            hidden = cast(
                npt.NDArray[np.float32],
                np.matmul(patches, patch_weight.reshape(config.vision_hidden_size, -1).T, dtype=np.float32),
            )
        hidden += self.checkpoint.fp32(f"{vision}.embeddings.patch_embedding.bias")
        hidden += self.checkpoint.fp32(f"{vision}.embeddings.position_embedding.weight")
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
                self._linear(attended, f"{prefix}.self_attn.out_proj.weight", f"{prefix}.self_attn.out_proj.bias"),
            )
            residual = hidden
            normalized = self._layer_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.layer_norm2.weight"),
                self.checkpoint.fp32(f"{prefix}.layer_norm2.bias"),
                config.vision_layer_norm_eps,
            )
            intermediate = self._linear(normalized, f"{prefix}.mlp.fc1.weight", f"{prefix}.mlp.fc1.bias")
            intermediate = self._gelu_tanh(intermediate)
            hidden = self._residual(
                residual,
                self._linear(intermediate, f"{prefix}.mlp.fc2.weight", f"{prefix}.mlp.fc2.bias"),
            )
        hidden = self._layer_norm(
            hidden,
            self.checkpoint.fp32(f"{vision}.post_layernorm.weight"),
            self.checkpoint.fp32(f"{vision}.post_layernorm.bias"),
            config.vision_layer_norm_eps,
        )
        factor = config.pixel_shuffle_factor
        side = config.patches_per_side
        shuffled = hidden.reshape(side, side, config.vision_hidden_size)
        shuffled = shuffled.reshape(side, side // factor, config.vision_hidden_size * factor)
        shuffled = shuffled.transpose(1, 0, 2)
        shuffled = shuffled.reshape(
            side // factor,
            side // factor,
            config.vision_hidden_size * factor**2,
        )
        shuffled = shuffled.transpose(1, 0, 2).reshape(config.image_token_count, -1)
        return self._linear(
            np.ascontiguousarray(shuffled),
            f"{self._root}vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
        )

    @staticmethod
    def _block_mask(padding: npt.NDArray[np.bool_], attention_ar: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        cumulative = np.cumsum(attention_ar, dtype=np.int32)
        return np.ascontiguousarray(
            (cumulative[None, :] <= cumulative[:, None]) & (padding[None, :] & padding[:, None])
        )

    def _prefix(self: Self, observation: SmolVLAObservation) -> tuple[npt.NDArray[np.float32], _PrefixCache]:
        config = self.config
        pieces: list[npt.NDArray[np.float32]] = []
        masks: list[npt.NDArray[np.bool_]] = []
        scale = np.float32(np.sqrt(config.vlm_hidden_size))
        for image, present in zip(observation.images, observation.image_masks, strict=True):
            if present:
                image_embedding = self._scale(self._vision_image(image), scale)
            else:
                # Every later attention mask excludes this entire token block.
                # Avoid executing the vision tower for an absent camera while
                # preserving the fixed prefix shape expected by the VLM.
                image_embedding = np.zeros(
                    (config.image_token_count, config.vlm_hidden_size),
                    dtype=np.float32,
                )
            pieces.append(image_embedding)
            masks.append(np.full((config.image_token_count,), present, dtype=np.bool_))
        embedding = self.checkpoint.fp32(f"{self._root}vlm_with_expert.vlm.model.text_model.embed_tokens.weight")
        language = self._scale(self._embedding(observation.language_tokens, embedding), scale)
        pieces.append(language)
        masks.append(observation.language_mask)
        padded_state = np.zeros((config.max_state_dim,), dtype=np.float32)
        padded_state[: config.state_dim] = observation.state
        state = self._linear(
            padded_state[None, :],
            f"{self._root}state_proj.weight",
            f"{self._root}state_proj.bias",
        )
        pieces.append(state)
        masks.append(np.ones((1,), dtype=np.bool_))
        hidden = np.ascontiguousarray(np.concatenate(pieces, axis=0))
        padding = np.ascontiguousarray(np.concatenate(masks))
        attention_ar = np.zeros((config.prefix_length,), dtype=np.bool_)
        attention_ar[-1] = True
        mask = self._block_mask(padding, attention_ar)
        positions = (np.cumsum(padding, dtype=np.int32) - 1).astype(np.int32)
        keys: list[npt.NDArray[np.float32]] = []
        values: list[npt.NDArray[np.float32]] = []
        text = f"{self._root}vlm_with_expert.vlm.model.text_model"
        for layer in range(config.vlm_num_layers):
            prefix = f"{text}.layers.{layer}"
            normalized = self._rms_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.input_layernorm.weight"),
                config.rms_norm_eps,
            )
            query = self._linear(normalized, f"{prefix}.self_attn.q_proj.weight").reshape(
                config.prefix_length, config.num_attention_heads, config.head_dim
            )
            key = self._linear(normalized, f"{prefix}.self_attn.k_proj.weight").reshape(
                config.prefix_length, config.num_key_value_heads, config.head_dim
            )
            value = self._linear(normalized, f"{prefix}.self_attn.v_proj.weight").reshape(
                config.prefix_length, config.num_key_value_heads, config.head_dim
            )
            query = self._split_half_rope(query, positions, config.rope_theta)
            key = self._split_half_rope(key, positions, config.rope_theta)
            keys.append(key)
            values.append(value)
            attended = self._attention(query, key, value, mask)
            hidden = self._residual(hidden, self._linear(attended, f"{prefix}.self_attn.o_proj.weight"))
            residual = hidden
            normalized = self._rms_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.post_attention_layernorm.weight"),
                config.rms_norm_eps,
            )
            gate = self._linear(normalized, f"{prefix}.mlp.gate_proj.weight")
            up = self._linear(normalized, f"{prefix}.mlp.up_proj.weight")
            activated = self._swiglu(gate, up)
            hidden = self._residual(residual, self._linear(activated, f"{prefix}.mlp.down_proj.weight"))
        hidden = self._rms_norm(
            hidden,
            self.checkpoint.fp32(f"{text}.norm.weight"),
            config.rms_norm_eps,
        )
        return hidden, _PrefixCache(tuple(keys), tuple(values), padding)

    def _suffix_embedding(
        self: Self, actions: npt.NDArray[np.float32], timestep: np.float32
    ) -> npt.NDArray[np.float32]:
        config = self.config
        action = self._linear(
            actions,
            f"{self._root}action_in_proj.weight",
            f"{self._root}action_in_proj.bias",
        )
        time = sinusoidal_time_embedding(
            timestep,
            config.expert_hidden_size,
            config.min_period,
            config.max_period,
        )
        joined = np.concatenate((action, np.broadcast_to(time, action.shape)), axis=1)
        hidden = self._linear(
            np.ascontiguousarray(joined),
            f"{self._root}action_time_mlp_in.weight",
            f"{self._root}action_time_mlp_in.bias",
        )
        hidden = self._silu(hidden)
        return self._linear(
            hidden,
            f"{self._root}action_time_mlp_out.weight",
            f"{self._root}action_time_mlp_out.bias",
        )

    def _denoise(
        self: Self,
        actions: npt.NDArray[np.float32],
        timestep: np.float32,
        cache: _PrefixCache,
    ) -> npt.NDArray[np.float32]:
        config = self.config
        hidden = self._suffix_embedding(actions, timestep)
        suffix_mask = np.ones((config.chunk_size,), dtype=np.bool_)
        suffix_ar = np.ones((config.chunk_size,), dtype=np.bool_)
        suffix_attention = self._block_mask(suffix_mask, suffix_ar)
        full_mask = np.concatenate(
            (
                np.broadcast_to(cache.padding_mask[None, :], (config.chunk_size, config.prefix_length)),
                suffix_attention,
            ),
            axis=1,
        )
        prefix_offset = int(np.sum(cache.padding_mask, dtype=np.int32))
        global_positions = np.arange(prefix_offset, prefix_offset + config.chunk_size, dtype=np.int32)
        local_positions = np.arange(config.chunk_size, dtype=np.int32)
        expert = f"{self._root}vlm_with_expert.lm_expert"
        for layer in range(config.expert_num_layers):
            prefix = f"{expert}.layers.{layer}"
            normalized = self._rms_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.input_layernorm.weight"),
                config.rms_norm_eps,
            )
            query = self._linear(normalized, f"{prefix}.self_attn.q_proj.weight").reshape(
                config.chunk_size, config.num_attention_heads, config.head_dim
            )
            if layer % config.self_attn_every_n_layers == 0:
                key = self._linear(normalized, f"{prefix}.self_attn.k_proj.weight").reshape(
                    config.chunk_size, config.num_key_value_heads, config.head_dim
                )
                value = self._linear(normalized, f"{prefix}.self_attn.v_proj.weight").reshape(
                    config.chunk_size, config.num_key_value_heads, config.head_dim
                )
                rotated_query = self._split_half_rope(query, global_positions, config.rope_theta)
                key = self._split_half_rope(key, global_positions, config.rope_theta)
                all_key = np.concatenate((cache.keys[layer], key), axis=0)
                all_value = np.concatenate((cache.values[layer], value), axis=0)
                attended = self._attention(rotated_query, all_key, all_value, full_mask)
            else:
                prefix_key = np.ascontiguousarray(cache.keys[layer].reshape(config.prefix_length, -1))
                prefix_value = np.ascontiguousarray(cache.values[layer].reshape(config.prefix_length, -1))
                key = self._linear(prefix_key, f"{prefix}.self_attn.k_proj.weight").reshape(
                    config.prefix_length, config.num_key_value_heads, config.head_dim
                )
                value = self._linear(prefix_value, f"{prefix}.self_attn.v_proj.weight").reshape(
                    config.prefix_length, config.num_key_value_heads, config.head_dim
                )
                rotated_query = self._split_half_rope(query, local_positions, config.rope_theta)
                cross_mask = np.broadcast_to(cache.padding_mask[None, :], (config.chunk_size, config.prefix_length))
                attended = self._attention(rotated_query, key, value, cross_mask)
            hidden = self._residual(hidden, self._linear(attended, f"{prefix}.self_attn.o_proj.weight"))
            residual = hidden
            normalized = self._rms_norm(
                hidden,
                self.checkpoint.fp32(f"{prefix}.post_attention_layernorm.weight"),
                config.rms_norm_eps,
            )
            gate = self._linear(normalized, f"{prefix}.mlp.gate_proj.weight")
            up = self._linear(normalized, f"{prefix}.mlp.up_proj.weight")
            hidden = self._residual(
                residual,
                self._linear(self._swiglu(gate, up), f"{prefix}.mlp.down_proj.weight"),
            )
        hidden = self._rms_norm(
            hidden,
            self.checkpoint.fp32(f"{expert}.norm.weight"),
            config.rms_norm_eps,
        )
        return self._linear(
            hidden,
            f"{self._root}action_out_proj.weight",
            f"{self._root}action_out_proj.bias",
        )

    def predict_action_chunk(self: Self, observation: SmolVLAObservation) -> SmolVLAActionChunk:
        """Run image-to-action flow inference with deterministic supplied noise."""
        observation.validate(self.checkpoint)
        stage_ns: dict[str, int] = {}
        started = perf_counter_ns()
        _, cache = self._prefix(observation)
        stage_ns["prefix"] = perf_counter_ns() - started
        actions = np.array(observation.noise, dtype=np.float32, copy=True)
        denoise_total = 0
        dt = np.float32(-1.0 / self.config.num_steps)
        for step in range(self.config.num_steps):
            timestep = np.float32(1.0 + step * float(dt))
            started = perf_counter_ns()
            velocity = self._denoise(actions, timestep, cache)
            denoise_total += perf_counter_ns() - started
            actions += dt * velocity
        stage_ns["denoise"] = denoise_total
        stage_ns["total"] = stage_ns["prefix"] + stage_ns["denoise"]
        padded = np.ascontiguousarray(actions)
        logical = np.ascontiguousarray(padded[:, : self.config.action_dim])
        if not np.all(np.isfinite(logical)):
            raise FloatingPointError("SmolVLA inference produced a non-finite action")
        return SmolVLAActionChunk(logical, padded, stage_ns)


__all__ = [
    "SmolVLAActionChunk",
    "SmolVLAObservation",
    "SmolVLAReferenceRuntime",
    "resize_with_top_left_padding_rgb",
    "sinusoidal_time_embedding",
]
