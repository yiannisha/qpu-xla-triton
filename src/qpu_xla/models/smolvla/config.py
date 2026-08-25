"""Architecture and inference contracts for the native SmolVLA baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Self

import numpy as np

UPSTREAM_LEROBOT_REVISION = "8b256a6c0d4769c3cc3e7e98f04940126398a391"
UPSTREAM_SMOLVLA_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
UPSTREAM_SMOLVLA_REPOSITORY = "lerobot/smolvla_base"


@dataclass(frozen=True, slots=True)
class SmolVLAConfig:
    """All topology fields required by inference and checkpoint validation.

    Defaults describe the pinned ``lerobot/smolvla_base`` checkpoint.  The
    fields remain configurable so small, complete graphs can be used for CPU
    and hardware differential tests without pretending to be the production
    model.
    """

    image_keys: tuple[str, ...] = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    input_image_height: int = 256
    input_image_width: int = 256
    image_size: int = 512
    patch_size: int = 16
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_layer_norm_eps: float = 1e-6
    pixel_shuffle_factor: int = 4
    vocab_size: int = 49_280
    vlm_hidden_size: int = 960
    vlm_intermediate_size: int = 2560
    vlm_num_layers: int = 16
    num_attention_heads: int = 15
    num_key_value_heads: int = 5
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    # The SmolVLM source config advertises 100_000, but the pinned LeRobot
    # ``apply_rope`` inference helper uses its own 10_000 default. This field
    # follows the executed upstream graph, which is the benchmark oracle.
    rope_theta: float = 10_000.0
    tokenizer_max_length: int = 48
    state_dim: int = 6
    action_dim: int = 6
    max_state_dim: int = 32
    max_action_dim: int = 32
    expert_hidden_size: int = 720
    expert_intermediate_size: int = 2048
    expert_num_layers: int = 16
    self_attn_every_n_layers: int = 2
    chunk_size: int = 50
    num_steps: int = 10
    min_period: float = 0.004
    max_period: float = 4.0
    source_prefix: str = "model."
    input_features: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self: Self) -> None:
        """Validate topology invariants required by the native kernels."""
        sizes = (
            len(self.image_keys),
            self.input_image_height,
            self.input_image_width,
            self.image_size,
            self.patch_size,
            self.vision_hidden_size,
            self.vision_intermediate_size,
            self.vision_num_layers,
            self.vision_num_heads,
            self.pixel_shuffle_factor,
            self.vocab_size,
            self.vlm_hidden_size,
            self.vlm_intermediate_size,
            self.vlm_num_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.tokenizer_max_length,
            self.state_dim,
            self.action_dim,
            self.max_state_dim,
            self.max_action_dim,
            self.expert_hidden_size,
            self.expert_intermediate_size,
            self.expert_num_layers,
            self.self_attn_every_n_layers,
            self.chunk_size,
            self.num_steps,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("SmolVLA architecture sizes must be positive")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("SmolVLA image keys must be unique")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.patches_per_side % self.pixel_shuffle_factor:
            raise ValueError("patch grid must be divisible by the pixel-shuffle factor")
        if self.vision_hidden_size % self.vision_num_heads:
            raise ValueError("vision hidden size must divide evenly into attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if self.vlm_hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("VLM hidden size must equal query heads times head_dim")
        if self.expert_num_layers != self.vlm_num_layers:
            raise ValueError("the native runtime currently requires one expert layer per VLM layer")
        if self.max_state_dim < self.state_dim or self.max_action_dim < self.action_dim:
            raise ValueError("padded state/action dimensions cannot be smaller than logical dimensions")
        if self.expert_hidden_size % 16 or self.vlm_hidden_size % 16 or self.head_dim % 16:
            raise ValueError("SmolVLA transformer widths must be 16-aligned for VideoCore kernels")
        positive_floats = (
            self.vision_layer_norm_eps,
            self.rms_norm_eps,
            self.rope_theta,
            self.min_period,
            self.max_period,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive_floats):
            raise ValueError("SmolVLA numerical constants must be finite and positive")
        if self.rope_theta <= 1 or self.max_period <= self.min_period:
            raise ValueError("SmolVLA RoPE/time-embedding ranges are invalid")

    @property
    def patches_per_side(self: Self) -> int:
        """Return the patch-grid width and height."""
        return self.image_size // self.patch_size

    @property
    def vision_token_count(self: Self) -> int:
        """Return the number of pre-connector patches in one square image."""
        return self.patches_per_side**2

    @property
    def image_token_count(self: Self) -> int:
        """Return the connector token count produced by one image."""
        return self.vision_token_count // self.pixel_shuffle_factor**2

    @property
    def prefix_length(self: Self) -> int:
        """Return image + language + state tokens in the fixed inference prefix."""
        return len(self.image_keys) * self.image_token_count + self.tokenizer_max_length + 1

    @property
    def key_value_size(self: Self) -> int:
        """Return the flattened grouped-key/value projection width."""
        return self.num_key_value_heads * self.head_dim

    @property
    def connector_input_size(self: Self) -> int:
        """Return the pixel-shuffled connector input feature width."""
        return self.vision_hidden_size * self.pixel_shuffle_factor**2

    @property
    def model_shape_class(self: Self) -> str:
        """Return an exact topology key for full-replay promotion evidence."""
        fields = (
            len(self.image_keys),
            self.input_image_height,
            self.input_image_width,
            self.image_size,
            self.patch_size,
            self.vision_hidden_size,
            self.vision_num_layers,
            self.vlm_hidden_size,
            self.vlm_num_layers,
            self.expert_hidden_size,
            self.expert_num_layers,
            self.prefix_length,
            self.chunk_size,
            self.num_steps,
        )
        return "x".join(map(str, fields))

    def expected_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        """Return the complete pinned checkpoint tensor-name and shape contract."""
        root = self.source_prefix
        vision = f"{root}vlm_with_expert.vlm.model.vision_model"
        text = f"{root}vlm_with_expert.vlm.model.text_model"
        expert = f"{root}vlm_with_expert.lm_expert"
        shapes: dict[str, tuple[int, ...]] = {
            f"{root}state_proj.weight": (self.vlm_hidden_size, self.max_state_dim),
            f"{root}state_proj.bias": (self.vlm_hidden_size,),
            f"{root}action_in_proj.weight": (self.expert_hidden_size, self.max_action_dim),
            f"{root}action_in_proj.bias": (self.expert_hidden_size,),
            f"{root}action_out_proj.weight": (self.max_action_dim, self.expert_hidden_size),
            f"{root}action_out_proj.bias": (self.max_action_dim,),
            f"{root}action_time_mlp_in.weight": (self.expert_hidden_size, self.expert_hidden_size * 2),
            f"{root}action_time_mlp_in.bias": (self.expert_hidden_size,),
            f"{root}action_time_mlp_out.weight": (self.expert_hidden_size, self.expert_hidden_size),
            f"{root}action_time_mlp_out.bias": (self.expert_hidden_size,),
            f"{vision}.embeddings.patch_embedding.weight": (
                self.vision_hidden_size,
                3,
                self.patch_size,
                self.patch_size,
            ),
            f"{vision}.embeddings.patch_embedding.bias": (self.vision_hidden_size,),
            f"{vision}.embeddings.position_embedding.weight": (
                self.vision_token_count,
                self.vision_hidden_size,
            ),
            f"{vision}.post_layernorm.weight": (self.vision_hidden_size,),
            f"{vision}.post_layernorm.bias": (self.vision_hidden_size,),
            f"{root}vlm_with_expert.vlm.model.connector.modality_projection.proj.weight": (
                self.vlm_hidden_size,
                self.connector_input_size,
            ),
            f"{text}.embed_tokens.weight": (self.vocab_size, self.vlm_hidden_size),
            f"{text}.norm.weight": (self.vlm_hidden_size,),
            f"{root}vlm_with_expert.vlm.lm_head.weight": (self.vocab_size, self.vlm_hidden_size),
            f"{expert}.norm.weight": (self.expert_hidden_size,),
        }
        for layer in range(self.vision_num_layers):
            prefix = f"{vision}.encoder.layers.{layer}"
            shapes.update(
                {
                    f"{prefix}.layer_norm1.weight": (self.vision_hidden_size,),
                    f"{prefix}.layer_norm1.bias": (self.vision_hidden_size,),
                    f"{prefix}.layer_norm2.weight": (self.vision_hidden_size,),
                    f"{prefix}.layer_norm2.bias": (self.vision_hidden_size,),
                    f"{prefix}.self_attn.q_proj.weight": (self.vision_hidden_size, self.vision_hidden_size),
                    f"{prefix}.self_attn.q_proj.bias": (self.vision_hidden_size,),
                    f"{prefix}.self_attn.k_proj.weight": (self.vision_hidden_size, self.vision_hidden_size),
                    f"{prefix}.self_attn.k_proj.bias": (self.vision_hidden_size,),
                    f"{prefix}.self_attn.v_proj.weight": (self.vision_hidden_size, self.vision_hidden_size),
                    f"{prefix}.self_attn.v_proj.bias": (self.vision_hidden_size,),
                    f"{prefix}.self_attn.out_proj.weight": (self.vision_hidden_size, self.vision_hidden_size),
                    f"{prefix}.self_attn.out_proj.bias": (self.vision_hidden_size,),
                    f"{prefix}.mlp.fc1.weight": (self.vision_intermediate_size, self.vision_hidden_size),
                    f"{prefix}.mlp.fc1.bias": (self.vision_intermediate_size,),
                    f"{prefix}.mlp.fc2.weight": (self.vision_hidden_size, self.vision_intermediate_size),
                    f"{prefix}.mlp.fc2.bias": (self.vision_hidden_size,),
                }
            )
        for layer in range(self.vlm_num_layers):
            prefix = f"{text}.layers.{layer}"
            shapes.update(self._decoder_layer_shapes(prefix, self.vlm_hidden_size, cross_attention=False))
        for layer in range(self.expert_num_layers):
            prefix = f"{expert}.layers.{layer}"
            shapes.update(
                self._decoder_layer_shapes(
                    prefix,
                    self.expert_hidden_size,
                    cross_attention=layer % self.self_attn_every_n_layers != 0,
                )
            )
        return shapes

    def _decoder_layer_shapes(
        self: Self,
        prefix: str,
        hidden_size: int,
        *,
        cross_attention: bool,
    ) -> dict[str, tuple[int, ...]]:
        key_input = self.key_value_size if cross_attention else hidden_size
        intermediate = (
            self.vlm_intermediate_size if hidden_size == self.vlm_hidden_size else self.expert_intermediate_size
        )
        return {
            f"{prefix}.input_layernorm.weight": (hidden_size,),
            f"{prefix}.self_attn.q_proj.weight": (self.vlm_hidden_size, hidden_size),
            f"{prefix}.self_attn.k_proj.weight": (self.key_value_size, key_input),
            f"{prefix}.self_attn.v_proj.weight": (self.key_value_size, key_input),
            f"{prefix}.self_attn.o_proj.weight": (hidden_size, self.vlm_hidden_size),
            f"{prefix}.post_attention_layernorm.weight": (hidden_size,),
            f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden_size),
            f"{prefix}.mlp.up_proj.weight": (intermediate, hidden_size),
            f"{prefix}.mlp.down_proj.weight": (hidden_size, intermediate),
        }

    def estimate_weight_bytes(self: Self, *, numerics: str = "fp32") -> int:
        """Estimate artifact bytes, including per-output W8A8 scales."""
        if numerics not in {"fp32", "w8a8"}:
            raise ValueError("SmolVLA numerics must be 'fp32' or 'w8a8'")
        total = 0
        for name, shape in self.expected_shapes().items():
            count = int(np.prod(shape, dtype=np.int64))
            quantized = (
                numerics == "w8a8"
                and len(shape) in {2, 4}
                and name.endswith(".weight")
                and not name.endswith(
                    (
                        "embed_tokens.weight",
                        "position_embedding.weight",
                        "lm_head.weight",
                    )
                )
            )
            if quantized:
                output_channels = shape[0]
                total += count + output_channels * np.dtype(np.float32).itemsize
            else:
                total += count * np.dtype(np.float32).itemsize
        return total

    @classmethod
    def from_huggingface_json(cls: type[SmolVLAConfig], path: str | PathLike[str]) -> SmolVLAConfig:
        """Load and validate the LeRobot policy fields used by this runtime."""
        location = Path(path)
        try:
            payload = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load SmolVLA config {location}") from exc
        if not isinstance(payload, dict) or payload.get("type") != "smolvla":
            raise ValueError("SmolVLA config must contain an object with type='smolvla'")
        input_features = payload.get("input_features")
        output_features = payload.get("output_features")
        if not isinstance(input_features, dict) or not isinstance(output_features, dict):
            raise ValueError("SmolVLA config is missing input/output feature maps")
        image_keys = tuple(
            name
            for name, feature in input_features.items()
            if isinstance(feature, dict) and feature.get("type") == "VISUAL"
        )
        state = input_features.get("observation.state")
        action = output_features.get("action")
        if not image_keys or not isinstance(state, dict) or not isinstance(action, dict):
            raise ValueError("SmolVLA config requires visual, state, and action features")
        image_shape = input_features[image_keys[0]].get("shape")
        if (
            not isinstance(image_shape, list)
            or len(image_shape) != 3
            or image_shape[0] != 3
            or any(input_features[name].get("shape") != image_shape for name in image_keys)
        ):
            raise ValueError("native SmolVLA requires equal three-channel image shapes")
        state_shape, action_shape = state.get("shape"), action.get("shape")
        if not isinstance(state_shape, list) or len(state_shape) != 1:
            raise ValueError("SmolVLA state feature must be rank one")
        if not isinstance(action_shape, list) or len(action_shape) != 1:
            raise ValueError("SmolVLA action feature must be rank one")
        resize = payload.get("resize_imgs_with_padding", [512, 512])
        if not isinstance(resize, list) or len(resize) != 2:
            raise ValueError("SmolVLA resize_imgs_with_padding must have width and height")
        if resize[0] != resize[1]:
            raise ValueError("native SmolVLA currently requires a square padded image target")
        required_semantics = {
            "adapt_to_pi_aloha": False,
            "add_image_special_tokens": False,
            "use_cache": True,
            "pad_language_to": "max_length",
            "prefix_length": 0,
            "n_obs_steps": 1,
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        }
        semantic_mismatches = {
            name: (payload.get(name, expected), expected)
            for name, expected in required_semantics.items()
            if payload.get(name, expected) != expected
        }
        if semantic_mismatches:
            raise ValueError(f"native SmolVLA does not implement policy semantics: {semantic_mismatches}")
        required_integer = {
            "chunk_size": ("chunk_size", 50),
            "num_steps": ("num_steps", 10),
            "tokenizer_max_length": ("tokenizer_max_length", 48),
            "max_state_dim": ("max_state_dim", 32),
            "max_action_dim": ("max_action_dim", 32),
            "vlm_num_layers": ("num_vlm_layers", 16),
            "expert_num_layers": ("num_vlm_layers", 16),
            "self_attn_every_n_layers": ("self_attn_every_n_layers", 2),
        }
        values: dict[str, int] = {}
        for field_name, (source_name, default) in required_integer.items():
            value = payload.get(source_name, default)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"SmolVLA config field {source_name} must be an integer")
            values[field_name] = value
        if payload.get("attention_mode", "cross_attn") != "cross_attn":
            raise ValueError("native SmolVLA currently implements the cross_attn checkpoint topology")
        expert_layers = payload.get("num_expert_layers", 0)
        if expert_layers not in {0, values["vlm_num_layers"]}:
            raise ValueError("native SmolVLA requires one action-expert layer per retained VLM layer")
        if float(payload.get("expert_width_multiplier", 0.75)) != 0.75:
            raise ValueError("native SmolVLA artifact contract requires the pinned 0.75 expert width")
        return cls(
            image_keys=image_keys,
            input_image_height=int(image_shape[1]),
            input_image_width=int(image_shape[2]),
            image_size=int(resize[0]),
            state_dim=int(state_shape[0]),
            action_dim=int(action_shape[0]),
            min_period=float(payload.get("min_period", 0.004)),
            max_period=float(payload.get("max_period", 4.0)),
            input_features={name: tuple(feature["shape"]) for name, feature in input_features.items()},
            chunk_size=values["chunk_size"],
            num_steps=values["num_steps"],
            tokenizer_max_length=values["tokenizer_max_length"],
            max_state_dim=values["max_state_dim"],
            max_action_dim=values["max_action_dim"],
            vlm_num_layers=values["vlm_num_layers"],
            expert_num_layers=values["expert_num_layers"],
            self_attn_every_n_layers=values["self_attn_every_n_layers"],
        )


__all__ = [
    "SmolVLAConfig",
    "UPSTREAM_LEROBOT_REVISION",
    "UPSTREAM_SMOLVLA_REPOSITORY",
    "UPSTREAM_SMOLVLA_REVISION",
]
