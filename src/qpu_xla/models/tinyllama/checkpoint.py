"""Strict TinyLlama checkpoint contracts and safe portable NPZ loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TinyLlamaConfig:
    """Architecture fields required to validate and budget a TinyLlama checkpoint."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0

    def __post_init__(self: Self) -> None:
        """Reject model topologies that cannot use grouped-query attention correctly."""
        integer_fields = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("TinyLlama configuration sizes must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if not np.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if not np.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("rope_theta must be finite and greater than one")

    @property
    def head_dim(self: Self) -> int:
        """Return the feature width of one attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def key_value_size(self: Self) -> int:
        """Return the projection width of the grouped key/value heads."""
        return self.num_key_value_heads * self.head_dim

    def expected_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        """Return the complete canonical tensor-name and shape contract."""
        shapes = {
            "model.embed_tokens.weight": (self.vocab_size, self.hidden_size),
            "model.norm.weight": (self.hidden_size,),
            "lm_head.weight": (self.vocab_size, self.hidden_size),
        }
        for layer in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (self.hidden_size,),
                    f"{prefix}.self_attn.q_proj.weight": (self.hidden_size, self.hidden_size),
                    f"{prefix}.self_attn.k_proj.weight": (self.key_value_size, self.hidden_size),
                    f"{prefix}.self_attn.v_proj.weight": (self.key_value_size, self.hidden_size),
                    f"{prefix}.self_attn.o_proj.weight": (self.hidden_size, self.hidden_size),
                    f"{prefix}.post_attention_layernorm.weight": (self.hidden_size,),
                    f"{prefix}.mlp.gate_proj.weight": (self.intermediate_size, self.hidden_size),
                    f"{prefix}.mlp.up_proj.weight": (self.intermediate_size, self.hidden_size),
                    f"{prefix}.mlp.down_proj.weight": (self.hidden_size, self.intermediate_size),
                }
            )
        return shapes

    @classmethod
    def from_huggingface_json(cls: type[TinyLlamaConfig], path: str | PathLike[str]) -> TinyLlamaConfig:
        """Load the architecture fields needed from a standard Hugging Face ``config.json``."""
        location = Path(path)
        try:
            payload = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load Hugging Face TinyLlama config {location}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Hugging Face TinyLlama config must contain an object")
        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
        if any(name not in payload for name in required):
            raise ValueError("Hugging Face TinyLlama config is missing required architecture fields")
        values = {name: payload[name] for name in required}
        values["rms_norm_eps"] = payload.get("rms_norm_eps", 1e-5)
        values["rope_theta"] = payload.get("rope_theta", 10_000.0)
        if not all(isinstance(values[name], int) and not isinstance(values[name], bool) for name in required):
            raise ValueError("Hugging Face TinyLlama architecture fields must be integers")
        if not all(isinstance(values[name], int | float) for name in ("rms_norm_eps", "rope_theta")):
            raise ValueError("Hugging Face TinyLlama numerical fields must be numbers")
        return cls(**values)

    def estimate_weight_bytes(self: Self, *, quantized: bool = False) -> int:
        """Estimate strict-checkpoint weight storage, including INT8 row scales when requested."""
        total = 0
        for shape in self.expected_shapes().values():
            if len(shape) == 1:
                total += shape[0] * np.dtype(np.float32).itemsize
            elif quantized:
                total += shape[0] * shape[1] + shape[0] * np.dtype(np.float32).itemsize
            else:
                total += shape[0] * shape[1] * np.dtype(np.float32).itemsize
        return total


@dataclass(frozen=True, slots=True)
class TinyLlamaCheckpoint:
    """Validated, contiguous FP32 tensors for the CPU-reference model runtime."""

    config: TinyLlamaConfig
    tensors: dict[str, npt.NDArray[np.float32]]

    def __post_init__(self: Self) -> None:
        """Enforce the entire architecture and dtype contract at load boundaries."""
        expected = self.config.expected_shapes()
        if set(self.tensors) != set(expected):
            missing = sorted(set(expected) - set(self.tensors))
            unexpected = sorted(set(self.tensors) - set(expected))
            raise ValueError(
                f"checkpoint tensor names do not match config; missing={missing}, unexpected={unexpected}"
            )
        for name, shape in expected.items():
            tensor = self.tensors[name]
            if tensor.dtype != np.dtype(np.float32) or tensor.shape != shape or not tensor.flags.c_contiguous:
                raise ValueError(f"checkpoint tensor {name!r} must be contiguous float32 with shape {shape}")

    @classmethod
    def from_npz(
        cls: type[TinyLlamaCheckpoint], path: str | PathLike[str], config: TinyLlamaConfig
    ) -> TinyLlamaCheckpoint:
        """Load a portable NPZ artifact without allowing pickle/object deserialization."""
        location = Path(path)
        try:
            with np.load(location, allow_pickle=False) as archive:
                tensors = {name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in archive.files}
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load TinyLlama NPZ checkpoint {location}") from exc
        return cls(config, tensors)

    @classmethod
    def from_safetensors(
        cls: type[TinyLlamaCheckpoint], path: str | PathLike[str], config: TinyLlamaConfig
    ) -> TinyLlamaCheckpoint:
        """Load a standard SafeTensors weight file through its optional dependency.

        The source format is not executed; values are converted to contiguous
        FP32 then checked against the same complete tensor contract as NPZ.
        """
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError(
                "SafeTensors loading requires the optional 'safetensors' package; "
                "install it before loading a production TinyLlama checkpoint"
            ) from exc
        location = Path(path)
        try:
            loaded = load_file(str(location))
            tensors = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in loaded.items()}
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"cannot load TinyLlama SafeTensors checkpoint {location}") from exc
        return cls(config, tensors)

    @classmethod
    def from_safetensors_index(
        cls: type[TinyLlamaCheckpoint], path: str | PathLike[str], config: TinyLlamaConfig
    ) -> TinyLlamaCheckpoint:
        """Load a Hugging Face SafeTensors shard index without trusting extra tensors."""
        index_path = Path(path)
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load TinyLlama SafeTensors index {index_path}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
            raise ValueError("SafeTensors index must contain an object weight_map")
        weight_map = payload["weight_map"]
        if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
            raise ValueError("SafeTensors index weight_map must map tensor names to shard names")
        expected_names = set(config.expected_shapes())
        if set(weight_map) != expected_names:
            raise ValueError("SafeTensors index tensor names do not match TinyLlama config")
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError(
                "SafeTensors loading requires the optional 'safetensors' package; "
                "install it before loading a production TinyLlama checkpoint"
            ) from exc
        loaded: dict[str, npt.NDArray[np.float32]] = {}
        for shard in sorted(set(weight_map.values())):
            try:
                shard_tensors = load_file(str(index_path.parent / shard))
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(f"cannot load TinyLlama SafeTensors shard {shard}") from exc
            for name, value in shard_tensors.items():
                if name in expected_names and weight_map[name] == shard:
                    if name in loaded:
                        raise ValueError(f"SafeTensors shard index provides duplicate tensor {name!r}")
                    loaded[name] = np.ascontiguousarray(value, dtype=np.float32)
        return cls(config, loaded)
