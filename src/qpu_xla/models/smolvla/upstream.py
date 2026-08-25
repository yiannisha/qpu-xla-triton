"""Pinned upstream LeRobot/PyTorch correctness and performance oracle."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.models.smolvla.config import (
    UPSTREAM_LEROBOT_REVISION,
    UPSTREAM_SMOLVLA_REPOSITORY,
    UPSTREAM_SMOLVLA_REVISION,
    SmolVLAConfig,
)
from qpu_xla.models.smolvla.reference import SmolVLAObservation


def _git_revision(checkout: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot determine LeRobot revision in {checkout}") from exc
    return result.stdout.strip()


class UpstreamTorchSmolVLAOracle:
    """Run the authoritative pinned ``SmolVLAPolicy`` on CPU in FP32."""

    label = "upstream_torch_cpu_fp32"
    _state_key: str
    _tokens_key: str
    _language_mask_key: str

    def __init__(self: Self, policy: Any, config: SmolVLAConfig, torch_module: Any) -> None:
        """Bind an already-loaded upstream policy and its matching native contract."""
        self.policy = policy
        self.config = config
        self.torch = torch_module

    @classmethod
    def open(
        cls: type[UpstreamTorchSmolVLAOracle],
        config: SmolVLAConfig,
        *,
        lerobot_checkout: str | Path,
        checkpoint: str | Path = UPSTREAM_SMOLVLA_REPOSITORY,
        cache_directory: str | Path | None = None,
        cpu_threads: int | None = None,
    ) -> UpstreamTorchSmolVLAOracle:
        """Load only the pinned LeRobot source and pinned model revision."""
        checkout = Path(lerobot_checkout).resolve()
        revision = _git_revision(checkout)
        if revision != UPSTREAM_LEROBOT_REVISION:
            raise ValueError(f"LeRobot checkout is {revision}, expected pinned revision {UPSTREAM_LEROBOT_REVISION}")
        source_root = checkout / "src"
        if not source_root.is_dir():
            raise ValueError(f"LeRobot checkout has no src directory: {checkout}")
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            torch = importlib.import_module("torch")
            constants = importlib.import_module("lerobot.utils.constants")
            configuration = importlib.import_module("lerobot.policies.smolvla.configuration_smolvla")
            modeling = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "upstream_torch_cpu_fp32 requires the pinned LeRobot SmolVLA optional dependencies"
            ) from exc
        if cpu_threads is not None:
            if cpu_threads <= 0:
                raise ValueError("upstream Torch CPU thread count must be positive")
            torch.set_num_threads(cpu_threads)
        upstream_config = configuration.SmolVLAConfig.from_pretrained(
            str(checkpoint),
            revision=UPSTREAM_SMOLVLA_REVISION,
            cache_dir=None if cache_directory is None else str(cache_directory),
        )
        upstream_config.device = "cpu"
        policy = modeling.SmolVLAPolicy.from_pretrained(
            str(checkpoint),
            config=upstream_config,
            revision=UPSTREAM_SMOLVLA_REVISION,
            cache_dir=None if cache_directory is None else str(cache_directory),
            strict=True,
        )
        policy.to(device="cpu", dtype=torch.float32)
        policy.eval()
        cls._validate_policy(policy, upstream_config, config)
        instance = cls(policy, config, torch)
        instance._state_key = constants.OBS_STATE
        instance._tokens_key = constants.OBS_LANGUAGE_TOKENS
        instance._language_mask_key = constants.OBS_LANGUAGE_ATTENTION_MASK
        return instance

    @staticmethod
    def _validate_policy(policy: Any, upstream_config: Any, config: SmolVLAConfig) -> None:
        """Reject an upstream graph that is not the native artifact's exact topology."""
        expected = config.expected_shapes()
        observed = {name: tuple(tensor.shape) for name, tensor in policy.state_dict().items()}
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            unexpected = sorted(set(observed) - set(expected))
            mismatched = sorted(
                name for name in set(expected) & set(observed) if expected[name] != observed[name]
            )
            raise ValueError(
                "upstream SmolVLA checkpoint topology does not match the native artifact; "
                f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
            )
        scalar_fields = {
            "chunk_size": config.chunk_size,
            "num_steps": config.num_steps,
            "tokenizer_max_length": config.tokenizer_max_length,
            "max_state_dim": config.max_state_dim,
            "max_action_dim": config.max_action_dim,
            "self_attn_every_n_layers": config.self_attn_every_n_layers,
        }
        mismatched_config: dict[str, object] = {
            name: (getattr(upstream_config, name, None), expected_value)
            for name, expected_value in scalar_fields.items()
            if getattr(upstream_config, name, None) != expected_value
        }
        image_keys = tuple(upstream_config.image_features)
        if image_keys != config.image_keys:
            mismatched_config["image_keys"] = (image_keys, config.image_keys)
        if mismatched_config:
            raise ValueError(f"upstream SmolVLA inference config mismatch: {mismatched_config}")

    def predict_action_chunk(self: Self, observation: SmolVLAObservation) -> npt.NDArray[np.float32]:
        """Execute upstream preprocessing and deterministic flow sampling."""
        observation.validate(cast(Any, _ConfigCheckpoint(self.config)))
        torch = self.torch
        batch: dict[str, Any] = {
            self._state_key: torch.from_numpy(observation.state[None]).to(dtype=torch.float32),
            self._tokens_key: torch.from_numpy(observation.language_tokens[None]),
            self._language_mask_key: torch.from_numpy(observation.language_mask[None]),
        }
        for key, image, present in zip(
            self.config.image_keys,
            observation.images,
            observation.image_masks,
            strict=True,
        ):
            chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
            chw *= np.float32(1.0 / 255.0)
            batch[key] = torch.from_numpy(chw[None])
            batch[f"{key}_padding_mask"] = torch.tensor([bool(present)], dtype=torch.bool)
        noise = torch.from_numpy(observation.noise[None])
        with torch.inference_mode():
            output = self.policy.predict_action_chunk(batch, noise=noise)
        result = output.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
        return np.ascontiguousarray(result)


class _ConfigCheckpoint:
    def __init__(self: Self, config: SmolVLAConfig) -> None:
        self.config = config


__all__ = ["UpstreamTorchSmolVLAOracle"]
