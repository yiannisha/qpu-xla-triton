"""Recorded, deterministic SmolVLA inference sessions."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from qpu_xla.models.smolvla.config import (
    UPSTREAM_LEROBOT_REVISION,
    UPSTREAM_SMOLVLA_REVISION,
    SmolVLAConfig,
)
from qpu_xla.models.smolvla.reference import SmolVLAObservation


@dataclass(frozen=True, slots=True)
class SmolVLAReplay:
    """A sequence of exact RGB/token/state/noise inputs and optional oracle actions."""

    observations: tuple[SmolVLAObservation, ...]
    upstream_actions: npt.NDArray[np.float32] | None = None
    model_revision: str = UPSTREAM_SMOLVLA_REVISION
    lerobot_revision: str = UPSTREAM_LEROBOT_REVISION
    episode_ids: tuple[str, ...] | None = None
    instruction_ids: tuple[str, ...] | None = None

    def validate(self: Self, config: SmolVLAConfig) -> None:
        """Validate all sessions against one native model configuration."""
        if not self.observations:
            raise ValueError("SmolVLA replay must contain at least one session")
        if self.model_revision != UPSTREAM_SMOLVLA_REVISION:
            raise ValueError("SmolVLA replay model revision does not match the pinned baseline")
        if self.lerobot_revision != UPSTREAM_LEROBOT_REVISION:
            raise ValueError("SmolVLA replay LeRobot revision does not match the pinned baseline")
        for observation in self.observations:
            # Validation only needs the topology, so use a small protocol
            # object instead of requiring a second checkpoint instance.
            observation.validate(_ConfigCheckpoint(config))
        for name, values in (("episode_ids", self.episode_ids), ("instruction_ids", self.instruction_ids)):
            if values is not None and (len(values) != len(self.observations) or any(not value for value in values)):
                raise ValueError(f"SmolVLA replay {name} must identify every observation")
        if self.upstream_actions is not None and (
            self.upstream_actions.dtype != np.dtype(np.float32)
            or self.upstream_actions.shape != (len(self.observations), config.chunk_size, config.action_dim)
            or not self.upstream_actions.flags.c_contiguous
            or not np.all(np.isfinite(self.upstream_actions))
        ):
            raise ValueError("recorded upstream actions do not match the logical action contract")

    @classmethod
    def load(cls: type[SmolVLAReplay], path: str | PathLike[str], config: SmolVLAConfig) -> SmolVLAReplay:
        """Load a non-pickled NPZ replay with strict dtypes and dimensions."""
        location = Path(path)
        try:
            archive = np.load(location, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load SmolVLA replay {location}") from exc
        with archive:
            required = {
                "images",
                "image_masks",
                "language_tokens",
                "language_mask",
                "state",
                "noise",
                "model_revision",
                "lerobot_revision",
            }
            if not required.issubset(archive.files):
                raise ValueError(f"SmolVLA replay is missing arrays: {sorted(required - set(archive.files))}")
            images = np.ascontiguousarray(archive["images"])
            image_masks = np.ascontiguousarray(archive["image_masks"])
            language_tokens = np.ascontiguousarray(archive["language_tokens"])
            language_mask = np.ascontiguousarray(archive["language_mask"])
            state = np.ascontiguousarray(archive["state"])
            noise = np.ascontiguousarray(archive["noise"])
            upstream = (
                np.ascontiguousarray(archive["upstream_actions"]) if "upstream_actions" in archive.files else None
            )
            model_revision = str(archive["model_revision"].item())
            lerobot_revision = str(archive["lerobot_revision"].item())
            episode_ids = (
                tuple(str(value) for value in archive["episode_ids"].tolist())
                if "episode_ids" in archive.files
                else None
            )
            instruction_ids = (
                tuple(str(value) for value in archive["instruction_ids"].tolist())
                if "instruction_ids" in archive.files
                else None
            )
        if images.ndim != 5:
            raise ValueError("SmolVLA replay images must be sessions by cameras by HWC")
        sessions = images.shape[0]
        arrays = (image_masks, language_tokens, language_mask, state, noise)
        if any(array.shape[0] != sessions for array in arrays):
            raise ValueError("SmolVLA replay arrays must have one leading row per session")
        observations = tuple(
            SmolVLAObservation(
                images=tuple(np.ascontiguousarray(images[index, camera]) for camera in range(images.shape[1])),
                image_masks=np.ascontiguousarray(image_masks[index]),
                language_tokens=np.ascontiguousarray(language_tokens[index]),
                language_mask=np.ascontiguousarray(language_mask[index]),
                state=np.ascontiguousarray(state[index]),
                noise=np.ascontiguousarray(noise[index]),
            )
            for index in range(sessions)
        )
        replay = cls(observations, upstream, model_revision, lerobot_revision, episode_ids, instruction_ids)
        replay.validate(config)
        return replay

    def save(self: Self, path: str | PathLike[str], config: SmolVLAConfig) -> None:
        """Save a deterministic, portable NPZ replay without Python objects."""
        self.validate(config)
        payload: dict[str, Any] = {
            "images": np.stack([np.stack(item.images) for item in self.observations]),
            "image_masks": np.stack([item.image_masks for item in self.observations]),
            "language_tokens": np.stack([item.language_tokens for item in self.observations]),
            "language_mask": np.stack([item.language_mask for item in self.observations]),
            "state": np.stack([item.state for item in self.observations]),
            "noise": np.stack([item.noise for item in self.observations]),
            "model_revision": np.array(self.model_revision),
            "lerobot_revision": np.array(self.lerobot_revision),
        }
        if self.upstream_actions is not None:
            payload["upstream_actions"] = self.upstream_actions
        if self.episode_ids is not None:
            payload["episode_ids"] = np.asarray(self.episode_ids)
        if self.instruction_ids is not None:
            payload["instruction_ids"] = np.asarray(self.instruction_ids)
        np.savez_compressed(Path(path), **payload)


@dataclass(frozen=True, slots=True)
class _ConfigCheckpoint:
    """Minimal validation adapter accepted by ``SmolVLAObservation``."""

    config: SmolVLAConfig


__all__ = ["SmolVLAReplay"]
