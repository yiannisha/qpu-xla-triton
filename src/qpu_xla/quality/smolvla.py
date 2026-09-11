"""Paired whole-action qualification for complete SmolVLA runtimes."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from qpu_xla.models.smolvla import SmolVLAReplay
from qpu_xla.models.smolvla.reference import SmolVLAObservation
from qpu_xla.quality.metrics import action_metrics


class ActionRuntime(Protocol):
    """Minimum complete-runtime protocol needed by replay qualification."""

    def predict_action_chunk(self, observation: SmolVLAObservation) -> Any:
        """Return a native action chunk or an FP32 array."""


def _actions(value: Any) -> npt.NDArray[np.float32]:
    result = value.actions if hasattr(value, "actions") else value
    return np.ascontiguousarray(result, dtype=np.float32)


def validate_smolvla_replay(
    replay: SmolVLAReplay,
    *,
    require_qualification_shape: bool = True,
) -> None:
    """Validate the sample and oracle requirements shared by all runners."""
    if require_qualification_shape:
        if len(replay.observations) != 100:
            raise ValueError("SmolVLA qualification requires exactly 100 replay observations")
        if replay.episode_ids is None or len(set(replay.episode_ids)) < 10:
            raise ValueError("SmolVLA qualification requires observations from at least 10 episodes")
    if replay.upstream_actions is None:
        raise ValueError("SmolVLA comparison requires recorded upstream actions")


def smolvla_replay_metrics(
    replay: SmolVLAReplay,
    candidate_actions: npt.ArrayLike,
    *,
    require_qualification_shape: bool = True,
) -> dict[str, Any]:
    """Score already-recorded candidate actions against the replay oracle."""
    validate_smolvla_replay(replay, require_qualification_shape=require_qualification_shape)
    assert replay.upstream_actions is not None
    return action_metrics(
        np.ascontiguousarray(replay.upstream_actions),
        np.ascontiguousarray(candidate_actions, dtype=np.float32),
        episode_ids=replay.episode_ids,
    )


def evaluate_smolvla_replay(
    replay: SmolVLAReplay,
    candidate: ActionRuntime,
    *,
    baseline: ActionRuntime | None = None,
    require_qualification_shape: bool = True,
) -> tuple[dict[str, Any], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Run matched replay inputs and report deltas against upstream CPU FP32."""
    if replay.upstream_actions is not None:
        validate_smolvla_replay(replay, require_qualification_shape=require_qualification_shape)
    elif require_qualification_shape:
        if len(replay.observations) != 100:
            raise ValueError("SmolVLA qualification requires exactly 100 replay observations")
        if replay.episode_ids is None or len(set(replay.episode_ids)) < 10:
            raise ValueError("SmolVLA qualification requires observations from at least 10 episodes")
    if replay.upstream_actions is None and baseline is None:
        raise ValueError("SmolVLA comparison requires recorded or executable upstream actions")
    baseline_values = (
        np.ascontiguousarray(replay.upstream_actions)
        if replay.upstream_actions is not None
        else np.stack([_actions(baseline.predict_action_chunk(item)) for item in replay.observations])  # type: ignore[union-attr]
    )
    candidate_values = np.stack([_actions(candidate.predict_action_chunk(item)) for item in replay.observations])
    metrics = action_metrics(baseline_values, candidate_values, episode_ids=replay.episode_ids)
    return metrics, baseline_values, candidate_values


__all__ = [
    "ActionRuntime",
    "evaluate_smolvla_replay",
    "smolvla_replay_metrics",
    "validate_smolvla_replay",
]
