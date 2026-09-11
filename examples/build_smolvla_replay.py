"""Build a deterministic 100-observation SmolVLA replay from a LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from qpu_xla.models.smolvla import (
    SmolVLAArtifact,
    SmolVLAObservation,
    SmolVLAReplay,
    UpstreamTorchSmolVLAOracle,
)
from qpu_xla.models.smolvla.config import UPSTREAM_LEROBOT_REVISION


def _stable_key(seed: int, value: object) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _checkout(checkout: Path) -> None:
    revision = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != UPSTREAM_LEROBOT_REVISION:
        raise ValueError(f"LeRobot checkout is {revision}, expected {UPSTREAM_LEROBOT_REVISION}")
    sys.path.insert(0, str(checkout / "src"))


def _selected_indices(dataset: Any, *, seed: int) -> tuple[list[int], list[str], list[str]]:
    by_episode: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index in range(len(dataset)):
        raw = dataset.get_raw_item(index)
        episode = str(raw.get("episode_index", ""))
        instruction = str(raw.get("task", raw.get("task_index", "")))
        if not episode or not instruction:
            raise ValueError("LeRobot replay source requires episode_index and task/task_index metadata")
        by_episode[episode].append((index, instruction))
    eligible = [episode for episode, values in by_episode.items() if len(values) >= 10]
    if len(eligible) < 10:
        raise ValueError("SmolVLA replay source needs at least 10 episodes with 10 observations each")
    episodes = sorted(eligible, key=lambda value: _stable_key(seed, value))[:10]
    selected: list[int] = []
    episode_ids: list[str] = []
    instruction_ids: list[str] = []
    for episode in episodes:
        values = sorted(by_episode[episode], key=lambda value: _stable_key(seed, value[0]))[:10]
        for index, instruction in values:
            selected.append(index)
            episode_ids.append(episode)
            instruction_ids.append(instruction)
    return selected, episode_ids, instruction_ids


def _selected_episodes(metadata: Any, *, seed: int) -> list[int]:
    """Choose episodes from metadata before downloading their data and videos."""
    metadata.ensure_readable()
    eligible = [
        int(row["episode_index"])
        for row in metadata.episodes
        if int(row["length"]) >= 10
    ]
    if len(eligible) < 10:
        raise ValueError("SmolVLA replay source needs at least 10 episodes with 10 observations each")
    return sorted(eligible, key=lambda value: _stable_key(seed, value))[:10]


def _to_uint8_hwc(value: Any, shape: tuple[int, int, int]) -> NDArray[np.uint8]:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    result = np.ascontiguousarray(array)
    if result.ndim != 3 or result.shape[2] != 3:
        raise ValueError(f"SmolVLA image has {result.shape}, expected an HWC RGB image")
    if result.shape != shape:
        target_height, target_width, _ = shape
        scale = min(target_width / result.shape[1], target_height / result.shape[0])
        resized_width = max(1, round(result.shape[1] * scale))
        resized_height = max(1, round(result.shape[0] * scale))
        resized = np.asarray(
            Image.fromarray(result).resize((resized_width, resized_height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        result = np.zeros(shape, dtype=np.uint8)
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2
        result[top : top + resized_height, left : left + resized_width] = resized
        result = np.ascontiguousarray(result)
    return result


def _image_mapping(values: list[str], expected: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            raise ValueError("--image-map values must use SOURCE=TARGET syntax")
        if source in mapping or target in mapping.values():
            raise ValueError("--image-map source and target keys must be unique")
        if target not in expected:
            raise ValueError(f"--image-map target {target!r} is not an artifact image key")
        mapping[source] = target
    if not mapping:
        raise ValueError("at least one explicit --image-map is required")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--lerobot-checkout", type=Path, required=True)
    parser.add_argument("--dataset-id", default="lerobot/svla_so100_pickplace")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--image-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="explicitly map a source dataset camera to an artifact camera; repeat per available view",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_910)
    parser.add_argument("--skip-upstream-actions", action="store_true")
    args = parser.parse_args()
    _checkout(args.lerobot_checkout.resolve())

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as UpstreamConfig
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    artifact = SmolVLAArtifact.open(args.artifact)
    config = artifact.checkpoint.config
    image_mapping = _image_mapping(args.image_map, config.image_keys)
    upstream_config = UpstreamConfig.from_pretrained(str(args.checkpoint))
    upstream_config.device = "cpu"
    metadata = LeRobotDatasetMetadata(args.dataset_id, root=args.dataset_root)
    missing_source_keys = sorted(set(image_mapping) - set(metadata.camera_keys))
    if missing_source_keys:
        raise ValueError(f"--image-map source keys are absent from the dataset: {missing_source_keys}")
    state_shape = tuple(metadata.features["observation.state"]["shape"])
    action_shape = tuple(metadata.features["action"]["shape"])
    if state_shape != (config.state_dim,) or action_shape != (config.action_dim,):
        raise ValueError(
            "dataset state/action shapes do not match the artifact: "
            f"state={state_shape}, action={action_shape}, expected={(config.state_dim,)}/{(config.action_dim,)}"
        )
    selected_episodes = _selected_episodes(metadata, seed=args.seed)
    dataset = LeRobotDataset(
        args.dataset_id,
        root=args.dataset_root,
        episodes=selected_episodes,
        return_uint8=True,
    )
    preprocessor, _ = make_pre_post_processors(
        upstream_config,
        pretrained_path=str(args.checkpoint),
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": image_mapping},
            "device_processor": {"device": "cpu"},
        },
    )
    indices, episode_ids, instruction_ids = _selected_indices(dataset, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    observations: list[SmolVLAObservation] = []
    expected_image_shape = (config.input_image_height, config.input_image_width, 3)
    for index in indices:
        processed = preprocessor(dataset[index])
        images = tuple(
            _to_uint8_hwc(processed[key], expected_image_shape)
            if key in processed
            else np.zeros(expected_image_shape, dtype=np.uint8)
            for key in config.image_keys
        )
        image_masks = np.asarray(
            [
                key in processed
                and bool(np.asarray(processed.get(f"{key}_padding_mask", True)).reshape(-1)[0])
                for key in config.image_keys
            ],
            dtype=np.bool_,
        )
        observations.append(
            SmolVLAObservation(
                images,
                np.ascontiguousarray(image_masks),
                np.ascontiguousarray(processed[OBS_LANGUAGE_TOKENS][0].detach().cpu().numpy(), dtype=np.int64),
                np.ascontiguousarray(processed[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().numpy(), dtype=np.bool_),
                np.ascontiguousarray(processed[OBS_STATE][0].detach().cpu().numpy(), dtype=np.float32),
                rng.standard_normal((config.chunk_size, config.max_action_dim), dtype=np.float32),
            )
        )
    actions = None
    if not args.skip_upstream_actions:
        oracle = UpstreamTorchSmolVLAOracle.open(
            config,
            lerobot_checkout=args.lerobot_checkout,
            checkpoint=args.checkpoint,
            cpu_threads=4,
        )
        actions = np.ascontiguousarray(
            np.stack([oracle.predict_action_chunk(observation) for observation in observations]), dtype=np.float32
        )
    replay = SmolVLAReplay(
        tuple(observations),
        actions,
        episode_ids=tuple(episode_ids),
        instruction_ids=tuple(instruction_ids),
    )
    replay.save(args.output, config)
    replay_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    replay_manifest = {
        "dataset_id": args.dataset_id,
        "dataset_revision": str(metadata.revision),
        "selection_seed": args.seed,
        "selected_episodes": selected_episodes,
        "image_mapping": image_mapping,
        "masked_artifact_images": sorted(set(config.image_keys) - set(image_mapping.values())),
        "image_adaptation": {
            "method": "aspect-preserving letterbox",
            "height": config.input_image_height,
            "width": config.input_image_width,
        },
        "observations": len(observations),
        "replay_sha256": replay_sha256,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved {len(observations)} observations from {len(set(episode_ids))} episodes to {args.output}")


if __name__ == "__main__":
    main()
