from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qpu_xla.models.smolvla import (
    SmolVLAArtifact,
    SmolVLAReplay,
    UpstreamTorchSmolVLAOracle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach deterministic pinned-upstream Torch actions to a SmolVLA replay"
    )
    parser.add_argument("--artifact", type=Path, required=True, help="native FP32 artifact (topology/provenance)")
    parser.add_argument("--input", type=Path, required=True, help="input replay without required oracle actions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lerobot-checkout", type=Path, required=True)
    parser.add_argument("--upstream-checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--upstream-cache", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    artifact = SmolVLAArtifact.open(args.artifact, verify=not args.skip_checksums)
    config = artifact.checkpoint.config
    replay = SmolVLAReplay.load(args.input, config)
    oracle = UpstreamTorchSmolVLAOracle.open(
        config,
        lerobot_checkout=args.lerobot_checkout,
        checkpoint=args.upstream_checkpoint,
        cache_directory=args.upstream_cache,
        cpu_threads=args.cpu_threads,
    )
    actions = np.ascontiguousarray(
        np.stack([oracle.predict_action_chunk(observation) for observation in replay.observations]),
        dtype=np.float32,
    )
    recorded = SmolVLAReplay(
        replay.observations,
        actions,
        replay.model_revision,
        replay.lerobot_revision,
    )
    recorded.save(args.output, config)
    print(f"recorded {len(replay.observations)} upstream action chunks in {args.output}")


if __name__ == "__main__":
    main()
