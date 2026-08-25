from __future__ import annotations

import argparse
from pathlib import Path

from qpu_xla.models.smolvla import (
    SmolVLAConfig,
    SmolVLANumerics,
    convert_smolvla_safetensors,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the pinned LeRobot SmolVLA SafeTensors file into a memory-mapped native artifact"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="downloaded model.safetensors")
    parser.add_argument("--policy-config", type=Path, required=True, help="downloaded LeRobot config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerics", choices=tuple(item.value for item in SmolVLANumerics), required=True)
    parser.add_argument(
        "--tokenizer-directory",
        type=Path,
        help="optional SmolVLM tokenizer/processor directory to copy into the artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SmolVLAConfig.from_huggingface_json(args.policy_config)
    output = convert_smolvla_safetensors(
        args.checkpoint,
        args.output,
        config,
        numerics=SmolVLANumerics(args.numerics),
        tokenizer_directory=args.tokenizer_directory,
    )
    print(f"created {args.numerics} SmolVLA artifact at {output}")


if __name__ == "__main__":
    main()
