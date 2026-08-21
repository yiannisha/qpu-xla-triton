#!/usr/bin/env python3
"""Generate exact llama.cpp graph-profile cases for available target models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_llama_cpp_evaluation_cases import (  # noqa: E402
    DEFAULT_GEMMA_BASE,
    DEFAULT_GEMMA_DRAFT,
)
from scripts.llama_cpp_common import write_json_atomic  # noqa: E402


def base_profile_cases(
    model: Path,
    *,
    batches: tuple[int, ...] = (1, 2, 3, 4, 5, 8),
    long_context_batches: tuple[int, ...] = (1, 4),
    long_contexts: tuple[int, ...] = (512, 2048, 4096),
) -> list[dict[str, Any]]:
    """Return base-model width and long-context graph cases without duplicates."""
    configurations = [(batch, 0) for batch in batches]
    configurations.extend(
        (batch, context)
        for batch in long_context_batches
        for context in long_contexts
    )
    return [
        {
            "name": f"gemma-base-b{batch}-c{context}",
            "model": str(model.resolve()),
            "model_role": "base",
            "batch": batch,
            "context_tokens": context,
            "context_size": 8192,
            "threads": 3,
            "warmups": 1,
            "iterations": 3,
            "context_type": "default",
        }
        for batch, context in configurations
    ]


def drafter_profile_cases(
    model: Path,
    other_model: Path,
    *,
    batches: tuple[int, ...] = (1,),
) -> list[dict[str, Any]]:
    """Return standalone MTP-context graph cases for the Gemma drafter."""
    return [
        {
            "name": f"gemma-drafter-b{batch}-c0",
            "model": str(model.resolve()),
            "other_model": str(other_model.resolve()),
            "model_role": "mtp-drafter",
            "batch": batch,
            "context_tokens": 0,
            "context_size": 8192,
            "threads": 3,
            "warmups": 1,
            "iterations": 3,
            "context_type": "mtp",
        }
        for batch in batches
    ]


def configure_gemma_attention_fixtures(
    cases: list[dict[str, Any]], fixture_root: Path
) -> None:
    """Capture one sliding-window and one global Gemma attention node in place."""
    for case in cases:
        if (
            case.get("model_role") == "base"
            and int(case["batch"]) in {1, 4}
            and int(case["context_tokens"]) in {0, 512, 2048, 4096}
        ):
            case["fixture_dir"] = str((fixture_root / case["name"]).resolve())
            case["fixture_pattern"] = r"^(node_30|node_235) FLASH_ATTN_EXT$"
            case["fixture_max_bytes"] = 32 * 1024 * 1024
            case["fixture_all_sources"] = True


def main() -> None:
    """Write available graph cases and explicit missing-model gaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-base", type=Path, default=DEFAULT_GEMMA_BASE)
    parser.add_argument("--gemma-draft", type=Path, default=DEFAULT_GEMMA_DRAFT)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument(
        "--attention-fixture-root",
        type=Path,
        help="attach bounded all-source captures to required Gemma attention cases",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    if args.gemma_base.is_file():
        cases.extend(base_profile_cases(args.gemma_base))
    else:
        gaps.append({"model": "gemma-4-e2b", "reason": "base model unavailable"})
    if args.gemma_draft.is_file():
        if args.gemma_base.is_file():
            cases.extend(drafter_profile_cases(args.gemma_draft, args.gemma_base))
        else:
            gaps.append(
                {
                    "model": "gemma-4-e2b-drafter",
                    "reason": "target base model unavailable for ctx_other",
                }
            )
    else:
        gaps.append({"model": "gemma-4-e2b-drafter", "reason": "MTP drafter unavailable"})
    if args.qwen_model is None or not args.qwen_model.is_file():
        gaps.append({"model": "qwen3.5-4b-mtp", "reason": "Qwen model unavailable"})
    else:
        qwen = args.qwen_model.resolve()
        for batch in (1, 2, 3, 4, 5, 8):
            for context in (0, 512, 2048, 4096):
                cases.append(
                    {
                        "name": f"qwen-base-b{batch}-c{context}",
                        "model": str(qwen),
                        "model_role": "base-with-embedded-mtp",
                        "batch": batch,
                        "context_tokens": context,
                        "context_size": 8192,
                        "threads": 4,
                        "warmups": 1,
                        "iterations": 3,
                        "context_type": "default",
                    }
                )
    if args.attention_fixture_root is not None:
        configure_gemma_attention_fixtures(cases, args.attention_fixture_root)
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-graph-profile-cases",
        "coverage_gaps": gaps,
        "cases": cases,
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(cases)} cases, {len(gaps)} coverage gaps")


if __name__ == "__main__":
    main()
