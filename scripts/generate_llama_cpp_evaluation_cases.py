#!/usr/bin/env python3
"""Generate the exact Gemma/Qwen llama.cpp CPU evaluation case matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import write_json_atomic  # noqa: E402

ARTICLE_PROMPT = "Explain photosynthesis in 300 words."
DEFAULT_GEMMA_BASE = Path(
    "/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/"
    "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
)
DEFAULT_GEMMA_DRAFT = Path(
    "/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/mtp-gemma-4-E2B-it.gguf"
)


def parse_integer_list(value: str, *, allow_zero: bool = True) -> tuple[int, ...]:
    """Parse a stable, duplicate-free comma-separated integer list."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {value!r}") from exc
    lower = 0 if allow_zero else 1
    if not values or any(item < lower for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"integer list must contain {qualifier} values")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer list must not contain duplicates")
    return values


def _base_case(
    *,
    name: str,
    model: str,
    mode: str,
    base_model: Path,
    threads: int,
    context_tokens: int,
    predict_tokens: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "model": model,
        "mode": mode,
        "placement": "cpu-only",
        "surface": "server",
        "base_model": str(base_model.resolve()),
        "prompt": ARTICLE_PROMPT,
        "predict_tokens": predict_tokens,
        "context_size": 8192,
        "context_tokens_target": context_tokens,
        "threads": threads,
        "threads_batch": threads,
        "flash_attention": True,
        "seed": 1234,
        "temperature": 0.0,
    }


def gemma_cases(
    base_model: Path,
    draft_model: Path,
    *,
    threads: tuple[int, ...] = (2, 3, 4),
    contexts: tuple[int, ...] = (0, 512, 2048),
    draft_depths: tuple[int, ...] = (2, 3),
    prompt_sizes: tuple[int, ...] = (10, 128, 512, 2048),
    predict_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Build the complete fixed Gemma CPU baseline sweep from the plan."""
    cases: list[dict[str, Any]] = []
    for context in contexts:
        for thread_count in threads:
            cases.append(
                _base_case(
                    name=f"gemma-plain-t{thread_count}-c{context}-decode",
                    model="gemma-4-e2b",
                    mode="plain",
                    base_model=base_model,
                    threads=thread_count,
                    context_tokens=context,
                    predict_tokens=predict_tokens,
                )
            )
            for depth in draft_depths:
                case = _base_case(
                    name=f"gemma-mtp-n{depth}-t{thread_count}-c{context}-decode",
                    model="gemma-4-e2b",
                    mode="mtp",
                    base_model=base_model,
                    threads=thread_count,
                    context_tokens=context,
                    predict_tokens=predict_tokens,
                )
                case.update(
                    {
                        "draft_model": str(draft_model.resolve()),
                        "draft_threads": thread_count,
                        "draft_n_max": depth,
                        "draft_n_min": 0,
                    }
                )
                cases.append(case)
    for prompt_size in prompt_sizes:
        for thread_count in threads:
            case = _base_case(
                name=f"gemma-prompt-p{prompt_size}-t{thread_count}",
                model="gemma-4-e2b",
                mode="prompt",
                base_model=base_model,
                threads=thread_count,
                context_tokens=0,
                predict_tokens=0,
            )
            case["prompt_tokens_target"] = prompt_size
            cases.append(case)
    return cases


def qwen_cases(
    base_model: Path,
    *,
    threads: tuple[int, ...] = (2, 3, 4),
    contexts: tuple[int, ...] = (0, 512, 2048, 4096),
    draft_depths: tuple[int, ...] = (0, 1, 2, 3, 7),
    prompt_sizes: tuple[int, ...] = (10, 128, 512, 2048),
    predict_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Build the complete fixed Qwen3.5 CPU baseline and MTP depth sweep."""
    cases: list[dict[str, Any]] = []
    for context in contexts:
        for thread_count in threads:
            plain = _base_case(
                name=f"qwen-plain-t{thread_count}-c{context}-decode",
                model="qwen3.5-4b",
                mode="plain",
                base_model=base_model,
                threads=thread_count,
                context_tokens=context,
                predict_tokens=predict_tokens,
            )
            plain["server_extra_arguments"] = ["--kv-unified"]
            cases.append(plain)
            for depth in draft_depths:
                case = _base_case(
                    name=f"qwen-mtp-n{depth}-t{thread_count}-c{context}-decode",
                    model="qwen3.5-4b",
                    mode="mtp",
                    base_model=base_model,
                    threads=thread_count,
                    context_tokens=context,
                    predict_tokens=predict_tokens,
                )
                case.update(
                    {
                        "draft_threads": thread_count,
                        "draft_n_max": depth,
                        "draft_n_min": 0,
                        "server_extra_arguments": ["--kv-unified"],
                    }
                )
                cases.append(case)
    for prompt_size in prompt_sizes:
        for thread_count in threads:
            case = _base_case(
                name=f"qwen-prompt-p{prompt_size}-t{thread_count}",
                model="qwen3.5-4b",
                mode="prompt",
                base_model=base_model,
                threads=thread_count,
                context_tokens=0,
                predict_tokens=0,
            )
            case.update(
                {
                    "prompt_tokens_target": prompt_size,
                    "server_extra_arguments": ["--kv-unified"],
                }
            )
            cases.append(case)
    return cases


def main() -> None:
    """Write cases for every locally available required model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-base", type=Path, default=DEFAULT_GEMMA_BASE)
    parser.add_argument("--gemma-draft", type=Path, default=DEFAULT_GEMMA_DRAFT)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--threads", type=lambda value: parse_integer_list(value, allow_zero=False), default=(2, 3, 4))
    parser.add_argument("--gemma-contexts", type=parse_integer_list, default=(0, 512, 2048))
    parser.add_argument("--qwen-contexts", type=parse_integer_list, default=(0, 512, 2048, 4096))
    parser.add_argument("--gemma-depths", type=parse_integer_list, default=(2, 3))
    parser.add_argument("--qwen-depths", type=parse_integer_list, default=(0, 1, 2, 3, 7))
    parser.add_argument(
        "--prompt-sizes",
        type=lambda value: parse_integer_list(value, allow_zero=False),
        default=(10, 128, 512, 2048),
    )
    parser.add_argument("--predict-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.predict_tokens <= 0:
        parser.error("predict-tokens must be positive")

    cases: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    if args.gemma_base.is_file() and args.gemma_draft.is_file():
        cases.extend(
            gemma_cases(
                args.gemma_base,
                args.gemma_draft,
                threads=args.threads,
                contexts=args.gemma_contexts,
                draft_depths=args.gemma_depths,
                prompt_sizes=args.prompt_sizes,
                predict_tokens=args.predict_tokens,
            )
        )
    else:
        gaps.append(
            {
                "model": "gemma-4-e2b",
                "reason": "base or MTP drafter model is unavailable",
                "required_paths": [str(args.gemma_base), str(args.gemma_draft)],
            }
        )
    if args.qwen_model is not None and args.qwen_model.is_file():
        cases.extend(
            qwen_cases(
                args.qwen_model,
                threads=args.threads,
                contexts=args.qwen_contexts,
                draft_depths=args.qwen_depths,
                prompt_sizes=args.prompt_sizes,
                predict_tokens=args.predict_tokens,
            )
        )
    else:
        gaps.append(
            {
                "model": "qwen3.5-4b-mtp",
                "reason": "Qwen3.5 MTP model was not supplied or is unavailable",
                "required_path": str(args.qwen_model) if args.qwen_model is not None else None,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-evaluation-cases",
        "measurement_surface": "isolated llama-server /completion",
        "source_workload": (
            "https://github.com/Mjrovai/EdgeML-with-Raspberry-Pi/blob/main/mtp-rasp/README.md"
        ),
        "configuration": {
            "predict_tokens": args.predict_tokens,
            "threads": list(args.threads),
            "gemma_contexts": list(args.gemma_contexts),
            "qwen_contexts": list(args.qwen_contexts),
            "gemma_draft_depths": list(args.gemma_depths),
            "qwen_draft_depths": list(args.qwen_depths),
            "prompt_token_sizes": list(args.prompt_sizes),
        },
        "coverage_gaps": gaps,
        "cases": cases,
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(cases)} cases, {len(gaps)} coverage gaps")


if __name__ == "__main__":
    main()
