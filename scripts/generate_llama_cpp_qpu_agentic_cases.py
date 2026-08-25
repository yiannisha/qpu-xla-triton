#!/usr/bin/env python3
"""Generate paired post-tool cases for overlapped QPU ``ffn_up`` suffixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import sha256_file, utc_now, write_json_atomic  # noqa: E402

DEFAULT_MODEL = Path(
    "/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/"
    "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
)


def integer_list(value: str) -> tuple[int, ...]:
    """Parse a duplicate-free comma-separated positive integer list."""
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {value!r}") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be positive and duplicate-free")
    return result


def fraction_map(value: str) -> dict[int, float]:
    """Parse suffix_tokens:qpu_output_fraction entries."""
    result: dict[int, float] = {}
    try:
        for entry in value.split(","):
            suffix_text, fraction_text = entry.split(":", 1)
            suffix = int(suffix_text)
            fraction = float(fraction_text)
            if suffix <= 0 or fraction <= 0.0 or fraction > 1.0:
                raise ValueError
            if suffix in result:
                raise ValueError
            result[suffix] = fraction
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "fractions must be unique suffix_tokens:fraction entries in (0, 1]"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one fraction is required")
    return result


def main() -> None:
    """Write exact cached-prefix CPU/QPU case pairs for the retained policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--plugin",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/libggml-qpu-inline.so",
        help="preload library implementing the CPU_REPACK/QPU overlap hook",
    )
    parser.add_argument(
        "--program-manifest",
        type=Path,
        default=ROOT / "integrations/llama_cpp/generated/manifest.json",
    )
    parser.add_argument("--prefixes", type=integer_list, default=integer_list("512,4096"))
    parser.add_argument("--suffixes", type=integer_list, default=integer_list("64,128,256"))
    parser.add_argument("--threads", type=integer_list, default=integer_list("4"))
    parser.add_argument(
        "--fractions",
        type=fraction_map,
        default=fraction_map("64:0.125,128:0.125,256:0.125"),
        help="QPU output-column fraction selected independently for each suffix",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(threads < 2 for threads in args.threads):
        parser.error("hybrid execution requires at least two GGML batch threads")
    unsupported_suffixes = [suffix for suffix in args.suffixes if suffix not in args.fractions]
    if unsupported_suffixes:
        parser.error(
            "suffixes do not have an exact output-column fraction: "
            + ",".join(str(value) for value in unsupported_suffixes)
        )
    for path in (args.model, args.plugin, args.program_manifest):
        if not path.is_file():
            parser.error(f"required artifact not found: {path}")

    manifest = json.loads(args.program_manifest.read_text(encoding="utf-8"))
    entries = [
        item
        for item in manifest.get("programs", [])
        if item.get("name") == "ggml-q4-0-q8-0-mx"
    ]
    if len(entries) != 1:
        parser.error("program manifest does not contain one ggml-q4-0-q8-0-mx entry")
    program = entries[0]
    evidence = {
        "program_manifest_path": str(args.program_manifest.resolve()),
        "program_manifest_sha256": sha256_file(args.program_manifest),
        "program": program["name"],
        "source_hash": program["source_hash"],
        "binary_sha256": program["binary_sha256"],
        "exact_shape": {
            "weight": "Q4_0",
            "activation": "CPU_REPACK_Q8_0x4",
            "output": "F32",
            "tile": [16, 16],
        },
    }
    context_size = max(args.prefixes) + max(args.suffixes) + 512
    cases: list[dict[str, Any]] = []
    for prefix in args.prefixes:
        for suffix in args.suffixes:
            for batch_threads in args.threads:
                fraction = args.fractions[suffix]
                partition = {
                    "axis": "output_columns",
                    "fraction": fraction,
                }
                base = {
                    "model": "gemma-4-e2b",
                    "mode": "plain",
                    "surface": "server",
                    "base_model": str(args.model.resolve()),
                    "prompt": "Use the newly returned tool evidence to answer with one short token.",
                    "context_filler": (
                        "A stable cached agent transcript records deterministic prior reasoning. "
                    ),
                    "suffix_filler": (
                        "Tool result: deterministic JSON evidence was returned successfully. "
                    ),
                    "workload": "decode",
                    "request_surface": "completion",
                    "predict_tokens": 1,
                    "context_size": context_size,
                    "context_tokens_target": prefix,
                    "suffix_tokens_target": suffix,
                    "threads": max(3, batch_threads),
                    "threads_batch": batch_threads,
                    "flash_attention": True,
                    "seed": 1234,
                    "temperature": 0.0,
                }
                workload = f"p{prefix}-s{suffix}-tb{batch_threads}"
                cpu_case = {
                    **base,
                    "name": f"gemma-agentic-{workload}-cpu",
                    "placement": "cpu-only",
                }
                qpu_case = {
                    **base,
                    "name": f"gemma-agentic-{workload}-qpu-up-overlap",
                    "placement": "hybrid",
                    "partition": partition,
                    "candidate_evidence": evidence,
                    "process_environment": {
                        "LD_PRELOAD": str(args.plugin.resolve()),
                        "GGML_QPU_UP_HYBRID": "1",
                        # The server appends one continuation-boundary row to the
                        # user-visible tool suffix at this graph boundary.
                        "GGML_QPU_UP_MAX_ROWS": str(max(args.suffixes) + 1),
                        "GGML_QPU_UP_MAX_FRACTION": str(max(args.fractions.values())),
                        "GGML_QPU_UP_FRACTION": str(fraction),
                    },
                }
                cases.extend((cpu_case, qpu_case))
    output = {
        "schema_version": 1,
        "kind": "llama-qpu-agentic-prefill-cases",
        "created_utc": utc_now(),
        "contract": {
            "scenario": "cached agent transcript followed by an exact-size tool-result suffix",
            "timed_boundary": "post-tool request through first generated token",
            "pairs": "identical model-visible tokens; only ffn_up output placement differs",
            "candidate": (
                "QPU computes a persistent Q4_0 ffn_up suffix while CPU_REPACK computes "
                "the prefix and gate; GEGLU is the join"
            ),
        },
        "cases": cases,
    }
    write_json_atomic(args.output, output)
    print(f"wrote {args.output}: {len(cases)} paired cases")


if __name__ == "__main__":
    main()
