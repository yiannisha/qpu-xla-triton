#!/usr/bin/env python3
"""Run isolated out-of-tree llama.cpp node profiles with environment evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from scripts.summarize_llama_cpp_graph_profile import summarize_profile  # noqa: E402


def normalize_case(case: dict[str, Any], index: int) -> dict[str, Any]:
    """Validate one graph configuration and make every scheduling input explicit."""
    normalized = dict(case)
    normalized.setdefault("name", f"profile-{index}")
    normalized.setdefault("batch", 1)
    normalized.setdefault("context_tokens", 0)
    normalized.setdefault("context_size", 8192)
    normalized.setdefault("threads", 3)
    normalized.setdefault("warmups", 1)
    normalized.setdefault("iterations", 3)
    normalized.setdefault("context_type", "default")
    normalized.setdefault(
        "prompt",
        "A deterministic graph profiling sentence records stable inference data. ",
    )
    if not normalized.get("model"):
        raise ValueError(f"case {normalized['name']!r} has no model")
    if normalized["context_type"] == "mtp" and not normalized.get("other_model"):
        raise ValueError(f"MTP case {normalized['name']!r} has no other_model target context")
    if normalized["context_type"] not in {"default", "mtp"}:
        raise ValueError(f"case {normalized['name']!r} has invalid context_type")
    for field in ("batch", "context_size", "threads", "iterations"):
        if int(normalized[field]) <= 0:
            raise ValueError(f"case {normalized['name']!r} has invalid {field}")
    if int(normalized["context_tokens"]) < 0 or int(normalized["warmups"]) < 0:
        raise ValueError(f"case {normalized['name']!r} has a negative count")
    if int(normalized["context_tokens"]) + int(normalized["batch"]) >= int(
        normalized["context_size"]
    ):
        raise ValueError(f"case {normalized['name']!r} exceeds context_size")
    if bool(normalized.get("fixture_dir")) != bool(normalized.get("fixture_pattern")):
        raise ValueError(
            f"case {normalized['name']!r} must set fixture_dir and fixture_pattern together"
        )
    normalized.setdefault("fixture_max_bytes", 64 * 1024 * 1024)
    if int(normalized["fixture_max_bytes"]) <= 0:
        raise ValueError(f"case {normalized['name']!r} has invalid fixture_max_bytes")
    return normalized


def select_cases(cases: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Select exact names while treating unknown names as errors."""
    if not names:
        return cases
    requested = set(names)
    selected = [case for case in cases if case["name"] in requested]
    missing = sorted(requested - {case["name"] for case in selected})
    if missing:
        raise ValueError(f"case names not found: {', '.join(missing)}")
    return selected


def build_command(binary: Path, case: dict[str, Any], output: Path) -> list[str]:
    """Build the complete native profiler command for one exact case."""
    command = [
        str(binary),
        "--model",
        str(case["model"]),
        "--output",
        str(output),
        "--batch",
        str(case["batch"]),
        "--context",
        str(case["context_tokens"]),
        "--context-size",
        str(case["context_size"]),
        "--threads",
        str(case["threads"]),
        "--warmups",
        str(case["warmups"]),
        "--iterations",
        str(case["iterations"]),
        "--context-type",
        str(case["context_type"]),
        "--prompt",
        str(case["prompt"]),
        "--quiet",
    ]
    if case.get("other_model"):
        command.extend(["--other-model", str(case["other_model"])])
    if case.get("fixture_dir"):
        command.extend(
            [
                "--fixture-dir",
                str(case["fixture_dir"]),
                "--fixture-pattern",
                str(case["fixture_pattern"]),
                "--fixture-max-bytes",
                str(case["fixture_max_bytes"]),
            ]
        )
    return command


def _throttle_value(environment: dict[str, Any]) -> int | None:
    stdout = environment["commands"]["throttling"].get("stdout", "")
    try:
        return int(stdout.split("throttled=", 1)[1].strip(), 0)
    except (IndexError, ValueError):
        return None


def validate_session(
    before: dict[str, Any],
    after: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate execution/environment while keeping serialized profiles non-promotable."""
    reasons: list[str] = []
    if any(record["returncode"] != 0 for record in records):
        reasons.append("one or more native profiler processes failed")
    if _throttle_value(before) != 0 or _throttle_value(after) != 0:
        reasons.append("throttling flags were unavailable or nonzero")
    if before["commands"]["swap"].get("stdout") != after["commands"]["swap"].get("stdout"):
        reasons.append("swapon state changed during the session")
    governors = {entry["governor"] for entry in before["cpu_frequency"]}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(value) for value in governors)}")
    return {
        "profile_execution_valid": not any(record["returncode"] != 0 for record in records),
        "environment_retained": not reasons,
        "timing_eligible_for_promotion": False,
        "timing_ineligibility_reason": (
            "the public eval callback serializes nodes and changes production scheduling"
        ),
        "rejection_reasons": reasons,
    }


def main() -> None:
    """Run selected cases and retain native output, raw logs, hashes, and environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--case", dest="case_names", action="append", default=[])
    parser.add_argument(
        "--profile-binary",
        type=Path,
        default=ROOT / "build/llama-qpu-runtime/qpu_llama_graph_profile",
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--session-id", default="graph-profile-session-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.case_file.is_file():
        parser.error(f"case file not found: {args.case_file}")
    if not args.profile_binary.is_file():
        parser.error(f"profile binary not found: {args.profile_binary}")
    payload = json.loads(args.case_file.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    try:
        cases = select_cases(
            [normalize_case(case, index) for index, case in enumerate(raw_cases)],
            args.case_names,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        parser.error("no cases selected")
    for case in cases:
        model = Path(case["model"])
        if not model.is_file():
            parser.error(f"model not found for {case['name']}: {model}")
        case["model"] = str(model.resolve())
        if case.get("other_model"):
            other_model = Path(case["other_model"])
            if not other_model.is_file():
                parser.error(f"other model not found for {case['name']}: {other_model}")
            case["other_model"] = str(other_model.resolve())
        if case.get("fixture_dir"):
            case["fixture_dir"] = str(Path(case["fixture_dir"]).resolve())

    raw_dir = (args.raw_dir or args.output.with_suffix("")).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    binary = args.profile_binary.resolve()
    model_paths = {
        str(path)
        for case in cases
        for path in (case.get("model"), case.get("other_model"))
        if path
    }
    model_hashes = {path: sha256_file(Path(path)) for path in sorted(model_paths)}
    before = collect_environment()
    records: list[dict[str, Any]] = []
    for case in cases:
        native_output = raw_dir / f"{case['name']}.json"
        command = build_command(binary, case, native_output)
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        native: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None
        if completed.returncode == 0 and native_output.is_file():
            native = json.loads(native_output.read_text(encoding="utf-8"))
            for fixture in native.get("fixtures", []):
                for tensor in fixture.get("tensors", []):
                    if tensor is not None and Path(tensor["path"]).is_file():
                        tensor["sha256"] = sha256_file(Path(tensor["path"]))
            summary = summarize_profile(native)
        records.append(
            {
                "case": case,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "native_output": (
                    {
                        "path": str(native_output),
                        "sha256": sha256_file(native_output),
                    }
                    if native_output.is_file()
                    else None
                ),
                "profile": native,
                "summary": summary,
            }
        )
    after = collect_environment()
    session = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-graph-profile-session",
        "created_utc": utc_now(),
        "session_id": args.session_id,
        "profile_binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "case_file": {
            "path": str(args.case_file.resolve()),
            "sha256": sha256_file(args.case_file),
        },
        "model_hashes": model_hashes,
        "records": records,
        "environment_before": before,
        "environment_after": after,
        "validation": validate_session(before, after, records),
    }
    write_json_atomic(args.output, session)
    failures = sum(record["returncode"] != 0 for record in records)
    print(f"wrote {args.output}: {len(records)} profiles, {failures} failures")


if __name__ == "__main__":
    main()
