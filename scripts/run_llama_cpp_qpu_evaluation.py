#!/usr/bin/env python3
"""Run isolated, reproducible llama.cpp end-to-end baseline/evaluation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from time import monotonic_ns
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    sha256_file,
    swap_used_bytes,
    utc_now,
    write_json_atomic,
)

PERF_RE = re.compile(
    r"llama_perf_context_print:\s+"
    r"(?P<name>load time|prompt eval time|eval time|total time)\s*=\s*"
    r"(?P<milliseconds>[0-9.]+)\s+ms"
    r"(?:\s*/\s*(?P<count>\d+)\s+(?P<count_unit>tokens|runs)"
    r"\s*\(\s*(?P<per_item_ms>[0-9.]+)\s+ms per token,\s*"
    r"(?P<tokens_per_second>[0-9.]+)\s+tokens per second\))?"
)
SPEC_RE = re.compile(
    r"statistics\s+(?P<kind>[\w-]+):.*?"
    r"#gen drafts\s*=\s*(?P<generated_drafts>\d+),\s*"
    r"#acc drafts\s*=\s*(?P<accepted_drafts>\d+),\s*"
    r"#gen tokens\s*=\s*(?P<generated_tokens>\d+),\s*"
    r"#acc tokens\s*=\s*(?P<accepted_tokens>\d+)"
    r"(?:,\s*#mean acc len\s*=\s*(?P<mean_accepted_length>[0-9.]+),\s*"
    r"#acc rate/pos\s*=\s*\((?P<acceptance_by_position>[^)]*)\))?"
)
DRAFT_ACCEPTANCE_RE = re.compile(
    r"draft acceptance\s*=\s*(?P<acceptance>[0-9.]+),\s*mean len\s*=\s*(?P<mean>[0-9.]+)"
)
ACCEPTANCE_POSITION_RE = re.compile(r"acc per pos\s*=\s*\((?P<positions>[^)]*)\)")
STRUCTURED_TOOL_TOKEN_CAP = 40
QPU_TELEMETRY_PREFIX = "qpu_llama_candidate_json:"


def parse_llama_metrics(output: str) -> dict[str, Any]:
    """Parse stable llama.cpp timing/stat lines while retaining raw logs separately."""
    performance: dict[str, Any] = {}
    for match in PERF_RE.finditer(output):
        key = match.group("name").replace(" ", "_")
        performance[key] = {
            "milliseconds": float(match.group("milliseconds")),
            "count": int(match.group("count")) if match.group("count") else None,
            "count_unit": match.group("count_unit"),
            "milliseconds_per_token": (
                float(match.group("per_item_ms")) if match.group("per_item_ms") else None
            ),
            "tokens_per_second": (
                float(match.group("tokens_per_second"))
                if match.group("tokens_per_second")
                else None
            ),
        }
    speculative: list[dict[str, Any]] = []
    for match in SPEC_RE.finditer(output):
        positions = match.group("acceptance_by_position")
        record = {
            "kind": match.group("kind"),
            "generated_drafts": int(match.group("generated_drafts")),
            "accepted_drafts": int(match.group("accepted_drafts")),
            "generated_tokens": int(match.group("generated_tokens")),
            "accepted_tokens": int(match.group("accepted_tokens")),
            "mean_accepted_length": (
                float(match.group("mean_accepted_length"))
                if match.group("mean_accepted_length")
                else None
            ),
            "acceptance_by_position": (
                [float(value.strip()) for value in positions.split(",") if value.strip()]
                if positions
                else []
            ),
        }
        eval_rate = performance.get("eval_time", {}).get("tokens_per_second")
        record["cycle_seconds"] = (
            record["mean_accepted_length"] / eval_rate
            if record["mean_accepted_length"] is not None and eval_rate
            else None
        )
        speculative.append(record)
    return {"performance": performance, "speculative": speculative}


def build_command(binary: Path, case: dict[str, Any]) -> list[str]:
    """Translate one explicit case into a deterministic llama-cli invocation."""
    command = [
        str(binary),
        "--model",
        str(case["base_model"]),
        "--prompt",
        str(case["prompt"]),
        "--n-predict",
        str(case["predict_tokens"]),
        "--ctx-size",
        str(case["context_size"]),
        "--threads",
        str(case["threads"]),
        "--threads-batch",
        str(case.get("threads_batch", case["threads"])),
        "--flash-attn",
        "on" if case.get("flash_attention", True) else "off",
        "--seed",
        str(case["seed"]),
        "--temp",
        str(case.get("temperature", 0.0)),
        "--color",
        "off",
        "--no-display-prompt",
        "--no-conversation",
        "--simple-io",
        "--perf",
        "--show-timings",
    ]
    if case["mode"] == "mtp":
        command.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(case["draft_n_max"]),
                "--spec-draft-n-min",
                str(case.get("draft_n_min", 0)),
            ]
        )
        draft_model = case.get("draft_model")
        if draft_model:
            command.extend(["--spec-draft-model", str(draft_model)])
        if "draft_threads" in case:
            command.extend(["--spec-draft-threads", str(case["draft_threads"])])
    for argument in case.get("extra_arguments", []):
        command.append(str(argument))
    return command


def build_server_command(binary: Path, case: dict[str, Any], port: int) -> list[str]:
    """Translate a case into the article's persistent llama-server surface."""
    command = [
        str(binary),
        "--model",
        str(case["base_model"]),
        "--threads",
        str(case["threads"]),
        "--threads-batch",
        str(case.get("threads_batch", case["threads"])),
        "--ctx-size",
        str(case["context_size"]),
        "--parallel",
        "1",
        "--flash-attn",
        "on" if case.get("flash_attention", True) else "off",
        "--n-gpu-layers",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-verbosity",
        "4",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    if case["mode"] == "mtp":
        command.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(case["draft_n_max"]),
                "--spec-draft-n-min",
                str(case.get("draft_n_min", 0)),
            ]
        )
        if case.get("draft_model"):
            command.extend(["--spec-draft-model", str(case["draft_model"])])
        if "draft_threads" in case:
            command.extend(["--spec-draft-threads", str(case["draft_threads"])])
    else:
        command.extend(["--spec-type", "none"])
    if case.get("workload") == "structured-tool-call":
        command.extend(["--tools", "all"])
    command.extend(str(argument) for argument in case.get("server_extra_arguments", []))
    return command


def normalize_case(case: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Validate fields that affect semantics and fill only explicit stable defaults."""
    normalized = dict(case)
    normalized.setdefault("name", f"case-{index}")
    normalized.setdefault("mode", "plain")
    normalized.setdefault("workload", "prompt" if normalized["mode"] == "prompt" else "decode")
    normalized.setdefault("request_surface", "completion")
    normalized.setdefault("predict_tokens", 256)
    normalized.setdefault("context_size", 8192)
    normalized.setdefault("threads", 3)
    normalized.setdefault("seed", 1234)
    normalized.setdefault("temperature", 0.0)
    normalized.setdefault("prompt", "Write one concise sentence about compiler optimization.")
    normalized.setdefault("flash_attention", True)
    normalized.setdefault("placement", "cpu-only")
    normalized.setdefault("surface", "server")
    normalized.setdefault("process_environment", {})
    if normalized["mode"] not in {"plain", "mtp", "prompt"}:
        raise ValueError(f"case {normalized['name']!r} has unsupported mode {normalized['mode']!r}")
    if normalized["surface"] not in {"server", "cli"}:
        raise ValueError(
            f"case {normalized['name']!r} has unsupported surface {normalized['surface']!r}"
        )
    if normalized["placement"] not in {"cpu-only", "qpu-only", "hybrid"}:
        raise ValueError(
            f"case {normalized['name']!r} has unsupported placement "
            f"{normalized['placement']!r}"
        )
    process_environment = normalized["process_environment"]
    if not isinstance(process_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in process_environment.items()
    ):
        raise ValueError(
            f"case {normalized['name']!r} process_environment must map strings to strings"
        )
    if normalized["workload"] not in {"decode", "prompt", "structured-tool-call"}:
        raise ValueError(
            f"case {normalized['name']!r} has unsupported workload {normalized['workload']!r}"
        )
    if normalized["request_surface"] not in {"completion", "chat-completions"}:
        raise ValueError(
            f"case {normalized['name']!r} has unsupported request_surface "
            f"{normalized['request_surface']!r}"
        )
    if not normalized.get("base_model"):
        raise ValueError(f"case {normalized['name']!r} has no base_model")
    if normalized["mode"] == "mtp" and "draft_n_max" not in normalized:
        raise ValueError(f"MTP case {normalized['name']!r} has no draft_n_max")
    if int(normalized["predict_tokens"]) < 0:
        raise ValueError(f"case {normalized['name']!r} has invalid predict_tokens")
    for field in ("context_size", "threads"):
        if int(normalized[field]) <= 0:
            raise ValueError(f"case {normalized['name']!r} has invalid {field}")
    context_target = int(normalized.get("context_tokens_target", 0))
    prompt_target = int(normalized.get("prompt_tokens_target", 0))
    suffix_target = int(normalized.get("suffix_tokens_target", 0))
    if context_target < 0 or context_target >= int(normalized["context_size"]):
        raise ValueError(f"case {normalized['name']!r} has invalid context_tokens_target")
    if prompt_target < 0 or prompt_target >= int(normalized["context_size"]):
        raise ValueError(f"case {normalized['name']!r} has invalid prompt_tokens_target")
    if context_target and prompt_target:
        raise ValueError(
            f"case {normalized['name']!r} cannot combine populated context and exact prompt targets"
        )
    if suffix_target < 0 or suffix_target >= int(normalized["context_size"]):
        raise ValueError(f"case {normalized['name']!r} has invalid suffix_tokens_target")
    if suffix_target and not context_target:
        raise ValueError(
            f"case {normalized['name']!r} requires a populated context for an exact suffix"
        )
    if prompt_target and normalized["mode"] != "prompt":
        raise ValueError(f"case {normalized['name']!r} uses prompt_tokens_target outside prompt mode")
    if normalized["mode"] == "prompt" and int(normalized["predict_tokens"]) != 0:
        raise ValueError(f"prompt case {normalized['name']!r} must set predict_tokens to zero")
    if normalized["mode"] == "prompt" and normalized["workload"] != "prompt":
        raise ValueError(f"prompt case {normalized['name']!r} must use prompt workload")
    if normalized["workload"] == "structured-tool-call":
        if normalized["surface"] != "server" or normalized["request_surface"] != "chat-completions":
            raise ValueError(
                f"structured tool case {normalized['name']!r} must use server chat-completions"
            )
        if int(normalized["predict_tokens"]) != STRUCTURED_TOOL_TOKEN_CAP:
            raise ValueError(
                f"structured tool case {normalized['name']!r} must use the fixed "
                f"{STRUCTURED_TOOL_TOKEN_CAP}-token cap"
            )
        if context_target or prompt_target:
            raise ValueError(
                f"structured tool case {normalized['name']!r} must start from an empty cache"
            )
        if not isinstance(normalized.get("messages"), list) or not normalized["messages"]:
            raise ValueError(f"structured tool case {normalized['name']!r} has no messages")
        if not isinstance(normalized.get("tools"), list) or not normalized["tools"]:
            raise ValueError(f"structured tool case {normalized['name']!r} has no tools")
        expected = normalized.get("expected_tool_call")
        if not isinstance(expected, dict) or not expected.get("name"):
            raise ValueError(
                f"structured tool case {normalized['name']!r} has no expected_tool_call"
            )
        if not isinstance(expected.get("arguments"), dict):
            raise ValueError(
                f"structured tool case {normalized['name']!r} expected arguments are not an object"
            )
        tool_names = {
            function.get("name")
            for tool in normalized["tools"]
            if isinstance(tool, dict)
            and isinstance((function := tool.get("function")), dict)
        }
        if expected["name"] not in tool_names:
            raise ValueError(
                f"structured tool case {normalized['name']!r} expected tool is not declared"
            )
    elif normalized["request_surface"] != "completion":
        raise ValueError(
            f"non-tool case {normalized['name']!r} must use the completion request surface"
        )
    return normalized


def select_cases(cases: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Select named cases in file order and reject typos instead of silently doing less work."""
    if not names:
        return cases
    requested = set(names)
    selected = [case for case in cases if case["name"] in requested]
    found = {case["name"] for case in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"case names not found: {', '.join(missing)}")
    return selected


def workload_semantics_sha256(case: dict[str, Any]) -> str:
    """Hash only fields that define model-visible workload semantics, not tuning knobs."""
    semantics = {
        "model": case.get("model"),
        "workload": case.get("workload"),
        "request_surface": case.get("request_surface"),
        "prompt": case.get("prompt"),
        "messages": case.get("messages"),
        "tools": case.get("tools"),
        "tool_choice": case.get("tool_choice"),
        "parallel_tool_calls": case.get("parallel_tool_calls"),
        "predict_tokens": case.get("predict_tokens"),
        "context_size": case.get("context_size"),
        "context_tokens_target": case.get("context_tokens_target"),
        "suffix_tokens_target": case.get("suffix_tokens_target"),
        "prompt_tokens_target": case.get("prompt_tokens_target"),
        "context_filler": case.get("context_filler"),
        "suffix_filler": case.get("suffix_filler"),
        "prompt_filler": case.get("prompt_filler"),
        "seed": case.get("seed"),
        "temperature": case.get("temperature"),
    }
    encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_candidate_evidence(case: dict[str, Any]) -> None:
    """Verify that a non-CPU case names an exact exported program and immutable bytes."""
    if case["placement"] == "cpu-only":
        return
    evidence = case.get("candidate_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"non-CPU case {case['name']!r} has no candidate_evidence object")
    required = {
        "program_manifest_path",
        "program_manifest_sha256",
        "program",
        "source_hash",
        "binary_sha256",
        "exact_shape",
    }
    missing = sorted(required - evidence.keys())
    if missing:
        raise ValueError(
            f"non-CPU case {case['name']!r} candidate_evidence is missing "
            f"{', '.join(missing)}"
        )
    if not isinstance(evidence["exact_shape"], dict) or not evidence["exact_shape"]:
        raise ValueError(f"non-CPU case {case['name']!r} has no exact_shape description")
    manifest_path = Path(str(evidence["program_manifest_path"])).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"candidate program manifest not found: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if evidence["program_manifest_sha256"] != manifest_sha256:
        raise ValueError(f"candidate program manifest hash mismatch for {case['name']!r}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("programs"), list):
        raise ValueError(f"candidate program manifest has no programs array: {manifest_path}")
    entries = [
        item
        for item in manifest["programs"]
        if isinstance(item, dict)
        if item.get("name") == evidence["program"]
    ]
    if len(entries) != 1:
        raise ValueError(
            f"candidate program {evidence['program']!r} was not unique in {manifest_path}"
        )
    entry = entries[0]
    if evidence["source_hash"] != entry.get("source_hash"):
        raise ValueError(f"candidate source hash mismatch for {case['name']!r}")
    if evidence["binary_sha256"] != entry.get("binary_sha256"):
        raise ValueError(f"candidate binary hash mismatch for {case['name']!r}")
    binary_path = manifest_path.parent / str(entry["binary"])
    if not binary_path.is_file() or sha256_file(binary_path) != evidence["binary_sha256"]:
        raise ValueError(f"candidate binary bytes do not match for {case['name']!r}")
    source_path = Path(str(entry["source"]))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if (
        not source_path.is_file()
        or sha256_file(source_path) != entry.get("source_file_sha256")
    ):
        raise ValueError(f"candidate source file does not match for {case['name']!r}")
    if case["placement"] == "hybrid" and not isinstance(case.get("partition"), dict):
        raise ValueError(f"hybrid case {case['name']!r} has no partition description")
    evidence["program_manifest_path"] = str(manifest_path)


def validate_candidate_execution(case: dict[str, Any], server_log: str) -> dict[str, Any]:
    """Require exact native telemetry instead of trusting a placement label."""
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in server_log.splitlines():
        if QPU_TELEMETRY_PREFIX not in line:
            continue
        encoded = line.split(QPU_TELEMETRY_PREFIX, 1)[1].strip()
        try:
            event = json.loads(encoded)
        except json.JSONDecodeError as exc:
            errors.append(f"malformed QPU telemetry: {exc.msg}")
            continue
        if not isinstance(event, dict):
            errors.append("QPU telemetry was not a JSON object")
            continue
        events.append(event)
    if case["placement"] == "cpu-only":
        if events:
            errors.append("CPU-only case reported QPU dispatch telemetry")
        return {"valid": not errors, "errors": errors, "events": events, "dispatch_count": 0}
    evidence = case["candidate_evidence"]
    if not events:
        errors.append("non-CPU case reported no QPU dispatch telemetry")
    dispatch_count = 0
    for event in events:
        for field in ("program", "source_hash", "binary_sha256", "exact_shape"):
            if event.get(field) != evidence[field]:
                errors.append(f"QPU telemetry {field} did not match candidate evidence")
        if event.get("placement") != case["placement"]:
            errors.append("QPU telemetry placement did not match the case")
        if event.get("partition") != case.get("partition"):
            errors.append("QPU telemetry partition did not match the case")
        value = event.get("dispatch_count")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append("QPU telemetry dispatch_count was not a positive integer")
        else:
            dispatch_count += value
    if events and dispatch_count <= 0:
        errors.append("non-CPU case did not attest any QPU dispatches")
    return {
        "valid": not errors,
        "errors": errors,
        "events": events,
        "dispatch_count": dispatch_count,
    }


def _throttle_value(environment: dict[str, Any]) -> int | None:
    stdout = environment["commands"]["throttling"].get("stdout", "")
    try:
        return int(stdout.split("throttled=", 1)[1].strip(), 0)
    except (IndexError, ValueError):
        return None


def validate_session(before: dict[str, Any], after: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply environmental and process-success retention gates without hiding failures."""
    reasons: list[str] = []
    before_throttle = _throttle_value(before)
    after_throttle = _throttle_value(after)
    if before_throttle is None or after_throttle is None or \
            before_throttle & 0xffff or after_throttle & 0xffff:
        reasons.append("current throttling flags were unavailable or nonzero")
    before_swap_configuration = before["commands"].get(
        "swap_configuration", before["commands"]["swap"]
    ).get("stdout")
    after_swap_configuration = after["commands"].get(
        "swap_configuration", after["commands"]["swap"]
    ).get("stdout")
    if before_swap_configuration != after_swap_configuration:
        reasons.append("swapon state changed during the session")
    before_swap_used = swap_used_bytes(before)
    after_swap_used = swap_used_bytes(after)
    if before_swap_used is None or after_swap_used is None:
        reasons.append("swap usage was unavailable")
    elif before_swap_used != 0 or after_swap_used != 0:
        reasons.append(
            f"swap was in use before/after the session ({before_swap_used}/{after_swap_used} bytes)"
        )
    governors = {entry["governor"] for entry in before["cpu_frequency"]}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(value) for value in governors)}")
    active_servers = before["commands"].get("llama_servers", {}).get("stdout", "").strip()
    if active_servers:
        reasons.append("one or more pre-existing llama-server processes were active")
    if any(sample["returncode"] != 0 for sample in samples):
        reasons.append("one or more llama.cpp processes failed")
    if any(not sample.get("context_population", {}).get("valid", True) for sample in samples):
        reasons.append("one or more populated-context requests did not reuse the requested prefix")
    if any(not sample.get("prompt_population", {}).get("valid", True) for sample in samples):
        reasons.append("one or more exact-prompt requests did not evaluate the requested token count")
    server_samples = [
        sample for sample in samples if str(sample.get("surface", "")).startswith("llama-server-")
    ]
    if any(
        not sample.get("generation_population", {}).get("valid", True)
        for sample in server_samples
    ):
        reasons.append("one or more server responses did not satisfy generated-token coverage")
    if any(
        validation is not None and not validation.get("valid", False)
        for sample in server_samples
        if (validation := sample.get("tool_call_validation")) is not None
    ):
        reasons.append("one or more structured tool-call responses violated the fixed contract")
    if any(
        sample.get("process_memory", {}).get("peak_rss_bytes") is None
        for sample in samples
    ):
        reasons.append("one or more process peak-RSS measurements were unavailable")
    if any(
        int(sample.get("process_memory", {}).get("swap_bytes") or 0) != 0
        for sample in samples
    ):
        reasons.append("one or more llama.cpp processes used swap")
    if any(
        not sample.get("candidate_execution", {}).get("valid", False)
        for sample in samples
    ):
        reasons.append("one or more cases lacked valid exact candidate execution telemetry")
    return {"retained": not reasons, "rejection_reasons": reasons}


def run_case(binary: Path, case: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Run one isolated llama-cli process and retain all output and timings."""
    command = build_command(binary, case)
    started_utc = utc_now()
    started_ns = monotonic_ns()
    environment = os.environ.copy()
    environment.update(case.get("process_environment", {}))
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False, env=environment
    )
    wall_ns = monotonic_ns() - started_ns
    combined = completed.stderr + "\n" + completed.stdout
    generated_sha256 = hashlib.sha256(completed.stdout.encode()).hexdigest()
    return {
        "sample_index": sample_index,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "command": command,
        "returncode": completed.returncode,
        "process_wall_ns": wall_ns,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": generated_sha256,
        "metrics": parse_llama_metrics(combined),
    }


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(port: int, process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:  # noqa: S310 - fixed loopback URL
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become healthy within {timeout_seconds:.1f} seconds")


def _parse_process_memory(status: str) -> dict[str, int | None]:
    """Parse Linux process peak/current RSS and swap into bytes."""
    fields = {"VmRSS": "rss_bytes", "VmHWM": "peak_rss_bytes", "VmSwap": "swap_bytes"}
    result: dict[str, int | None] = {name: None for name in fields.values()}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in fields:
            continue
        parts = value.split()
        if not parts:
            continue
        try:
            amount = int(parts[0])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) == 1 or parts[1].lower() == "kb" else 1
        result[fields[key]] = amount * multiplier
    return result


def _process_memory(pid: int) -> dict[str, int | None]:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"rss_bytes": None, "peak_rss_bytes": None, "swap_bytes": None}
    return _parse_process_memory(status)


def _server_speculative_metrics(log: str, predicted_per_second: float | None) -> dict[str, Any]:
    acceptance_matches = list(DRAFT_ACCEPTANCE_RE.finditer(log))
    position_matches = list(ACCEPTANCE_POSITION_RE.finditer(log))
    if not acceptance_matches:
        parsed = parse_llama_metrics(log)["speculative"]
        if not parsed:
            return {}
        result = dict(parsed[-1])
        if result.get("mean_accepted_length") is not None and predicted_per_second:
            result["cycle_seconds"] = result["mean_accepted_length"] / predicted_per_second
        return result
    match = acceptance_matches[-1]
    mean = float(match.group("mean"))
    positions = (
        [float(value.strip()) for value in position_matches[-1].group("positions").split(",")]
        if position_matches
        else []
    )
    return {
        "draft_acceptance": float(match.group("acceptance")),
        "mean_accepted_length": mean,
        "acceptance_by_position": positions,
        "cycle_seconds": mean / predicted_per_second if predicted_per_second else None,
    }


def _post_json(port: int, path: str, payload: dict[str, Any], timeout: float) -> tuple[bytes, dict[str, Any]]:
    encoded = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed loopback URL
        raw = response.read()
    return raw, json.loads(raw)


def _build_server_request(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the endpoint-specific request while preserving a raw-token correctness surface."""
    common = {
        "cache_prompt": False,
        "temperature": float(case.get("temperature", 0.0)),
        "seed": int(case["seed"]),
        "return_tokens": True,
    }
    if case["request_surface"] == "completion":
        return (
            "/completion",
            {
                **common,
                "prompt": str(case["prompt"]),
                "n_predict": int(case["predict_tokens"]),
            },
        )
    if case["request_surface"] == "chat-completions":
        return (
            "/v1/chat/completions",
            {
                **common,
                "messages": case["messages"],
                "max_tokens": int(case["predict_tokens"]),
                "tools": case["tools"],
                "tool_choice": case.get("tool_choice", "required"),
                "parallel_tool_calls": bool(case.get("parallel_tool_calls", False)),
                "parse_tool_calls": True,
                "stream": False,
                "verbose": True,
            },
        )
    raise ValueError(f"unsupported request surface {case['request_surface']!r}")


def _normalized_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Drop nondeterministic IDs and parse function arguments for exact comparisons."""
    if not isinstance(raw_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            calls.append(
                {
                    "name": None,
                    "arguments": raw_call,
                    "arguments_json_valid": False,
                }
            )
            continue
        function = raw_call.get("function")
        function = function if isinstance(function, dict) else raw_call
        arguments = function.get("arguments")
        arguments_json_valid = isinstance(arguments, dict)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
                arguments_json_valid = isinstance(arguments, dict)
            except json.JSONDecodeError:
                arguments_json_valid = False
        calls.append(
            {
                "name": function.get("name"),
                "arguments": arguments,
                "arguments_json_valid": arguments_json_valid,
            }
        )
    return calls


def _response_semantics(response: dict[str, Any], request_surface: str) -> dict[str, Any]:
    """Extract comparable generated tokens and model-visible output from either endpoint."""
    timings_value = response.get("timings")
    timings: dict[str, Any] = timings_value if isinstance(timings_value, dict) else {}
    if request_surface == "completion":
        content = str(response.get("content", ""))
        token_ids = response.get("tokens")
        token_count = response.get("tokens_predicted", timings.get("predicted_n"))
        stop_reason = response.get("stop_type")
        tool_calls: list[dict[str, Any]] = []
    elif request_surface == "chat-completions":
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        choice = choice if isinstance(choice, dict) else {}
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        content_value = message.get("content")
        content = "" if content_value is None else str(content_value)
        verbose = response.get("__verbose")
        verbose = verbose if isinstance(verbose, dict) else {}
        token_ids = verbose.get("tokens")
        usage_value = response.get("usage")
        usage: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
        token_count = usage.get("completion_tokens", timings.get("predicted_n"))
        stop_reason = choice.get("finish_reason", verbose.get("stop_type"))
        tool_calls = _normalized_tool_calls(message.get("tool_calls"))
    else:
        raise ValueError(f"unsupported request surface {request_surface!r}")
    normalized_tokens = (
        [int(token) for token in token_ids]
        if isinstance(token_ids, list) and all(isinstance(token, int) for token in token_ids)
        else None
    )
    semantic_payload = {"content": content, "tool_calls": tool_calls}
    semantic_json = json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
    return {
        **semantic_payload,
        "token_ids": normalized_tokens,
        "token_count": int(token_count) if token_count is not None else None,
        "stop_reason": stop_reason,
        "semantic_sha256": hashlib.sha256(semantic_json.encode()).hexdigest(),
    }


def _validate_structured_tool_response(
    case: dict[str, Any], semantics: dict[str, Any]
) -> dict[str, Any]:
    """Validate the fixed application workload independently of timing success."""
    expected = case["expected_tool_call"]
    calls = semantics["tool_calls"]
    errors: list[str] = []
    if len(calls) != 1:
        errors.append(f"expected exactly one tool call, received {len(calls)}")
    else:
        call = calls[0]
        if call.get("name") != expected["name"]:
            errors.append(
                f"expected tool {expected['name']!r}, received {call.get('name')!r}"
            )
        if not call.get("arguments_json_valid"):
            errors.append("tool arguments were not a JSON object")
        elif call.get("arguments") != expected["arguments"]:
            errors.append("tool arguments did not match the fixed expected object")
    if semantics["content"]:
        errors.append("structured tool response contained assistant prose")
    if semantics["stop_reason"] != "tool_calls":
        errors.append(f"expected tool_calls stop reason, received {semantics['stop_reason']!r}")
    token_count = semantics["token_count"]
    if token_count is None or token_count <= 0 or token_count > STRUCTURED_TOOL_TOKEN_CAP:
        errors.append(
            f"generated token count {token_count!r} was outside 1..{STRUCTURED_TOOL_TOKEN_CAP}"
        )
    token_ids = semantics["token_ids"]
    if token_ids is None or token_count is None or len(token_ids) != token_count:
        errors.append("raw generated token IDs were unavailable or incomplete")
    return {
        "valid": not errors,
        "errors": errors,
        "token_cap": STRUCTURED_TOOL_TOKEN_CAP,
        "generated_tokens": token_count,
        "expected": expected,
        "observed": calls,
    }


def _has_complete_generated_tokens(semantics: dict[str, Any]) -> bool:
    """Return whether a response retained every generated token ID it counted."""
    token_ids = semantics.get("token_ids")
    token_count = semantics.get("token_count")
    return (
        isinstance(token_ids, list)
        and isinstance(token_count, int)
        and token_count > 0
        and len(token_ids) == token_count
    )


def _context_prompt(
    port: int, case: dict[str, Any], timeout: float
) -> tuple[str | list[int], list[int], list[int]]:
    target = int(case.get("context_tokens_target", 0))
    if target <= 0:
        return str(case["prompt"]), [], []
    filler_text = case.get(
        "context_filler",
        "A deterministic context sentence records a stable benchmark fact. ",
    )
    _, filler_response = _post_json(
        port,
        "/tokenize",
        {"content": str(filler_text) * (target + 1), "add_special": True},
        timeout,
    )
    filler_tokens = [int(token) for token in filler_response["tokens"]]
    if len(filler_tokens) < target:
        raise RuntimeError(f"tokenized context filler produced {len(filler_tokens)} tokens, need {target}")
    prefix_tokens = filler_tokens[:target]
    suffix_target = int(case.get("suffix_tokens_target", 0))
    if suffix_target > 0:
        suffix_source = str(
            case.get(
                "suffix_filler",
                "A tool returned deterministic structured evidence for the agent to inspect. ",
            )
        )
        _, suffix_response = _post_json(
            port,
            "/tokenize",
            {
                "content": (suffix_source + " ") * (suffix_target + 1),
                "add_special": False,
            },
            timeout,
        )
        suffix_tokens = [int(token) for token in suffix_response["tokens"]]
        if len(suffix_tokens) < suffix_target:
            raise RuntimeError(
                f"tokenized suffix produced {len(suffix_tokens)} tokens, need {suffix_target}"
            )
        suffix_tokens = suffix_tokens[:suffix_target]
    else:
        _, task_response = _post_json(
            port,
            "/tokenize",
            {"content": str(case["prompt"]), "add_special": False},
            timeout,
        )
        suffix_tokens = [int(token) for token in task_response["tokens"]]
    return prefix_tokens + suffix_tokens, prefix_tokens, suffix_tokens


def _expected_timed_suffix_tokens(
    prefix_target: int,
    request_cache_tokens: int,
    suffix_target: int,
) -> tuple[int, int]:
    """Return native prefix replay and total timed tokens for an exact suffix."""
    replay = max(0, prefix_target - request_cache_tokens)
    return replay, suffix_target + replay


def _exact_prompt(port: int, case: dict[str, Any], timeout: float) -> list[int] | None:
    """Tokenize and truncate a deterministic prompt to the requested exact size."""
    target = int(case.get("prompt_tokens_target", 0))
    if target <= 0:
        return None
    source = str(case.get("prompt_filler", case["prompt"]))
    if not source:
        raise ValueError("prompt_filler must not be empty")
    _, response = _post_json(
        port,
        "/tokenize",
        {"content": (source + " ") * (target + 1), "add_special": True},
        timeout,
    )
    tokens = [int(token) for token in response["tokens"]]
    if len(tokens) < target:
        raise RuntimeError(f"tokenized prompt produced {len(tokens)} tokens, need {target}")
    return tokens[:target]


def run_server_case(
    binary: Path,
    case: dict[str, Any],
    sample_index: int,
    *,
    startup_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    """Run one isolated server, wait for readiness, and time one endpoint request."""
    port = _available_loopback_port()
    command = build_server_command(binary, case, port)
    request_endpoint, request_payload = _build_server_request(case)
    started_utc = utc_now()
    process_started_ns = monotonic_ns()
    response_bytes = b""
    response_payload: dict[str, Any] | None = None
    error: str | None = None
    request_wall_ns: int | None = None
    startup_ns: int | None = None
    process_memory: dict[str, int | None] = {
        "rss_bytes": None,
        "peak_rss_bytes": None,
        "swap_bytes": None,
    }
    context_population: dict[str, Any] = {
        "target_tokens": int(case.get("context_tokens_target", 0)),
        "prefix_token_count": 0,
        "prefill_wall_ns": None,
        "prefill_response": None,
        "suffix_tokens_target": int(case.get("suffix_tokens_target", 0)),
        "suffix_token_count": 0,
    }
    prompt_population: dict[str, Any] = {
        "target_tokens": int(case.get("prompt_tokens_target", 0)),
        "request_token_count": None,
        "request_tokens_evaluated": None,
        "timed_prompt_tokens": None,
    }
    with tempfile.TemporaryFile(mode="w+b") as log_file:
        environment = os.environ.copy()
        environment.update(case.get("process_environment", {}))
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            _wait_for_health(port, process, startup_timeout)
            startup_ns = monotonic_ns() - process_started_ns
            if case["request_surface"] == "completion":
                exact_prompt = _exact_prompt(port, case, request_timeout)
                if exact_prompt is None:
                    prompt, prefix_tokens, suffix_tokens = _context_prompt(
                        port, case, request_timeout
                    )
                else:
                    prompt, prefix_tokens, suffix_tokens = exact_prompt, [], []
                    prompt_population["request_token_count"] = len(exact_prompt)
                request_payload["prompt"] = prompt
                if prefix_tokens:
                    prefill_payload = {
                        "prompt": prefix_tokens,
                        "n_predict": 0,
                        "cache_prompt": True,
                        "temperature": 0.0,
                        "seed": int(case["seed"]),
                    }
                    prefill_started_ns = monotonic_ns()
                    _, prefill_response = _post_json(
                        port,
                        "/completion",
                        prefill_payload,
                        request_timeout,
                    )
                    context_population = {
                        "target_tokens": int(case["context_tokens_target"]),
                        "prefix_token_count": len(prefix_tokens),
                        "prefill_wall_ns": monotonic_ns() - prefill_started_ns,
                        "prefill_response": prefill_response,
                        "suffix_tokens_target": int(case.get("suffix_tokens_target", 0)),
                        "suffix_token_count": len(suffix_tokens),
                    }
                    request_payload["cache_prompt"] = True
            request_started_ns = monotonic_ns()
            response_bytes, response_payload = _post_json(
                port,
                request_endpoint,
                request_payload,
                request_timeout,
            )
            request_wall_ns = monotonic_ns() - request_started_ns
        except (OSError, RuntimeError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            process_memory = _process_memory(process.pid)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            log_file.flush()
            log_file.seek(0)
            server_log = log_file.read().decode(errors="replace")
    candidate_execution = validate_candidate_execution(case, server_log)
    timings = response_payload.get("timings", {}) if response_payload else {}
    predicted_per_second_value = timings.get("predicted_per_second")
    predicted_per_second = (
        float(predicted_per_second_value) if predicted_per_second_value is not None else None
    )
    target_context_tokens = int(context_population["target_tokens"])
    request_cache_tokens = int(timings.get("cache_n", 0))
    prefill_response = context_population.get("prefill_response") or {}
    prefill_cached_tokens = int(prefill_response.get("tokens_cached", 0))
    verbose_response_value = (
        response_payload.get("__verbose", {}) if isinstance(response_payload, dict) else {}
    )
    verbose_response = verbose_response_value if isinstance(verbose_response_value, dict) else {}
    usage_value = response_payload.get("usage", {}) if isinstance(response_payload, dict) else {}
    usage = usage_value if isinstance(usage_value, dict) else {}
    request_prompt_tokens = int(
        response_payload.get(
            "tokens_evaluated",
            verbose_response.get("tokens_evaluated", usage.get("prompt_tokens", 0)),
        )
        if response_payload
        else 0
    )
    context_population["request_cache_n"] = request_cache_tokens
    context_population["prefill_tokens_cached"] = prefill_cached_tokens
    context_population["request_tokens_evaluated"] = request_prompt_tokens
    context_population["timed_prompt_tokens"] = int(timings.get("prompt_n", 0))
    suffix_target = int(context_population.get("suffix_tokens_target", 0))
    # llama.cpp may deliberately retain fewer than all prefix tokens at a KV-cache
    # checkpoint.  Those prefix tokens are replayed alongside the new suffix, so
    # prompt_n is suffix size plus the native cache overlap rather than suffix size
    # alone.  The model-visible tool suffix remains exactly suffix_target tokens.
    cache_replay_tokens, expected_timed_prompt_tokens = _expected_timed_suffix_tokens(
        target_context_tokens,
        request_cache_tokens,
        suffix_target,
    )
    context_population["cache_replay_tokens"] = cache_replay_tokens
    context_population["expected_timed_prompt_tokens"] = expected_timed_prompt_tokens
    context_population["valid"] = (
        target_context_tokens == 0
        or (
            int(context_population["prefix_token_count"]) == target_context_tokens
            and prefill_cached_tokens == target_context_tokens
            and request_prompt_tokens >= target_context_tokens
            and (
                suffix_target == 0
                or (
                    int(context_population.get("suffix_token_count", 0)) == suffix_target
                    and request_cache_tokens <= target_context_tokens
                    and int(context_population["timed_prompt_tokens"])
                    == expected_timed_prompt_tokens
                )
            )
        )
    )
    target_prompt_tokens = int(prompt_population["target_tokens"])
    timed_prompt_tokens = int(timings.get("prompt_n", 0))
    prompt_population["request_tokens_evaluated"] = request_prompt_tokens
    prompt_population["timed_prompt_tokens"] = timed_prompt_tokens
    prompt_population["valid"] = (
        target_prompt_tokens == 0
        or (
            int(prompt_population["request_token_count"] or 0) == target_prompt_tokens
            and request_prompt_tokens == target_prompt_tokens
            and timed_prompt_tokens == target_prompt_tokens
        )
    )
    semantics = _response_semantics(response_payload or {}, case["request_surface"])
    tool_call_validation = (
        _validate_structured_tool_response(case, semantics)
        if case["workload"] == "structured-tool-call"
        else None
    )
    generation_target = int(case["predict_tokens"])
    full_target_required = case["workload"] == "decode" and generation_target >= 256
    complete_generated_tokens = _has_complete_generated_tokens(semantics)
    generation_population = {
        "target_tokens": generation_target,
        "actual_tokens": semantics["token_count"],
        "raw_token_ids_complete": complete_generated_tokens,
        "full_target_required": full_target_required,
        "valid": (
            generation_target == 0
            or (
                complete_generated_tokens
                and (
                    not full_target_required
                    or int(semantics["token_count"]) >= generation_target
                )
            )
        ),
    }
    return {
        "sample_index": sample_index,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "surface": f"llama-server-{request_endpoint}",
        "command": command,
        "request_endpoint": request_endpoint,
        "request": request_payload,
        "context_population": context_population,
        "prompt_population": prompt_population,
        "generation_population": generation_population,
        "returncode": 0 if response_payload is not None and error is None else 1,
        "server_exit_after_termination": process.returncode,
        "startup_ns": startup_ns,
        "request_wall_ns": request_wall_ns,
        "process_memory": process_memory,
        "error": error,
        "response_raw": response_bytes.decode(errors="replace"),
        "response": response_payload,
        "response_semantics": semantics,
        "tool_call_validation": tool_call_validation,
        "candidate_execution": candidate_execution,
        "server_log": server_log,
        "content_sha256": hashlib.sha256(semantics["content"].encode()).hexdigest(),
        "semantic_sha256": semantics["semantic_sha256"],
        "metrics": {
            "timings": timings,
            "speculative": _server_speculative_metrics(server_log, predicted_per_second),
        },
    }


def compare_greedy_semantics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare fixed-seed greedy token IDs, text bytes, counts, and stop reason to plain CPU."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        case = result["case"]
        key = (
            case.get("model"),
            case.get("workload"),
            case.get("request_surface"),
            case.get("prompt"),
            json.dumps(case.get("messages"), sort_keys=True, separators=(",", ":")),
            json.dumps(case.get("tools"), sort_keys=True, separators=(",", ":")),
            case.get("seed"),
            case.get("temperature"),
            case.get("predict_tokens"),
            case.get("context_tokens_target"),
            case.get("prompt_tokens_target"),
        )
        groups.setdefault(key, []).append(result)
    comparisons: list[dict[str, Any]] = []
    for key, members in groups.items():
        baselines = [
            member
            for member in members
            if member["case"]["mode"] == "plain" and member["case"]["placement"] == "cpu-only"
        ]
        if not baselines:
            continue
        baseline = baselines[0]
        baseline_samples = baseline["samples"]
        for member in members:
            for index, candidate_sample in enumerate(member["samples"]):
                if index >= len(baseline_samples):
                    break
                baseline_sample = baseline_samples[index]
                baseline_semantics = baseline_sample.get("response_semantics") or _response_semantics(
                    baseline_sample.get("response") or {},
                    str(baseline["case"].get("request_surface", "completion")),
                )
                candidate_semantics = candidate_sample.get(
                    "response_semantics"
                ) or _response_semantics(
                    candidate_sample.get("response") or {},
                    str(member["case"].get("request_surface", "completion")),
                )
                token_ids_available = (
                    _has_complete_generated_tokens(baseline_semantics)
                    and _has_complete_generated_tokens(candidate_semantics)
                )
                token_ids_identical = (
                    token_ids_available
                    and baseline_semantics["token_ids"] == candidate_semantics["token_ids"]
                )
                semantic_output_identical = (
                    baseline_semantics["semantic_sha256"]
                    == candidate_semantics["semantic_sha256"]
                )
                text_bytes_identical = (
                    baseline_semantics["content"] == candidate_semantics["content"]
                )
                token_count_identical = (
                    baseline_semantics["token_count"] == candidate_semantics["token_count"]
                )
                stop_identical = (
                    baseline_semantics["stop_reason"] == candidate_semantics["stop_reason"]
                )
                tool_calls_identical = (
                    baseline_semantics["tool_calls"] == candidate_semantics["tool_calls"]
                )
                comparisons.append(
                    {
                        "group": list(key),
                        "baseline_case": baseline["case"]["name"],
                        "candidate_case": member["case"]["name"],
                        "sample_index": index,
                        "token_ids_available": token_ids_available,
                        "token_ids_identical": token_ids_identical,
                        "text_bytes_identical": text_bytes_identical,
                        "semantic_output_identical": semantic_output_identical,
                        "tool_calls_identical": tool_calls_identical,
                        "token_count_identical": token_count_identical,
                        "stop_reason_identical": stop_identical,
                        "identical": all(
                            (
                                token_ids_identical,
                                semantic_output_identical,
                                tool_calls_identical,
                                token_count_identical,
                                stop_identical,
                            )
                        ),
                    }
                )
    return comparisons


def main() -> None:
    """Load explicit cases, execute them, and write a self-contained session record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument(
        "--llama-cli",
        type=Path,
        default=Path("/home/yiannis/side/llama.cpp/build/bin/llama-cli"),
    )
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=Path("/home/yiannis/side/llama.cpp/build/bin/llama-server"),
    )
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--case",
        dest="case_names",
        action="append",
        default=[],
        help="run only this exact case name; repeat for multiple cases",
    )
    parser.add_argument("--session-id", default="session-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    if not args.case_file.is_file():
        parser.error(f"case file not found: {args.case_file}")
    if not args.llama_cli.is_file():
        parser.error(f"llama-cli not found: {args.llama_cli}")
    if not args.llama_server.is_file():
        parser.error(f"llama-server not found: {args.llama_server}")
    raw_cases = json.loads(args.case_file.read_text(encoding="utf-8"))
    if isinstance(raw_cases, dict):
        raw_cases = raw_cases.get("cases", [])
    cases = [normalize_case(case, index=index) for index, case in enumerate(raw_cases)]
    try:
        cases = select_cases(cases, args.case_names)
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        parser.error("case file has no cases")
    for case in cases:
        base_model = Path(case["base_model"])
        if not base_model.is_file():
            parser.error(f"base model not found for {case['name']}: {base_model}")
        if case.get("draft_model") and not Path(case["draft_model"]).is_file():
            parser.error(f"draft model not found for {case['name']}: {case['draft_model']}")
        case["base_model"] = str(base_model.resolve())
        if case.get("draft_model"):
            case["draft_model"] = str(Path(case["draft_model"]).resolve())
        try:
            validate_candidate_evidence(case)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

    before = collect_environment()
    model_paths = {
        str(Path(path).resolve())
        for case in cases
        for path in (case.get("base_model"), case.get("draft_model"))
        if path
    }
    model_hashes = {path: sha256_file(Path(path)) for path in sorted(model_paths)}
    results: list[dict[str, Any]] = []
    for case in cases:
        if case["surface"] == "server":
            samples = [
                run_server_case(
                    args.llama_server.resolve(),
                    case,
                    index,
                    startup_timeout=args.startup_timeout,
                    request_timeout=args.request_timeout,
                )
                for index in range(args.samples)
            ]
        elif case["surface"] == "cli":
            samples = [run_case(args.llama_cli.resolve(), case, index) for index in range(args.samples)]
        else:
            raise ValueError(f"unsupported evaluation surface {case['surface']!r}")
        results.append(
            {
                "case": case,
                "case_sha256": hashlib.sha256(
                    json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "workload_semantics_sha256": workload_semantics_sha256(case),
                "prompt_sha256": hashlib.sha256(str(case["prompt"]).encode()).hexdigest(),
                "base_model_sha256": model_hashes[case["base_model"]],
                "draft_model_sha256": (
                    model_hashes[case["draft_model"]] if case.get("draft_model") else None
                ),
                "samples": samples,
            }
        )
    after = collect_environment()
    flat_samples = [sample for result in results for sample in result["samples"]]
    semantic_comparisons = compare_greedy_semantics(results)
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-qpu-end-to-end-session",
        "created_utc": utc_now(),
        "session_id": args.session_id,
        "measurement_contract": {
            "server_startup": "cold model load and initialization through health readiness",
            "server_request": (
                "one endpoint-specific request after health readiness; /completion for raw "
                "decode/prompt and /v1/chat/completions for structured tool calls"
            ),
            "cli_process_wall": "cold-start fallback surface including model load and initialization",
            "separation": "startup and request/inference timings are never collapsed",
            "correctness": (
                "server requests retain raw generated token IDs; structured tool calls also "
                "validate the parsed function name, exact arguments, stop reason, and token cap"
            ),
            "memory": (
                "server samples retain Linux process peak RSS and swap; retained sessions "
                "require zero system and per-process swap"
            ),
            "candidate_execution": (
                "non-CPU cases require exact qpu_llama_candidate_json telemetry matching the "
                "retained program, hashes, shape, placement, partition, and dispatch count"
            ),
        },
        "llama_cli": {
            "path": str(args.llama_cli.resolve()),
            "sha256": sha256_file(args.llama_cli),
        },
        "llama_server": {
            "path": str(args.llama_server.resolve()),
            "sha256": sha256_file(args.llama_server),
        },
        "case_file": {
            "path": str(args.case_file.resolve()),
            "sha256": sha256_file(args.case_file),
        },
        "model_hashes": model_hashes,
        "results": results,
        "greedy_semantic_comparisons": semantic_comparisons,
        "environment_before": before,
        "environment_after": after,
        "validation": validate_session(before, after, flat_samples),
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output}: {len(cases)} cases, {len(flat_samples)} samples")


if __name__ == "__main__":
    main()
