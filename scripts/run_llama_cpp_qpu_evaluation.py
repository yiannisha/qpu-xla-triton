#!/usr/bin/env python3
"""Run isolated, reproducible llama.cpp end-to-end baseline/evaluation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    command.extend(str(argument) for argument in case.get("server_extra_arguments", []))
    return command


def normalize_case(case: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Validate fields that affect semantics and fill only explicit stable defaults."""
    normalized = dict(case)
    normalized.setdefault("name", f"case-{index}")
    normalized.setdefault("mode", "plain")
    normalized.setdefault("predict_tokens", 256)
    normalized.setdefault("context_size", 8192)
    normalized.setdefault("threads", 3)
    normalized.setdefault("seed", 1234)
    normalized.setdefault("temperature", 0.0)
    normalized.setdefault("prompt", "Write one concise sentence about compiler optimization.")
    normalized.setdefault("flash_attention", True)
    normalized.setdefault("placement", "cpu-only")
    normalized.setdefault("surface", "server")
    if normalized["mode"] not in {"plain", "mtp", "prompt"}:
        raise ValueError(f"case {normalized['name']!r} has unsupported mode {normalized['mode']!r}")
    if normalized["surface"] not in {"server", "cli"}:
        raise ValueError(
            f"case {normalized['name']!r} has unsupported surface {normalized['surface']!r}"
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
    if context_target < 0 or context_target >= int(normalized["context_size"]):
        raise ValueError(f"case {normalized['name']!r} has invalid context_tokens_target")
    if prompt_target < 0 or prompt_target >= int(normalized["context_size"]):
        raise ValueError(f"case {normalized['name']!r} has invalid prompt_tokens_target")
    if context_target and prompt_target:
        raise ValueError(
            f"case {normalized['name']!r} cannot combine populated context and exact prompt targets"
        )
    if prompt_target and normalized["mode"] != "prompt":
        raise ValueError(f"case {normalized['name']!r} uses prompt_tokens_target outside prompt mode")
    if normalized["mode"] == "prompt" and int(normalized["predict_tokens"]) != 0:
        raise ValueError(f"prompt case {normalized['name']!r} must set predict_tokens to zero")
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


def _throttle_value(environment: dict[str, Any]) -> int | None:
    stdout = environment["commands"]["throttling"].get("stdout", "")
    try:
        return int(stdout.split("throttled=", 1)[1].strip(), 0)
    except (IndexError, ValueError):
        return None


def validate_session(before: dict[str, Any], after: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply environmental and process-success retention gates without hiding failures."""
    reasons: list[str] = []
    if _throttle_value(before) != 0 or _throttle_value(after) != 0:
        reasons.append("throttling flags were unavailable or nonzero")
    if before["commands"]["swap"].get("stdout") != after["commands"]["swap"].get("stdout"):
        reasons.append("swapon state changed during the session")
    governors = {entry["governor"] for entry in before["cpu_frequency"]}
    if governors != {"performance"}:
        reasons.append(f"CPU governors were {sorted(str(value) for value in governors)}")
    if any(sample["returncode"] != 0 for sample in samples):
        reasons.append("one or more llama.cpp processes failed")
    if any(not sample.get("context_population", {}).get("valid", True) for sample in samples):
        reasons.append("one or more populated-context requests did not reuse the requested prefix")
    if any(not sample.get("prompt_population", {}).get("valid", True) for sample in samples):
        reasons.append("one or more exact-prompt requests did not evaluate the requested token count")
    return {"retained": not reasons, "rejection_reasons": reasons}


def run_case(binary: Path, case: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Run one isolated llama-cli process and retain all output and timings."""
    command = build_command(binary, case)
    started_utc = utc_now()
    started_ns = monotonic_ns()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
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


def _server_speculative_metrics(log: str, predicted_per_second: float | None) -> dict[str, Any]:
    acceptance_matches = list(DRAFT_ACCEPTANCE_RE.finditer(log))
    position_matches = list(ACCEPTANCE_POSITION_RE.finditer(log))
    if not acceptance_matches:
        parsed = parse_llama_metrics(log)["speculative"]
        if not parsed:
            return {}
        result = parsed[-1]
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


def _context_prompt(port: int, case: dict[str, Any], timeout: float) -> tuple[str | list[int], list[int]]:
    target = int(case.get("context_tokens_target", 0))
    if target <= 0:
        return str(case["prompt"]), []
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
    _, task_response = _post_json(
        port,
        "/tokenize",
        {"content": str(case["prompt"]), "add_special": False},
        timeout,
    )
    task_tokens = [int(token) for token in task_response["tokens"]]
    return prefix_tokens + task_tokens, prefix_tokens


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
    """Run one isolated server, wait for readiness, and time one completion request."""
    port = _available_loopback_port()
    command = build_server_command(binary, case, port)
    request_payload = {
        "prompt": str(case["prompt"]),
        "n_predict": int(case["predict_tokens"]),
        "cache_prompt": False,
        "temperature": float(case.get("temperature", 0.0)),
        "seed": int(case["seed"]),
    }
    started_utc = utc_now()
    process_started_ns = monotonic_ns()
    response_bytes = b""
    response_payload: dict[str, Any] | None = None
    error: str | None = None
    request_wall_ns: int | None = None
    startup_ns: int | None = None
    context_population: dict[str, Any] = {
        "target_tokens": int(case.get("context_tokens_target", 0)),
        "prefix_token_count": 0,
        "prefill_wall_ns": None,
        "prefill_response": None,
    }
    prompt_population: dict[str, Any] = {
        "target_tokens": int(case.get("prompt_tokens_target", 0)),
        "request_token_count": None,
        "request_tokens_evaluated": None,
        "timed_prompt_tokens": None,
    }
    with tempfile.TemporaryFile(mode="w+b") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            _wait_for_health(port, process, startup_timeout)
            startup_ns = monotonic_ns() - process_started_ns
            exact_prompt = _exact_prompt(port, case, request_timeout)
            if exact_prompt is None:
                prompt, prefix_tokens = _context_prompt(port, case, request_timeout)
            else:
                prompt, prefix_tokens = exact_prompt, []
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
                }
                request_payload["cache_prompt"] = True
            request_started_ns = monotonic_ns()
            response_bytes, response_payload = _post_json(
                port,
                "/completion",
                request_payload,
                request_timeout,
            )
            request_wall_ns = monotonic_ns() - request_started_ns
        except (OSError, RuntimeError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
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
    timings = response_payload.get("timings", {}) if response_payload else {}
    predicted_per_second_value = timings.get("predicted_per_second")
    predicted_per_second = (
        float(predicted_per_second_value) if predicted_per_second_value is not None else None
    )
    target_context_tokens = int(context_population["target_tokens"])
    request_cache_tokens = int(timings.get("cache_n", 0))
    prefill_response = context_population.get("prefill_response") or {}
    prefill_cached_tokens = int(prefill_response.get("tokens_cached", 0))
    request_prompt_tokens = int(response_payload.get("tokens_evaluated", 0)) if response_payload else 0
    context_population["request_cache_n"] = request_cache_tokens
    context_population["prefill_tokens_cached"] = prefill_cached_tokens
    context_population["request_tokens_evaluated"] = request_prompt_tokens
    context_population["valid"] = (
        target_context_tokens == 0
        or (
            int(context_population["prefix_token_count"]) == target_context_tokens
            and prefill_cached_tokens == target_context_tokens
            and request_prompt_tokens >= target_context_tokens
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
    content = str(response_payload.get("content", "")) if response_payload else ""
    return {
        "sample_index": sample_index,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "surface": "llama-server-/completion",
        "command": command,
        "request": request_payload,
        "context_population": context_population,
        "prompt_population": prompt_population,
        "returncode": 0 if response_payload is not None and error is None else 1,
        "server_exit_after_termination": process.returncode,
        "startup_ns": startup_ns,
        "request_wall_ns": request_wall_ns,
        "error": error,
        "response_raw": response_bytes.decode(errors="replace"),
        "response": response_payload,
        "server_log": server_log,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
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
            case.get("prompt"),
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
                baseline_response = baseline_sample.get("response") or {}
                candidate_response = candidate_sample.get("response") or {}
                token_ids_identical = baseline_response.get("tokens") == candidate_response.get("tokens")
                text_identical = baseline_sample.get("content_sha256") == candidate_sample.get("content_sha256")
                token_count_identical = baseline_response.get("tokens_predicted") == candidate_response.get(
                    "tokens_predicted"
                )
                stop_identical = baseline_response.get("stop_type") == candidate_response.get("stop_type")
                comparisons.append(
                    {
                        "group": list(key),
                        "baseline_case": baseline["case"]["name"],
                        "candidate_case": member["case"]["name"],
                        "sample_index": index,
                        "token_ids_identical": token_ids_identical,
                        "text_bytes_identical": text_identical,
                        "token_count_identical": token_count_identical,
                        "stop_reason_identical": stop_identical,
                        "identical": all(
                            (token_ids_identical, text_identical, token_count_identical, stop_identical)
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
            "server_request": "one /completion request after health readiness",
            "cli_process_wall": "cold-start fallback surface including model load and initialization",
            "separation": "startup and request/inference timings are never collapsed",
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
