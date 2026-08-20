#!/usr/bin/env python3
"""Build or audit the pinned llama.cpp CPU baseline and retain exact metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    command_stdout,
    parse_cmake_cache,
    run_capture,
    sha256_file,
    utc_now,
    write_json_atomic,
)

PINNED_COMMIT = "91d2fc387529940230555abd297a8b5e99737d3f"
REQUIRED_CACHE = {"CMAKE_BUILD_TYPE": "Release", "GGML_NATIVE": "ON"}


def _configure(llama_root: Path, build_dir: Path, *, kleidiai: bool) -> dict[str, Any]:
    command = [
        "cmake",
        "-S",
        str(llama_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DGGML_CPU_KLEIDIAI={'ON' if kleidiai else 'OFF'}",
    ]
    return run_capture(command)


def _build(build_dir: Path, jobs: int) -> dict[str, Any]:
    return run_capture(["cmake", "--build", str(build_dir), "--parallel", str(jobs)])


def _compile_evidence(build_dir: Path) -> dict[str, Any]:
    compile_commands = build_dir / "compile_commands.json"
    text = compile_commands.read_text(encoding="utf-8", errors="replace") if compile_commands.is_file() else ""
    cpu_commands: list[str] = []
    if text:
        entries = json.loads(text)
        cpu_commands = [
            str(entry.get("command", ""))
            for entry in entries
            if "ggml-cpu" in str(entry.get("file", "")) or "ggml-cpu" in str(entry.get("command", ""))
        ]
    joined = "\n".join(cpu_commands)
    return {
        "cpu_compile_command_count": len(cpu_commands),
        "cortex_a76": "cortex-a76" in joined,
        "neon": any(token in joined for token in ("+simd", "+neon", "cortex-a76")),
        "dot_product": "+dotprod" in joined,
        "fp16_vector": "cortex-a76" in joined and "+nofp16" not in joined,
        "llamafile": "GGML_USE_LLAMAFILE" in joined,
        "kleidiai": "GGML_USE_CPU_KLEIDIAI" in joined,
        "representative_cpu_command": cpu_commands[0] if cpu_commands else None,
    }


def _model_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    """Build when requested, validate the checkout, and emit a baseline manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path, default=Path.home() / "side/llama.cpp")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", default=PINNED_COMMIT)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--kleidiai", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    llama_root = args.llama_root.resolve()
    build_dir = (args.build_dir or llama_root / "build").resolve()
    if not (llama_root / ".git").exists():
        parser.error(f"not a git checkout: {llama_root}")
    if args.build:
        args.configure = True

    head_record = run_capture(["git", "rev-parse", "HEAD"], cwd=llama_root, check=True)
    head = command_stdout(head_record)
    status_record = run_capture(["git", "status", "--porcelain", "--untracked-files=no"], cwd=llama_root, check=True)
    dirty = bool(command_stdout(status_record))
    if head != args.expected_commit:
        parser.error(f"expected llama.cpp {args.expected_commit}, found {head}")
    if dirty and not args.allow_dirty:
        parser.error("llama.cpp has tracked changes; pass --allow-dirty only for a deliberately non-baseline audit")

    actions: list[dict[str, Any]] = []
    if args.configure:
        configure = _configure(llama_root, build_dir, kleidiai=args.kleidiai)
        actions.append(configure)
        if configure["returncode"] != 0:
            write_json_atomic(args.output, {"status": "configure-failed", "actions": actions})
            raise SystemExit(int(configure["returncode"] or 1))
    if args.build:
        build = _build(build_dir, args.jobs)
        actions.append(build)
        if build["returncode"] != 0:
            write_json_atomic(args.output, {"status": "build-failed", "actions": actions})
            raise SystemExit(int(build["returncode"] or 1))

    cache = parse_cmake_cache(build_dir / "CMakeCache.txt")
    cache_checks = {key: cache.get(key) == value for key, value in REQUIRED_CACHE.items()}
    cli = build_dir / "bin/llama-cli"
    bench = build_dir / "bin/llama-bench"
    version_record = run_capture([str(cli), "--version"]) if cli.is_file() else None
    devices_record = run_capture([str(bench), "--list-devices"]) if bench.is_file() else None
    compile_evidence = _compile_evidence(build_dir)
    version_text = ""
    if version_record is not None:
        version_text = f"{version_record['stdout']}\n{version_record['stderr']}"
    build_number = None
    if match := re.search(r"version:\s+(\d+)", version_text):
        build_number = int(match.group(1))

    failures = [key for key, passed in cache_checks.items() if not passed]
    failures.extend(
        key
        for key in ("cortex_a76", "neon", "dot_product", "fp16_vector", "llamafile")
        if not compile_evidence[key]
    )
    if args.kleidiai and not compile_evidence["kleidiai"]:
        failures.append("kleidiai")
    missing_models = [str(path) for path in args.model if not path.is_file()]
    if missing_models:
        failures.append("model-files")
    payload = {
        "schema_version": 1,
        "kind": "llama-cpp-pinned-baseline-bootstrap",
        "status": "valid" if not failures else "invalid",
        "created_utc": utc_now(),
        "failures": failures,
        "checkout": {
            "root": str(llama_root),
            "head": head,
            "expected_commit": args.expected_commit,
            "tracked_worktree_dirty": dirty,
            "git_status": status_record,
        },
        "build": {
            "directory": str(build_dir),
            "cache": cache,
            "required_cache_checks": cache_checks,
            "compile_evidence": compile_evidence,
            "build_number": build_number,
            "llama_cli_sha256": sha256_file(cli) if cli.is_file() else None,
            "version": version_record,
            "devices": devices_record,
            "actions": actions,
        },
        "models": [_model_record(path) for path in args.model if path.is_file()],
        "missing_models": missing_models,
        "environment": collect_environment(),
    }
    write_json_atomic(args.output, payload)
    print(f"{payload['status']}: wrote {args.output}")
    if failures:
        raise SystemExit(f"baseline validation failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
