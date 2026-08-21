"""Shared helpers for reproducible llama.cpp/QPU experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return a stable UTC timestamp for retained JSON records."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Hash a file without loading a model-sized input into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON without leaving a valid-looking partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_capture(command: Sequence[str], *, cwd: Path | None = None, check: bool = False) -> dict[str, Any]:
    """Run a command and retain its complete result in a JSON-safe record."""
    started = utc_now()
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        record: dict[str, Any] = {
            "command": list(command),
            "cwd": str(cwd) if cwd is not None else None,
            "started_utc": started,
            "finished_utc": utc_now(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as exc:
        record = {
            "command": list(command),
            "cwd": str(cwd) if cwd is not None else None,
            "started_utc": started,
            "finished_utc": utc_now(),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    if check and record["returncode"] != 0:
        raise RuntimeError(
            f"command failed with status {record['returncode']}: {' '.join(command)}\n{record['stderr']}"
        )
    return record


def read_text(path: Path) -> str | None:
    """Read a small system metadata file when it exists."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def parse_cmake_cache(path: Path) -> dict[str, str]:
    """Parse user-facing CMake cache entries."""
    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        entries[key] = value
    return entries


def _cpu_frequency_state() -> list[dict[str, str | None]]:
    root = Path("/sys/devices/system/cpu/cpufreq")
    result: list[dict[str, str | None]] = []
    for policy in sorted(root.glob("policy*")):
        result.append(
            {
                "policy": policy.name,
                "governor": read_text(policy / "scaling_governor"),
                "current_khz": read_text(policy / "scaling_cur_freq"),
                "minimum_khz": read_text(policy / "scaling_min_freq"),
                "maximum_khz": read_text(policy / "scaling_max_freq"),
            }
        )
    return result


def _thermal_state() -> list[dict[str, str | int | None]]:
    result: list[dict[str, str | int | None]] = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        raw = read_text(zone / "temp")
        result.append(
            {
                "zone": zone.name,
                "type": read_text(zone / "type"),
                "millidegrees_c": int(raw) if raw and re.fullmatch(r"-?\d+", raw) else None,
            }
        )
    return result


def collect_environment() -> dict[str, Any]:
    """Collect the environment fields required by retained benchmark sessions."""
    commands = {
        "lscpu": run_capture(["lscpu"]),
        "process_affinity": run_capture(
            ["taskset", "--pid", "--cpu-list", str(os.getpid())]
        ),
        "memory": run_capture(["free", "--bytes"]),
        "swap": run_capture(["swapon", "--show", "--bytes", "--output-all"]),
        "swap_used_bytes": run_capture(
            ["swapon", "--show", "--bytes", "--noheadings", "--output", "USED"]
        ),
        "kernel": run_capture(["uname", "-a"]),
        "firmware": run_capture(["vcgencmd", "version"]),
        "throttling": run_capture(["vcgencmd", "get_throttled"]),
        "temperature": run_capture(["vcgencmd", "measure_temp"]),
        "cpu_clock": run_capture(["vcgencmd", "measure_clock", "arm"]),
        "v3d_clock": run_capture(["vcgencmd", "measure_clock", "v3d"]),
        "zram": run_capture(["zramctl", "--json"]),
        "llama_servers": run_capture(["pgrep", "-a", "-x", "llama-server"]),
        "system_load": run_capture(["uptime"]),
    }
    return {
        "captured_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_frequency": _cpu_frequency_state(),
        "thermal_zones": _thermal_state(),
        "commands": commands,
    }


def command_stdout(record: dict[str, Any]) -> str:
    """Return normalized stdout from a run record."""
    value = record.get("stdout", "")
    return str(value).strip()


def swap_used_bytes(environment: dict[str, Any]) -> int | None:
    """Sum active swap usage from the dedicated machine-readable environment probe."""
    record = environment.get("commands", {}).get("swap_used_bytes", {})
    if record.get("returncode") != 0:
        return None
    values = str(record.get("stdout", "")).split()
    try:
        return sum(int(value) for value in values)
    except ValueError:
        return None
