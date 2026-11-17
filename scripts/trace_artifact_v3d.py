#!/usr/bin/env python3
"""Capture V3D submission tracepoints in a private instance for fault diagnosis.

Use outside performance measurements. The command runs as the invoking user;
only tracefs access uses existing noninteractive lab privileges.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llama_cpp_common import gpu_error_lines  # noqa: E402
from scripts.run_llama_cpp_ffn_breakdown import capture as run_capture  # noqa: E402
from scripts.run_rebuttal_diagnostics import MMU_PARAMETER, observable_mmu_errors  # noqa: E402

TRACEFS = Path("/sys/kernel/tracing")


def capture(command: list[str], output: Path, timeout: int) -> dict:
    """Retain a bounded diagnostic trace and remove only our own trace instance."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    instance = TRACEFS / "instances" / f"qpu-artifact-{os.getpid()}-{time.monotonic_ns()}"
    record = {
        "format": "artifact-v3d-trace-v1",
        "command": command,
        "instance": str(instance),
        "scope": "Diagnostic only; tracing overhead invalidates use as paper performance evidence.",
        "started_epoch_seconds": time.time(),
        "started_monotonic_seconds": time.monotonic(),
        "completed": False,
    }

    def sudo(*args, value=None):
        return subprocess.run(["sudo", "-n", *args], input=value, text=True, capture_output=True, check=True).stdout

    def write(path, value):
        sudo("tee", str(path), value=value)

    created = False
    try:
        sudo("mkdir", str(instance))
        created = True
        write(instance / "tracing_on", "0\n")
        write(instance / "trace_clock", "mono\n")
        write(instance / "buffer_size_kb", "4096\n")
        write(instance / "events/v3d/enable", "1\n")
        formats = {}
        for event in ("v3d_submit_csd_ioctl", "v3d_submit_cl_ioctl", "v3d_submit_csd", "v3d_csd_irq"):
            formats[event] = sudo("cat", str(instance / "events/v3d" / event / "format"))
        (output / "event-formats.json").write_text(json.dumps(formats, indent=2) + "\n")
        write(instance / "tracing_on", "1\n")
        record["mmu_logging"] = {}

        def write_mmu(value):
            sudo("tee", str(MMU_PARAMETER), value=value + "\n")

        with observable_mmu_errors(record["mmu_logging"], write_mmu):
            result = run_capture(command, dict(os.environ), output / "run", timeout_seconds=timeout)
        record["result"] = result
        record["returncode"] = result["returncode"]
        record["completed"] = result["returncode"] == 0 and result["invalid_reason"] is None
        record["kernel_error_lines"] = gpu_error_lines(result["kernel"])

    except (Exception, KeyboardInterrupt) as error:
        record["error"] = str(error)
    finally:
        if created:
            try:
                write(instance / "tracing_on", "0\n")
                trace = sudo("cat", str(instance / "trace"))
                (output / "trace.txt").write_text(trace)
                # Per-CPU overrun counts qualify whether the retained trace is complete.
                record["per_cpu_stats"] = sudo(
                    "sh", "-c", 'for path in "$1"/per_cpu/cpu*/stats; do cat "$path"; done', "sh", str(instance)
                )
                lost = re.findall(
                    r"^(?:overrun|commit overrun|dropped events):\s*(\d+)",
                    record["per_cpu_stats"],
                    re.MULTILINE,
                )
                record["reported_lost_events"] = sum(map(int, lost))
                record["trace_complete"] = bool(lost) and record["reported_lost_events"] == 0
                record["compute_ioctl_events"] = trace.count("v3d_submit_csd_ioctl:")
                record["render_ioctl_events"] = trace.count("v3d_submit_cl_ioctl:")
            except (OSError, subprocess.CalledProcessError) as error:
                record["capture_error"] = str(error)
            try:
                sudo("rmdir", str(instance))
                record["instance_removed"] = True
            except subprocess.CalledProcessError as error:
                record["cleanup_error"] = str(error)
        record["finished_epoch_seconds"] = time.time()
        (output / "trace-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    """Run a user-selected diagnostic command without changing global tracing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("supply a positive timeout and a command after --")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    record = capture(command, args.output.resolve(), args.timeout)
    print(json.dumps(record, indent=2))
    valid = (
        record.get("completed")
        and record.get("instance_removed")
        and record.get("trace_complete")
        and not record.get("capture_error")
        and not record.get("kernel_error_lines")
    )
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
