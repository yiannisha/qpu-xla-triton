"""Run one diagnostic command with performance/swap controls and live kernel logs.

Example: python scripts/run_rebuttal_diagnostics.py --output /tmp/run -- COMMAND ...
The command is executed directly, without shell expansion. Every attempt is retained.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llama_cpp_common import collect_environment, gpu_error_lines, utc_now, write_json_atomic  # noqa: E402
from scripts.run_llama_cpp_ffn_breakdown import capture  # noqa: E402
from scripts.run_llama_cpp_qpu_ffn_island_eval import require_no_competing_workloads, wait_until_cool  # noqa: E402


def main():
    """Execute the requested experiment and retain its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command after --")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    record = {"command": command, "started_utc": utc_now(), "environment_original": collect_environment()}

    def save():
        write_json_atomic(out / "process.json", record)

    def sudo(*words, value=None):
        subprocess.run(["sudo", "-n", *words], input=value, text=True, check=True, stdout=subprocess.DEVNULL)

    policies = {
        p: p.read_text().strip() for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")
    }
    swaps = [line.split() for line in Path("/proc/swaps").read_text().splitlines()[1:]]
    save()
    try:
        require_no_competing_workloads("rebuttal diagnostic")
        for policy in policies:
            sudo("tee", str(policy), value="performance\n")
        sudo("systemctl", "mask", "--runtime", "dev-zram0.swap")
        for swap in swaps:
            sudo("swapoff", swap[0])
        wait_until_cool(65, 600)
        record["environment_before"] = collect_environment()
        save()
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"}
        record["result"] = capture(command, env, out / "run", timeout_seconds=args.timeout)
        record["completed"] = record["result"]["returncode"] == 0 and record["result"]["invalid_reason"] is None
        record["kernel_error_lines"] = gpu_error_lines(record["result"]["kernel"])
        record["retained_for_latency"] = record["completed"] and not record["kernel_error_lines"]
    except Exception as error:
        record["completed"] = False
        record["error"] = str(error)
    finally:
        for policy, governor in policies.items():
            sudo("tee", str(policy), value=governor + "\n")
        sudo("systemctl", "unmask", "--runtime", "dev-zram0.swap")
        subprocess.run(
            ["sudo", "-n", "systemctl", "reset-failed", "dev-zram0.swap", "systemd-zram-setup@zram0.service"],
            check=False,
        )
        if any(s[0] == "/dev/zram0" for s in swaps):
            sudo("systemctl", "start", "dev-zram0.swap")
        record["environment_after"] = collect_environment()
        record["finished_utc"] = utc_now()
        save()
    print(str(out / "process.json"), flush=True)
    if not record.get("retained_for_latency"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
