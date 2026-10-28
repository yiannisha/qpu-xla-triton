"""Run a new exploratory integrated fraction screen with fixed numerical gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llama_cpp_common import collect_environment, utc_now, write_json_atomic  # noqa: E402
from scripts.run_llama_cpp_ffn_breakdown import capture  # noqa: E402
from scripts.run_llama_cpp_qpu_ffn_island_eval import (  # noqa: E402
    git_record,
    require_no_competing_workloads,
    wait_until_cool,
)


def main():
    """Validate each fraction before running the exploratory complete-prompt grid."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument(
        "--ignore-undervoltage",
        action="store_true",
        help="record voltage events without aborting; diagnostic evidence only",
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--llama-root", type=Path, required=True)
    p.add_argument("--llama-bench", type=Path)
    p.add_argument("--profile-build", type=Path, default=ROOT / "build/llama-qpu-profile")
    args = p.parse_args()
    out = args.output.resolve()
    if args.repetitions < 1:
        p.error("repetitions must be positive")
    if out.exists() and any(out.iterdir()):
        p.error("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    args.model = args.model.resolve()
    bench = (args.llama_bench or args.llama_root / "build/bin/llama-bench").resolve()
    profile = args.profile_build.resolve()
    for artifact in [
        args.model,
        bench,
        profile / "qpu_ffn_island_model_smoke",
        profile / "qpu_repack_stride_smoke",
        profile / "qpu_ffn_island_graph_smoke",
        profile / "libggml-qpu-inline.so",
    ]:
        if not artifact.is_file():
            p.error(f"required artifact not found: {artifact}")
    record = {
        "kind": "new-exploratory-FFN-split-screen",
        "started_utc": utc_now(),
        "rows": [129, 257],
        "fractions": [0.0625, 0.125, 0.1875],
        "repetitions": args.repetitions,
        "confirmation": False,
        "preflights": [],
        "environment_original": collect_environment(),
    }
    record["ignore_undervoltage"] = args.ignore_undervoltage
    record["source"] = {"repository": git_record(ROOT), "llama_cpp": git_record(args.llama_root)}

    def save():
        write_json_atomic(out / "screen-attempt.json", record)

    def sudo(*words, value=None):
        subprocess.run(["sudo", "-n", *words], input=value, text=True, check=True, stdout=subprocess.DEVNULL)

    policies = {
        p: p.read_text().strip() for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")
    }
    swaps = [line.split() for line in Path("/proc/swaps").read_text().splitlines()[1:]]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GGML_QPU_") and k not in ["LD_PRELOAD", "LD_LIBRARY_PATH"]
    }
    env.update({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"})
    save()
    try:
        sudo("true")
        require_no_competing_workloads("before split screen")
        for policy in policies:
            sudo("tee", str(policy), value="performance\n")
        sudo("systemctl", "mask", "--runtime", "dev-zram0.swap")
        for swap in swaps:
            sudo("swapoff", swap[0])
        record["environment_before"] = collect_environment()
        save()
        valid = []
        for fraction in record["fractions"]:
            require_no_competing_workloads("before split numerical check")
            wait_until_cool(65, 600)
            command = [str(profile / "qpu_ffn_island_model_smoke"), str(args.model), str(fraction)]
            record["active_run"] = {"fraction": fraction, "command": command, "started_utc": utc_now()}
            save()
            print(f"{utc_now()} numerical preflight fraction={fraction}", flush=True)
            result = capture(
                command,
                env,
                out / f"fraction-{fraction}-validation",
                timeout_seconds=300,
                abort_on_undervoltage=not args.ignore_undervoltage,
            )
            result["fraction"] = fraction
            result["command"] = command
            record["preflights"].append(result)
            save()
            if result["invalid_reason"]:
                raise RuntimeError(result["invalid_reason"])
            if result["returncode"] == 0:
                valid.append(fraction)
        record["numerically_accepted_fractions"] = valid
        save()
        if not valid:
            raise RuntimeError("no fraction passed the fixed model numerical gates")
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/run_llama_cpp_qpu_ffn_island_eval.py"),
            "--rows",
            "129,257",
            "--fractions",
            ",".join(map(str, valid)),
            "--repetitions",
            str(args.repetitions),
            "--threads",
            "4",
            "--wgs",
            "24",
            "--plugin",
            str(profile / "libggml-qpu-inline.so"),
            "--model",
            str(args.model),
            "--llama-bench",
            str(bench),
            "--llama-root",
            str(args.llama_root.resolve()),
            "--stride-smoke",
            str(profile / "qpu_repack_stride_smoke"),
            "--graph-smoke",
            str(profile / "qpu_ffn_island_graph_smoke"),
            "--model-smoke",
            str(profile / "qpu_ffn_island_model_smoke"),
            "--skip-verification",
            "--skip-fallback",
            "--output",
            str(out / "screen.json"),
        ]
        record["screen_command"] = command
        save()
        result = capture(
            command, env, out / "screen", timeout_seconds=3600, abort_on_undervoltage=not args.ignore_undervoltage
        )
        record["execution"] = result
        record["completed"] = result["returncode"] == 0 and not result["invalid_reason"]
    except Exception as error:
        record["completed"] = False
        record["error"] = str(error)
        print(str(error), flush=True)
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
        for swap in swaps:
            if swap[0] != "/dev/zram0":
                sudo("swapon", "--priority", swap[4], swap[0])
        record["environment_after"] = collect_environment()
        record["finished_utc"] = utc_now()
        save()
        record["power_clean"] = not any(
            r.get("undervoltage_event_count", 0) for r in record["preflights"] + [record.get("execution", {})]
        )
        save()
    if not record.get("completed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
