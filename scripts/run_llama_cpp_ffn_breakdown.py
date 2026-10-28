"""Collect diagnostic FFN spans and branch timelines in independent processes."""

from __future__ import annotations

import argparse
import json
import os
import random
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llama_cpp_common import collect_environment, sha256_file, utc_now, write_json_atomic  # noqa: E402
from scripts.run_llama_cpp_qpu_ffn_island_eval import (  # noqa: E402
    git_record,
    manifest_programs,
    parse_prefixed_json,
    require_no_competing_workloads,
    validate_candidate,
    wait_until_cool,
)


def capture(command, env, log_prefix, timeout_seconds=180, abort_on_undervoltage=True):
    """Capture a process group and a live kernel stream with bounded cleanup."""
    kernel = subprocess.Popen(
        ["sudo", "-n", "dmesg", "--follow-new", "--time-format", "iso"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    streams, files = {}, []
    for stream, label in [(child.stdout, "stdout"), (child.stderr, "stderr"), (kernel.stdout, "kernel")]:
        f = Path(str(log_prefix) + f"-{label}.log").open("wb", buffering=0)
        selector.register(stream, selectors.EVENT_READ, (label, f))
        streams[label] = bytearray()
        files.append(f)
    start, reason = time.monotonic(), None
    try:
        while child.poll() is None:
            for key, _ in selector.select(0.25):
                chunk = os.read(key.fileobj.fileno(), 65536)
                label, f = key.data
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                f.write(chunk)
                streams[label].extend(chunk)
                if label == "kernel":
                    os.fsync(f.fileno())
                    if abort_on_undervoltage and b"Undervoltage detected!" in streams[label]:
                        reason = "new kernel undervoltage event during diagnostic"
            if time.monotonic() - start > timeout_seconds:
                reason = "diagnostic process timeout"
            if kernel.poll() is not None:
                reason = "kernel monitor exited; check sudo/dmesg access in the kernel log"
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
        for key in list(selector.get_map().values()):
            label, f = key.data
            if label != "kernel":
                chunk = key.fileobj.read()
                streams[label].extend(chunk)
                f.write(chunk)
        return {
            "returncode": child.wait(),
            "invalid_reason": reason,
            "undervoltage_event_count": streams["kernel"].count(b"Undervoltage detected!"),
            **{k: v.decode(errors="replace") for k, v in streams.items()},
        }
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        if kernel.poll() is None:
            os.killpg(kernel.pid, signal.SIGTERM)
        kernel.wait(timeout=5)
        for f in files:
            os.fsync(f.fileno())
            f.close()


def main():
    """Run independent instrumented CPU/QPU prompt pairs and save every attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument(
        "--ignore-undervoltage",
        action="store_true",
        help="record voltage events without aborting; diagnostic evidence only",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-bench", type=Path, help="Defaults to LLAMA_ROOT/build/bin/llama-bench")
    parser.add_argument("--cpu-profile", type=Path, default=ROOT / "build/ffn-profile-cpu")
    parser.add_argument("--profile-build", type=Path, default=ROOT / "build/llama-qpu-profile")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.pairs < 1:
        parser.error("pairs must be positive")
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty; preserve earlier attempts")
    out.mkdir(parents=True, exist_ok=True)
    args.model = args.model.resolve()
    args.llama_bench = (args.llama_bench or args.llama_root / "build/bin/llama-bench").resolve()
    cpu = args.cpu_profile.resolve()
    plugin = args.profile_build.resolve() / "libggml-qpu-inline.so"
    for path in [args.model, args.llama_bench, cpu / "libggml-cpu.so.0", plugin]:
        if not path.is_file():
            parser.error(f"required artifact not found: {path}")
    programs = manifest_programs(ROOT / "integrations/llama_cpp/generated/manifest.json")
    record = {
        "kind": "instrumented-ffn-breakdown-diagnostic",
        "started_utc": utc_now(),
        "timing_eligible_for_headline": False,
        "planned_pairs_per_shape": args.pairs,
        "rows": [129, 257],
        "fraction": 0.125,
        "runs": [],
        "ignore_undervoltage": args.ignore_undervoltage,
        "source": {"repository": git_record(ROOT), "llama_cpp": git_record(args.llama_root)},
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "llama_bench_sha256": sha256_file(args.llama_bench),
        "cpu_library_sha256": sha256_file(cpu / "libggml-cpu.so.0"),
        "plugin_sha256": sha256_file(plugin),
        "environment_original": collect_environment(),
    }
    path = out / "breakdown.json"

    def save():
        write_json_atomic(path, record)
        with path.open("rb") as f:
            os.fsync(f.fileno())

    policies = {
        p: p.read_text().strip() for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")
    }
    swaps = [line.split() for line in Path("/proc/swaps").read_text().splitlines()[1:]]
    save()

    def sudo(*words, value=None):
        subprocess.run(["sudo", "-n", *words], input=value, text=True, check=True, stdout=subprocess.DEVNULL)

    try:
        sudo("true")
        for p in policies:
            sudo("tee", str(p), value="performance\n")
        sudo("systemctl", "mask", "--runtime", "dev-zram0.swap")
        for swap in swaps:
            sudo("swapoff", swap[0])
        record["environment_before"] = collect_environment()
        rng = random.Random(20261028)
        for rows in [129, 257]:
            for pair in range(args.pairs):
                order = [False, True]
                rng.shuffle(order)
                for candidate in order:
                    require_no_competing_workloads("before FFN diagnostic")
                    wait_until_cool(65, 600)
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if not k.startswith("GGML_QPU_") and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")
                    }
                    env.update(
                        {
                            "LD_LIBRARY_PATH": str(cpu),
                            "GGML_QPU_FFN_PROFILE": "1",
                            "OPENBLAS_NUM_THREADS": "4",
                            "OMP_NUM_THREADS": "4",
                        }
                    )
                    if candidate:
                        env.update(
                            {
                                "LD_PRELOAD": str(plugin),
                                "GGML_QPU_FFN_ISLAND": "1",
                                "GGML_QPU_FFN_ISLAND_FRACTION": "0.125",
                                "GGML_QPU_FFN_ISLAND_WGS": "24",
                                "GGML_QPU_TELEMETRY": "1",
                            }
                        )
                    command = [
                        str(args.llama_bench),
                        "-m",
                        str(args.model),
                        "-p",
                        str(rows),
                        "-n",
                        "0",
                        "-t",
                        "4",
                        "-r",
                        "1",
                        "-o",
                        "json",
                    ]
                    label = f"m{rows}-pair{pair}-{'qpu' if candidate else 'cpu'}"
                    print(f"{utc_now()} {label}", flush=True)
                    record["active_run"] = {"label": label, "command": command, "started_utc": utc_now()}
                    save()
                    run = capture(command, env, out / label, abort_on_undervoltage=not args.ignore_undervoltage)
                    run.update(
                        {
                            "label": label,
                            "rows": rows,
                            "pair": pair,
                            "candidate": candidate,
                            "command": command,
                            "finished_utc": utc_now(),
                        }
                    )
                    record["runs"].append(run)
                    save()
                    require_no_competing_workloads("after FFN diagnostic")
                    if run["invalid_reason"] or run["returncode"]:
                        raise RuntimeError(run["invalid_reason"] or f"process failed: {run['returncode']}")
                    run["benchmark"] = json.loads(run["stdout"])[0]
                    run["regions"] = parse_prefixed_json(run["stderr"], "cpu_ffn_region_json:")
                    if len(run["regions"]) != 70:
                        raise RuntimeError(f"expected warmup + measured 35-layer FFN spans: {len(run['regions'])}")
                    if candidate:
                        run["candidate_validation"] = validate_candidate(
                            parse_prefixed_json(run["stderr"], "qpu_llama_candidate_json:"),
                            parse_prefixed_json(run["stderr"], "qpu_llama_weight_json:"),
                            rows=rows,
                            minimum_rows=64,
                            maximum_rows=528,
                            programs=programs,
                        )
                        if not run["candidate_validation"]["valid"]:
                            raise RuntimeError(str(run["candidate_validation"]))
                    save()
        record["completed"] = True
    except Exception as error:
        record["completed"] = False
        record["error"] = str(error)
        print(str(error), flush=True)
    finally:
        for p, governor in policies.items():
            sudo("tee", str(p), value=governor + "\n")
        sudo("systemctl", "unmask", "--runtime", "dev-zram0.swap")
        subprocess.run(
            ["sudo", "-n", "systemctl", "reset-failed", "dev-zram0.swap", "systemd-zram-setup@zram0.service"],
            check=False,
        )
        if any(swap[0] == "/dev/zram0" for swap in swaps):
            sudo("systemctl", "start", "dev-zram0.swap")
        for swap in swaps:
            if swap[0] != "/dev/zram0":
                sudo("swapon", "--priority", swap[4], swap[0])
        record["environment_after"] = collect_environment()
        record["power_clean"] = not any(run.get("undervoltage_event_count", 0) for run in record["runs"])
        record["finished_utc"] = utc_now()
        save()
    if not record.get("completed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
