"""Validate and measure matched compaction, residency and concurrency variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.llama_cpp_common import (  # noqa: E402
    collect_environment,
    gpu_error_lines,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from scripts.run_llama_cpp_ffn_breakdown import capture  # noqa: E402
from scripts.run_llama_cpp_qpu_ffn_island_eval import (  # noqa: E402
    manifest_programs,
    parse_prefixed_json,
    require_no_competing_workloads,
    validate_candidate,
    wait_until_cool,
)


def main():
    """Execute the requested experiment and retain its evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--llama-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    if args.rounds < 1:
        p.error("rounds must be positive")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    build = ROOT / "build/llama-qpu-ablation"
    cpu = ROOT / "build/ffn-ablation-cpu"
    modes = ["full", "compact", "materialized", "serialized"]
    common = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GGML_QPU_") and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")
    }
    common.update({"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "LD_LIBRARY_PATH": str(cpu)})
    record = {
        "kind": "matched-all-channel-FFN-ablation",
        "started_utc": utc_now(),
        "modes": modes,
        "rows": 129,
        "rounds": args.rounds,
        "fraction": 0.125,
        "preflights": [],
        "runs": [],
        "plugin_sha256": sha256_file(build / "libggml-qpu-inline.so"),
        "model_sha256": sha256_file(args.model),
        "cpu_library_sha256": sha256_file(cpu / "libggml-cpu.so.0"),
        "environment_original": collect_environment(),
    }

    def save():
        write_json_atomic(out / "ablation.json", record)

    def sudo(*words, value=None):
        subprocess.run(["sudo", "-n", *words], input=value, text=True, check=True, stdout=subprocess.DEVNULL)

    policies = {
        x: x.read_text().strip() for x in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")
    }
    swaps = [x.split() for x in Path("/proc/swaps").read_text().splitlines()[1:]]
    programs = manifest_programs(ROOT / "integrations/llama_cpp/generated/manifest.json")
    save()
    try:
        for policy in policies:
            sudo("tee", str(policy), value="performance\n")
        sudo("systemctl", "mask", "--runtime", "dev-zram0.swap")
        for swap in swaps:
            sudo("swapoff", swap[0])
        record["environment_before"] = collect_environment()
        save()
        reference = {}
        for mode in modes:
            require_no_competing_workloads("ablation preflight")
            wait_until_cool(65, 600)
            env = {**common, "GGML_QPU_FFN_ABLATION": mode, "GGML_QPU_FFN_DUMP_DIR": str(out / "logits" / mode)}
            print(f"{utc_now()} validating {mode}", flush=True)
            record["active"] = {"phase": "preflight", "mode": mode}
            save()
            result = capture(
                [str(build / "qpu_ffn_island_model_smoke"), str(args.model.resolve()), "0.125"],
                env,
                out / f"preflight-{mode}",
                timeout_seconds=300,
            )
            result["mode"] = mode
            result["gpu_error_lines"] = gpu_error_lines(result["kernel"])
            record["preflights"].append(result)
            save()
            if result["returncode"] or result["invalid_reason"] or result["gpu_error_lines"]:
                raise RuntimeError(f"{mode} numerical preflight failed")
            hashes = {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (out / "logits" / mode).glob("*.f32")
            }
            assert len(hashes) == 6
            if mode == "full":
                reference = hashes
            result["logit_hashes"] = hashes
            result["bitwise_matches_full"] = hashes == reference
            save()
            if hashes != reference:
                raise RuntimeError(f"{mode} changed candidate logits; matched ablation rejected")
        rng = random.Random(20261028)
        for round_index in range(args.rounds):
            order = modes.copy()
            rng.shuffle(order)
            for mode in order:
                require_no_competing_workloads("ablation timing")
                wait_until_cool(65, 600)
                env = {
                    **common,
                    "LD_PRELOAD": str(build / "libggml-qpu-inline.so"),
                    "GGML_QPU_FFN_ISLAND": "1",
                    "GGML_QPU_FFN_ISLAND_FRACTION": "0.125",
                    "GGML_QPU_FFN_ISLAND_WGS": "24",
                    "GGML_QPU_TELEMETRY": "1",
                    "GGML_QPU_FFN_ABLATION": mode,
                }
                command = [
                    str(args.llama_root.resolve() / "build/bin/llama-bench"),
                    "-m",
                    str(args.model.resolve()),
                    "-p",
                    "129",
                    "-n",
                    "0",
                    "-t",
                    "4",
                    "-r",
                    "1",
                    "-o",
                    "json",
                ]
                print(f"{utc_now()} round={round_index} mode={mode}", flush=True)
                record["active"] = {"phase": "timing", "mode": mode, "round": round_index}
                save()
                result = capture(command, env, out / f"round-{round_index}-{mode}")
                result.update({"mode": mode, "round": round_index, "command": command})
                result["gpu_error_lines"] = gpu_error_lines(result["kernel"])
                record["runs"].append(result)
                save()
                if result["returncode"] or result["invalid_reason"] or result["gpu_error_lines"]:
                    raise RuntimeError(f"{mode} process failed")
                result["benchmark"] = json.loads(result["stdout"])[0]
                result["validation"] = validate_candidate(
                    parse_prefixed_json(result["stderr"], "qpu_llama_candidate_json:"),
                    parse_prefixed_json(result["stderr"], "qpu_llama_weight_json:"),
                    rows=129,
                    minimum_rows=64,
                    maximum_rows=528,
                    programs=programs,
                )
                assert result["validation"]["valid"], result["validation"]
                a = parse_prefixed_json(result["stderr"], "qpu_ffn_ablation_json:")
                assert len(a) == 70
                assert all(x["mode"] == mode and x["exact_restoration"] for x in a)
                if mode == "materialized":
                    assert all(x["materialized_bytes"] > 0 for x in a)
                save()
        record["completed"] = True
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
        record["environment_after"] = collect_environment()
        record["finished_utc"] = utc_now()
        save()
    if not record.get("completed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
