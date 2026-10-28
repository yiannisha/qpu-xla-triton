"""Replay the retained production MTP node through CPU_REPACK and its inline hook.

Use run_rebuttal_diagnostics.py to supply host controls and kernel monitoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_llama_cpp_qpu_ffn_island_eval import parse_prefixed_json  # noqa: E402


def main():
    """Execute the requested experiment and retain its evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--plugin", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--activation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--samples", type=int, default=31)
    args = p.parse_args()
    if args.pairs < 1 or args.samples < 1:
        p.error("pairs and samples must be positive")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "shape": {"m": 1, "k": 256, "n": 2048},
        "cpu_threads": 1,
        "warmups": 5,
        "samples_per_process": args.samples,
        "pairs": args.pairs,
        "runs": [],
        "boundary": (
            "persistent real Q4_0 weights, F32 activation through CPU_REPACK graph; "
            "candidate hook consumes CPU-prepared Q8_0"
        ),
        "unresolved": (
            "device unpacking, memory access and arithmetic are not separated; "
            "wait includes device execution and observation"
        ),
        "files": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (
                ("binary", args.binary),
                ("plugin", args.plugin),
                ("weights", args.weights),
                ("activation", args.activation),
            )
        },
    }
    rng = random.Random(20261028)
    for pair in range(args.pairs):
        order = ["cpu", "qpu"]
        rng.shuffle(order)
        values = {}
        for mode in order:
            prefix = out / f"pair-{pair}-{mode}"
            command = [
                str(args.binary.resolve()),
                "--weights",
                str(args.weights.resolve()),
                "--activation-f32",
                str(args.activation.resolve()),
                "--output-bin",
                str(prefix) + ".f32",
                "--weight-type",
                "q4_0",
                "--input-columns",
                "256",
                "--output-columns",
                "2048",
                "--rows",
                "1",
                "--cpu-threads",
                "1",
                "--warmups",
                "5",
                "--samples",
                str(args.samples),
            ]
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("GGML_QPU_") and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")
            }
            env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            if mode == "qpu":
                env.update(
                    {
                        "LD_PRELOAD": str(args.plugin.resolve()),
                        "GGML_QPU_M1_INLINE": "1",
                        "GGML_QPU_M1_TENSORS": "fixture.weight",
                        "GGML_QPU_TELEMETRY": "1",
                    }
                )
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
            Path(str(prefix) + "-stdout.log").write_text(process.stdout)
            Path(str(prefix) + "-stderr.log").write_text(process.stderr)
            if process.returncode:
                raise RuntimeError(process.stderr)
            data = json.loads(process.stdout)
            assert "REPACK" in data["weight_buffer_type"].upper(), data
            telemetry = parse_prefixed_json(process.stderr, "qpu_llama_candidate_json:")
            if mode == "qpu":
                assert len(telemetry) == args.samples + 5, (len(telemetry), args.samples + 5)
                assert all(t["m"] == 1 and t["k"] == 256 and t["n"] == 2048 and not t["fallback"] for t in telemetry)
            else:
                assert not telemetry
            values[mode] = np.fromfile(str(prefix) + ".f32", dtype=np.float32)
            record["runs"].append(
                {
                    "pair": pair,
                    "mode": mode,
                    "order": order,
                    "command": command,
                    "benchmark": data,
                    "telemetry_including_warmup": telemetry,
                }
            )
        np.testing.assert_allclose(values["qpu"], values["cpu"], atol=1e-4, rtol=1e-4)
        record.setdefault("numerical", []).append(
            {"pair": pair, "max_abs_error": float(np.max(np.abs(values["qpu"] - values["cpu"]))), "passed": True}
        )
        (out / "mtp.json").write_text(json.dumps(record, indent=2) + "\n")
    record["completed"] = True
    (out / "mtp.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
