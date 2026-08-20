#!/usr/bin/env python3
"""Run the exhaustive W8A8 matrix as isolated, memory-sized subprocess jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from qpu_xla.workloads import LLAMA_DENSE_V1, YOLO_DETECTION_V1, LlamaWorkload

ROOT = Path(__file__).resolve().parents[1]
DENSE_SCRIPT = ROOT / "examples/benchmark_qpu_xla_w8a8.py"
YOLO_SCRIPT = ROOT / "examples/benchmark_qpu_xla_yolo_w8a8.py"
RUNTIME_SCRIPT = ROOT / "examples/benchmark_qpu_xla_tinyllama_runtime.py"
MATRIX_SCRIPT = ROOT / "scripts/w8a8_evaluation_matrix.py"
PROJECTIONS = ("hidden", "kv", "up", "down", "lm_head")
EPILOGUES = ("cpu", "standalone-qpu", "fused-qpu")
SQUARE_CASES = ("square-64x64x64", "square-512x512x512")


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    dry_run: bool,
    retries: int = 0,
) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    for attempt in range(retries + 1):
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            print(
                f"job failed; retrying ({attempt + 1}/{retries})",
                file=sys.stderr,
                flush=True,
            )


def _run_benchmark(
    command: list[str],
    *,
    output: Path,
    env: dict[str, str],
    dry_run: bool,
    resume: bool,
    retries: int,
) -> None:
    candidate_output = output.with_suffix(".candidates.json")
    if resume and output.is_file() and candidate_output.is_file():
        print(f"resume: keeping {output.name}", flush=True)
        return
    _run(command, env=env, dry_run=dry_run, retries=retries)


def _runtime_arena_mib(case: LlamaWorkload) -> int:
    """Conservatively size all persistent one-layer projection plans and activations."""
    rows = (case.tokens + 15) // 16 * 16
    kv = case.kv_heads * case.head_dim
    shapes = (
        *((case.hidden_size, case.hidden_size),) * 2,
        *((case.hidden_size, kv),) * 2,
        *((case.hidden_size, case.intermediate_size),) * 2,
        (case.intermediate_size, case.hidden_size),
        (case.hidden_size, case.vocabulary_size),
    )
    total = 0
    for inputs, outputs in shapes:
        padded_inputs = (inputs + 15) // 16 * 16
        padded_outputs = (outputs + 15) // 16 * 16
        total += (
            padded_inputs * padded_outputs
            + rows * padded_inputs
            + rows * padded_outputs * 4
            + (rows + padded_outputs) * 4
        )
    activation_elements = rows * (case.hidden_size * 6 + case.intermediate_size * 3 + kv * 2 + case.vocabulary_size)
    total += activation_elements * 4
    return max(128, (total * 5 // 4 + (1 << 20) - 1) // (1 << 20))


def _supported_cases(registry_path: Path) -> tuple[str, ...]:
    payload: dict[str, Any] = json.loads(registry_path.read_text(encoding="utf-8"))
    names = {
        case.name
        for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)
        for record in payload["records"]
        if record.get("status") == "supported-win" and case.name in str(record.get("name"))
    }
    return tuple(sorted(names))


def main() -> None:
    """Run isolated stage jobs, render matrices, and execute model regressions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--family", choices=("all", "dense", "yolo"), default="all")
    parser.add_argument("--skip-runtime-regression", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.retries < 0:
        parser.error("warmup/retries must be non-negative and repeat must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["OMP_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = str(ROOT / "src")

    if args.family in {"all", "dense"}:
        for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout):
            assert isinstance(case, LlamaWorkload)
            for projection in PROJECTIONS:
                for epilogue in EPILOGUES:
                    output = root / f"dense-{case.name}-{projection}-{epilogue}.json"
                    _run_benchmark(
                        [
                            sys.executable,
                            str(DENSE_SCRIPT),
                            "--case",
                            case.name,
                            "--projection",
                            projection,
                            "--epilogue",
                            epilogue,
                            "--warmup",
                            str(args.warmup),
                            "--repeat",
                            str(args.repeat),
                            "--seed",
                            str(args.seed),
                            "--output",
                            str(output),
                        ],
                        output=output,
                        env=env,
                        dry_run=args.dry_run,
                        resume=args.resume,
                        retries=args.retries,
                    )
        for case_name in SQUARE_CASES:
            for epilogue in EPILOGUES:
                output = root / f"dense-{case_name}-{epilogue}.json"
                _run_benchmark(
                    [
                        sys.executable,
                        str(DENSE_SCRIPT),
                        "--case",
                        case_name,
                        "--epilogue",
                        epilogue,
                        "--warmup",
                        str(args.warmup),
                        "--repeat",
                        str(args.repeat),
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(output),
                    ],
                    output=output,
                    env=env,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    retries=args.retries,
                )

    if args.family in {"all", "yolo"}:
        for case in (*YOLO_DETECTION_V1.tuning, *YOLO_DETECTION_V1.holdout):
            output = root / f"yolo-{case.name}.json"
            _run_benchmark(
                [
                    sys.executable,
                    str(YOLO_SCRIPT),
                    "--case",
                    case.name,
                    "--warmup",
                    str(args.warmup),
                    "--repeat",
                    str(args.repeat),
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(output),
                ],
                output=output,
                env=env,
                dry_run=args.dry_run,
                resume=args.resume,
                retries=args.retries,
            )

    matrix_command = [sys.executable, str(MATRIX_SCRIPT), "--logs", str(root)]
    if args.family == "all":
        matrix_command.append("--require-complete")
    _run(
        matrix_command,
        env=env,
        dry_run=args.dry_run,
    )

    registry = root / "w8a8-calibrated.candidates.json"
    if args.family in {"all", "dense"} and not args.skip_runtime_regression and not args.dry_run:
        for case_name in _supported_cases(registry):
            case = next(case for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout) if case.name == case_name)
            assert isinstance(case, LlamaWorkload)
            runtime_output = root / f"runtime-{case.name}-l1.json"
            if args.resume and runtime_output.is_file():
                print(f"resume: keeping {runtime_output.name}", flush=True)
            else:
                _run(
                    [
                        sys.executable,
                        str(RUNTIME_SCRIPT),
                        "--case",
                        case.name,
                        "--layers",
                        "1",
                        "--warmup",
                        str(args.warmup),
                        "--repeat",
                        str(args.repeat),
                        "--data-area-mib",
                        str(_runtime_arena_mib(case)),
                        "--candidates",
                        str(registry),
                        "--output",
                        str(runtime_output),
                    ],
                    env=env,
                    dry_run=False,
                    retries=args.retries,
                )
        _run(
            matrix_command,
            env=env,
            dry_run=False,
        )


if __name__ == "__main__":
    main()
