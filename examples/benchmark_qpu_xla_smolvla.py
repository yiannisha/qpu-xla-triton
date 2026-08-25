from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from qpu_xla.benchmark import CandidateRegistry, collect_metadata
from qpu_xla.models.smolvla import (
    SmolVLAActionChunk,
    SmolVLAActionMetrics,
    SmolVLAArtifact,
    SmolVLAPlacementPolicy,
    SmolVLAReferenceRuntime,
    SmolVLAReplay,
    SmolVLARuntime,
    UpstreamTorchSmolVLAOracle,
)

MODES = (
    "upstream_torch_cpu_fp32",
    "native_cpu_fp32",
    "qpu_fp32",
    "hybrid_fp32",
    "native_cpu_w8a8",
    "qpu_w8a8",
    "hybrid_w8a8",
    "auto_fp32",
    "auto_w8a8",
)


def _trace_summary(trace: dict[str, list[dict[str, object]]], invocations: int) -> dict[str, object]:
    totals: dict[str, float] = {}
    categories: dict[str, float] = {}
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        raw_duration = event.get("dur")
        if not isinstance(raw_duration, int | float):
            continue
        duration = float(raw_duration) / 1_000_000.0
        name, category = str(event.get("name", "unknown")), str(event.get("cat", "unknown"))
        totals[name] = totals.get(name, 0.0) + duration
        categories[category] = categories.get(category, 0.0) + duration
    divisor = max(invocations, 1)
    return {
        "mean_seconds_per_inference_by_event": {name: value / divisor for name, value in sorted(totals.items())},
        "mean_seconds_per_inference_by_category": {
            name: value / divisor for name, value in sorted(categories.items())
        },
    }


def _mode_artifact(args: argparse.Namespace, mode: str) -> Path:
    if mode.endswith("w8a8"):
        if args.w8a8_artifact is None:
            raise ValueError(f"mode {mode} requires --w8a8-artifact")
        return cast(Path, args.w8a8_artifact)
    return cast(Path, args.fp32_artifact)


def _policy(mode: str, candidates: CandidateRegistry) -> SmolVLAPlacementPolicy:
    if mode.startswith("qpu_"):
        return SmolVLAPlacementPolicy.forced_qpu()
    if mode.startswith("hybrid_"):
        return SmolVLAPlacementPolicy.hybrid()
    if mode.startswith("auto_"):
        return SmolVLAPlacementPolicy.auto(candidates)
    return SmolVLAPlacementPolicy.cpu()


def _actions(result: SmolVLAActionChunk | npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return result.actions if isinstance(result, SmolVLAActionChunk) else result


def _run_worker(args: argparse.Namespace) -> None:
    mode = args.mode
    config_artifact = SmolVLAArtifact.open(args.fp32_artifact, verify=not args.skip_checksums)
    replay = SmolVLAReplay.load(args.replay, config_artifact.checkpoint.config)
    candidates = CandidateRegistry.load(args.candidates) if args.candidates is not None else CandidateRegistry()
    runtime: Any
    artifact: SmolVLAArtifact | None = None
    started = perf_counter()
    if mode == "upstream_torch_cpu_fp32":
        if args.lerobot_checkout is None:
            raise ValueError("upstream_torch_cpu_fp32 requires --lerobot-checkout")
        runtime = UpstreamTorchSmolVLAOracle.open(
            config_artifact.checkpoint.config,
            lerobot_checkout=args.lerobot_checkout,
            checkpoint=args.upstream_checkpoint,
            cache_directory=args.upstream_cache,
            cpu_threads=args.cpu_threads,
        )
    else:
        artifact = SmolVLAArtifact.open(_mode_artifact(args, mode), verify=not args.skip_checksums)
        if artifact.checkpoint.config != config_artifact.checkpoint.config:
            raise ValueError("FP32 and W8A8 artifacts do not describe the same model topology")
        if mode.startswith("native_cpu_"):
            runtime = SmolVLAReferenceRuntime(artifact.checkpoint)
        else:
            runtime = SmolVLARuntime.open(
                artifact,
                policy=_policy(mode, candidates),
                memory_limit_bytes=int(args.memory_limit_gib * 1024**3),
            )
    cold_result = runtime.predict_action_chunk(replay.observations[0])
    cold_seconds = perf_counter() - started

    for index in range(args.warmup):
        runtime.predict_action_chunk(replay.observations[index % len(replay.observations)])
    durations: list[float] = []
    outputs: list[npt.NDArray[np.float32]] = []
    stage_seconds: dict[str, list[float]] = {}
    for _ in range(args.repeat):
        outputs.clear()
        for observation in replay.observations:
            started = perf_counter()
            result = runtime.predict_action_chunk(observation)
            durations.append(perf_counter() - started)
            outputs.append(np.array(_actions(result), dtype=np.float32, copy=True, order="C"))
            if isinstance(result, SmolVLAActionChunk):
                for stage, duration_ns in result.stage_ns.items():
                    stage_seconds.setdefault(stage, []).append(duration_ns / 1e9)
    device = runtime.device if isinstance(runtime, SmolVLARuntime) else None
    metadata = collect_metadata(
        device,
        extra={
            "mode": mode,
            "semantics": "cold-start plus persistent steady-state full action chunk",
            "upstream_baseline": UpstreamTorchSmolVLAOracle.label,
            "model_revision": replay.model_revision,
            "lerobot_revision": replay.lerobot_revision,
        },
    )
    trace = (
        _trace_summary(runtime.queue.chrome_trace(), args.warmup + args.repeat * len(replay.observations) + 1)
        if isinstance(runtime, SmolVLARuntime)
        else {}
    )
    memory = (
        {
            "artifact_bytes": runtime.memory_plan.artifact_bytes,
            "qpu_arena_bytes": runtime.memory_plan.qpu_arena_bytes,
            "activation_bytes": runtime.memory_plan.activation_bytes,
            "projected_peak_rss_bytes": runtime.memory_plan.projected_peak_rss_bytes,
        }
        if isinstance(runtime, SmolVLARuntime)
        else {}
    )
    if isinstance(runtime, SmolVLARuntime):
        runtime.close()
    payload = {
        "mode": mode,
        "metadata": metadata,
        "sessions": len(replay.observations),
        "cold_start_seconds": [cold_seconds],
        "steady_state_seconds": durations,
        "steady_state_median_seconds": median(durations),
        "stage_seconds": stage_seconds,
        "actions": np.stack(outputs).tolist(),
        "cold_action": _actions(cold_result).tolist(),
        "trace": trace,
        "memory": memory,
    }
    args.worker_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_modes(requested: tuple[str, ...], has_recorded_upstream: bool) -> tuple[str, ...]:
    modes = list(requested)
    if any(mode.endswith("fp32") and mode != "upstream_torch_cpu_fp32" for mode in modes):
        modes.append("native_cpu_fp32")
    if any(mode.endswith("w8a8") for mode in modes):
        modes.extend(("native_cpu_w8a8", "native_cpu_fp32"))
    if not has_recorded_upstream and "upstream_torch_cpu_fp32" in requested:
        modes.append("upstream_torch_cpu_fp32")
    return tuple(dict.fromkeys(modes))


def _worker_command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--fp32-artifact",
        str(args.fp32_artifact),
        "--replay",
        str(args.replay),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--memory-limit-gib",
        str(args.memory_limit_gib),
        "--cpu-threads",
        str(args.cpu_threads),
        "--upstream-checkpoint",
        str(args.upstream_checkpoint),
        "--worker-output",
        str(output),
    ]
    for option, value in (
        ("--w8a8-artifact", args.w8a8_artifact),
        ("--candidates", args.candidates),
        ("--lerobot-checkout", args.lerobot_checkout),
        ("--upstream-cache", args.upstream_cache),
    ):
        if value is not None:
            command.extend((option, str(value)))
    if args.skip_checksums:
        command.append("--skip-checksums")
    return command


def _metrics(actual: Any, expected: Any) -> SmolVLAActionMetrics:
    return SmolVLAActionMetrics.calculate(
        np.ascontiguousarray(actual, dtype=np.float32),
        np.ascontiguousarray(expected, dtype=np.float32),
    )


def _add_comparisons(
    reports: dict[str, dict[str, Any]],
    replay: SmolVLAReplay,
    requested: tuple[str, ...],
) -> None:
    upstream = (
        replay.upstream_actions
        if replay.upstream_actions is not None
        else np.asarray(reports["upstream_torch_cpu_fp32"]["actions"], dtype=np.float32)
        if "upstream_torch_cpu_fp32" in reports
        else None
    )
    fp32 = (
        np.asarray(reports["native_cpu_fp32"]["actions"], dtype=np.float32) if "native_cpu_fp32" in reports else None
    )
    w8a8 = (
        np.asarray(reports["native_cpu_w8a8"]["actions"], dtype=np.float32) if "native_cpu_w8a8" in reports else None
    )
    for mode, report in reports.items():
        if mode not in requested:
            report["supporting_oracle_only"] = True
        actual = np.asarray(report["actions"], dtype=np.float32)
        comparisons: dict[str, object] = {}
        if upstream is not None and mode != "upstream_torch_cpu_fp32":
            value = _metrics(actual, upstream)
            upstream_comparison = value.to_dict()
            if mode.endswith("w8a8"):
                upstream_comparison["passes_w8a8_quality_gate"] = value.passes_w8a8_quality
            else:
                upstream_comparison["passes_fp32_gate"] = value.passes_fp32_upstream
            comparisons["vs_upstream_torch_cpu_fp32"] = upstream_comparison
            upstream_timing = reports.get("upstream_torch_cpu_fp32", {}).get("steady_state_median_seconds")
            if isinstance(upstream_timing, int | float):
                report["speedup_over_upstream_torch_cpu_fp32"] = float(upstream_timing) / float(
                    report["steady_state_median_seconds"]
                )
        if fp32 is not None and mode.endswith("w8a8"):
            value = _metrics(actual, fp32)
            comparisons["vs_native_cpu_fp32"] = {
                **value.to_dict(),
                "passes_w8a8_quality_gate": value.passes_w8a8_quality,
            }
        if w8a8 is not None and mode in {"qpu_w8a8", "hybrid_w8a8", "auto_w8a8"}:
            value = _metrics(actual, w8a8)
            comparisons["vs_native_cpu_w8a8"] = {
                **value.to_dict(),
                "passes_same_contract_gate": value.passes_w8a8_contract,
            }
        report["correctness"] = comparisons
        accelerated = mode.startswith(("qpu_", "hybrid_", "auto_"))
        full_speedup = report.get("speedup_over_upstream_torch_cpu_fp32", 0.0)
        numerical_pass = all(
            bool(item.get("passes_fp32_gate", True))
            and bool(item.get("passes_w8a8_quality_gate", True))
            and bool(item.get("passes_same_contract_gate", True))
            for item in comparisons.values()
            if isinstance(item, dict)
        )
        report["auto_promotion"] = {
            "full_replay_win": accelerated and float(full_speedup) >= 1.05,
            "numerical_gates_pass": numerical_pass,
            "eligible": False,
            "reason": "AUTO also requires exact supported-win stage and enclosing-block records",
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete native SmolVLA CPU/QPU graph against pinned upstream Torch"
    )
    parser.add_argument("--fp32-artifact", type=Path, required=True)
    parser.add_argument("--w8a8-artifact", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--modes", default=",".join(MODES[:-2]))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--lerobot-checkout", type=Path)
    parser.add_argument("--upstream-checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--upstream-cache", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--memory-limit-gib", type=float, default=6.0)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("smolvla-qpu-benchmark.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.cpu_threads <= 0 or args.memory_limit_gib <= 0:
        parser.error("repeat, cpu-threads, and memory limit must be positive; warmup may be zero")
    if args.worker and (args.mode is None or args.worker_output is None):
        parser.error("internal worker requires --mode and --worker-output")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker:
        _run_worker(args)
        return
    requested = tuple(dict.fromkeys(part.strip() for part in args.modes.split(",") if part.strip()))
    if not requested or any(mode not in MODES for mode in requested):
        raise ValueError(f"--modes must contain only: {', '.join(MODES)}")
    config_artifact = SmolVLAArtifact.open(args.fp32_artifact, verify=not args.skip_checksums)
    replay = SmolVLAReplay.load(args.replay, config_artifact.checkpoint.config)
    modes = _required_modes(requested, replay.upstream_actions is not None)
    if "upstream_torch_cpu_fp32" in modes and args.lerobot_checkout is None:
        raise ValueError("an upstream comparison requires --lerobot-checkout at the pinned revision")
    environment = dict(os.environ)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = str(args.cpu_threads)
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="smolvla-benchmark-") as directory:
        root = Path(directory)
        for mode in modes:
            output = root / f"{mode}.json"
            subprocess.run(_worker_command(args, mode, output), check=True, env=environment)
            reports[mode] = json.loads(output.read_text(encoding="utf-8"))
    _add_comparisons(reports, replay, requested)
    payload = {
        "name": "smolvla-e2e-v1",
        "semantics": {
            "cold_start": "artifact/model load + allocation/assembly + first complete action chunk",
            "steady_state": "persistent runtime, complete RGB-to-action chunk including transfers",
            "upstream_reference": UpstreamTorchSmolVLAOracle.label,
            "auto_gate": ">=1.05x exact stage + block + full replay supported-win evidence",
        },
        "requested_modes": requested,
        "model_shape_class": config_artifact.checkpoint.config.model_shape_class,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for mode in requested:
        report = reports[mode]
        print(
            f"{mode}: cold {report['cold_start_seconds'][0]:.3f}s, "
            f"steady median {report['steady_state_median_seconds']:.3f}s"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
