"""Command-line entry point for full-model CPU-versus-QPU quality reports."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from qpu_xla import Device
from qpu_xla.models.vision import (
    ResNet18Artifact,
    ResNet18Runtime,
    VisionMode,
    YoloV8Artifact,
    YoloV8Runtime,
    torchvision_preprocess,
)
from qpu_xla.models.vision.yolov8 import result_to_coco
from qpu_xla.quality.coco import (
    coco_bbox_metrics,
    coco_per_category_ap,
    detection_agreement,
    paired_coco_bootstrap,
    save_coco_predictions,
)
from qpu_xla.quality.manifest import SampleManifest, imagenet_stratified_manifest, sha256_file
from qpu_xla.quality.metrics import classification_metrics
from qpu_xla.quality.mmlu import compare_mmlu, load_mmlu_jsonl, run_llama_mmlu
from qpu_xla.quality.report import QualityReport, RunIntegrity
from qpu_xla.quality.smolvla import smolvla_replay_metrics, validate_smolvla_replay


def _provenance(**values: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        **values,
    }


def _write_markdown(report: QualityReport, path: Path) -> None:
    """Write a compact human-readable mirror without inventing a verdict."""
    lines = [
        f"# {report.task}: {report.candidate} vs {report.baseline}",
        "",
        f"Run integrity: `{'valid' if report.integrity.valid_run else 'invalid'}`",
        "",
        "```json",
        json.dumps(report.metrics, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_report(report: QualityReport, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report.save(output / "report.json")
    _write_markdown(report, output / "README.md")


def _open_device(stack: ExitStack, mode: VisionMode, data_area_size: int) -> Device | None:
    return stack.enter_context(Device.open(data_area_size=data_area_size)) if mode.uses_qpu else None


def _convert_resnet18(args: argparse.Namespace) -> None:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("ResNet-18 conversion requires the optional torchvision dependency") from exc
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    ResNet18Artifact.create(args.output, model)


def _all(args: argparse.Namespace) -> None:
    """Run a declarative list of independent qualification commands."""
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError("all-run config must contain a non-empty 'runs' list")
    allowed = {"mmlu", "resnet18", "yolov8n", "smolvla"}
    for run in runs:
        if not isinstance(run, dict) or run.get("command") not in allowed or not isinstance(run.get("args"), list):
            raise ValueError("each all-run entry requires a supported command and string args list")
        if not all(isinstance(value, str) for value in run["args"]):
            raise ValueError("all-run command arguments must be strings")
        subprocess.run(
            [sys.executable, "-m", "qpu_xla.quality.cli", str(run["command"]), *run["args"]],
            check=True,
        )


def _mmlu(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    clean_environment = os.environ.copy()
    island_keys = (
        "LD_PRELOAD",
        "GGML_QPU_FFN_ISLAND",
        "GGML_QPU_FFN_ISLAND_FRACTION",
        "GGML_QPU_FFN_ISLAND_MIN_ROWS",
        "GGML_QPU_FFN_ISLAND_MAX_ROWS",
        "GGML_QPU_FFN_ISLAND_WGS",
        "GGML_QPU_TELEMETRY",
    )
    for key in island_keys:
        clean_environment.pop(key, None)
    baseline_path = output / "cpu.jsonl"
    candidate_path = output / "qpu_hybrid.jsonl"
    reuse_baseline = False
    if args.resume and baseline_path.exists():
        try:
            reuse_baseline = len(load_mmlu_jsonl(baseline_path)) == args.tasks
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            reuse_baseline = False
    if not reuse_baseline:
        baseline_process = run_llama_mmlu(
            args.binary,
            args.model,
            args.dataset,
            baseline_path,
            tasks=args.tasks,
            threads=args.threads,
            environment=clean_environment,
        )
        (output / "cpu.stderr.log").write_text(baseline_process.stderr, encoding="utf-8")
    candidate_environment = clean_environment | {
        "LD_PRELOAD": str(args.plugin.resolve()),
        "GGML_QPU_FFN_ISLAND": "1",
        "GGML_QPU_FFN_ISLAND_FRACTION": str(args.fraction),
        "GGML_QPU_FFN_ISLAND_MIN_ROWS": str(args.minimum_rows),
        "GGML_QPU_FFN_ISLAND_MAX_ROWS": str(args.maximum_rows),
        "GGML_QPU_FFN_ISLAND_WGS": str(args.wgs),
        "GGML_QPU_TELEMETRY": "1",
    }
    candidate_process = run_llama_mmlu(
        args.binary,
        args.model,
        args.dataset,
        candidate_path,
        tasks=args.tasks,
        threads=args.threads,
        environment=candidate_environment,
    )
    (output / "qpu_hybrid.stderr.log").write_text(candidate_process.stderr, encoding="utf-8")
    events = [line for line in candidate_process.stderr.splitlines() if line.startswith("qpu_llama_candidate_json:")]
    report = QualityReport(
        task="gemma-mmlu",
        baseline="llama.cpp_cpu_repack",
        candidate="llama.cpp_qpu_cpu_ffn_island",
        metrics=compare_mmlu(load_mmlu_jsonl(baseline_path), load_mmlu_jsonl(candidate_path)),
        integrity=RunIntegrity(
            args.tasks,
            args.tasks,
            True,
            qpu_dispatches=len(events),
            cpu_operations=len(events),
            requires_qpu=True,
            requires_cpu=True,
            errors=() if events else ("candidate emitted no QPU FFN-island telemetry",),
        ),
        provenance=_provenance(
            binary_sha256=sha256_file(args.binary),
            model_sha256=sha256_file(args.model),
            dataset_sha256=sha256_file(args.dataset),
            plugin_sha256=sha256_file(args.plugin),
            selection_seed=1,
            tasks=args.tasks,
            threads=args.threads,
        ),
        artifacts={"baseline": baseline_path.name, "candidate": candidate_path.name},
    )
    _save_report(report, output)


def _imagenet(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = ResNet18Artifact.open(args.artifact)
    manifest_path = args.manifest or args.output / "samples.json"
    if manifest_path.exists():
        manifest = SampleManifest.load(manifest_path)
    else:
        manifest = imagenet_stratified_manifest(args.dataset, hash_contents=args.hash_samples)
        manifest.save(manifest_path)
    mode = VisionMode(args.mode)
    labels = np.asarray([record.label for record in manifest.samples], dtype=np.int64)
    groups = [record.group or str(record.label) for record in manifest.samples]
    sample_count = len(manifest.samples)
    if sample_count != 5_000:
        raise ValueError("ImageNet qualification requires exactly 5000 manifest samples")
    cached_baseline: npt.NDArray[np.float32] | None = None
    if args.baseline_logits is not None:
        with np.load(args.baseline_logits, allow_pickle=False) as archive:
            cached_baseline = np.ascontiguousarray(archive["baseline"], dtype=np.float32)
        if cached_baseline.shape != (sample_count, 1_000) or not np.all(np.isfinite(cached_baseline)):
            raise ValueError("--baseline-logits does not contain 5000 finite 1000-class rows")
    baseline_model = None if cached_baseline is not None else artifact.torch_model()

    def baseline(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert baseline_model is not None
        with torch.inference_mode():
            return cast(npt.NDArray[np.float32], baseline_model(torch.from_numpy(values)).detach().numpy())

    before_path = args.output / "baseline.partial.npy"
    after_path = args.output / "candidate.partial.npy"
    progress_path = args.output / "progress.json"
    previous_telemetry: dict[str, int] = {}
    processed = 0
    if args.resume and before_path.exists() and after_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        processed = int(progress["processed_samples"])
        previous_telemetry = {str(name): int(value) for name, value in progress["telemetry"].items()}
        before = np.lib.format.open_memmap(before_path, mode="r+")
        after = np.lib.format.open_memmap(after_path, mode="r+")
        if before.shape != (sample_count, 1_000) or after.shape != before.shape or not 0 <= processed <= sample_count:
            raise ValueError("ImageNet resume checkpoint has incompatible shapes or progress")
    else:
        before = np.lib.format.open_memmap(before_path, mode="w+", dtype=np.float32, shape=(sample_count, 1_000))
        after = np.lib.format.open_memmap(after_path, mode="w+", dtype=np.float32, shape=(sample_count, 1_000))

    with ExitStack() as stack:
        device = _open_device(stack, mode, args.data_area_size)
        runtime = stack.enter_context(ResNet18Runtime.from_artifact(artifact, mode=mode, device=device))
        for index in range(processed, sample_count):
            record = manifest.samples[index]
            values = torchvision_preprocess(args.dataset / record.relative_path)
            before[index] = (
                cached_baseline[index]
                if cached_baseline is not None
                else np.asarray(baseline(values), dtype=np.float32).reshape(-1)
            )
            after[index] = np.asarray(runtime.predict_logits(values), dtype=np.float32).reshape(-1)
            completed = index + 1
            if completed % 10 == 0 or completed == sample_count:
                before.flush()
                after.flush()
                current = runtime.context.telemetry.to_dict()
                cumulative = {
                    name: (
                        value
                        if name == "eligible_layers"
                        else previous_telemetry.get(name, 0) + value
                    )
                    for name, value in current.items()
                }
                temporary = progress_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps({"processed_samples": completed, "telemetry": cumulative}, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(progress_path)
                if completed % 100 == 0 or completed == sample_count:
                    print(f"ImageNet {mode.value}: {completed}/{sample_count}", flush=True)
        current = runtime.context.telemetry.to_dict()
        telemetry = {
            name: value if name == "eligible_layers" else previous_telemetry.get(name, 0) + value
            for name, value in current.items()
        }
    before_values = np.ascontiguousarray(before)
    after_values = np.ascontiguousarray(after)
    metrics = classification_metrics(before_values, after_values, labels, groups=groups)
    raw_path = args.output / "logits.npz"
    np.savez_compressed(raw_path, baseline=before_values, candidate=after_values)
    report = QualityReport(
        task="resnet18-imagenet",
        baseline="torchvision_fp32",
        candidate=mode.value,
        metrics=metrics,
        integrity=RunIntegrity(
            5_000,
            before_values.shape[0],
            bool(np.all(np.isfinite(after_values))),
            qpu_dispatches=telemetry["qpu_dispatches"],
            cpu_operations=telemetry["cpu_calls"] + telemetry["hybrid_calls"],
            unexpected_fallbacks=telemetry["unexpected_fallbacks"],
            requires_qpu=mode.uses_qpu,
            requires_cpu=mode.hybrid,
        ),
        provenance=_provenance(
            artifact_state_sha256=artifact.metadata["state_sha256"],
            manifest_sha256=sha256_file(manifest_path),
            mode=mode.value,
            telemetry=telemetry,
        ),
        artifacts={"logits": raw_path.name, "samples": os.path.relpath(manifest_path, args.output)},
    )
    _save_report(report, args.output)


def _predict_ultralytics(model: Any, image: Path, image_id: int) -> list[dict[str, Any]]:
    with torch.inference_mode():
        result = model.predict(
            source=str(image),
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            agnostic_nms=False,
            rect=False,
            verbose=False,
            device="cpu",
        )
    return result_to_coco(result[0], image_id=image_id)


def _load_coco_batches(
    path: Path, expected_images: list[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    """Load a durable one-record-per-image prediction checkpoint."""
    predictions: list[dict[str, Any]] = []
    completed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if completed >= len(expected_images) or int(item["image_id"]) != int(expected_images[completed]["id"]):
                raise ValueError("COCO prediction checkpoint is not an ordered image prefix")
            values = item["predictions"]
            if not isinstance(values, list) or any(
                int(value["image_id"]) != int(item["image_id"]) for value in values
            ):
                raise ValueError("COCO prediction checkpoint contains an invalid image batch")
            predictions.extend(values)
            completed += 1
    return completed, predictions


def _append_coco_batch(stream: Any, image_id: int, predictions: list[dict[str, Any]]) -> None:
    stream.write(json.dumps({"image_id": image_id, "predictions": predictions}, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _coco(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("COCO qualification requires the optional ultralytics dependency") from exc
    annotation_payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = sorted(annotation_payload["images"], key=lambda item: int(item["id"]))
    if len(images) != 5_000:
        raise ValueError(f"COCO val2017 qualification requires 5000 images, found {len(images)}")
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = YoloV8Artifact.open(args.checkpoint, expected_sha256=args.checkpoint_sha256)
    mode = VisionMode(args.mode)
    baseline_predictions: list[dict[str, Any]]
    baseline_completed: int
    if args.baseline_predictions is not None:
        baseline_predictions = json.loads(args.baseline_predictions.read_text(encoding="utf-8"))
        baseline_completed = len(images)
    else:
        baseline_predictions, baseline_completed = [], 0
    cpu_partial = args.output / "cpu_predictions.partial.jsonl"
    candidate_partial = args.output / "candidate_predictions.partial.jsonl"
    progress_path = args.output / "progress.json"
    previous_telemetry: dict[str, int] = {}
    if args.resume:
        if args.baseline_predictions is None and cpu_partial.exists():
            baseline_completed, baseline_predictions = _load_coco_batches(cpu_partial, images)
        if candidate_partial.exists():
            candidate_completed, candidate_predictions = _load_coco_batches(candidate_partial, images)
        else:
            candidate_completed, candidate_predictions = 0, []
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if int(progress["processed_samples"]) != candidate_completed:
                raise ValueError("COCO telemetry progress does not match the prediction checkpoint")
            previous_telemetry = {str(name): int(value) for name, value in progress["telemetry"].items()}
    else:
        candidate_completed, candidate_predictions = 0, []
        cpu_partial.write_text("", encoding="utf-8")
        candidate_partial.write_text("", encoding="utf-8")
    baseline_model = None if baseline_completed == len(images) else YOLO(str(artifact.checkpoint))
    cpu_mode = "a" if baseline_completed else "w"
    candidate_mode = "a" if candidate_completed else "w"
    with ExitStack() as stack:
        device = _open_device(stack, mode, args.data_area_size)
        candidate = stack.enter_context(YoloV8Runtime(artifact, mode=mode, device=device))
        with cpu_partial.open(cpu_mode, encoding="utf-8") as cpu_stream, candidate_partial.open(
            candidate_mode, encoding="utf-8"
        ) as candidate_stream:
            for index, item in enumerate(images):
                image_id = int(item["id"])
                path = args.images / str(item["file_name"])
                if index >= baseline_completed:
                    assert baseline_model is not None
                    values = _predict_ultralytics(baseline_model, path, image_id)
                    baseline_predictions.extend(values)
                    _append_coco_batch(cpu_stream, image_id, values)
                if index >= candidate_completed:
                    values = candidate.predict(path, image_id=image_id)
                    candidate_predictions.extend(values)
                    _append_coco_batch(candidate_stream, image_id, values)
                    completed = index + 1
                    current = candidate.context.telemetry.to_dict()
                    cumulative = {
                        name: value if name == "eligible_layers" else previous_telemetry.get(name, 0) + value
                        for name, value in current.items()
                    }
                    temporary = progress_path.with_suffix(".json.tmp")
                    temporary.write_text(
                        json.dumps({"processed_samples": completed, "telemetry": cumulative}, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(progress_path)
                    if completed % 100 == 0 or completed == len(images):
                        print(f"COCO {mode.value}: {completed}/{len(images)}", flush=True)
        current = candidate.context.telemetry.to_dict()
        telemetry = {
            name: value if name == "eligible_layers" else previous_telemetry.get(name, 0) + value
            for name, value in current.items()
        }
    before_path = args.output / "cpu_predictions.json"
    after_path = args.output / "candidate_predictions.json"
    save_coco_predictions(before_path, baseline_predictions)
    save_coco_predictions(after_path, candidate_predictions)
    baseline_metrics = coco_bbox_metrics(args.annotations, baseline_predictions)
    candidate_metrics = coco_bbox_metrics(args.annotations, candidate_predictions)
    baseline_categories = coco_per_category_ap(args.annotations, baseline_predictions)
    candidate_categories = coco_per_category_ap(args.annotations, candidate_predictions)
    metrics: dict[str, Any] = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {name: candidate_metrics[name] - baseline_metrics[name] for name in baseline_metrics},
        "per_category_ap_delta": {
            name: candidate_categories[name] - baseline_categories[name] for name in baseline_categories
        },
        "prediction_agreement": detection_agreement(baseline_predictions, candidate_predictions),
    }
    if args.bootstrap_replicates:
        metrics["paired_bootstrap"] = paired_coco_bootstrap(
            args.annotations,
            baseline_predictions,
            candidate_predictions,
            replicates=args.bootstrap_replicates,
        )
    report = QualityReport(
        task="yolov8n-coco-val2017",
        baseline="ultralytics_fp32",
        candidate=mode.value,
        metrics=metrics,
        integrity=RunIntegrity(
            5_000,
            len(images),
            True,
            qpu_dispatches=telemetry["qpu_dispatches"],
            cpu_operations=telemetry["cpu_calls"] + telemetry["hybrid_calls"],
            unexpected_fallbacks=telemetry["unexpected_fallbacks"],
            requires_qpu=mode.uses_qpu,
            requires_cpu=mode.hybrid,
        ),
        provenance=_provenance(
            checkpoint_sha256=artifact.checkpoint_sha256,
            source_revision=artifact.source_revision,
            annotations_sha256=sha256_file(args.annotations),
            mode=mode.value,
            telemetry=telemetry,
            validation={"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300},
        ),
        artifacts={"baseline_predictions": before_path.name, "candidate_predictions": after_path.name},
    )
    _save_report(report, args.output)


def _smolvla(args: argparse.Namespace) -> None:
    from qpu_xla.models.smolvla import (
        SmolVLAArtifact,
        SmolVLAPlacementPolicy,
        SmolVLAReferenceRuntime,
        SmolVLAReplay,
        SmolVLARuntime,
    )

    fp32_artifact = SmolVLAArtifact.open(args.fp32_artifact)
    replay = SmolVLAReplay.load(args.replay, fp32_artifact.checkpoint.config)
    validate_smolvla_replay(replay)
    if args.mode.endswith("w8a8") and args.w8a8_artifact is None:
        raise ValueError("--w8a8-artifact is required for a W8A8 SmolVLA mode")
    selected = fp32_artifact if args.mode.endswith("fp32") else SmolVLAArtifact.open(args.w8a8_artifact)
    if selected.checkpoint.config != fp32_artifact.checkpoint.config:
        raise ValueError("SmolVLA FP32 and W8A8 artifacts have different topology")
    runtime: Any
    if args.mode.startswith("native_cpu_"):
        runtime = SmolVLAReferenceRuntime(selected.checkpoint)
    else:
        policy = (
            SmolVLAPlacementPolicy.hybrid() if args.mode.startswith("hybrid_") else SmolVLAPlacementPolicy.forced_qpu()
        )
        runtime = SmolVLARuntime.open(selected, policy=policy, memory_limit_bytes=int(args.memory_limit_gib * 1024**3))
    args.output.mkdir(parents=True, exist_ok=True)
    partial_path = args.output / "candidate.partial.npy"
    progress_path = args.output / "progress.json"
    expected_shape = (
        len(replay.observations),
        selected.checkpoint.config.chunk_size,
        selected.checkpoint.config.action_dim,
    )
    processed = 0
    previous_qpu_dispatches = 0
    previous_cpu_operations = 0
    if args.resume and partial_path.exists() and progress_path.exists():
        after = np.lib.format.open_memmap(partial_path, mode="r+")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        processed = int(progress["processed_observations"])
        previous_qpu_dispatches = int(progress["qpu_dispatches"])
        previous_cpu_operations = int(progress["cpu_operations"])
        if after.shape != expected_shape or not 0 <= processed <= len(replay.observations):
            raise ValueError("SmolVLA resume checkpoint has incompatible shape or progress")
    else:
        after = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=expected_shape)

    try:
        for index in range(processed, len(replay.observations)):
            value = runtime.predict_action_chunk(replay.observations[index])
            result = value.actions if hasattr(value, "actions") else value
            after[index] = np.ascontiguousarray(result, dtype=np.float32)
            after.flush()
            if isinstance(runtime, SmolVLARuntime):
                qpu_counts = runtime.queue.consume_completed_event_counts()
                cpu_counts = runtime.cpu_queue.consume_completed_event_counts()
                previous_qpu_dispatches += qpu_counts.get("qpu", 0)
                previous_cpu_operations += cpu_counts.get("host", 0)
                qpu_dispatches = previous_qpu_dispatches
                cpu_operations = previous_cpu_operations
            else:
                qpu_dispatches = 0
                cpu_operations = index + 1
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "processed_observations": index + 1,
                        "qpu_dispatches": qpu_dispatches,
                        "cpu_operations": cpu_operations,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(progress_path)
            print(f"SmolVLA {args.mode}: {index + 1}/{len(replay.observations)}", flush=True)
        if isinstance(runtime, SmolVLARuntime):
            qpu_counts = runtime.queue.consume_completed_event_counts()
            cpu_counts = runtime.cpu_queue.consume_completed_event_counts()
            qpu_dispatches = previous_qpu_dispatches + qpu_counts.get("qpu", 0)
            cpu_operations = previous_cpu_operations + cpu_counts.get("host", 0)
        else:
            qpu_dispatches, cpu_operations = 0, len(replay.observations)
    finally:
        if isinstance(runtime, SmolVLARuntime):
            runtime.close()
    before = np.ascontiguousarray(replay.upstream_actions)
    after_values = np.ascontiguousarray(after)
    metrics = smolvla_replay_metrics(replay, after_values)
    actions_path = args.output / "actions.npz"
    np.savez_compressed(actions_path, baseline=before, candidate=after_values)
    qpu_mode = args.mode.startswith(("qpu_", "hybrid_"))
    hybrid = args.mode.startswith("hybrid_")
    report = QualityReport(
        task="smolvla-replay",
        baseline="upstream_torch_cpu_fp32",
        candidate=args.mode,
        metrics=metrics,
        integrity=RunIntegrity(
            100,
            len(replay.observations),
            bool(np.all(np.isfinite(after_values))),
            qpu_dispatches=qpu_dispatches,
            cpu_operations=cpu_operations,
            requires_qpu=qpu_mode,
            requires_cpu=hybrid,
        ),
        provenance=_provenance(
            replay_sha256=sha256_file(args.replay),
            model_revision=replay.model_revision,
            lerobot_revision=replay.lerobot_revision,
        ),
        artifacts={"actions": actions_path.name},
    )
    _save_report(report, args.output)


def _add_common_vision(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=tuple(mode.value for mode in VisionMode), required=True)
    parser.add_argument("--data-area-size", type=int, default=2 * 1024**3)
    parser.add_argument("--output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-resnet18")
    convert.add_argument("--output", type=Path, required=True)
    convert.set_defaults(run=_convert_resnet18)
    mmlu = commands.add_parser("mmlu")
    mmlu.add_argument("--binary", type=Path, required=True)
    mmlu.add_argument("--model", type=Path, required=True)
    mmlu.add_argument("--dataset", type=Path, required=True)
    mmlu.add_argument("--plugin", type=Path, required=True)
    mmlu.add_argument("--tasks", type=int, default=1_000)
    mmlu.add_argument("--threads", type=int, default=4)
    mmlu.add_argument("--fraction", type=float, default=0.125)
    mmlu.add_argument("--minimum-rows", type=int, default=64)
    mmlu.add_argument("--maximum-rows", type=int, default=528)
    mmlu.add_argument("--wgs", type=int, default=24)
    mmlu.add_argument("--resume", action="store_true", help="reuse an existing complete CPU JSONL")
    mmlu.add_argument("--output", type=Path, required=True)
    mmlu.set_defaults(run=_mmlu)
    imagenet = commands.add_parser("resnet18")
    imagenet.add_argument("--artifact", type=Path, required=True)
    imagenet.add_argument("--dataset", type=Path, required=True)
    imagenet.add_argument("--manifest", type=Path)
    imagenet.add_argument("--baseline-logits", type=Path, help="reuse the baseline array from a completed logits.npz")
    imagenet.add_argument("--hash-samples", action="store_true")
    imagenet.add_argument("--resume", action="store_true")
    _add_common_vision(imagenet)
    imagenet.set_defaults(run=_imagenet)
    coco = commands.add_parser("yolov8n")
    coco.add_argument("--checkpoint", type=Path, required=True)
    coco.add_argument("--checkpoint-sha256")
    coco.add_argument("--images", type=Path, required=True)
    coco.add_argument("--annotations", type=Path, required=True)
    coco.add_argument("--baseline-predictions", type=Path, help="reuse a completed cpu_predictions.json")
    coco.add_argument("--bootstrap-replicates", type=int, default=10_000)
    coco.add_argument("--resume", action="store_true")
    _add_common_vision(coco)
    coco.set_defaults(run=_coco)
    smol = commands.add_parser("smolvla")
    smol.add_argument("--fp32-artifact", type=Path, required=True)
    smol.add_argument("--w8a8-artifact", type=Path)
    smol.add_argument("--replay", type=Path, required=True)
    smol.add_argument(
        "--mode",
        choices=("native_cpu_fp32", "qpu_fp32", "hybrid_fp32", "native_cpu_w8a8", "qpu_w8a8", "hybrid_w8a8"),
        required=True,
    )
    smol.add_argument("--memory-limit-gib", type=float, default=6.0)
    smol.add_argument("--resume", action="store_true")
    smol.add_argument("--output", type=Path, required=True)
    smol.set_defaults(run=_smolvla)
    all_runs = commands.add_parser("all")
    all_runs.add_argument("--config", type=Path, required=True)
    all_runs.set_defaults(run=_all)
    return parser


def main() -> None:
    """Run one explicitly configured qualification workload."""
    args = _parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
