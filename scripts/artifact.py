#!/usr/bin/env python3
"""Prepare and exercise the local research artifact; never publish it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.artifact_common import contained, read_record, sha256, verify_sums  # noqa: E402

REBUTTAL = ROOT / "experiment_logs/20261028-rebuttal"


def write_json(path: Path, data: object) -> None:
    """Save a readable record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def command_output(command: list[str]) -> str:
    """Collect a bounded read-only system probe, including failures."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def check_environment(hardware: bool = False) -> dict:
    """Record host facts; portable checks never change host state."""
    packages = {}
    for name in ("numpy", "torch", "ioctl-opt", "pytest", "torchvision", "ultralytics", "pycocotools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    nodes = [Path("/dev/dri/renderD128"), Path("/dev/dri/by-path/platform-1002000000.v3d-card")]
    model_path = Path("/proc/device-tree/model")
    facts = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "board": model_path.read_text().strip("\0\n") if model_path.exists() else None,
        "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
        "memory": Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None,
        "storage_free_bytes": shutil.disk_usage(ROOT).free,
        "packages": packages,
        "tools": {
            name: command_output([name, "--version"]).splitlines()[:1] for name in ("uv", "cc", "cmake", "make")
        },
        "device_nodes": {str(p): {"exists": p.exists(), "read_write": os.access(p, os.R_OK | os.W_OK)} for p in nodes},
        "governors": {
            str(p): p.read_text().strip()
            for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")
        },
        "swap": Path("/proc/swaps").read_text() if Path("/proc/swaps").exists() else None,
        "temperature_millicelsius": {
            str(p): p.read_text().strip() for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")
        },
        "firmware": command_output(["vcgencmd", "version"]) if shutil.which("vcgencmd") else None,
        "throttled": command_output(["vcgencmd", "get_throttled"]) if shutil.which("vcgencmd") else None,
    }
    errors = []
    for name in ("numpy", "torch", "ioctl-opt"):
        if packages[name] is None:
            errors.append(f"missing package: {name}; run uv sync --locked --group dev")
    if hardware:
        if "Raspberry Pi 5" not in (facts["board"] or ""):
            errors.append("hardware profile requires a Raspberry Pi 5")
        if not all(os.access(p, os.R_OK | os.W_OK) for p in nodes):
            errors.append("current driver needs render-node and V3D card access; inspect render/video groups")
    return {
        "kind": "artifact-environment-v1",
        "hardware_requested": hardware,
        "facts": facts,
        "errors": errors,
        "ok": not errors,
    }


def verify() -> dict:
    """Verify retained evidence and the explicit release inventory."""
    counts = {"rebuttal_package_files": verify_sums(REBUTTAL, REBUTTAL / "PACKAGE_SHA256SUMS")}
    records = json.loads((REBUTTAL / "provenance.json").read_text())["records"]
    for record in records:
        path = contained(REBUTTAL, record["archive_path"])
        data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256_uncompressed"]:
            raise ValueError(f"rebuttal provenance mismatch: {path}")
    counts["rebuttal_provenance_records"] = len(records)
    evidence = ROOT / "artifact/evidence"
    counts["selected_evidence_files"] = verify_sums(evidence, evidence / "SHA256SUMS")
    inventory = json.loads((ROOT / "artifact/release-files.json").read_text())
    reserved = {"SHA256SUMS", "PKG-INFO", "artifact/RELEASE.json"}
    if len(inventory["files"]) != len(set(inventory["files"])) or reserved.intersection(inventory["files"]):
        raise ValueError("release inventory contains duplicate or reserved generated paths")
    for relative in inventory["files"]:
        if not contained(ROOT, relative).is_file():
            raise ValueError(f"missing release file: {relative}")
    counts["release_files"] = len(inventory["files"])
    claims = json.loads((ROOT / "artifact/claims.json").read_text())
    for claim in claims["claims"]:
        for relative in claim["evidence"]:
            if not contained(ROOT, relative).is_file():
                raise ValueError(f"missing evidence for {claim['id']}: {relative}")
    counts["claim_groups"] = len(claims["claims"])
    if (ROOT / "SHA256SUMS").exists():
        counts["release_checksum_files"] = verify_sums(ROOT, ROOT / "SHA256SUMS")
    return {"kind": "artifact-verification-v1", "ok": True, **counts}


def analyze(output: Path) -> dict:
    """Regenerate retained summaries without QPU hardware or external models."""
    from scripts.summarize_llama_cpp_ffn_breakdown import summarize as breakdown
    from scripts.summarize_llama_cpp_ffn_split_screen import summarize as split_screen
    from scripts.summarize_rebuttal_remaining import summarize as remaining

    output.mkdir(parents=True, exist_ok=True)
    result = {
        "headline": headline_results(),
        "todo1": breakdown(read_record(REBUTTAL, "todo1/breakdown.json")),
        "todo2": split_screen(REBUTTAL),
        "todo3_9": remaining(REBUTTAL),
    }
    write_json(output / "rebuttal.json", result)
    suite = ROOT / "experiment_logs/20260819-qpu-xla-kernel-suite"
    projections = []
    for name in ("decode-h2048-c512", "prefill-h1024-t64", "prefill-h512-t16"):
        projections.extend(json.loads((suite / f"{name}-fp32-linear.json").read_text())["results"])
    projection_counts = {
        "configurations": len(projections),
        "supported_wins": sum(r["status"] == "supported-win" for r in projections),
        "other_cases": sum(r["status"] != "supported-win" for r in projections),
        "max_abs_error": max(r["max_abs_error"] for r in projections),
    }
    write_json(output / "projections.json", {"counts": projection_counts, "full_grid": projections})
    quality = {}
    for path in sorted((ROOT / "artifact/evidence/model-quality").glob("*/report.json")):
        quality[path.parent.name] = json.loads(path.read_text())
    write_json(output / "quality-reports.json", quality)
    numerical = quality_numerical()
    write_json(output / "quality-numerical.json", numerical)
    # These are regenerated report views. The associated raw outputs are retained
    # in evidence, but expensive task metrics/bootstraps are not rerun by this command.
    lines = [
        "# Retained artifact evidence",
        "",
        "Offline analysis; no new hardware measurements or independent validation.",
        "",
        "## Corrected headline campaigns",
        "",
        "| Boundary | CPU ms | Candidate ms | Speedup | 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for name, interval in result["headline"].items():
        lines.append(
            f"| {name} | {interval['baseline_median_ns'] / 1e6:.3f} | "
            f"{interval['candidate_median_ns'] / 1e6:.3f} | {interval['median_speedup']:.5f} | "
            f"[{interval['bootstrap_95_low']:.5f}, {interval['bootstrap_95_high']:.5f}] |"
        )
    lines.extend(
        [
            "",
            "| Projection cases | Supported wins | Other cases | Max absolute error |",
            "|---:|---:|---:|---:|",
            f"| {len(projections)} | {projection_counts['supported_wins']} | {projection_counts['other_cases']} | "
            f"{projection_counts['max_abs_error']:.6g} |",
            "",
            "## Rebuttal split screen",
            "",
            "| M | QPU fraction | Speedup | 95% interval |",
            "|---:|---:|---:|---|",
        ]
    )
    for row in result["todo2"]["grid"]:
        lines.append(
            f"| {row['rows']} | {row['fraction']} | {row['median_speedup']:.5f} | "
            f"[{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] |"
        )
    lines.extend(
        [
            "",
            "No cell is independent confirmation of a newly selected policy. "
            "Keep instrumented diagnostics separate from headline campaigns.",
            "",
            "## Numerical checks recomputed from raw model outputs",
            "",
            "```json",
            json.dumps(numerical, indent=2),
            "```",
            "",
            "## Quality report views",
            "",
            "These views preserve recorded metrics; they do not recompute task metrics from raw outputs.",
            "",
        ]
    )
    for name, report in quality.items():
        lines.extend([f"### {name}", "", "```json", json.dumps(report["metrics"], indent=2), "```", ""])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return {"ok": True, "projection_counts": projection_counts, "quality_reports": len(quality), "output": str(output)}


def headline_results() -> dict:
    """Recompute original medians and intervals with the original resampling designs."""
    from scripts.run_llama_cpp_qpu_ffn_island_agentic_eval import prompt_ns
    from scripts.run_llama_cpp_qpu_ffn_island_eval import paired_speedup_interval, speedup_interval

    directory = ROOT / "experiment_logs/20260910-ffn-island-stride-fix"
    prompt = json.loads((directory / "corrected-independent-processes-v7.json").read_text())
    agentic = json.loads((directory / "corrected-agentic-15-v2.json").read_text())
    if not prompt["retention"]["retained"] or not agentic["retention"]["retained"]:
        raise ValueError("headline records are not retained")
    results = {}
    for entry in prompt["comparisons"]:
        rows, fraction = entry["rows"], entry["fraction"]
        groups = [
            sorted(
                (r for r in prompt["runs"] if r["rows"] == rows and r["fraction"] == share),
                key=lambda r: r["process_index"],
            )
            for share in (None, fraction)
        ]
        expected = entry["speedup"]
        value = speedup_interval(
            *[[r["samples_ns"][0] for r in group] for group in groups],
            seed=expected["seed"],
            resamples=expected["resamples"],
        )
        if value != expected:
            raise ValueError(f"headline prompt summary differs from retained result at M={rows}")
        results[f"prompt-M{rows}"] = value
    for entry in agentic["pair_results"]:
        groups = [
            sorted(
                (
                    r
                    for r in agentic["samples"]
                    if r["suffix"] == entry["suffix_tokens"] and r["placement"] == placement
                ),
                key=lambda r: r["sample_index"],
            )
            for placement in ("cpu", "qpu")
        ]
        for name, field in (("prompt", "prompt_processing_speedup"), ("request-wall", "request_wall_speedup")):
            expected = entry[field]
            value = paired_speedup_interval(
                *[[prompt_ns(r) if name == "prompt" else r["request_wall_ns"] for r in group] for group in groups],
                seed=expected["seed"],
                resamples=expected["resamples"],
            )
            if value != expected:
                raise ValueError(f"agentic {name} summary differs from retained result")
            results[f"cached-512-suffix{entry['suffix_tokens']}-{name}"] = value
    return results


def quality_numerical() -> dict:
    """Recompute inexpensive output metrics; preserve published CI/AP reports separately."""
    import numpy as np

    evidence = ROOT / "artifact/evidence"
    manifest = json.loads((evidence / "manifests/imagenet-samples-5000.json").read_text())
    labels = np.array([s["label"] for s in manifest["samples"]])
    results = {}
    for path in sorted((evidence / "model-quality").glob("*/logits.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            baseline, candidate = arrays["baseline"], arrays["candidate"]
        if baseline.shape != (len(labels), 1000) or candidate.shape != baseline.shape:
            raise ValueError(f"classification output shape mismatch: {path}")
        if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
            raise ValueError(f"nonfinite classifier outputs: {path}")
        cpu1, qpu1 = baseline.argmax(axis=1), candidate.argmax(axis=1)
        results[path.parent.name] = {
            "baseline_top1": float(np.mean(cpu1 == labels)),
            "candidate_top1": float(np.mean(qpu1 == labels)),
            "baseline_top5": float(
                np.mean(np.any(np.argpartition(baseline, -5, axis=1)[:, -5:] == labels[:, None], axis=1))
            ),
            "candidate_top5": float(
                np.mean(np.any(np.argpartition(candidate, -5, axis=1)[:, -5:] == labels[:, None], axis=1))
            ),
            "top1_agreement": float(np.mean(cpu1 == qpu1)),
            "logit_max_abs_error": float(np.max(np.abs(candidate.astype(np.float64) - baseline))),
        }
    mmlu = evidence / "model-quality/gemma-mmlu-1000"
    paired = [
        [json.loads(line) for line in gzip.decompress((mmlu / f"{name}.jsonl.gz").read_bytes()).splitlines()]
        for name in ("cpu", "qpu_hybrid")
    ]
    before, after = paired
    if (
        len(before) != 1000
        or len(after) != 1000
        or any(
            (a["task_id"], a["correct_index"]) != (b["task_id"], b["correct_index"])
            for a, b in zip(before, after, strict=True)
        )
    ):
        raise ValueError("MMLU outputs are not the same 1000 paired questions")
    results["gemma-mmlu-1000"] = {
        "baseline_accuracy": sum(a["correct_index"] == a["predicted_index"] for a in before) / len(before),
        "candidate_accuracy": sum(a["correct_index"] == a["predicted_index"] for a in after) / len(after),
        "answer_agreement": sum(
            a["predicted_index"] == b["predicted_index"] for a, b in zip(before, after, strict=True)
        )
        / len(before),
    }
    with np.load(evidence / "model-quality/smolvla-replay100-qpu-fp32/actions.npz", allow_pickle=False) as arrays:
        baseline, candidate = arrays["baseline"].astype(np.float64), arrays["candidate"].astype(np.float64)
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("nonfinite retained SmolVLA actions")
    difference = candidate - baseline
    results["smolvla-replay100-qpu-fp32"] = {
        "max_abs_error": float(np.max(np.abs(difference))),
        "normalized_rmse": float(np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(baseline**2))),
        "cosine_similarity": float(
            np.sum(baseline * candidate) / (np.linalg.norm(baseline) * np.linalg.norm(candidate))
        ),
    }
    return results


def smoke(output: Path, hardware: bool, native_build: Path | None) -> dict:
    """Run a bounded numerical check, with explicit CPU/fake or QPU placement."""
    import numpy as np

    from qpu_xla import Device
    from qpu_xla.ops import matmul
    from qpu_xla.scheduler import Placement

    environment = check_environment(hardware)
    write_json(output / "environment.json", environment)
    if not environment["ok"]:
        raise ValueError("environment check failed; see environment.json")
    evidence = verify()
    errors = {}
    p, q, r = (32, 8, 32) if hardware else (17, 19, 18)
    with Device.open(data_area_size=4 * 1024**2) if hardware else Device.fake() as device, device.queue() as queue:
        for dtype in (np.float32, np.int32):
            left = device.tensor((p, q), dtype)
            right = device.tensor((q, r), dtype)
            destination = device.tensor((p, r), dtype)
            left.numpy()[:] = (np.arange(p * q).reshape(p, q) % 7 - 3).astype(dtype)
            right.numpy()[:] = (np.arange(q * r).reshape(q, r) % 5 - 2).astype(dtype)
            expected = left.numpy() @ right.numpy()
            matmul(
                destination, left, right, queue=queue, placement=Placement.QPU if hardware else Placement.CPU
            ).wait()
            actual = destination.numpy()
            errors[np.dtype(dtype).name] = float(np.max(np.abs(actual.astype(np.float64) - expected)))
            np.testing.assert_allclose(
                actual, expected, atol=3e-4, rtol=3e-4
            ) if dtype == np.float32 else np.testing.assert_array_equal(actual, expected)
    if hardware:
        from examples.tiledconv2d import numpy_conv2d_nchw, tiledconv2d_int16

        x = (np.arange(60).reshape(1, 2, 6, 5) % 7 - 3).astype(np.int16)
        weight = (np.arange(72).reshape(4, 2, 3, 3) % 5 - 2).astype(np.int16)
        expected = numpy_conv2d_nchw(
            x.astype(np.int32), weight.astype(np.int32), padding=1, compute_dtype=np.int32, out_dtype=np.int32
        )
        actual = tiledconv2d_int16(x, weight, padding=1)
        errors["int16_conv_int32_accumulation"] = float(np.max(np.abs(actual.astype(np.int64) - expected)))
        np.testing.assert_array_equal(actual, expected)
    native = []
    if native_build:
        if not hardware:
            raise ValueError("--native-build requires --hardware")
        for name in ("qpu_repack_stride_smoke", "qpu_ffn_island_graph_smoke"):
            command = [str(native_build.resolve() / name)]
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            native.append(name)
    result = {
        "kind": "artifact-smoke-v1",
        "ok": True,
        "hardware": hardware,
        "matmul_shape": [p, q, r],
        "max_abs_error": errors,
        "native_checks": native,
        "evidence": evidence,
    }
    write_json(output / "smoke.json", result)
    return result


def prepare_mtp(model: Path, output: Path) -> dict:
    """Extract the exact native weight range from a user-acquired hashed GGUF."""
    manifest = json.loads((ROOT / "experiment_logs/20260819-llama-cpp-qpu/gemma-mtp-gguf-manifest.json").read_text())
    if sha256(model) != manifest["model"]["sha256"]:
        raise ValueError("MTP GGUF does not match the retained model hash")
    tensor = next(t for t in manifest["tensors"] if t["name"] == "blk.0.ffn_gate.weight")
    with model.open("rb") as stream:
        stream.seek(tensor["file_offset"])
        data = stream.read(tensor["bytes"])
    expected = read_record(REBUTTAL, "todo4/mtp.json")["files"]
    if hashlib.sha256(data).hexdigest() != expected["weights"]["sha256"]:
        raise ValueError("extracted MTP tensor hash differs from the measured fixture")
    activation = ROOT / "artifact/evidence/mtp/activation.f32"
    if sha256(activation) != expected["activation"]["sha256"]:
        raise ValueError("MTP activation hash differs from the measured fixture")
    output.mkdir(parents=True, exist_ok=False)
    (output / "weights.q4_0.bin").write_bytes(data)
    shutil.copyfile(activation, output / "activation.f32")
    result = {
        "model_sha256": manifest["model"]["sha256"],
        "tensor": tensor,
        "weights_sha256": expected["weights"]["sha256"],
        "activation_sha256": expected["activation"]["sha256"],
    }
    write_json(output / "provenance.json", result)
    return {"ok": True, "output": str(output), "weights_bytes": len(data)}


def check_assets(config: Path, only: str | None = None) -> dict:
    """Check only explicitly supplied assets; report omitted assets as missing."""
    locations = json.loads(config.read_text())
    manifest = json.loads((ROOT / "artifact/assets.json").read_text())
    selected = set(only.split(",")) if only else None
    file_ids = {a["id"] for a in manifest["assets"] if a["kind"] == "file"}
    if selected and selected - file_ids:
        raise ValueError(f"unknown file assets: {sorted(selected - file_ids)}")
    results = []
    for asset in manifest["assets"]:
        if asset["kind"] != "file" or (selected is not None and asset["id"] not in selected):
            continue
        value = locations.get(asset["id"])
        path = Path(value).expanduser() if value else None
        exists = path is not None and path.is_file()
        digest = sha256(path) if exists else None
        results.append(
            {
                "id": asset["id"],
                "path": str(path) if path else None,
                "sha256": digest,
                "ok": exists and digest == asset["sha256"],
            }
        )
    return {"kind": "artifact-assets-v1", "ok": all(r["ok"] for r in results), "assets": results}


def build_release(output: Path) -> dict:
    """Build a local tarball from an explicit inventory, never directory recursion."""
    verify()
    if output.exists():
        raise ValueError(f"release output already exists: {output}")
    inventory = json.loads((ROOT / "artifact/release-files.json").read_text())["files"]
    hashes = {relative: sha256(contained(ROOT, relative)) for relative in sorted(inventory)}
    modes = {name: 0o755 if os.access(ROOT / name, os.X_OK) else 0o644 for name in inventory}
    revision = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        previous = ROOT / "artifact/RELEASE.json"
        revision = json.loads(previous.read_text())["source_base_commit"] if previous.exists() else None
    content_hash = hashlib.sha256(
        json.dumps({"files": hashes, "modes": modes, "source_base_commit": revision}, sort_keys=True).encode()
    ).hexdigest()
    version = f"0.0.0+artifact.{content_hash[:12]}"
    prefix = f"qpu-xla-artifact-{content_hash[:12]}"
    release = {
        "format": "qpu-xla-artifact-release-v1",
        "version": version,
        "source_base_commit": revision,
        "content_sha256": content_hash,
        "files": hashes,
        "modes": modes,
        "publication": "local-only; no DOI or badges claimed",
    }
    generated = {
        "artifact/RELEASE.json": (json.dumps(release, indent=2, sort_keys=True) + "\n").encode(),
        "PKG-INFO": f"Metadata-Version: 2.1\nName: py-videocore7\nVersion: {version}\n".encode(),
    }
    all_hashes = {**hashes, **{name: hashlib.sha256(data).hexdigest() for name, data in generated.items()}}
    generated["SHA256SUMS"] = "".join(f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as target,
        gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w|") as archive,
    ):
        for name in sorted(set(inventory) | set(generated)):
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.mode = modes.get(name, 0o644)
            if name in generated:
                data = generated[name]
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                # Recheck the bytes actually archived against the frozen inventory.
                data = contained(ROOT, name).read_bytes()
                if hashlib.sha256(data).hexdigest() != hashes[name]:
                    raise ValueError(f"file changed during release: {name}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256(output)}  {output.name}\n")
    return {
        "ok": True,
        "archive": str(output),
        "bytes": output.stat().st_size,
        "version": version,
        "content_sha256": content_hash,
        "files": len(all_hashes) + 1,
    }


def run_profile(args) -> dict:
    """Plan or sequentially execute existing runners using argument arrays."""
    config = json.loads(args.config.read_text())
    profile = config["profiles"][args.profile]
    values = {**os.environ, "python": sys.executable, "root": str(ROOT), "output": str(args.output.resolve())}
    values.setdefault("BUILD_ROOT", str(ROOT / "build"))
    values.setdefault("INTEGRATION_BUILD", str(Path(values["BUILD_ROOT"]) / "llama-qpu-runtime"))
    values.setdefault("MTP_FIXTURE", str(ROOT / "build/artifact-mtp"))
    stages = profile["stages"]
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {s["id"] for s in stages}
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        stages = [s for s in stages if s["id"] in wanted]
    else:
        stages = [s for s in stages if s.get("enabled", True)]
    plan = []
    for stage in stages:
        plan.append({**stage, "argv": [arg.format_map(values) for arg in stage["argv"]]})
    result = {"profile": args.profile, "executed": False, "stages": plan}
    if not args.execute:
        for stage in plan:
            print(f"{stage['id']}: {shlex.join(stage['argv'])}")
        return result
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("profile execution requires a new or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    result["environment"] = check_environment(profile.get("hardware", False))
    if not result["environment"]["ok"]:
        write_json(args.output / "campaign.json", result)
        raise ValueError("profile environment failed; see campaign.json")
    result["executed"] = True
    result["completed"] = False
    write_json(args.output / "campaign.json", result)
    for stage in plan:
        start = time.monotonic()
        # Leave signal handling, numerical gates, state restoration and failed-run
        # retention to the established controlled runners.
        stage["returncode"] = subprocess.run(stage["argv"], cwd=ROOT).returncode
        stage["elapsed_seconds"] = time.monotonic() - start
        gate_failed = False
        if any(Path(arg).name == "run_rebuttal_diagnostics.py" for arg in stage["argv"]):
            control = Path(stage["argv"][stage["argv"].index("--output") + 1]) / "process.json"
            if not control.is_absolute():
                control = ROOT / control
            if control.is_file():
                status = json.loads(control.read_text())
                stage["control"] = {
                    "record": str(control),
                    "completed": status.get("completed", False),
                    "retained_for_latency": status.get("retained_for_latency", False),
                    "kernel_error_lines": status.get("kernel_error_lines", []),
                }
            else:
                stage["control"] = {"record": str(control), "retained_for_latency": False, "error": "missing record"}
            gate_failed = not stage["control"]["retained_for_latency"]
        if stage["returncode"] or gate_failed:
            result["failed_stage"] = stage["id"]
        write_json(args.output / "campaign.json", result)
        if stage["returncode"] or gate_failed:
            raise ValueError(f"stage {stage['id']} failed; retain its output and inspect campaign.json")
    result["completed"] = True
    write_json(args.output / "campaign.json", result)
    return result


def main() -> None:
    """Dispatch explicit artifact preparation and evaluation actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--hardware", action="store_true")
    check.add_argument("--output", type=Path)
    sub.add_parser("verify")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--output", type=Path, required=True)
    basic = sub.add_parser("smoke")
    basic.add_argument("--hardware", action="store_true")
    basic.add_argument("--native-build", type=Path)
    basic.add_argument("--output", type=Path, required=True)
    assets = sub.add_parser("assets")
    assets.add_argument("--config", type=Path, required=True)
    assets.add_argument("--only", help="Comma-separated file asset IDs; default checks all")
    mtp = sub.add_parser("prepare-mtp")
    mtp.add_argument("--model", type=Path, required=True)
    mtp.add_argument("--output", type=Path, required=True)
    release = sub.add_parser("build")
    release.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("profile", choices=("smoke", "paper", "rebuttal", "quality"))
    run.add_argument("--config", type=Path, default=ROOT / "artifact/profiles.json")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--only", help="Comma-separated stage IDs")
    run.add_argument("--execute", action="store_true", help="Execute sequentially; default prints the plan")
    args = parser.parse_args()
    try:
        match args.command:
            case "check":
                result = check_environment(args.hardware)
                if args.output:
                    write_json(args.output, result)
            case "verify":
                result = verify()
            case "analyze":
                result = analyze(args.output)
            case "smoke":
                result = smoke(args.output, args.hardware, args.native_build)
            case "assets":
                result = check_assets(args.config, args.only)
            case "prepare-mtp":
                result = prepare_mtp(args.model, args.output)
            case "build":
                result = build_release(args.output)
            case "run":
                result = run_profile(args)
        print(json.dumps(result, indent=2))
        if result.get("ok") is False:
            raise SystemExit(2)
    except (
        OSError,
        ValueError,
        KeyError,
        AssertionError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        parser.exit(2, f"artifact: {error}\n")


if __name__ == "__main__":
    main()
