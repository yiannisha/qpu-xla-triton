"""Build the pinned native integration in a dedicated llama.cpp checkout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "91d2fc387529940230555abd297a8b5e99737d3f"


def patched_sources(llama: Path, patches: list[Path]) -> dict[str, bytes]:
    """Build the expected complete patch-series state away from the checkout."""
    names = sorted({name for patch in patches for name in re.findall(r"^\+\+\+ b/(.+)$", patch.read_text(), re.M)})
    with tempfile.TemporaryDirectory(prefix="qpu-artifact-patches-") as temporary:
        root = Path(temporary)
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "-C", str(llama), "show", f"{PIN}:{name}"]))
        for patch in patches:
            subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        return {name: (root / name).read_bytes() for name in names}


def prepare(llama: Path, rebuttal: bool, jobs: int, build_root: Path) -> None:
    """Check the base and patch identities before configuring persistent builds."""
    revision = subprocess.check_output(["git", "-C", str(llama), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"llama.cpp HEAD is {revision}, expected {PIN}; use a dedicated checkout")
    build_root.mkdir(parents=True, exist_ok=True)
    output = build_root / "artifact-native-build.json"
    commands = []
    record = {
        "llama_base": PIN,
        "llama_root": str(llama),
        "build_root": str(build_root),
        "started_utc": datetime.now(UTC).isoformat(),
        "commands": commands,
        "rebuttal": rebuttal,
        "completed": False,
    }
    started = time.monotonic()

    def save():
        output.write_text(json.dumps(record, indent=2) + "\n")

    def run(argv):
        entry = {"argv": argv}
        commands.append(entry)
        print(" ".join(argv), flush=True)
        save()
        begin = time.monotonic()
        process = subprocess.run(argv, cwd=ROOT)
        entry.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - begin)
        save()
        process.check_returncode()

    patches = sorted((ROOT / "integrations/llama_cpp/patches").glob("000[1-5]-*.patch"))
    if len(patches) != 5:
        raise ValueError("expected all five integration patches")
    expected = patched_sources(llama, patches)
    changed = subprocess.check_output(["git", "-C", str(llama), "diff", "HEAD", "--name-only"], text=True).splitlines()
    if all((llama / name).read_bytes() == data for name, data in expected.items()):
        if set(changed) - set(expected):
            raise ValueError("unrelated tracked changes in llama.cpp; use a dedicated checkout")
    else:
        if changed:
            raise ValueError("partial or different llama.cpp edits; use a clean dedicated pinned checkout")
        for patch in patches:
            base = ["git", "-C", str(llama), "apply"]
            run(base + ["--check", str(patch)])
            run(base + [str(patch)])
    run(
        [
            "cmake",
            "-S",
            str(llama),
            "-B",
            str(llama / "build"),
            "-G",
            "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_CPU_REPACK=ON",
            "-DGGML_NATIVE=ON",
            "-DGGML_OPENMP=ON",
            "-DGGML_BLAS=OFF",
            "-DGGML_BACKEND_DL=OFF",
            "-DGGML_CPU_ALL_VARIANTS=OFF",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(llama / "build"),
            "--parallel",
            str(jobs),
            "--target",
            "llama-bench",
            "llama-server",
            "llama-perplexity",
        ]
    )
    builds = ["llama-qpu-runtime"] + (["llama-qpu-profile", "llama-qpu-ablation"] if rebuttal else [])
    for name in builds:
        directory = build_root / name
        run(
            [
                "cmake",
                "-S",
                str(ROOT / "integrations/llama_cpp"),
                "-B",
                str(directory),
                f"-DLLAMA_CPP_ROOT={llama}",
                f"-DLLAMA_CPP_BUILD_DIR={llama / 'build'}",
                "-DQPU_LLAMA_BUILD_GGML_BACKEND=ON",
                "-DQPU_LLAMA_BUILD_HARDWARE_TESTS=ON",
                "-DQPU_LLAMA_BUILD_BENCHMARK=ON",
            ]
        )
        run(
            [
                "cmake",
                "--build",
                str(directory),
                "--parallel",
                str(jobs),
                "--target",
                "ggml-qpu-inline",
                "qpu_repack_stride_smoke",
                "qpu_ffn_island_graph_smoke",
                "qpu_ffn_island_model_smoke",
                "qpu_llama_cpu_repack_bench",
                *(["ggml-qpu", "qpu_ggml_geglu_bench"] if name == "llama-qpu-runtime" else []),
            ]
        )
    if rebuttal:
        for name in ("ffn-profile-cpu", "ffn-ablation-cpu"):
            run(
                [
                    sys.executable,
                    "scripts/build_llama_cpp_ffn_profile.py",
                    "--llama-root",
                    str(llama),
                    "--output",
                    str(build_root / name),
                ]
            )
    record.update(
        completed=True, elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(UTC).isoformat()
    )
    save()


def main() -> None:
    """Build only; measurements and privileged host controls are separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--rebuttal", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--build-root", type=Path, default=ROOT / "build", help="Separate integration/profile output root"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    try:
        prepare(args.llama_root.resolve(), args.rebuttal, args.jobs, args.build_root.resolve())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"native preparation: {error}\n")


if __name__ == "__main__":
    main()
