"""Build an isolated CPU backend with diagnostic FFN region timestamps.

Reuse the retained build's flags and other objects; never overwrite its sources,
objects, or libraries. Select the result with LD_LIBRARY_PATH for diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

HELPERS = r"""
#include <time.h>
static uint64_t ffn_profile_clock(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (uint64_t) stamp.tv_sec * 1000000000ULL + (uint64_t) stamp.tv_nsec;
}
static _Thread_local uint64_t ffn_profile_starts[256];
static _Thread_local int64_t ffn_profile_widths[256];
"""
BEFORE = r"""
        int ffn_profile_layer = -1;
        const char * ffn_profile_role = NULL;
        const char * ffn_profile_enabled = getenv("GGML_QPU_FFN_PROFILE");
        if (state->ith == 0 && ffn_profile_enabled && strcmp(ffn_profile_enabled, "1") == 0 &&
            node->op == GGML_OP_MUL_MAT && node->src[0] &&
            sscanf(node->src[0]->name, "blk.%d.", &ffn_profile_layer) == 1 &&
            ffn_profile_layer >= 0 && ffn_profile_layer < 256) {
            const char * weight_name = node->src[0]->name;
            if (strstr(weight_name, ".ffn_gate.weight") || strstr(weight_name, ".ffn_up.weight")) {
                ffn_profile_role = "start";
                if (ffn_profile_starts[ffn_profile_layer] == 0) {
                    ffn_profile_widths[ffn_profile_layer] = node->src[0]->ne[1];
                    ffn_profile_starts[ffn_profile_layer] = ffn_profile_clock();
                }
            } else if (strstr(weight_name, ".ffn_down.weight")) {
                ffn_profile_role = "finish";
            }
        }
"""
AFTER = r"""
        if (state->ith == 0 && ffn_profile_role && strcmp(ffn_profile_role, "finish") == 0) {
            const uint64_t finished = ffn_profile_clock();
            fprintf(stderr,
                "cpu_ffn_region_json:{\"layer\":%d,\"m\":%lld,\"width\":%lld,"
                "\"begin_ns\":%llu,\"end_ns\":%llu,\"wall_ns\":%llu,"
                "\"clock\":\"CLOCK_MONOTONIC\"}\n",
                ffn_profile_layer, (long long) node->ne[1],
                (long long) ffn_profile_widths[ffn_profile_layer],
                (unsigned long long) ffn_profile_starts[ffn_profile_layer],
                (unsigned long long) finished,
                (unsigned long long) (finished - ffn_profile_starts[ffn_profile_layer]));
            ffn_profile_starts[ffn_profile_layer] = 0;
        }
"""


def main() -> None:
    """Compile and link isolated timestamped copies of the retained CPU backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-build", type=Path, help="Unix Makefiles build; defaults to LLAMA_ROOT/build")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.llama_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = root / "ggml/src/ggml-cpu/ggml-cpu.c"
    source = original.read_text()
    marker = "static thread_ret_t ggml_graph_compute_thread(void * data) {"
    assert source.count(marker) == 1
    source = source.replace(marker, HELPERS + "\n" + marker)
    start = source.index(marker)
    end = source.index("static inline bool ggml_graph_compute_thread_ready", start)
    region = source[start:end]
    before = "        // TODO: move fused-op detection into ggml_graph_plan"
    assert region.count(before) == 1
    region = region.replace(before, BEFORE + "\n" + before)
    after = "            ggml_barrier(state->threadpool);\n        }\n    }"
    assert region.count(after) == 1
    region = region.replace(after, "            ggml_barrier(state->threadpool);\n        }\n" + AFTER + "\n    }")
    source = source[:start] + region + source[end:]
    instrumented = out / "ggml-cpu-profile.c"
    instrumented.write_text(source)
    build = (args.llama_build.resolve() if args.llama_build else root / "build") / "ggml/src"
    flags = {}
    for line in (build / "CMakeFiles/ggml-cpu.dir/flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    obj = out / "ggml-cpu-profile.c.o"
    compile_command = [
        "/usr/bin/cc",
        *flags["C_DEFINES"],
        *flags["C_INCLUDES"],
        *flags["C_FLAGS"],
        "-c",
        str(instrumented),
        "-o",
        str(obj),
    ]
    subprocess.run(compile_command, cwd=build, check=True)
    repack_original = root / "ggml/src/ggml-cpu/repack.cpp"
    repack = repack_original.read_text()
    anchor = "                ggml_qpu_ffn_island_cpu_output_columns_for(src0->name));"
    assert repack.count(anchor) == 1
    repack = ('extern "C" void ggml_qpu_ffn_island_profile_cpu_started() __attribute__((weak));\n' + repack).replace(
        anchor,
        anchor + "\n            if (ith == 0 && ggml_qpu_ffn_island_profile_cpu_started != nullptr) {\n"
        "                ggml_qpu_ffn_island_profile_cpu_started();\n            }",
    )
    repack_path, repack_obj = out / "repack-profile.cpp", out / "repack-profile.cpp.o"
    repack_path.write_text(repack)
    repack_compile = [
        "/usr/bin/c++",
        *flags["CXX_DEFINES"],
        *flags["CXX_INCLUDES"],
        *flags["CXX_FLAGS"],
        "-c",
        str(repack_path),
        "-o",
        str(repack_obj),
    ]
    subprocess.run(repack_compile, cwd=build, check=True)
    link = shlex.split((build / "CMakeFiles/ggml-cpu.dir/link.txt").read_text())
    indexes = [i for i, value in enumerate(link) if value.endswith("ggml-cpu.c.o")]
    assert len(indexes) == 1
    link[indexes[0]] = str(obj)
    repack_indexes = [i for i, value in enumerate(link) if value.endswith("ggml-cpu/repack.cpp.o")]
    assert len(repack_indexes) == 1
    link[repack_indexes[0]] = str(repack_obj)
    link[link.index("-o") + 1] = str(out / "libggml-cpu.so.0")
    link = [word for word in link if not word.startswith("-Wl,--dependency-file=")]
    subprocess.run(link, cwd=build, check=True)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (out / "build.json").write_text(
        json.dumps(
            {
                "original_source": str(original),
                "original_source_sha256": digest(original),
                "instrumented_source_sha256": digest(instrumented),
                "library_sha256": digest(out / "libggml-cpu.so.0"),
                "compile_command": compile_command,
                "link_command": link,
                "repack_compile_command": repack_compile,
                "repack_original_sha256": digest(repack_original),
                "repack_instrumented_sha256": digest(repack_path),
                "boundary": "first FFN projection entry through down-projection post-barrier; diagnostic only",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
