# Installation

This is the installation entry point promised by Table 7 and Appendix H of
*Practical: Building and Evaluating a Native CPU–QPU Runtime for Edge AI
Inference on Raspberry Pi 5*. The supplied manuscript is identified in
[artifact/manuscript.json](artifact/manuscript.json).

Use 64-bit Linux, Python 3.12+, `uv`, `git`, a C/C++ compiler, Python development
headers, CMake and Make. QPU execution requires an actively cooled Raspberry
Pi 5 with reliable power and V3D device access; the model experiments use an
8 GB board. Allow several GB for dependencies on ordinary disk; models,
datasets and native builds add to that. Avoid a small RAM-backed `/tmp`.

From the extracted source root:

```sh
uv sync --locked --group dev
uv run --locked python scripts/artifact.py verify
uv run --locked python scripts/artifact.py smoke --output build/install-smoke
```

The smoke returns `ok: true`, verifies retained checksums and checks numerical
contracts using a fake backend. It runs on a machine without QPU hardware.
For actual QPU execution:

```sh
uv run --locked python scripts/artifact.py check --hardware --output build/pi-environment.json
```

The current driver needs access to both `/dev/dri/renderD128` and
`/dev/dri/by-path/platform-1002000000.v3d-card`. Check the host `render` and
`video` groups. See [artifact/README.md](artifact/README.md#pi-preparation-and-basic-test)
for the dedicated pinned `llama.cpp` checkout, ordered patch preparation and
native smoke commands. Set `BUILD_ROOT` to the new isolated build directory
for subsequent profiles. Do not reuse unrelated modified checkouts.

Install task-quality dependencies only when evaluating those workloads:

```sh
uv sync --locked --group dev --extra evaluation
```

External input revisions, hashes and preparation commands are in
[artifact/ASSETS.md](artifact/ASSETS.md). [HOWTO.md](HOWTO.md) selects paper
results and fresh experiments. Measured setup costs and known MMU/stability
limitations are recorded in [artifact/VALIDATION.md](artifact/VALIDATION.md).
The optional container recipe has not been validated; the native path has.
