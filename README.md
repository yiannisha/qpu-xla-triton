# py-videocore7 and QPU-XLA

This repository has two layers:

- `py-videocore7`: a low-level Python assembler and driver for programming the
  Raspberry Pi 5 VideoCore VII QPU;
- `qpu_xla`: a newer ML-oriented heterogeneous runtime built on that driver.

The current development focus is `qpu_xla`. It provides host-mapped tensors,
CPU/QPU queues and events, reusable tiled kernels, operator contracts,
shape-aware placement, explicit CPU/QPU hybrid execution, persistent plans,
video preprocessing, and TinyLlama-oriented runtime building blocks.

For the current architecture, supported operators, runtime semantics, and
known limitations, read [QPU-XLA.md](QPU-XLA.md).

## Current status

The repository is a systems prototype, not a finished XLA implementation.

Implemented runtime pieces include:

- shared host/QPU-visible `Device`, `Buffer`, and `Tensor` storage;
- in-order `Queue` workers with explicit `Event` dependencies;
- packaged FP32/INT32 GEMM, pooling, min/max, and copy kernels;
- CPU, QPU, and calibrated whole-operator placement;
- explicit concurrent CPU/QPU row-split matmul;
- GEMM-backed FP32/INT32 convolution, including a no-window-duplication 1x1
  path;
- unnormalized attention and mixed QPU-GEMM/CPU-softmax SDPA;
- persistent INT32 convolution, MLP, and attention plans;
- native GGML Q4_0×Q8_0 M=1/M=4 QPU kernels and an out-of-tree C runtime for
  pinned `llama.cpp` evaluation;
- a constrained QPU-XLA DSL, differential candidate runner, video contracts,
  and TinyLlama model/runtime scaffolding.

The runtime is not yet a general XLA compiler. Hybrid partitioning is explicit,
not automatically selected by the scheduler. The native Q4_0 candidates are
hardware-differential tested but currently slower than the exact optimized CPU
nodes, so they remain experimental and are not automatic placements.

## Installation and hardware access

The project targets Raspberry Pi 5 hardware with VideoCore VII and a usable V3D
render node. Install `uv`, clone the repository, and run from its root:

```console
sudo apt update
sudo apt install git
git clone https://github.com/Idein/py-videocore7.git
cd py-videocore7
uv sync
```

Hardware execution requires access to `/dev/dri/renderD128` and membership in
the appropriate `render`/`video` device group for the host image. Check the
local device permissions before running QPU tests.

## Running the QPU-XLA runtime benchmark

The runtime matrix compares CPU-only, QPU-only, automatic placement, explicit
CPU/QPU row splits, and mixed attention execution:

```console
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run examples/benchmark_qpu_xla_matrix.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run examples/benchmark_qpu_xla_matrix.py \
  --size 512 --warmup 2 --repeat 7 --output fp32-matmul-attention-s512.json
```

FP32 kernel coverage, current exact-shape results, promotion evidence, and
reproduction commands are maintained together in the
[FP32 kernel-suite directory](experiment_logs/20260819-qpu-xla-kernel-suite/README.md).
General runtime architecture and the legacy benchmark inventory remain in
[QPU-XLA.md](QPU-XLA.md) and [EXPERIMENTS_REGISTRY.md](EXPERIMENTS_REGISTRY.md).

Run the exhaustive W8A8 dense/convolution matrix in isolated jobs with:

```console
uv run scripts/run_w8a8_evaluation.py --output-root experiment_logs/w8a8-matrix
```

This covers all Llama and YOLO manifest shapes, square 64³/512³ GEMMs, CPU/QPU
row and output splits, CPU/standalone-QPU/fused-QPU dequantization, all four CPU
references, raw samples, FP32 quality metrics, calibrated registry generation,
and one-layer regressions for standalone deployment winners.

## Running tests

Run the full test suite on a development host:

```console
uv run pytest -vs tests
```

Tests marked `hardware` require the VideoCore VII render node. CPU/fake-backend
tests cover runtime contracts, queue/event behavior, placement, DSL validation,
model loading, and reference execution without QPU hardware.

## Legacy examples

The original low-level and hand-written operator examples remain available for
assembly experiments and historical comparisons:

- `examples/sgemm.py`, `examples/sgemm_fast.py`, and `examples/igemm.py`;
- `examples/minmax.py`, `examples/pool2d.py`, and `examples/scopy.py`;
- `examples/tiledconv2d.py`, `examples/tiledattention.py`,
  `examples/tiledmlp.py`, and `examples/tiledlenet5.py`.

They are benchmarkable legacy paths, but they are not the canonical API for the
new runtime. Their commands and timing categories are listed in
[EXPERIMENTS_REGISTRY.md](EXPERIMENTS_REGISTRY.md).

## Low-level VideoCore VII background

Raspberry Pi 5 (BCM2712) includes a VideoCore VII QPU in its SoC. This project
communicates with the hardware through the DRM V3D driver.

For the underlying assembler and driver, see the original
[py-videocore7 documentation](https://github.com/Idein/py-videocore7). Related
references include the
[Linux V3D driver](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/gpu/drm/v3d)
and the [Mesa QPU sources](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/broadcom/qpu).
