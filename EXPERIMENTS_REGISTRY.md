# Experiments Registry

Commands below assume you are in the repo root:

```bash
cd /home/yiannis/side/py-videocore7
```

Recommended runner:

```bash
uv run examples/<script>.py
```

Notes:

- Most examples exercise the VideoCore VII QPU path and are intended for supported Raspberry Pi hardware with V3D access.
- Several benchmarks optionally compare against PyTorch. If `torch` is not installed, those comparisons print `n/a`.
- For scripts with flags, `uv run examples/<script>.py --help` prints the CLI.

## Current QPU-XLA runtime benchmark

The newer runtime-facing benchmark is:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run examples/benchmark_qpu_xla_matrix.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run examples/benchmark_qpu_xla_matrix.py \
  --size 512 --warmup 2 --repeat 7 --output fp32-matmul-attention-s512.json
```

It is the canonical benchmark for the current `src/qpu_xla` work. It compares
CPU-only, QPU-only, uncalibrated automatic placement, explicit concurrent
CPU/QPU matmul row splits, CPU-only SDPA, the QPU GEMM attention core, and
mixed QPU-GEMM/CPU-softmax SDPA.

The maintained FP32 inventory, exact-shape results, promotion evidence, and
reproduction commands are in the
[`FP32 kernel-suite directory`](experiment_logs/20260819-qpu-xla-kernel-suite/README.md).
For general runtime architecture and operator status, see
[`QPU-XLA.md`](QPU-XLA.md).

## Native llama.cpp Q4_0 and end-to-end evaluation

Build the out-of-tree C runtime and its live hardware differential tests with:

```bash
cmake -S integrations/llama_cpp -B build/llama-qpu-runtime \
  -DQPU_LLAMA_BUILD_HARDWARE_TESTS=ON \
  -DQPU_LLAMA_BUILD_BENCHMARK=ON \
  -DLLAMA_CPP_ROOT=/home/yiannis/side/llama.cpp
cmake --build build/llama-qpu-runtime --parallel 4
ctest --test-dir build/llama-qpu-runtime --output-on-failure
```

The exact-shape native operator matrix is driven from bounded payloads in the
GGUF manifests:

```bash
python examples/benchmark_llama_cpp_qpu_ops.py \
  experiment_logs/20260819-llama-cpp-qpu/gemma-mtp-gguf-manifest.json \
  --tensor blk.0.ffn_gate.weight --rows 4 \
  --output experiment_logs/20260819-llama-cpp-qpu/operator-session.json
```

Generate and selectively execute the CPU end-to-end matrix with:

```bash
python scripts/generate_llama_cpp_evaluation_cases.py \
  --output experiment_logs/20260819-llama-cpp-qpu/full-evaluation-cases.json
python scripts/run_llama_cpp_qpu_evaluation.py \
  --case-file experiment_logs/20260819-llama-cpp-qpu/full-evaluation-cases.json \
  --case gemma-mtp-n2-t3-c0-decode --samples 31 \
  --session-id gemma-mtp-session-1 \
  --output experiment_logs/20260819-llama-cpp-qpu/gemma-mtp-session-1.json
```

Only performance-governor, unthrottled, no-swap-change sessions qualify for
retention. Current records are diagnostics rejected for the `ondemand`
governor; no native Q4_0 candidate is promoted. The generated result is
[`LLAMA_CPP_QPU_MATRIX.md`](experiment_logs/20260819-llama-cpp-qpu/LLAMA_CPP_QPU_MATRIX.md).

## Packed W8A8 model-kernel benchmarks

Dense Llama projection matrix:

```bash
uv run examples/benchmark_qpu_xla_w8a8.py --case prefill-h512-t16 --epilogue fused-qpu
uv run examples/benchmark_qpu_xla_w8a8.py --case prefill-h1024-t64 --epilogue standalone-qpu
uv run examples/benchmark_qpu_xla_w8a8.py --case square-512x512x512 --epilogue cpu
uv run examples/benchmark_qpu_xla_w8a8.py --case decode-h2048-c512 --projection hidden
uv run examples/benchmark_qpu_xla_llama_stages.py --case prefill-h1024-t64
uv run examples/benchmark_qpu_xla_tinyllama_runtime.py --case prefill-h1024-t64
uv run examples/benchmark_qpu_xla_tinyllama_runtime.py --case decode-h2048-c512
uv run scripts/build_llama_candidate_registry.py
uv run scripts/run_w8a8_evaluation.py --output-root experiment_logs/w8a8-matrix
```

The script records NumPy/Torch dynamic-W8A8, NumPy/OpenBLAS/Torch FP32, host
quantization/packing, kernel-only, dequantization, whole-operation, and hybrid
raw samples. `same-contract-win` means a 1.05x win over dynamic W8A8 CPU;
`supported-win` additionally clears FP32 quality gates and beats the fastest
deployable FP32 CPU backend by 1.05x. Only the latter enters AUTO placement.

YOLO convolution candidates:

```bash
uv run examples/benchmark_qpu_xla_yolo_w8a8.py --case p3-1x1
uv run examples/benchmark_qpu_xla_yolo_w8a8.py --case p3-3x3-s1
uv run examples/benchmark_qpu_xla_yolo_w8a8.py --case depthwise-p3
```

This benchmark keeps NumPy/Torch dynamic W8A8, NumPy lowered FP32, Torch native
FP32 conv2d, host preparation, QPU execution events, row/output hybrids, and
steady-state total timings distinct. Unsupported grouped/depthwise output
splits are retained explicitly in JSON.
Current YOLO candidates are correct but slower and remain experimental.

The merged archive for this machine is:

```text
experiment_logs/20260819-qpu-xla-w8a8/llama-calibrated.candidates.json
```

It contains 48 winning and slower Llama records. The builder also demotes
standalone wins when their measured one-layer composition regresses versus
FP32 CPU; only the remaining 17 `supported-win` entries are visible to AUTO.

The entries below are the legacy example-script inventory. They remain useful
for low-level kernel comparisons but are not the canonical API description of
the new runtime.

## `examples/pctr_gpu_clock.py`

What it does:

- Measures V3D clock frequency with hardware performance counters.

CLI:

```bash
uv run examples/pctr_gpu_clock.py
```

## `examples/payload.py`

What it does:

- Dumps compute-shader payload/workgroup metadata written by a tiny QPU program.
- Useful for understanding workgroup IDs, invocation IDs, and payload layout.

CLI:

```bash
uv run examples/payload.py
```

## `examples/scopy.py`

What it does:

- Benchmarks a simple memory copy kernel.
- Runs three built-in experiments: CPU copy baseline, 1-QPU copy, and 12-QPU copy.
- Uses a fixed length of `24 * 1024 * 1024` `uint32` elements.

CLI:

```bash
uv run examples/scopy.py
```

## `examples/sgemm.py`

What it does:

- Benchmarks naive FP32 SGEMM on QPU against NumPy and optional Torch.
- Runs two built-in suites:
- Square SGEMM sweep with sizes `256, 512, 768, 1024, 1536, 2048`.
- Batched single-dispatch SGEMM sweep with sizes `128, 256, 512` and batches `1, 2, 4, 8, 16, 32`.

CLI:

```bash
uv run examples/sgemm.py
```

## `examples/sgemm_fast.py`

What it does:

- Benchmarks optimized FP32 SGEMM variants against NumPy, optional Torch, and the naive kernel from `examples/sgemm.py`.
- Built-in problem size is `1024 x 1024` times `1024 x 1024`.
- Prints accuracy summaries for naive, fast-payload, and fast-qpu-aware variants.

CLI:

```bash
uv run examples/sgemm_fast.py
```

## `examples/sgemm_batched_small.py`

What it does:

- Benchmarks a specialized small batched FP32 SGEMM kernel against the generic batched kernel, NumPy, and optional Torch.
- Default sweep uses sizes `128, 256` and batches `1, 2, 4, 8, 16`.

CLI:

```bash
uv run examples/sgemm_batched_small.py
uv run examples/sgemm_batched_small.py --size 128 --batch 8
uv run examples/sgemm_batched_small.py --batch 16
uv run examples/sgemm_batched_small.py --trials 10
uv run examples/sgemm_batched_small.py --help
```

Flags:

- `--size {128,256}`: run one size instead of the full sweep.
- `--batch INT`: restrict to one batch or use with `--size` for one point.
- `--trials INT`: number of timing trials.

## `examples/benchmark_qpu_xla_matrix.py`

What it does:

- Benchmarks the packaged QPU-XLA runtime rather than the legacy one-shot kernels.
- Compares CPU-only, QPU-only, uncalibrated automatic placement, and concurrent CPU/QPU row splits for FP32 matmul.
- Benchmarks CPU-only SDPA, the QPU-GEMM attention core, and mixed SDPA with QPU GEMM stages plus CPU softmax.
- Reports median whole-operation latency, throughput, and maximum absolute error.

CLI:

```bash
uv run examples/benchmark_qpu_xla_matrix.py
uv run examples/benchmark_qpu_xla_matrix.py --size 512 --warmup 2 --repeat 7
uv run examples/benchmark_qpu_xla_matrix.py --help
```

The default `512` square size is deliberate: on the tested hardware, the
`128/512` QPU-row FP32 split has demonstrated a lower wall-clock latency than
CPU-only matmul. The best split is machine- and shape-dependent, so use the
matrix rather than assuming that a larger QPU fraction is better.

## `examples/igemm.py`

What it does:

- Benchmarks INT32 GEMM on QPU against NumPy.
- Uses `smul24`; inputs must stay within signed 24-bit range.
- Default sweep uses square sizes `256, 512, 768, 1024`.

CLI:

```bash
uv run examples/igemm.py
```

## `examples/igemm_int16.py`

What it does:

- Benchmarks packed INT16-input GEMM with INT32 accumulation against NumPy.
- Uses packed int16 pairs in int32 words; this is not an int8/int16 dot-product kernel.
- Default sweep uses square sizes `256, 512, 768, 1024`.

CLI:

```bash
uv run examples/igemm_int16.py
```

## `examples/minmax.py`

What it does:

- Benchmarks elementwise `min` and `max` kernels against NumPy and optional Torch.
- Default length is `4 * 1024 * 1024` scalar elements.

CLI:

```bash
uv run examples/minmax.py
uv run examples/minmax.py --length 1048576 --repeat 10
uv run examples/minmax.py --num-qpus 1
uv run examples/minmax.py --dtypes fp32 int32 int16
uv run examples/minmax.py --help
```

Flags:

- `--length INT`: number of scalar elements.
- `--num-qpus {1,12}`: choose 1 or 12 QPUs.
- `--repeat INT`: repetitions after warmup.
- `--seed INT`: RNG seed.
- `--dtypes ...`: choose one or more of `fp32`, `int32`, `int16`.

## `examples/pool2d.py`

What it does:

- Benchmarks NCHW `2x2`, stride-2, no-padding pooling against NumPy and optional Torch.
- Covers maxpool and avgpool for selected dtypes.
- Default shape is `N=1, C=32, H=144, W=144`.

CLI:

```bash
uv run examples/pool2d.py
uv run examples/pool2d.py --batch 4 --channels 64 --height 112 --width 112
uv run examples/pool2d.py --num-qpus 1 --repeat 5
uv run examples/pool2d.py --dtypes fp32 int32
uv run examples/pool2d.py --help
```

Flags:

- `--batch INT`
- `--channels INT`
- `--height INT`
- `--width INT`
- `--num-qpus {1,12}`
- `--repeat INT`
- `--seed INT`
- `--dtypes ...`: choose one or more of `fp32`, `int32`, `int16`.

## `examples/tiledattention.py`

What it does:

- Benchmarks tiled single-head attention core `O = (Q @ K^T) @ V`.
- Runs two built-in suites:
- FP32 with `Q=128x128`, `K=128x128`, `V=128x128`.
- INT32 with the same dimensions and signed-24-bit contract checks.
- Reports per-stage timings for score, value, and total attention, plus steady-state QPU timings.

CLI:

```bash
uv run examples/tiledattention.py
```

## `examples/tiledconv2d.py`

What it does:

- Benchmarks the current tiled GEMM-backed `conv2d` path.
- Runs three built-in suites:
- FP32: `N=1, C_in=32, H=W=34, C_out=64, K=3x3, stride=1, pad=1`.
- INT32: `N=1, C_in=32, H=W=18, C_out=64, K=3x3, stride=1, pad=1`.
- INT16 packed: same shape as INT32, accumulated into INT32.
- Prints NumPy, Torch native conv2d, QPU host prep, QPU cached total, QPU execute only, QPU prep+cached total, and max error.

CLI:

```bash
uv run examples/tiledconv2d.py
```

## `examples/tiledlenet5.py`

What it does:

- Benchmarks a persistent LeNet-5-style inference pipeline against NumPy and optional Torch.
- Architecture is `Conv5x5 -> ReLU -> AvgPool -> Conv5x5 -> ReLU -> AvgPool -> FC -> ReLU -> FC -> ReLU -> FC`.
- Default batch is `48`.
- Default dtype set is `fp32 int32`.

CLI:

```bash
uv run examples/tiledlenet5.py
uv run examples/tiledlenet5.py --batch 96 --repeat 5
uv run examples/tiledlenet5.py --num-qpus 1
uv run examples/tiledlenet5.py --dtypes fp32
uv run examples/tiledlenet5.py --dtypes int32 --seed 7
uv run examples/tiledlenet5.py --help
```

Flags:

- `--batch INT`
- `--repeat INT`
- `--seed INT`
- `--num-qpus {1,12}`
- `--dtypes fp32 int32`

## `examples/tiledmlp.py`

What it does:

- Benchmarks a tiled MLP `Linear -> ReLU -> Linear`.
- Runs two built-in suites:
- FP32 with `batch=256`, `in=1024`, `hidden=1024`, `out=512`.
- INT32 with the same dimensions and signed-24-bit contract checks.
- Reports timings for total MLP, layer 1, ReLU, and layer 2.

CLI:

```bash
uv run examples/tiledmlp.py
```

## Summary By Script

Quick index:

- `examples/pctr_gpu_clock.py`: V3D clock measurement.
- `examples/payload.py`: compute payload/workgroup introspection.
- `examples/scopy.py`: memory copy bandwidth.
- `examples/sgemm.py`: naive FP32 GEMM sweeps.
- `examples/sgemm_fast.py`: optimized FP32 GEMM comparison.
- `examples/sgemm_batched_small.py`: specialized small batched FP32 GEMM.
- `examples/igemm.py`: INT32 GEMM.
- `examples/igemm_int16.py`: packed INT16-input GEMM with INT32 accumulation.
- `examples/minmax.py`: elementwise min/max.
- `examples/pool2d.py`: 2D maxpool/avgpool.
- `examples/tiledattention.py`: tiled attention core.
- `examples/tiledconv2d.py`: tiled GEMM-backed conv2d.
- `examples/tiledlenet5.py`: end-to-end LeNet-5-style inference pipeline.
- `examples/tiledmlp.py`: tiled two-layer MLP.
