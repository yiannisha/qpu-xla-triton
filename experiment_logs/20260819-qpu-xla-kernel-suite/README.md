# FP32 QPU-XLA Kernel Suite

This directory is the canonical home for FP32 kernel-suite documentation,
retained benchmark logs, and generated reports. CPU-only implementations are
references, not kernel-suite deliverables. Quantized results appear only where
they provide a cross-backend comparison; their detailed reports live in the
separate W8A8 experiment directories. W4A8 is intentionally deferred.

## Document Map

- [`README.md`](README.md): maintained FP32 inventory, conclusions, and
  reproduction commands.
- [`FP32_EVALUATION_MATRIX.md`](FP32_EVALUATION_MATRIX.md): generated FP32
  projection, stage, and post-op evaluation matrix.
- [`KERNEL_BACKEND_LATENCY_MATRIX.md`](KERNEL_BACKEND_LATENCY_MATRIX.md):
  generated compact backend comparison; this is the required canonical
  latency-matrix path.
- [`KERNEL_SCOREBOARD.md`](KERNEL_SCOREBOARD.md): generated retained-candidate
  scoreboard with raw-sample summaries.

The three matrix files are generated artifacts. Update the benchmark logs and
rerun the commands below instead of hand-editing their result rows.

## Llama Kernel Coverage

| Operation | Dtype | QPU-only | CPU/QPU split | Partition | Current conclusion |
|---|---|---|---|---|---|
| Dense matmul/linear | FP32 | `vc7.tiled_fp32_gemm`, `vc7.fp32_gemv` | Yes | prefill rows; decode output columns | QPU-only and 16-row hybrid prefill wins; decode remains CPU; exact-shape calibration required |
| RMSNorm | FP32 | `vc7.rms_norm_fp32` | Yes | rows | Vec4 QPU wins at 128×2048 and 256×2048; a 224/256-row hybrid also wins |
| RoPE | FP32 | `vc7.rope_fp32` | Yes | rows | Vec4 is up to 10.2× faster than the prior QPU kernel, but CPU remains fastest |
| SwiGLU | FP32 | `vc7.swiglu_fp32` | Yes | rows | Full-QPU and high-row hybrid wins at the 128- and 256-token prefill shapes |
| Softmax | FP32 | `vc7.softmax_fp32` | Yes | rows | Full-QPU and 7168/8192-row hybrid wins at the 256-token prefill shape |
| Attention score/value GEMMs | FP32 | tiled GEMM or decode GEMV | Yes | prefill rows; decode output columns | Score, softmax, and value placement are independently selectable; scaling/mask prep remains a host stage |
| KV-cache append | FP32 | `vc7.copy_words` | Yes | tokens | Exact; small append loses, exact-stripe large copy wins |
| Residual add | FP32 | `vc7.residual_add_fp32` | Yes | rows | Exact; small tensors lose, exact-stripe large tensor wins |
| Embedding lookup | FP32 | `vc7.embedding_lookup_fp32` | Yes | tokens | Exact; correct but slower at the tested shape |
| Sampling/argmax | FP32→I32 | `vc7.argmax_fp32` | Yes | rows | Exact, including first-index ties; correct but slower at the tested shape |

## Dtype Contracts

- FP32 kernels consume and produce `float32`.
- INT32 GEMM remains an internal/reference-capable path with the signed 24-bit
  `smul24` input restriction.
- W4A8/INT4 is deferred and is not advertised by the suite.

The VideoCore VII instruction is named `v8dot` in this assembler. It is used
by W8A8 GEMM/GEMV only. Applying it to FP32 would change the numerical contract,
so FP32 GEMM/GEMV correctly use `fmul`/`fadd` instead.

## Benchmark and Promotion Rules

The retained matrix is in
[`KERNEL_SCOREBOARD.md`](KERNEL_SCOREBOARD.md).
It contains raw samples for CPU, QPU-only, and every stable hybrid partition.
The newer FP32 projection and post-op sweeps are consolidated in
[`FP32_EVALUATION_MATRIX.md`](FP32_EVALUATION_MATRIX.md).
The compact cross-backend view, including exact shapes, QPU-only latency, the
best hybrid partition, W8A8, YOLO conv2d, and explicit coverage gaps, is in
[`KERNEL_BACKEND_LATENCY_MATRIX.md`](KERNEL_BACKEND_LATENCY_MATRIX.md).

A candidate is automatic-placement eligible only when all of these hold:

1. Its source hash matches the current implementation.
2. Its exact shape, dtype, layout, placement, and partition were measured.
3. It passes the operation's numerical tolerance with no NaN/Inf output.
4. Median whole-operation speedup is at least 1.05× over the fastest measured
   CPU reference.

Kernel-only time is retained separately where available. It is not used to
promote a candidate whose host preparation, submission, or epilogue makes the
whole operation slower.

Every benchmark now retains both CPU backends:

- NumPy 2.2.2, linked to `scipy-openblas` 0.3.28 (`neoversen1`,
  `USE64BITINT`, dynamic architecture);
- Torch 2.11.0 native CPU operators.

Both were run with four CPU threads (`OPENBLAS_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, and Torch reporting four intra-op threads). The exact
backend versions, BLAS configuration, and thread settings are stored in every
new JSON log. NumPy matrix products are the OpenBLAS comparison. NumPy
elementwise, reduction, gather, and copy measurements use NumPy itself and do
not invoke BLAS merely because the NumPy build is OpenBLAS-linked.

The FP32 Llama stage Torch baselines are:

- `torch.nn.functional.silu` followed by multiplication for SwiGLU;
- `torch.nn.functional.rms_norm` for RMSNorm;
- the faster of Torch complex multiplication and Torch pairwise arithmetic for
  RoPE;
- `torch.softmax` for softmax.

The corresponding NumPy implementation is also timed for every stage. The
faster Torch or NumPy result is selected independently for each exact shape.
Hybrid stage candidates run that selected implementation on the CPU partition
concurrently with the QPU partition.

There are two deliberately distinct CPU meanings:

- Runtime CPU fallback in `matmul` and `PreparedFP32Linear` uses NumPy
  `matmul`, because Torch is not a core library dependency.
- Performance benchmarks measure both NumPy and Torch and compare against the
  faster result for that exact shape. The selected reference is recorded per
  row in the JSON and Markdown matrices.

## Current FP32 Projection Results

The latest complete sweeps found measured wins in ten exact
shape/placement classes. Aligned prefill shapes now execute directly from the
caller's source into its destination, avoiding the plan's padded staging copy:

| Shape | Projection | Implementation | CPU ms | Candidate ms | Speedup |
|---|---|---|---:|---:|---:|
| `16x1536x512` | down | QPU-only tiled GEMM | 1.750 OpenBLAS | 1.472 | 1.189x |
| `16x512x1536` | gate | QPU-only tiled GEMM | 1.516 OpenBLAS | 1.357 | 1.117x |
| `16x512x1536` | up | QPU-only tiled GEMM | 1.505 OpenBLAS | 1.342 | 1.121x |
| `16x512x32000` | LM head | QPU-only tiled GEMM | 32.774 Torch | 24.166 | 1.356x |
| `64x2816x1024` | down | QPU 16-row prefix + NumPy tail | 7.052 Torch | 6.165 | 1.144x |
| `64x1024x2816` | gate | QPU 16-row prefix + NumPy tail | 7.398 Torch | 6.083 | 1.216x |
| `64x1024x2816` | up | QPU 16-row prefix + NumPy tail | 7.572 Torch | 6.002 | 1.261x |
| `64x1024x32000` | LM head | QPU 16-row prefix + NumPy tail | 81.400 Torch | 66.777 | 1.219x |
| `64x1024x1024` | output | QPU 16-row prefix + NumPy tail | 3.344 OpenBLAS | 2.484 | 1.346x |
| `64x1024x1024` | query | QPU 16-row prefix + NumPy tail | 2.988 Torch | 2.502 | 1.194x |

All tested one-token FP32 GEMV projections remain slower than both CPU options,
so decode remains CPU by default. These measurements are steady-state
whole-operation times, including submission and result handling, not
raw-kernel-only claims.

For `prefill-h512-t16`, residual add, embedding, argmax, and KV append were all
exact but slower than their fastest CPU references. They remain available for
explicit evaluation and are not automatic placements.

## Current Eight-Case Llama Stage Sweep

The 2026-08-19 sweep retained 240 FP32 stage candidates:

| Operation | Candidates | Supported wins | Best observed speedup |
|---|---:|---:|---:|
| RMSNorm | 48 | 3 | 2.040× |
| RoPE | 72 | 0 | 0.803× |
| Softmax | 72 | 2 | 1.240× |
| SwiGLU | 48 | 6 | 2.300× |

These are exact-shape results on the current machine, not general hardware
claims. The vec4 kernels process four contiguous words per SIMD lane at widths
divisible by 64 and retain scalar fallbacks. SwiGLU and RoPE distribute uneven
block counts over all 12 QPUs. Softmax uses `wgs_per_sg=48`; the retained
settings for RMSNorm and RoPE remain 24 after hardware sweeps.

The strongest `prefill-h2048-t256` results are 1.821 ms QPU versus 4.190 ms
Torch for SwiGLU (2.300×), 0.619 ms QPU versus 1.264 ms NumPy for RMSNorm
(2.040×), and 2.841 ms QPU versus 3.523 ms Torch for softmax (1.240×). RoPE
improved from 10.023 ms to 0.958 ms with exact output, but Torch remains faster
at 0.769 ms. Cached uniform blocks for these vector kernels and FP32 GEMM
occupy 256 bytes so subsequent persistent allocations retain the tensor
alignment required for full vec4 TMU throughput. Maximum observed errors for
the winning QPU paths are 2.86e-6 for SwiGLU, 1.43e-6 for RMSNorm, and 2.98e-7
for softmax.

## Current FP32 Pooling Results

The prepared 2x2/stride-2 FP32 path caches its source-address metadata and
uses the largest QPU count that exactly divides the vector stream. This removes
the former 1-or-12-QPU dispatch cliff. Exact hardware results include:

| Input shape | Operation | CPU ms | QPU ms | Speedup |
|---|---|---:|---:|---:|
| `1x32x144x144` | max-pool | 1.146 NumPy | 0.866 | 1.323x |
| `1x48x128x128` | avg-pool | 1.742 NumPy | 1.059 | 1.645x |
| `1x48x128x128` | max-pool | 1.706 NumPy | 1.041 | 1.640x |
| `1x64x112x112` | max-pool | 1.397 NumPy | 1.281 | 1.091x |

All reported pool errors are zero. Average pooling remains CPU-preferred at
the tested `1x32x144x144` and `1x64x112x112` shapes, so these records must stay
shape-specific.

## Current FP32 Epilogue Results

The bias/ReLU family now has separate hardware coverage for bias-only,
ReLU-only, and fused bias+ReLU. Bias-bearing operations retain the 16x16 tiled
layout, while ReLU-only selects an exact-stripe contiguous four-word TMU path
with a scalar fallback. The no-bias uniform stream was also corrected so its
third word is the destination address rather than the unused bias slot.

| Shape | Operation | CPU ms | QPU ms | Speedup |
|---|---|---:|---:|---:|
| `128x5632` | bias+ReLU | 1.020 NumPy | 0.812 | 1.256x |
| `256x8192` | bias+ReLU | 3.162 NumPy | 2.285 | 1.383x |
| `64x32000` | bias+ReLU | 2.956 NumPy | 1.768 | 1.672x |
| `16x262128` | ReLU | 3.425 NumPy | 2.808 | 1.220x |
| `16x262128` | bias+ReLU | 10.048 NumPy | 5.234 | 1.920x |

All reported errors are zero. Bias-only remains below the 1.05x gate; its best
wide-row result is 1.022x. ReLU-only remains CPU-preferred at the three smaller
shapes but wins on the exact-stripe `16x262128` shape. Fused bias+ReLU wins at
all four retained shapes because it combines two CPU passes into one QPU
traversal.

## Current FP32 Min/Max Results

The contiguous min/max family now selects between its original single-word
TMU loop and a four-word TMU vector specialization. The vector path is used
only when the shape has at least six exact QPU stripes; other shapes retain the
generic scalar path.

| Length | Operation | CPU ms | QPU ms | Speedup |
|---|---|---:|---:|---:|
| `4194048` | minimum | 6.808 NumPy | 4.392 | 1.550x |
| `4194048` | maximum | 6.748 NumPy | 4.447 | 1.517x |

Both vectorized results have zero error. The nearby `4194240` shape does not
have enough exact four-word QPU stripes, falls back to the scalar path, and
measures 0.908x/0.909x. Promotion therefore remains shape-specific.

The same exact-stripe TMU layout also improves the other contiguous bandwidth
primitives:

| Length | Operation | CPU ms | QPU ms | Speedup |
|---|---|---:|---:|---:|
| `4194048` | copy | 3.465 NumPy | 3.000 | 1.155x |
| `4194048` | residual add | 6.590 NumPy | 4.675 | 1.410x |

Both errors are zero. These large-tensor wins do not promote the small
`16x64` KV append or `16x512` residual records, which remain CPU-preferred.

## Matmul and Attention Spot Checks

For raw square FP32 matmul, the fastest CPU backend changes with shape. At
64×64 Torch measured 0.017 ms versus 0.030 ms for OpenBLAS; QPU-only measured
0.240 ms. At 512×512 OpenBLAS measured 3.886 ms versus 6.151 ms for Torch; the
best heterogeneous split measured 4.294 ms and QPU-only measured 12.745 ms.
Neither accelerated candidate beats the fastest raw CPU backend.

For full staged SDPA, Torch wins at 64×64 (0.086 ms), while NumPy/OpenBLAS wins
at 512×512 (11.449 ms). QPU-only staged SDPA measured 1.117 ms and 37.886 ms,
respectively. The independently selectable CPU/QPU stage splits were also
slower. The retained records are `fp32-matmul-attention-s64.json` and
`fp32-matmul-attention-s512.json`.

## Reproduction

Run one FP32 stage matrix:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_llama_stages.py \
  --case prefill-h1024-t64 --warmup 5 --repeat 15 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/prefill-h1024-t64-stages.json
```

Run the FP32 projection matrix (Q, K, V, O, gate, up, down, and LM head):

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_fp32_linear.py \
  --case prefill-h1024-t64 --projection all --warmup 2 --repeat 7 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/prefill-h1024-t64-fp32-linear.json
```

Run residual, embedding, argmax, and KV-append matrices:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_llama_post_ops.py \
  --case prefill-h512-t16 --warmup 2 --repeat 7 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/prefill-h512-t16-post-ops.json
```

Run the prepared FP32 pool matrix:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_pool2d_fp32.py \
  --channels 64 --height 112 --width 112 --warmup 3 --repeat 21 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/fp32-pool2d-n1-c64-h112-w112.json
```

Run a prepared-output FP32 epilogue matrix:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_fp32_epilogues.py \
  --rows 128 --columns 5632 --warmup 3 --repeat 21 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/fp32-epilogues-r128-c5632.json
```

Run the FP32 min/max specialization benchmark:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_fp32_minmax.py \
  --length 4194048 --warmup 3 --repeat 15 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/fp32-minmax-n4194048.json
```

Run the FP32 copy and residual bandwidth benchmarks:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_fp32_copy.py \
  --length 4194048 --warmup 3 --repeat 15 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/fp32-copy-n4194048.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_fp32_residual.py \
  --length 4194048 --warmup 3 --repeat 15 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/fp32-residual-n4194048.json
```

Run the square matmul and full staged-attention matrix:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_matrix.py \
  --size 512 --warmup 2 --repeat 7 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/fp32-matmul-attention-s512.json
```

Regenerate the consolidated FP32 matrix:

```bash
.venv/bin/python scripts/fp32_evaluation_matrix.py \
  experiment_logs/20260819-qpu-xla-kernel-suite/*-fp32-linear.json \
  experiment_logs/20260819-qpu-xla-kernel-suite/*-post-ops.json \
  experiment_logs/20260819-qpu-xla-kernel-suite/*-stages.json \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/FP32_EVALUATION_MATRIX.md
```

Regenerate the compact cross-backend latency matrix:

```bash
.venv/bin/python scripts/kernel_backend_latency_matrix.py \
  --logs experiment_logs/20260819-qpu-xla-kernel-suite \
  --yolo-logs experiment_logs/w8a8-evaluation-20260819 \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/KERNEL_BACKEND_LATENCY_MATRIX.md
```

Regenerate the scoreboard:

```bash
.venv/bin/python scripts/kernel_scoreboard.py \
  --logs experiment_logs/20260819-qpu-xla-kernel-suite \
  --output experiment_logs/20260819-qpu-xla-kernel-suite/KERNEL_SCOREBOARD.md
```
