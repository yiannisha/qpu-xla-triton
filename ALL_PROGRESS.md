# Complete QPU worklog and findings

Date of audit: 2026-08-28

Primary activity window: 2026-08-13 through 2026-08-25

Repository: `qpu-xla-triton` / `py-videocore7` lineage

## Scope and audit method

This document collects the notable work, experiments, implementations, and
negative results visible in the repository history and reports, with emphasis
on the last two weeks. It covers:

- `main`;
- `origin/feat/smolvla-qpu-runtime`;
- `origin/qpu/llama-cpp-pipeline-acceleration`;
- all narrative reports and the relevant structured benchmark archives on
  those refs;
- the older custom-kernel work that the two-week runtime and integration work
  builds on;
- the inherited low-level history, summarized separately from the recent work.

As requested, `origin/feat/llama-cpp-ggml-qpu-ops` was not reviewed as a
separate branch. Its commits are nevertheless part of `main` through merge
commit `2da3b62`, so the state and evidence that are actually present on
`main` are included.

The audit refreshed the remote refs and inspected the history graph: 126
commits are reachable from `main`, and 135 unique commits are reachable across
the three included tips. There are 20 commits dated August 13-25 across those
tips. The two requested feature branches both fork from `8936307`: SmolVLA
adds three unique commits and the pipeline branch adds six unique commits.

Evidence is classified as follows:

1. **Current contract:** documented code behavior with current tests.
2. **Retained result:** a result that passed the experiment's declared
   environmental, correctness, sample-count, and promotion rules.
3. **Diagnostic or screen:** useful engineering evidence, but not a promotion
   or publication-quality performance claim.
4. **Historical result:** an older snapshot that may have been corrected or
   superseded later.
5. **Implementation without production measurement:** code exists, but the
   required model, replay, hardware run, or clean measurement is absent.

This hierarchy matters. Several local kernels win in isolation while their
enclosing layer or request loses. Several April paths were fast but numerically
wrong. Conversely, some current kernels are exact and useful even though they
are slower and deliberately unavailable to automatic placement.

## Executive summary

The work transformed a low-level Raspberry Pi 5 VideoCore VII assembler and
DRM driver into a heterogeneous ML systems prototype with four increasingly
ambitious layers:

1. handwritten QPU kernels and persistent executor experiments;
2. an importable `qpu_xla` runtime with shared CPU/QPU-visible memory, queues,
   events, operators, placement, tracing, a constrained DSL, and autotuning;
3. calibrated FP32 and W8A8 model-oriented kernels plus TinyLlama and video
   scaffolding;
4. native `llama.cpp` and full SmolVLA integration experiments.

The strongest positive findings are:

- Large, contiguous, exact-stripe FP32 kernels can beat four-thread CPU
  references: copy, residual, min/max, several pool shapes, wide fused
  bias+ReLU, and selected Llama prefill stages.
- Exact-shape FP32 prefill linear placement can win either as QPU-only work or
  as a small QPU row prefix concurrent with a CPU tail. Ten measured
  projection/placement classes clear the 1.05x gate.
- Native signed `v8dot` makes the packed W8A8 prefill family genuinely useful.
  Selected projections and one-layer TinyLlama-like compositions beat the
  fastest measured FP32 CPU reference, with exact dynamic-W8A8 accumulation
  and conservative exact-shape dispatch.
- A true native SmolVLA graph was implemented from RGB input through a 50x6
  action chunk, including real QPU and concurrent CPU/QPU stage paths, strict
  artifact/replay provenance, and a memory preflight suitable for an 8 GiB Pi.
- The llama.cpp work built far more than a kernel demo: a native C V3D runtime,
  persistent quantized operators, generated/hash-validated programs, GGML
  backends and inline hooks, failure recovery, telemetry, exact-token
  evaluators, graph profiles, retained paired campaigns, and several fused or
  persistent architectural probes.
- The pipeline work found a potentially interesting **component** boundary:
  channel-partitioned complete FFN islands measured 1.10-1.19x weighted
  FFN-region estimates at larger M, with individual screens as high as 1.407x.
  This is the most plausible future llama.cpp direction, but it is not yet a
  production hook or end-to-end win.

The strongest negative findings are equally important:

- The older naive/fast GEMM, packed INT16, tiled conv, attention, MLP, and
  LeNet benchmark snapshot contained very large errors, NaNs, or Infs. Timing
  from those paths was not valid acceleration evidence.
- QPU-only raw FP32 square GEMM and full staged SDPA lose to the fastest CPU
  backend at the tested 64 and 512 sizes.
- Single-token decode is a poor fit for the current kernels. FP32 GEMV, W8A8
  GEMV, RMSNorm, RoPE, softmax, and SwiGLU mostly lose because fixed dispatch
  and boundary cost dominate.
- GEMM-backed W8A8 convolution is exact but badly loses to Torch native conv:
  0.13x for 1x1, 0.064x for 3x3, and 0.012x for depthwise in the retained
  summary. Direct spatial kernels are required.
- Every main-branch native llama.cpp Q4_0/Q4_K/Q6_K/Q8_0 and fused-attention
  candidate lost to the exact pinned `ggml-cpu` node.
- Exact inline GEGLU won at selected one-thread node shapes but the paired
  request was 0.969x. Row-hybrid GEGLU was below 1.0x at every tested M.
- The exact `ffn_up` overlap reached all 35 Gemma layers and hid most QPU time,
  but full requests measured only 1.006x, 1.011x, and 0.986x across the three
  suffix sizes.
- Approximate column-W8 reached retained full-model execution with 630
  attested layer dispatches, but measured only 0.9996x at M=129 and 1.0161x at
  M=257; both confidence intervals crossed 1.0 and neither cleared 1.05x.
- Batched fused attention, the current QPU down projection, M=1 MTP drafting,
  and persistent launch collapse were all decisively insufficient.
- No llama.cpp QPU placement is enabled by default and no end-to-end llama.cpp
  acceleration claim is currently supported.
- SmolVLA has no production-size benchmark result in the repo because the
  checkpoint and robot replay are intentionally absent. Its implementation is
  complete enough to benchmark, but its production latency and upstream
  action-quality result remain open.

## Repository and branch timeline

### Inherited low-level history

The January-May 2025 history established the original `py-videocore7`
foundation:

- package initialization, typing, Ruff, and module naming;
- assembler and instruction tests for ALU, dual issue, branches, labels,
  conditions, signals, rotate/shuffle/broadcast, replicated and quad
  registers;
- TMU load/store, prefetch, cache/atomic operations, UNIFA, and rectangular
  strided reads;
- compute payload/workgroup introspection, multithread submission, 48-thread
  tests, and workgroup tests;
- basic `scopy` and `sgemm` examples, including pretransformed/strided TMU
  loading improvements;
- DRM/V3D register access, performance counters, buffer-object allocation,
  CSD submission, and packaging of the low-level extension;
- later assembler safety fixes, raw-instruction export, label-position export,
  and documentation/device-path corrections.

The January 2026 commits fixed CSD workgroup sizing by adding explicit local
invocation information and changed the default workgroup to `(1,1,1)`.

These are inherited platform capabilities, not new two-week ML claims, but
they are the substrate used by every recent kernel.

### Earlier custom ML kernel commits

The March/April 2026 feature sequence added the first ML-oriented stack:

| Commit | Work |
|---|---|
| `b7535e5` | packed INT16-input GEMM experiment |
| `769333d` | faster FP32 SGEMM variants |
| `849aace` | elementwise min/max kernels |
| `df49f76` | average/max pooling |
| `e7b67af` | tiled GEMM-backed Conv2D |
| `fd8504e` | tiled two-layer MLP |
| `d19e764` | tiled attention core |
| `5c13e97` | persistent end-to-end LeNet-5-style pipeline |

### Last-two-week commits

| Date | Commit | Branch-visible contribution |
|---|---|---|
| Aug 13 | `1fad3d1` | QPU-XLA runtime, tests, reports, DSL/autotune, TinyLlama/video scaffolding, legacy experiment archive |
| Aug 19 | `0e7612c` | broad FP32/INT8 operator and kernel expansion, scheduler/candidate work |
| Aug 19 | `feea049` | calibrated TinyLlama W8A8 runtime and benchmarks |
| Aug 19 | `39c1689` | native llama.cpp QPU runtime, generated programs, evaluators, graph profiles |
| Aug 19 | `2d54e73` | candidate matrices, scoreboards, and benchmark tooling |
| Aug 19 | `0acadad` | archived benchmark evidence |
| Aug 19 | `11dfb00` | reproducible graph fixtures |
| Aug 21 | `bb706e1` | native GGML Q4_K, Q6_K, Q8_0, attention and fixture/operator candidates |
| Aug 21 | `7eb8830` | expanded llama.cpp profiling, replay, safety, and evaluation tooling |
| Aug 21 | `8936307` | recorded llama.cpp evaluation results |
| Aug 21 | `2da3b62` | merged the GGML operator work to `main` |
| Aug 25 | `001aec5` | SmolVLA-specific heterogeneous QPU operators |
| Aug 25 | `a267b79` | full native SmolVLA CPU/QPU runtime, conversion, replay, benchmark |
| Aug 25 | `6828971` | SmolVLA CPU/hardware tests and workflow documentation |
| Aug 25 | `77fbe9e` | arbitrary-M Q4, batched attention, GEGLU prefill kernel candidates |
| Aug 25 | `077d6ce` | GGML backend and inline prefill integration |
| Aug 25 | `cd6b15e` | hybrid, fused FFN, agentic-prefill evaluation |
| Aug 25 | `88bc303` | FFN-island, MTP inline, large-M, additional evaluation paths |
| Aug 25 | `a472092` | persistent exact multiphase Q4 projection prototype |
| Aug 25 | `6993b06` | consolidated expanded acceleration results and raw records |

## How the QPU is used

The hardware target is the Raspberry Pi 5 BCM2712 VideoCore VII. The inherited
README models it as three slices, four QPUs per slice, four physical cores per
QPU, dual issue at a nominal 800 MHz, for a theoretical 76.8 GFLOP/s. In this
repository, labels such as “12 QPUs” or `--num-qpus 12` refer to the twelve QPU
cores used by the software dispatch, not twelve GPUs.

The low-level flow is:

1. A Python function decorated with `@qpu` emits VideoCore instructions using
   injected ALU operations, registers, labels, branches, signals, and raw
   encodings.
2. `Driver.program()` assembles the function and writes the 64-bit instruction
   words into a DRM V3D buffer object.
3. `Driver.alloc()` creates NumPy arrays backed by mapped BO memory and exposes
   their GPU-visible addresses. Host and QPU can therefore refer to the same
   allocation.
4. Uniform streams contain tensor addresses, dimensions, strides, tile counts,
   and other launch metadata.
5. `Driver.execute()` submits a DRM V3D compute-shader dispatch with workgroup,
   local-invocation, workgroups-per-supergroup, thread, NaN, segment, and
   threading controls, then waits for the BO through the dispatcher context.
6. QPU kernels use 16 SIMD lanes, TMU memory operations, ALU dual issue, and
   explicit synchronization/barrier patterns. `smul24` supplies signed
   24-bit integer multiplication; `v8dot` supplies signed packed-byte dot
   products for W8A8.

The original `Driver` is synchronous and owns one linear code/data arena. The
new `qpu_xla` layer retains that hardware backend but adds application-level
asynchrony, ownership, dependencies, placement, and reusable plans.

## Legacy custom-kernel findings

The archived April campaign attempted 20 benchmark runs. Eighteen completed;
the two recorded failures were the GPU-clock performance counter opening
`/dev/mem` without the required permission and the default twelve-QPU min/max
length violating its chunk-size contract. A later corrected-core rerun kept
the invalid default as a negative fixture and added valid one- and twelve-core
cases.

### Memory copy and basic parallel scaling

The April full rerun copied 24 Mi `uint32` elements at:

- CPU: 4,746 MB/s;
- one QPU core: 1,756 MB/s;
- twelve QPU cores: 4,171 MB/s.

This established the recurring pattern that enough exact parallel stripes can
approach CPU bandwidth, while one QPU core is not competitive.

### Naive and fast FP32 SGEMM

The naive size sweep reached roughly 21-23 GFLOP/s from 512 through 2048, but
its April output was already badly wrong at smaller sizes and became NaN/Inf at
large sizes. The fast 1024 variant improved 18.77 to 22.68 GFLOP/s (1.208x),
but produced hundreds of NaN/Inf values. That speedup is not a valid numerical
result.

The specialized small-batch kernel was essentially identical to the generic
batched implementation: measured speedups ranged around 0.991-1.004x. It did
not provide a material specialization win, and both paths retained large
absolute errors in that snapshot.

### INT32 and packed INT16 GEMM

The historical INT32 QPU timing beat the NumPy integer baseline by large
factors, reaching 16.7-20.5 Gop/s at 512-768, but the April maximum errors were
approximately 0.88-1.62 billion. The packed INT16 experiment reached about
21.7 Gop/s at 1024 but had 62-101 million maximum errors. These results
motivated the later signed-24-bit contracts, differential tests, and
quarantine rules.

The **current** public Conv2D-related INT16 contract is different: inputs and
weights are widened on the host and passed through the validated INT32
microkernel, producing exact INT32 accumulation for the tested shape. The
packed assembly entry point remains present only as a quarantined experiment.

### Min/max and pooling

The early min/max implementation showed a strong twelve-core scaling effect.
At about four million elements, one core was roughly 0.92-1.06 GiB/s, while a
valid twelve-core length reached roughly 5.74-6.60 GiB/s. The script's default
length was invalid for the twelve-core FP32 chunk (not divisible by 192), an
important benchmark/default bug that was preserved in the report.

For 2x2/stride-2 pooling at `1x32x144x144`, twelve QPU cores were close to CPU
for FP32 max/average and substantially faster than the scalar CPU integer
average implementations. One core lost on nearly every case. Integer average
pooling explicitly uses truncation toward zero to match the selected Torch
contract.

### GEMM-backed Conv2D

The legacy Conv2D design is `im2col + OIHW reshape + padded GEMM + NCHW
reshape`, not a direct convolution microkernel. The common 1x1/stride-1/no-pad
packaged path avoids spatial-window duplication but is still GEMM-backed.

The April live rerun reported attractive execute-only numbers but large errors:
37.75 FP32, 670,088 INT32, and 3,993,623 packed INT16. Later work therefore
kept the packed INT16 path quarantined and added exact hardware differential
coverage for the packaged paths. The current guide reports FP32 and INT32 as
good for tested shapes and public widened INT16 as exact, but warns against
generalizing beyond those shapes.

Benchmark taxonomy was corrected during this work:

- `numpy` means a lowered `im2col + dot` CPU baseline, not native convolution;
- Torch `conv2d` is the optimized native CPU comparison;
- host prep, cached total, execute-only, and prep+cached total are separate;
- cold-start setup must not be mixed with steady-state execution.

### Tiled MLP, attention, and LeNet

The reusable substrate produced persistent executors for:

- `Linear -> ReLU -> Linear`;
- the unnormalized attention core `(Q @ K.T) @ V`;
- a LeNet-5-style `Conv/Pool/FC` pipeline with QPU-side gather/lowering,
  pooling, persistent weights, and stage timing.

Architecturally, LeNet was an important full-pipeline milestone: steady state
performed no CPU compute in the QPU path and reused device buffers and
metadata. Twelve cores reduced the pipeline from roughly 31.5 ms to 10.1 ms.
Numerically, the April snapshot was not usable: FP32 LeNet produced NaN,
INT32 errors exceeded 81 million, tiled attention errors ranged from 19.6 to
78 million across stages/dtypes, and tiled MLP produced NaN or billion-scale
errors. These are valuable systems prototypes and failure records, not valid
model-acceleration claims.

## The QPU-XLA runtime created in the last two weeks

### Runtime and memory model

`src/qpu_xla` introduces an installable runtime above the assembler/driver:

- `Device`, `Buffer`, `Tensor`, and `AccessMode` wrap host-mapped,
  QPU-visible allocations;
- `Queue` has one in-order worker and can submit QPU kernels or declared CPU
  host tasks;
- `Event` carries dependencies, timestamps, errors, and cross-queue joins;
- buffer access declarations include read/write modes and byte ranges;
- trace data can be exported in Chrome trace format;
- `Device.fake()` provides a CPU-only contract backend;
- the public API is asynchronous even though the underlying driver wait is
  synchronous inside the queue worker.

This is a heterogeneous operator runtime, not a complete XLA implementation.

### Placement and calibration

The scheduler represents CPU, QPU, AUTO, and explicit hybrid candidates. It
stores exact-shape median samples in serializable cost models. Hybrid matmul
assigns a tile-compatible output-row prefix to QPU and a disjoint tail to
NumPy on a second queue, allowing useful overlap without merging partial sums.
Single-row projections can instead partition output columns.

AUTO is conservative: an unmeasured shape remains CPU. Later candidate
registries strengthen the rule further by requiring current source hashes,
exact shape/dtype/layout/partition, numerical validity, and at least 1.05x
whole-operation speedup over the fastest measured CPU reference.

### Compiler DSL and autotuning

The new compiler package captures a constrained Python surface into typed IR,
checks allowed operations and source locations, and lowers a small primitive
set toward VC7. Primitives include load/store, dot, reductions, select,
program IDs, ranges, and barriers. It is intentionally narrow and should not
be described as a general Python-to-QPU or XLA compiler.

The autotuner validates candidates, runs a CPU reference, optionally performs
hardware differentials, times repeated candidates, and persists results. This
is the beginning of safe kernel exploration rather than an online production
autotuner.

### Video and TinyLlama scaffolding

The video layer defines NV12 frame ownership/contracts, NV12-to-NCHW-FP32
preprocessing, a frame ring, and a headless runner. It establishes pipeline
boundaries but is not a complete camera application.

The TinyLlama layer includes checkpoint/config/artifact loading, a transparent
CPU reference, causal grouped-query prefill and incremental decode, persistent
KV caches, quantization, generation, sampling, calibrated packed projections,
and a mixed CPU/QPU runtime. Only projections with an exact promoted record
are packed at construction; CPU-only projections retain FP32 weights.

## Packaged kernel and operator inventory

The current `main` kernel family includes the following distinct work. Some
files expose multiple scalar/vector or dtype variants behind one descriptor.

| Family | Implemented kernel/operator | Contract or important boundary |
|---|---|---|
| Copy | `vc7.copy_words` | contiguous words; scalar or four-word TMU; exact stripe distribution |
| FP32 GEMM | `vc7.tiled_fp32_gemm` | logical 16x16 output tile, K tile 4 |
| INT32 GEMM | `vc7.tiled_int32_gemm` | 16x16x4; signed-24-bit `smul24` inputs |
| FP32 GEMV | `vc7.fp32_gemv` | single-row decode candidate |
| W8A8 GEMM | `vc7.tiled_w8a8_gemm` | packed four INT8/word, 16x16x16, exact INT32 `v8dot` accumulation |
| Fused W8A8 | `vc7.tiled_w8a8_gemm_dequantize` | row/column FP32 scaling fused into store |
| W8A8 GEMV | `vc7.w8a8_gemv` | single-row packed decode candidate |
| W8A8 epilogue | `vc7.w8a8_dequantize` | separate INT32-to-FP32 tiled scaling |
| Bias/ReLU | FP32/INT32 bias, ReLU, bias+ReLU | fused wide one-pass epilogue; tiled bias layout and contiguous ReLU path |
| Activations | `vc7.swiglu_fp32` | fused SiLU(gate)*up, vec4 and 12-QPU balancing |
| Normalization | `vc7.rms_norm_fp32` | row-parallel, scalar/vec4 |
| Position | `vc7.rope_fp32` | cached trig tables, scalar/pairwise vec4 |
| Reduction | `vc7.softmax_fp32` | stable row softmax |
| Residual | `vc7.residual_add_fp32` | scalar/vec4 contiguous add |
| Elementwise | `vc7.minimum_words`, `vc7.maximum_words` | FP32/INT32, exact-stripe vec4 or generic scalar |
| Pooling | FP32/INT32 max/average 2D | prepared address metadata, largest exact QPU divisor |
| Lookup | `vc7.embedding_lookup_fp32` | token/address-table gather |
| Sampling | `vc7.argmax_fp32` | exact first-index tie behavior |
| Quantized conv | `vc7.depthwise_w8a8_3x3` plus GEMM-backed plans | experimental; direct spatial performance not achieved |
| GGML Q4_0 | `vc7.ggml_q4_0_q8_0_linear` | native 18-byte Q4_0 weights and 34-byte Q8_0 activations |
| GGML Q4_K | M=4 Q4_KxQ8_K | exact native-format experiment |
| GGML Q6_K | M=4 Q6_KxQ8_K | exact native-format experiment |
| GGML Q8_0 | M=4 Q8_0xQ8_0 | exact native-format experiment |
| GGML attention | Gemma F16 M=1 fused attention | exact supported subset, online softmax, no score materialization |

The packaged operator surface composes these into matmul/linear, FP32 and
INT32 Conv2D, W8A8 dense/grouped Conv2D, FP32/INT32 attention cores, mixed
QPU-GEMM/CPU-softmax SDPA, INT32 MLP plans, FP32 MLP helpers, pooling,
embedding, residual, RoPE, sampling, and elementwise operations. Persistent
INT32 Conv2D, MLP, and attention plans keep transformed weights and workspaces.

## Modern FP32 result matrix

All values in this section are steady-state whole-operation medians from the
August kernel-suite reports, compared independently against the faster NumPy
or Torch CPU reference for the exact shape.

### Projection wins

Ten exact projection/placement classes cleared the 1.05x gate:

| Shape and operation | Placement | CPU ms | Candidate ms | Speedup |
|---|---|---:|---:|---:|
| `16x1536x512` down | QPU only | 1.750 | 1.472 | 1.189x |
| `16x512x1536` gate | QPU only | 1.516 | 1.357 | 1.117x |
| `16x512x1536` up | QPU only | 1.505 | 1.342 | 1.121x |
| `16x512x32000` LM head | QPU only | 32.774 | 24.166 | 1.356x |
| `64x2816x1024` down | 16/64 QPU rows | 7.052 | 6.165 | 1.144x |
| `64x1024x2816` gate | 16/64 QPU rows | 7.398 | 6.083 | 1.216x |
| `64x1024x2816` up | 16/64 QPU rows | 7.572 | 6.002 | 1.261x |
| `64x1024x32000` LM head | 16/64 QPU rows | 81.400 | 66.777 | 1.219x |
| `64x1024x1024` output | 16/64 QPU rows | 3.344 | 2.484 | 1.346x |
| `64x1024x1024` query | 16/64 QPU rows | 2.988 | 2.502 | 1.194x |

The key placement lesson is that “more QPU” is not monotonically better. At
64 rows, a small 16-row QPU prefix can overlap well with an optimized CPU tail.
Narrow K/V projections still lose because the fixed QPU boundary overwhelms
their smaller compute volume.

### Llama stage wins and losses

The eight-case stage sweep retained 240 candidates:

- RMSNorm: 3 supported wins out of 48, best 2.040x;
- RoPE: 0 out of 72, best 0.803x despite a 10.2x improvement over the earlier
  QPU kernel;
- softmax: 2 out of 72, best 1.240x;
- SwiGLU: 6 out of 48, best 2.300x.

The strongest `prefill-h2048-t256` results were:

- SwiGLU: 1.821 ms QPU versus 4.190 ms Torch, 2.300x, max error 2.86e-6;
- RMSNorm: 0.619 ms QPU versus 1.264 ms NumPy, 2.040x, max error 1.43e-6;
- softmax: 2.841 ms QPU versus 3.523 ms Torch, 1.240x, max error 2.98e-7;
- RoPE: improved to 0.958 ms QPU but still lost to 0.769 ms Torch.

Every tested one-token FP32 GEMV projection and decode-stage primitive lost to
the fastest CPU backend. Example `decode-h2048-c512` projection acceleration
ranged only about 0.089-0.309x, and decode stage operations were generally
0.08-0.30x. Decode remains CPU by default.

### Bandwidth, pooling, and epilogue wins

Exact large-tensor wins include:

- copy length 4,194,048: 3.000 vs 3.465 ms, 1.155x;
- residual add: 4.675 vs 6.590 ms, 1.410x;
- minimum: 4.392 vs 6.808 ms, 1.550x;
- maximum: 4.447 vs 6.748 ms, 1.517x.

A nearby min/max length, 4,194,240, could not use enough exact four-word
stripes and fell to about 0.908-0.909x. This is strong evidence that alignment
and stripe geometry—not just element count—determine viability.

Supported FP32 pool wins were 1.323x for `1x32x144x144` max-pool, 1.640-1.645x
for both operations at `1x48x128x128`, and 1.091x for `1x64x112x112`
max-pool. Average pooling lost to Torch at the first and last shapes.

Fused bias+ReLU won on all four retained wide shapes, from 1.256x to 1.920x,
because it replaced two CPU passes with one QPU traversal. Bias-only did not
clear the gate; ReLU-only won only on the exact-stripe `16x262128` shape.

### FP32 GEMM and SDPA losses

At 64 cubed, QPU-only matmul was 0.240 ms against 0.017 ms Torch. At 512
cubed, QPU-only was 12.745 ms and the best hybrid was 4.294 ms against 3.886
ms OpenBLAS. Neither was promoted.

Full staged SDPA was also slower: 1.117 ms QPU versus 0.086 ms Torch at size
64, and 37.886 ms QPU versus 11.449 ms OpenBLAS at size 512. Independently
selecting score, softmax, and value placements did not rescue the full
operator.

## Packed W8A8 and TinyLlama findings

### What was built

Weights are quantized per output channel, activations dynamically per input
row, packed four signed bytes per `uint32`, and accumulated exactly in INT32
with `v8dot`. `PreparedW8A8Linear` owns padded packed weights and workspaces.
The fused kernel applies row and output scales while storing FP32. Row and
output-column hybrids are mutually exclusive; grouped output splits require
16 aligned outputs per group, and depthwise output splits are rejected.

Every W8A8 assembly path has a hardware differential test. The host checks
worst-case INT32 accumulation before launch. AUTO requires an exact
`supported-win` record; a same-quantized-contract win alone is insufficient.

### Positive results

The current summary reports promoted fused prefill projection ranges of:

- 1.21-1.68x at 64x1024;
- 1.09-1.30x at 128x2048;
- 3.10-5.32x at 16x4096.

The one-layer mixed runtime measured 2.86x, 1.54x, and 1.16x over the FP32
NumPy reference for hidden sizes 512, 1024, and 2048 in the tuning cases. The
legacy merged registry retains 48 evaluated Llama stage records but exposes
only 17 prefill winners through AUTO.

### Negative and demoted results

The 256-token holdout measured 0.83x at the full one-layer boundary, so a
standalone SwiGLU win was demoted. This is an explicit example of an operator
win being rejected by a larger composition test.

Persistent-cache decode was exact against the calibrated CPU oracle but only:

- 0.60x at cache 512;
- 0.48x at cache 2048;
- 0.38x for hidden 3072/cache 4096.

AUTO therefore uses CPU for all measured decode shapes.

The YOLO candidates proved that packed GEMM correctness does not solve
convolution layout and boundary cost:

- p3 1x1: 0.13x Torch native FP32 conv;
- p3 3x3: 0.064x;
- p3 depthwise: 0.012x.

All remain explicit research candidates. Direct 1x1, 3x3, and depthwise
spatial kernels are the real next step.

## Native llama.cpp work on `main`

### Reproducible integration boundary

The integration pins llama.cpp commit
`91d2fc387529940230555abd297a8b5e99737d3f` (build 10073) and records model
SHA-256 values for Gemma base, Gemma MTP drafter, and Qwen. Generated program
artifacts carry source, binary, uniform, and launch hashes.

The out-of-tree C runtime implements:

- V3D render-node discovery and capability checks;
- BO allocation/mapping/GPU-address lookup;
- program validation, persistent program/uniform buffers, submission, wait,
  timeout, and cleanup;
- persistent selected weights, activation staging, output staging, and
  prepared Q4_0, Q4_K, Q6_K, and Q8_0 ABIs;
- per-context locking and injected failures at device, allocation, hash,
  submit, and wait boundaries;
- staging output so a failed launch cannot expose partial model state.

CPU baselines and CPU halves of hybrids call the same exported optimized
`ggml-cpu` block-dot path as the pinned build. The operator evaluator extracts
bounded fixtures from real GGUF manifests and scans every aligned output-column
partition.

The checked-in Pi result records 354 passing Python tests, all 11 native CTest
targets, and seven VideoCore VII hardware tests covering numerical smoke and
failure recovery. These validate the integration machinery; they do not turn
the losing latency rows below into promoted candidates.

### Graph profiling

Serialized callback profiles are diagnostic because synchronizing every node
changes execution. They nevertheless identify the right families.

For Gemma batch 1/context 0, the serialized shares were:

- FFN gate/up linear 43.1%;
- FFN down 20.4%;
- LM head 20.0%;
- attention linear 5.9%;
- everything else individually below the 5% elimination gate.

For batch 4, FFN gate/up was 38.4%, LM head 19.1%, FFN down 18.7%,
elementwise 8.4%, and attention linear 6.1%. This justified prioritizing
projection and FFN regions instead of isolated RoPE/norm work.

### Native quantized projection results

All tested outputs matched the exact CPU_REPACK reference within the declared
FP32 tolerance, but every QPU-only and hybrid path lost:

| Candidate | Shape | CPU ms | QPU ms | Best non-CPU |
|---|---:|---:|---:|---:|
| Gemma drafter Q4_0 LM head | `1x256x262144` | 2.981 | 37.762 | 3.990 at 1/12 QPU |
| same, M=4 | `4x256x262144` | 5.232 | 39.703 | 12.857 at 1/4 QPU |
| Qwen Q4_K FFN gate | `4x2560x9216` | 1.174 | 19.648 | 5.028 at 1/6 QPU |
| Qwen Q6_K FFN down | `4x9216x2560` | 1.811 | 73.232 | 15.245 at 0.08125 QPU |
| Qwen Q8_0 SSM output | `4x4096x2560` | 0.839 | 10.192 | 3.091 at 0.08125 QPU |

The captured Gemma drafter LM-head replay proved production-graph
correctness: CPU_REPACK was bitwise equal to the capture, QPU max error was
5.72e-6, and greedy argmax was unchanged. It remained correctness-only
because the graph callback serialized execution.

### Fused attention result

The fused M=1 Gemma subset accepts FP32 query, native FP16 K/V/mask, performs
online softmax, and emits FP32 without scores. It supports 256-wide heads, one
KV head, and no ALiBi, softcap, or sinks.

It measured only 0.094x, 0.100x, 0.124x, and 0.187x CPU for contexts 256,
512, 2048, and 4096. At context 4096, even one QPU query head took about 8.53
ms while all eight CPU heads took about 1.74 ms, proving that a head hybrid
could not win even with ideal overlap.

### End-to-end evaluation infrastructure and result

The runner separates startup, prefill, request wall time, prompt evaluation,
decode, speculative acceptance, and MTP cycles. It supports exact native token
arrays and a deterministic structured tool-call fixture. It hashes cases,
models, candidates, and model-visible workload semantics, records RSS/swap,
thermal/throttle/governor state, raw output tokens, and native dispatch
telemetry.

Promotion requires at least five independent retained sessions on each side,
performance governor, zero swap/current throttling/competing server, identical
greedy semantics, exact candidate hashes and shape, positive matching native
telemetry, at least 1.05x median request speedup, and bootstrap 95% lower bound
above 1.0.

The main report contained 113 operator rows, four attention diagnostics, four
end-to-end CPU smoke rows, and 150 planned cases, but zero provisional
operator wins, zero retained non-CPU end-to-end rows, and zero promoted
configurations. The sessions used `ondemand`, full zram, and a competing Qwen
server, so their timing was rejected. Qwen production graph coverage and a
Gemma vision projector/image were also absent.

## SmolVLA native runtime branch

Branch: `origin/feat/smolvla-qpu-runtime`

Unique commits: `001aec5`, `a267b79`, `6828971`

This branch turns the generic runtime into a model-specific executable graph,
not merely a collection of disconnected operators. It pins the upstream
LeRobot source at commit `8b256a6c0d4769c3cc3e7e98f04940126398a391` and
records checkpoint provenance at
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`. The audited model has
450,046,176 parameters and runs from three RGB
camera frames to one `50x6` action chunk.

### Reconstructed model and preprocessing contract

The branch records the complete inference structure and several easy-to-miss
compatibility details:

- each of the three `256x256` RGB views is resized/padded with the upstream
  top-left convention to `512x512`, then normalized;
- the SigLIP vision tower has 12 layers at width 768;
- each camera produces 1,024 patch tokens and the pixel-shuffle stage reduces
  these to 64 tokens, giving 192 visual tokens across three cameras;
- the VLM path is width 960 with 16 layers, and the constructed prefix length
  is 241 tokens;
- the action expert is width 720 with 16 layers;
- denoising performs ten Euler steps over a `50x32` latent and returns the
  first six channels as the `50x6` action chunk;
- compatibility follows the *executed* upstream RoPE theta of 10,000 even
  though a configuration value says 100,000. Treating the config value as the
  runtime truth would silently change the model.

The native artifact converter streams tensors instead of materializing a
second full model copy, emits a manifest with hashes and tensor metadata, and
supports mmap-backed loading. Dynamic W8 activation quantization uses per-row
scales and the same signed packed-byte convention as the validated W8A8
kernels.

### Operations implemented on the QPU

The SmolVLA work adds or composes real QPU paths for:

- image resize/pad/normalize;
- vision patch gathering and patch projection;
- FP32 and W8A8 linear projections;
- RMS/layer normalization, activations, affine operations, and residual adds;
- token embedding;
- split-half rotary embedding;
- grouped attention;
- pixel shuffle and token rearrangement.

The runtime can select upstream CPU FP32, native CPU FP32, QPU FP32, hybrid
FP32, native CPU W8, QPU W8, hybrid W8, or AUTO execution. CPU and QPU work is
scheduled through declared tensor dependencies so independent partitions can
overlap.

Some boundaries intentionally remain on the CPU: mask construction, timestep
table work, the Euler update, final action slicing, and a host-visible mask
boundary between attention score and softmax. Thus “QPU mode” means a native
heterogeneous graph with substantial QPU execution, not that every scalar or
control operation has been forced onto the accelerator.

### Correctness, placement, and memory gates

The branch does not promote a fast isolated stage by itself. AUTO requires an
exact-shape `supported-win` record for the stage, a win for the enclosing
block, and a passing full-replay result with the same placement. Its numerical
gates are:

- upstream FP32 versus native FP32: NRMSE at most `1e-3` and cosine similarity
  at least `0.9999`;
- QPU W8 versus native CPU W8: maximum absolute error at most `1e-4`;
- W8 versus FP32: NRMSE at most `0.10` and cosine similarity at least `0.99`.

The memory preflight is as important as the kernels on an 8 GiB Pi. The report
projects approximately:

| Mode | Artifact | QPU arena | Projected peak |
|---|---:|---:|---:|
| FP32 | 1.677 GiB | 0.906 GiB | 3.259 GiB |
| W8A8 | 0.687 GiB | 0.625 GiB | 1.989 GiB |

The launcher also applies a 6 GiB process limit before driver initialization,
so a bad allocation plan fails predictably instead of destabilizing the Pi.
The report explicitly rejects a much larger Pi0.5-style 3.5-4B parameter
baseline: FP32 weights alone would exceed about 13 GiB and are not a practical
8 GiB target.

### What is proven and what remains unmeasured

The branch contains a full native implementation, converter, replay format,
benchmark harness, provenance checks, memory planner, program exporter, and
CPU/hardware tests. The checked-in branch report records 376 Python tests, 18
native tests, 12 hardware tests, and 16 exported programs in its target Pi
environment.

It does **not** contain the production checkpoint or a real robot replay. That
is deliberate: the heavyweight/private artifacts are inputs to the workflow,
not repository fixtures. Consequently the repository cannot support a claim
about production SmolVLA latency, action-vector agreement, or end-to-end QPU
speedup yet. This branch should be described as an implementation ready for a
provenance-locked production run, not as a measured model acceleration.

## llama.cpp pipeline-acceleration branch

Branch: `origin/qpu/llama-cpp-pipeline-acceleration`

Unique commits: `77fbe9e`, `077d6ce`, `cd6b15e`, `88bc303`, `a472092`,
`6993b06`

This branch is a systematic search for a better integration boundary after
the main-branch quantized nodes proved too slow. It explores progressively
larger and more overlapped regions: arbitrary-M projections, graph backends,
inline GEGLU, row and output-column hybrids, fused producer/consumer chains,
complete FFN islands, batched attention, MTP, and persistent multi-projection
dispatch. All experimental hooks finish disabled.

### 1. Arbitrary-M exact Q4_0/Q8_0 projection kernels

The first step generalized the native quantized path beyond one row and
changed work distribution and tensor layout. On `M=16, K=1536, N=6144`, the
progression was:

| Variant | QPU ms | Best hybrid ms | CPU ms |
|---|---:|---:|---:|
| original Q4 up projection | 20.950 | 10.813 | 0.914 |
| output-outer, 16-row work | 4.189 | 4.330 | 0.913 |
| tiled 16-row form | 7.491 | 5.987 | 0.980 |

The output-outer layout was a real kernel improvement, but even it remained
about 4.6x slower than the exact CPU node at this representative shape.
Larger-M sweeps remained roughly 4-22x behind depending on shape. Later NEON
staging reduced exact activation preparation from about 1.3 ms to roughly
0.12-0.16 ms at `M=257`, showing that packing was fixable but was not the
dominant gap.

### 2. Conventional GGML backend integration

The branch next implemented a device backend with GGML graph splitting. The
QPU GEGLU path was exact, but it introduced 35 graph splits for the relevant
Gemma request. A 21-token request became approximately 1.85x slower than CPU.
This established that the ordinary backend boundary was too expensive and
motivated an inline weak hook that keeps the surrounding CPU graph intact.

### 3. Inline exact GEGLU

The inline hook accepts only split, contiguous FP32 GEGLU with width divisible
by 768, uses the pinned one-batch-thread CPU contract, and explicitly allows
rows 17, 33, 129, and 257. The Q4 route was disabled. It reproduces the CPU's
FP16 lookup-table semantics, uses cacheable DMA/PRIME fences, and retains its
program, LUT, and scratch allocations.

Isolated operator screens reported speedups of 1.134x, 1.113x, 1.190x, and
1.124x for those four row counts. These were useful exact node wins, but the
environment was diagnostic rather than promotion quality and two of four CPU
reruns were faster than the earlier baseline. The paired request result was:

- CPU: 1,400.429 ms;
- QPU candidate: 1,445.608 ms;
- speedup: 0.969x;
- native dispatches: 35.

Therefore the exact GEGLU node win did not survive its full-model boundary.

### 4. Row-partitioned exact GEGLU

Splitting rows between CPU and QPU was intended to hide accelerator latency.
It lost at all tested row counts: 0.980x at 17, 0.951x at 33, 0.964x at 129,
and 0.971x at 257. No end-to-end campaign was warranted.

### 5. Exact `ffn_up` output-column overlap

The next experiment used a more promising boundary: CPU and QPU concurrently
produce disjoint output-column ranges for every `ffn_up` layer. It reached all
35 layers with exact output and no fallback. The paired full-model results
were:

| M / QPU fraction | Speedup | 95% CI |
|---|---:|---:|
| 65 / 0.0625 | 1.006x | 0.977-1.036 |
| 129 / 0.125 | 1.011x | 0.992-1.030 |
| 257 / 0.0625 | 0.986x | 0.975-0.997 |

The per-layer telemetry explains why a seemingly good overlap was still too
small. At M=65, QPU work was 7.44 ms, 6.97 ms overlapped, and only 0.48 ms was
visible as a tail. At M=129 the corresponding values were 13.75, 12.68, and
1.06 ms. At M=257/fraction 0.0625 they were 31.85, 26.71, and 5.14 ms. Raising
the fraction to 0.25 made the QPU 40.66 ms, overlap 21.43 ms, and exposed tail
19.23 ms. A roughly 1.9x kernel improvement would have been needed to bring
that tail back to the overlap boundary and predict only about 1.06x overall.

### 6. Approximate column-W8 and W8A8 hybrids

Quantizing activation rows inside the timed path cost roughly 5 ms at M=257,
so a row-W8 route was not attractive. The retained alternative quantized
selected output columns and added a performance governor plus paired-session
controls. It is approximate relative to the FP32 graph, even though its
native dispatch and token checks passed.

The campaign retained all generated tokens exactly across 18 pairs and
attested 630 layer dispatches. Results were:

| M | CPU ms | Candidate ms | Speedup | 95% CI | QPU memory |
|---|---:|---:|---:|---:|---:|
| 129 | 3,019.758 | 3,021.002 | 0.9996x | 0.9466-1.0236 | 121.5 MiB |
| 257 | 5,603.841 | 5,515.050 | 1.0161x | 0.6587-1.1098 | 124.3 MiB |

The M=257 interval is especially unstable because it contains a retained
0.659x outlier. Neither result clears 1.05x or has a confidence interval above
1.0. Exact tokens for this short fixture are not a substitute for the missing
perplexity and long-generation quality evaluation required by the approximate
weight/activation contract.

### 7. Batched fused attention

One QPU launch processed all heads and rows and was exact, but the implementation
lost decisively:

| Context | M | CPU ms | QPU ms | Speedup |
|---:|---:|---:|---:|---:|
| 512 | 65 | 15.581 | 26.521 | 0.588x |
| 512 | 257 | 35.752 | 109.117 | 0.328x |
| 4,096 | 65 | 124.670 | 220.574 | 0.565x |
| 4,096 | 257 | 288.954 | 861.595 | 0.335x |

A workgroup sweep did not rescue it: 12-way workgroups fell to roughly
0.335x, while 24 and 48 were only around 0.58x in the relevant screen.

### 8. Fused GEGLU to resident-Q8 producer chain

This experiment tried to eliminate an intermediate host boundary by having
the GEGLU producer leave Q8 activations resident for the next projection. The
producer alone reached 0.921x, 1.050x, and 1.139x at M=65, 129, and 257, but
the complete chain achieved only 0.216x, 0.254x, and 0.268x.

At the later M=272 validation, two output bytes differed by one because a
reciprocal ULP crossed a half-rounding boundary. The maximum dequantized error
was 0.004749. Attempts to correct the rounding moved the mismatch rather than
eliminating it, so the exact source was restored and the resident-Q8 route was
quarantined as approximate. Workgroup counts from 8 through 192 left chain
time around 69.9-108 ms and did not change the conclusion.

### 9. Channel-partitioned complete FFN islands

The most promising component experiment assigns complete channel slices to
CPU and QPU through gate, up, GEGLU, and down projections. This removes most
cross-device intermediates; only a partial FP32 hidden result crosses the join.
The screen was optimistic because QPU activation preparation was outside the
timed region, so it is an architectural signal rather than a deployable win.

At M=257, narrow and wide slices measured 1.154x at `3/16` and 1.088x at
`1/8`. The larger-M estimates were:

| M | Narrow slice | Wide slice | Work-weighted FFN estimate |
|---:|---:|---:|---:|
| 513 | 1.407x | 1.102x | 1.188x |
| 1,025 | 1.063x | 1.117x | 1.102x |
| 2,049 | 1.123x | 1.121x | 1.122x |

This is the best remaining llama.cpp hypothesis in the branch, but it lacks a
production GGML hook, timed preparation, full-layer numerical validation, and
a retained end-to-end model campaign.

### 10. Full-model M=513 FFN experiment

The first telemetry was invalid: it reported 70 dispatches at M=508/512
because the model's actual prefix shape differed from the requested nominal
shape. The evaluator correctly rejected it rather than treating nearby shapes
as evidence.

With an exact `ubatch` control, the 35 expected M=513 layer dispatches were
attested. At a `3/16` slice, calibration measured 9,090.719 ms CPU versus
8,871.950 ms candidate (1.0247x), and held-out measurement was 9,053.511
versus 8,925.507 ms (1.0143x). QPU memory was 262.7 MiB. Raising the workgroup
count to 192 improved a projection suffix by about 1.041-1.048x but was still
insufficient for promotion.

### 11. Multi-token prediction experiments

The dynamic backend was exact but dispatched 1,753 tiny M=1 QPU operations.
A 64-token request took 50.845 seconds versus 22.476 seconds on CPU, only
0.442x.

The inline weak hook then added exact shape/fallback controls and excluded
embedding tensors. Against the relevant one-thread CPU boundary, the M=1
operator was about 25 microseconds on CPU and 350 microseconds on QPU: roughly
14x slower. Earlier operator screens that compared with a four-thread CPU team
startup made QPU look faster, but that was the wrong baseline and was
explicitly rejected.

### 12. Persistent 24-thread projection subroutine

The final prototype keeps 24 threads resident behind a global barrier and
executes two exact projections in one CSD submission. Both outputs were
bitwise identical to the reference. For N=6,144 with a 1,152-column suffix,
two separate launches took 48.176 ms and the persistent launch took 46.502 ms
(1.036x). For N=12,288 with a 2,304-column suffix, time changed from 96.187 to
92.776 ms (1.0368x).

This proves that multi-phase launch collapse and the global barrier are
feasible, but the approximately four-percent scheduler saving is not the
order-of-magnitude improvement the native quantized projection needs. Qwen
was not rerun: the main results had QPU-only speedups of only about 0.060x for
Q4_K gate, 0.025x for Q6_K down, and 0.082x for Q8 SSM output, which a 1.04x
launch improvement cannot plausibly reverse.

### Pipeline branch conclusion

No explored route satisfies the repository's promotion rules. Exact node wins
were erased by graph boundaries or exposed tails; approximate W8 runs lacked a
statistically significant request win and still need quality evaluation;
attention and MTP were clear losses; and persistent dispatch saved only about
four percent. All hooks remain opt-in or disabled. The useful artifact is the
engineering map of where time moves, with the complete FFN island retained as
the one hypothesis worth a stricter next experiment.

## Consolidated result ledger

This ledger separates genuinely reusable conclusions from attractive numbers
that should not be used as acceleration claims.

### Positive and reusable

- The V3D DRM path, assembler, BO allocation, uniform ABI, CSD submission,
  dependency handling, and persistent program/buffer lifecycle are real and
  repeatedly exercised foundations.
- Shared host/QPU-visible buffers plus explicit access ranges are sufficient
  to build asynchronous CPU/QPU graphs and deterministic failure recovery.
- Exact large-stripe elementwise/reduction work can scale across QPUs and beat
  the selected CPU baseline when alignment and workload size are favorable.
- FP32 prefill projection has ten exact-shape wins, including QPU-only and
  overlapped row-hybrid placements; these are safe only behind the exact
  registry entries that measured them.
- Packed signed W8A8 with `v8dot` is the strongest current arithmetic family.
  It has exact INT32 accumulation, differential-tested assembly, fused
  scaling, reusable prepared plans, and real selected prefill/one-layer wins.
- Public INT16 Conv2D is correct by widening to the validated INT32 path. It is
  a correctness baseline while the packed experiment stays quarantined.
- Persistent multi-phase QPU programs and a 24-thread global barrier work and
  can remove a small amount of repeated launch overhead.
- The SmolVLA branch demonstrates that a substantial multimodal diffusion
  policy can be reconstructed as a provenance-locked native heterogeneous
  graph within a plausible 8 GiB memory budget.
- Exact-shape telemetry, hash validation, paired sessions, environment gates,
  fallback accounting, and whole-layer/model replay prevent local kernel wins
  from being mistaken for product wins.

### Correct but slower

- Main native llama.cpp Q4_0, Q4_K, Q6_K, Q8_0 projections and fused attention.
- W8A8 single-row GEMV, dequantization-only stages, and several convolution
  candidates retained for explicit evaluation.
- FP32 raw GEMM and staged SDPA at the audited shapes.
- Most decode-shape norm, RoPE, softmax, activation, and projection paths.
- Batched all-head attention, exact row-hybrid GEGLU, and exact M=1 MTP hooks.
- The persistent two-projection program: exact and slightly faster than two
  QPU launches, but nowhere near CPU competitiveness.

### Fast-looking but invalid, superseded, or incomplete

- April naive/fast FP32 SGEMM timing with large errors and NaN/Inf output.
- April INT32/packed-INT16 GEMM timing with million- to billion-scale errors.
- April tiled Conv2D, MLP, attention, and LeNet speed numbers with invalid
  numerical outputs.
- The min/max default twelve-QPU size, which violated its 192-element chunk
  divisibility assumption.
- Early QPU benchmarks that charged driver creation, assembly, allocation,
  host lowering, upload, execution, and readback while timing only CPU compute.
- A four-thread-team-startup comparison that made the tiny M=1 QPU path look
  favorable; the relevant one-thread CPU node was about 14x faster.
- The first M=513 full-model record, whose telemetry actually ran M=508/512.
- Resident-Q8 “exact” chaining, disproved by two one-byte differences at M=272
  and therefore reclassified as approximate.
- Isolated stage wins whose enclosing one-layer/model regressions lost,
  including the 2,048-hidden/256-token SwiGLU case.
- SmolVLA production performance and action-quality claims: the runtime exists,
  but the production artifact/replay is absent.
- FFN-island weighted estimates: promising, but activation prep was omitted and
  there is no production hook or retained model run.

## Cross-cutting technical lessons

### Workload shape matters more than nominal arithmetic throughput

The QPU performs best on enough aligned, contiguous work to amortize launch
and memory boundaries. Tiny decode nodes and irregular graph splits lose even
when their arithmetic maps cleanly. Exact shape must therefore be part of the
dispatch identity; nearby dimensions are not evidence.

### Integration boundaries dominate model results

Moving one operator can introduce packing, fences, graph splits, upload,
readback, synchronization, or an exposed concurrent tail. The experiments
repeatedly show this hierarchy:

`kernel win -> stage win -> block win -> full replay win -> paired request win`

Promotion is justified only at the right enclosing boundary. GEGLU and
`ffn_up` are the clearest examples of good local results that vanished at the
request boundary.

### Concurrency is more promising than QPU replacement

The useful FP32 hybrids give the QPU a tile-aligned row or output-column region
while optimized CPU code handles a disjoint region. The FFN-island experiment
extends this idea across a whole producer/consumer region. In contrast,
replacing highly optimized NEON `ggml-cpu` quantized matmul outright has lost
by large factors.

### Native packed-byte math is the credible quantized route

The historical packed INT16 work was neither correct nor bandwidth-efficient
enough. The current W8A8 family aligns the host layout, logical tile, and
hardware `v8dot` primitive; that produces exact accumulation and the clearest
prefill wins. It still does not make small decode or GEMM-backed convolution
automatically good.

### Direct kernels are needed for convolution and attention

`im2col` duplicates data and turns convolution into a memory-heavy GEMM. The
W8A8 convolution results confirm that a good GEMM primitive alone cannot beat
native CPU convolution, particularly for depthwise. Likewise, materializing
and staging full attention remains too expensive. Hot 1x1/3x3/depthwise
spatial kernels and more resident/fused attention dataflow are architectural
work, not tuning exercises.

### Correctness and provenance mechanisms are first-class output

The generated program hashes, model/tensor manifests, source locks, numerical
gates, exact native telemetry, failure injection, and memory preflight are not
benchmark decoration. They caught real errors: wrong shapes, wrong baseline
threads, stale or mismatched artifacts, a two-byte quantization discrepancy,
and invalid system conditions.

## What appears most notable now

If this work is being reduced to a shorter public or internal narrative, the
highest-value points are:

1. **The runtime and measurement discipline are a substantial deliverable.**
   This is no longer an isolated assembler demo: it has memory ownership,
   asynchronous heterogeneous scheduling, persistent plans, exact candidate
   placement, traces, artifact hashes, native llama.cpp telemetry, and robust
   fallbacks.
2. **W8A8 prefill is the clearest hardware success.** Native `v8dot`, a layout
   designed around the instruction, and exact-shape promotion produced real
   projection and one-layer wins. This is stronger evidence than the older
   high-throughput but numerically invalid GEMM experiments.
3. **Exact FP32 stripes and CPU/QPU overlap establish a viable scheduling
   pattern.** The accelerator is useful for carefully selected disjoint work,
   not as a universal replacement for optimized ARM libraries.
4. **The llama.cpp integration is a high-quality negative result.** Multiple
   integration boundaries, quantization formats, fusions, partitions, and
   persistent launches were actually implemented and measured. They converge
   on the conclusion that current decode and native quantized projection paths
   are not deployable wins.
5. **The complete FFN island is the best unclosed llama.cpp lead.** It moves
   the boundary far enough to show 1.10-1.19x estimated region-level benefit,
   but must be retested with all prep included and a real model hook before it
   can be called a win.
6. **SmolVLA is the broadest systems implementation, not yet a performance
   result.** Its graph reconstruction, memory feasibility, converter, and
   replay gates are notable; production latency and action quality remain the
   decisive missing run.
7. **Direct kernels are the next architectural step.** More tuning of
   GEMM-backed Conv2D or staged attention is unlikely to change their outcome.

Lower-priority or closed lines are M=1 QPU decode/MTP, the current native GGML
Q4/Q6/Q8 kernels as CPU replacements, batched fused attention in its current
dataflow, exact row-hybrid GEGLU, and packed INT16 until its hardware
differential is exact.

## Open experiments implied by the evidence

These are gaps, not work claimed as completed:

- Run the SmolVLA converter and upstream recorder with the pinned production
  checkpoint and a real robot replay; then execute CPU FP32, QPU FP32, and W8
  modes under the documented memory and numerical gates.
- Turn the FFN island into one production GGML integration boundary, include
  activation preparation and joins in the timed region, prove the full layer
  numerically, and run a retained paired request campaign.
- Revisit W8A8 prefill only at exact shapes supported by the registry and add
  model-level holdouts before exposing any new isolated stage win.
- Implement direct 1x1, 3x3, and depthwise convolution dataflows instead of
  further tuning `im2col`; keep public widened INT16 as the oracle until packed
  INT16 passes an exact Pi hardware differential.
- Pursue more resident/fused attention only if it removes materialized score
  or intermediate boundaries; the current all-head kernel and staged SDPA are
  already clear losses.
- Repeat any publishable performance campaign with `performance` governor,
  no current throttling, no swap/zram pressure, no competing server, exact
  workload hashes, and enough independent paired sessions for the declared
  confidence gate.

## Evidence map

The following narrative sources were read from the included refs and
reconciled into this document:

- root history and architecture: `README.md`, `REPORT.md`,
  `EXPERIMENTS_REPORT.md`, `EXPERIMENTS_REGISTRY.md`, `QPU-XLA.md`,
  `QPU-XLA-Rewrite-and-Roadmap-Plan.md`, `LLAMA-CPP-QPU-ACCELERATION-PLAN.md`,
  and `AGENTS.md`;
- modern FP32/W8A8: all Markdown summaries under
  `experiment_logs/20260819-qpu-xla-kernel-suite`,
  `experiment_logs/20260819-qpu-xla-w8a8`, and the merged
  `experiment_logs/KERNEL_SCOREBOARD.md`;
- llama.cpp on main: `integrations/llama_cpp/README.md`, every Markdown graph
  profile, `LLAMA_CPP_QPU_MATRIX.md`, and `RESULTS.md` under
  `experiment_logs/20260819-llama-cpp-qpu`;
- SmolVLA branch: `SMOLVLA-QPU.md` plus the branch's benchmark, conversion,
  recorder, runtime, operator, exporter, and test sources;
- pipeline branch: `integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md`,
  `integrations/llama_cpp/README.md`, `HYBRID_RESULTS.md`, and
  `UP_OVERLAP_RESULTS.md`.

The corresponding JSON candidate files, scoreboards, graph profiles, manifests,
fixture records, evaluation cases, and pipeline diagnostics were inspected to
cross-check the summary tables and to distinguish raw screens from retained
campaigns. Generated binaries and large/private model artifacts were checked
through their manifests and hashes; they were not treated as report text.

Remote refs present during the audit were `origin/main`,
`origin/feat/smolvla-qpu-runtime`,
`origin/qpu/llama-cpp-pipeline-acceleration`, and the symbolic `origin` HEAD.
`origin/feat/llama-cpp-ggml-qpu-ops` was intentionally ignored as a separate
tip per the request; its merged main-branch result remains covered.

## Local audit verification

The requested branches were each checked out in detached state for source,
report, and test inspection, then the worktree was restored to `main`.

On restored `main`, the portable focused suite completed with 176 passes and
99 hardware deselections. It emitted 21 NumPy divide/overflow/invalid warnings
from synthetic extreme-value matmul references, but no test failed.

On the pipeline branch, the portable focused suite completed with 192 passes
and 103 hardware deselections:

```text
PYTHONPATH=. uv run pytest -q -m 'not hardware' \
  tests/test_qpu_xla*.py tests/test_llama_cpp*.py \
  tests/test_export_qpu_programs.py
```

On the SmolVLA branch, its two focused modules completed with five passes and
five hardware deselections. The reduced synthetic graph emitted expected
NumPy overflow/divide warnings; no test failed.

A broader non-hardware pipeline collection had 51 failures because several
legacy tests are not marked `hardware` and directly open `/dev/dri`. This
audit ran on macOS without a V3D render node, so those are environment-limited
hardware attempts, not newly discovered kernel regressions. Pi-only assembly
differentials and production performance numbers remain those recorded in the
checked-in reports.

## Final assessment

In two weeks, the repository advanced from a broad kernel/runtime prototype to
two serious application integrations and an unusually complete body of
negative performance evidence. The most defensible acceleration result is
selected exact-shape prefill—especially packed W8A8—and the most defensible
systems result is the provenance-checked runtime/integration infrastructure.

There is not yet a defensible end-to-end llama.cpp QPU speedup or a measured
production SmolVLA speedup. That distinction is a strength of the work: failed
or inconclusive paths were retained with enough telemetry to prevent them from
being rediscovered or accidentally promoted. The next wins, if they exist,
are most likely to come from larger resident regions such as a complete FFN
island, direct convolution, or truly fused attention—not from more isolated
decode nodes or relabeling diagnostic screens as model results.
