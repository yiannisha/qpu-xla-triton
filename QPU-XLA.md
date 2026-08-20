# QPU-XLA runtime and current status

`qpu_xla` is the newer ML runtime layer in this repository. It sits above the
VideoCore VII assembler and driver and provides shared-memory tensors, queued
CPU/QPU work, reusable kernels, operator contracts, placement decisions, and
model/runtime building blocks.

It is best described as a hand-written heterogeneous operator/runtime
prototype. It is not yet a complete XLA compiler and it should not be described
as one.

## Architecture

The current stack has five layers:

1. `Device`, `Tensor`, `Buffer`, and `AccessMode` manage host-mapped,
   QPU-visible allocations.
2. `Queue` and `Event` provide asynchronous submission, dependencies,
   host-task execution, buffer access declarations, and Chrome trace export.
3. Packaged `Kernel` adapters retain assembled programs and reusable uniform
   storage for supported QPU kernels.
4. Operators select CPU, QPU, or explicit heterogeneous implementations and
   preserve shape, dtype, layout, and arithmetic contracts.
5. Plans, model code, video pipelines, benchmarking, and the constrained DSL
   build higher-level workloads on those primitives.

The public runtime entry points are collected in
[`src/qpu_xla/__init__.py`](src/qpu_xla/__init__.py). Hardware execution uses
`Device.open()` and the existing VideoCore driver; `Device.fake()` provides a
CPU-only backend for API and scheduling tests.

## Runtime semantics

Allocations are host-mapped and QPU-visible. A NumPy view and a QPU address can
refer to the same tensor storage, so a CPU host task can prepare or consume a
buffer without an explicit host/device copy.

Each queue has one worker and is in-order. A queue can submit either a QPU
kernel or a declared host task. Events express cross-queue dependencies, and
the runtime records submission, start, finish, and dependency information for
tracing. The current implementation is asynchronous at the application API,
but the underlying VideoCore driver dispatch remains synchronous inside the
queue worker.

Buffer access declarations are part of the contract. They identify reads,
writes, and byte ranges so the runtime can validate ownership and represent
future hazard tracking. Separate queues are required for explicit CPU/QPU
hybrid work.

## Packaged QPU kernels

The reusable kernel layer in [`src/qpu_xla/kernels`](src/qpu_xla/kernels)
currently contains:

- `vc7.copy_words`: exact-stripe multi-QPU word copy with scalar and four-word
  TMU paths;
- `vc7.tiled_fp32_gemm`: tiled FP32 GEMM;
- `vc7.tiled_int32_gemm`: tiled INT32 GEMM using the VideoCore `smul24`
  arithmetic path;
- `vc7.tiled_w8a8_gemm`: four-byte-packed INT8 GEMM using native signed
  `v8dot`, with exact INT32 accumulation;
- `vc7.tiled_w8a8_gemm_dequantize`: the same GEMM with fused INT32-to-FP32
  row/column scaling in the store path;
- `vc7.w8a8_gemv`: single-row packed decode projection candidate;
- `vc7.w8a8_dequantize`: optional tiled INT32-to-FP32 scaling epilogue;
- `vc7.swiglu_fp32`: fused FP32 SiLU-gate multiplication with scalar and
  four-word TMU paths plus balanced 12-QPU striping;
- `vc7.rms_norm_fp32`: row-parallel FP32 RMSNorm with scalar and four-word TMU
  paths;
- `vc7.rope_fp32`: FP32 rotary embedding with cached trigonometric tables,
  scalar fallback, and a four-value pairwise path;
- `vc7.softmax_fp32`: stable row-parallel FP32 softmax with scalar and
  four-word TMU paths;
- `vc7.bias_fp32` and `vc7.bias_relu_fp32`: tiled FP32 epilogues, including the
  fused one-pass bias-plus-ReLU path;
- `vc7.relu_fp32`: exact-stripe contiguous ReLU with scalar and four-word TMU
  paths;
- `vc7.residual_add_fp32`: contiguous FP32 residual addition with scalar and
  four-word TMU paths;
- FP32 and INT32 min/max word kernels, including an exact-stripe four-word TMU
  vector specialization with a generic scalar fallback;
- FP32 and INT32 2D max-pool and average-pool kernels.

The internal GEMM tile contract is 16 output rows, 16 output columns, and a
reduction tile of 4 values. The INT32 kernels retain the signed-24-bit operand
contract imposed by `smul24`.

The packed W8A8 GEMM has a separate logical 16x16x16 contract: four signed
bytes occupy each input word, and four packed words are consumed per reduction
iteration. The host rejects shapes whose worst-case INT32 accumulation could
overflow.

Programs and uniform storage are cached per hardware backend where the kernel
adapter supports it. This is separate from cold-start setup: assembly,
allocation, and program creation should not be charged to steady-state kernel
execution. The vec4 FP32 stage kernels and tiled FP32 GEMM reserve 256-byte
uniform blocks. The low-level allocator is linear, so this padding preserves
the 256-byte tensor-base alignment needed for full vec4 TMU throughput across
later lazy program creation.

## Operator surface

The stable operator entry points are in
[`src/qpu_xla/ops`](src/qpu_xla/ops).

### Matmul and placement

`matmul()` supports CPU and tiled-QPU implementations. `plan_matmul()` exposes
the selected candidate, while `calibrate_matmul()` records whole-operation CPU
and QPU timings in a shape-specific `CostModel`.

The scheduler represents CPU, QPU, and `Placement.HYBRID` candidates. Hybrid
candidates carry an explicit partition and are excluded from `AUTO` until an
exact-shape calibration sample exists.

`hybrid_matmul()` is a compatibility entry point for the first-class hybrid
matmul placement. It assigns a QPU-compatible
prefix of output rows to the tiled GEMM and the remaining rows to NumPy on a
separate queue. The output regions are disjoint and share the input storage,
so the CPU and QPU can overlap.

The FP32 dtype/placement inventory, reproduction commands, and current
exact-shape results are maintained together in the
[`FP32 kernel-suite directory`](experiment_logs/20260819-qpu-xla-kernel-suite/README.md).

### Convolution

`conv2d_fp32()` and `conv2d_int32()` are GEMM-backed NCHW convolution paths.
General spatial windows use `im2col`; the common 1x1, stride-1, no-padding,
unit-dilation case packs pixels directly into GEMM rows and avoids window
duplication. Neither path is a native direct-convolution microkernel.

`Conv2dInt32Plan` persists padded workspaces and the transformed weight matrix
for repeated fixed-shape INT32 inference. The current persistent plan is
INT32-only.

`Conv2dW8A8Plan` is the experimental quantized counterpart. It retains packed
weights and workspaces, handles grouped convolution, and uses vectorized
im2col for general windows. It is bit-exact to its dynamic W8A8 reference, but
the measured 1x1, 3x3, and depthwise YOLO cases are slower than Torch native
convolution and are therefore not scheduler-visible. Direct spatial kernels
are still required.

### Attention

`attention_fp32()` and `attention_int32()` implement the unnormalized core
`(Q @ K.T) @ V`. They are not full transformer attention and do not include
softmax or causal masking.

`scaled_dot_product_attention_fp32()` adds scale, optional causal masking, and
row-wise softmax. Its GEMM stages use the QPU when dimensions meet the tiled
contract, while the numerically sensitive softmax remains an explicit CPU host
task. This is a CPU/QPU pipeline, but its stages are dependency-ordered rather
than independently parallel.

`AttentionInt32Plan` persists fixed-shape INT32 attention preparation and
workspace state.

### MLP, pooling, and elementwise operations

`mlp_int32()` and `MlpInt32Plan` implement a tiled INT32 linear/ReLU/linear
path with persistent-plan support. FP32 MLP functionality remains primarily in
the legacy executor examples rather than the packaged `qpu_xla.ops` surface.

`pool2d_fp32()` and `pool2d_int32()` provide packaged 2x2 stride-2 pooling.
`PreparedPool2DFP32` caches the address metadata for repeated fixed-shape
execution, and the FP32 kernel selects the largest exact QPU divisor up to 12.
`copy()`, `minimum()`, and `maximum()` provide queue-integrated CPU fallbacks
and contiguous word-oriented QPU specializations.

## Scheduling and heterogeneous execution

[`src/qpu_xla/scheduler`](src/qpu_xla/scheduler) provides:

- `Placement.CPU`, `Placement.QPU`, and `Placement.AUTO`;
- capability checks;
- deterministic candidate selection;
- exact-shape median timing samples;
- JSON-serializable cost models.

The scheduler supports whole-operator placement and explicit calibrated
partitions. Matmul/linear, SwiGLU, RMSNorm, RoPE, and softmax can split row
prefixes between QPU and CPU; single-row linear can instead split output
columns. The runtime can execute the disjoint regions concurrently, but it
does not yet perform an online partition search. Automatic placement requires
an exact-shape retained result for the selected partition.

## Compiler DSL and autotuning

The constrained DSL in [`src/qpu_xla/compiler`](src/qpu_xla/compiler) captures
a restricted set of QPU-XLA primitives into a typed IR, validates source
locations and allowed operations, and lowers the supported forms toward VC7.
It currently covers primitives such as load/store, dot, reductions, select,
program IDs, ranges, and barriers. It is a narrow verified kernel-generation
surface, not a general Python-to-QPU compiler.

[`src/qpu_xla/autotune`](src/qpu_xla/autotune) provides candidate validation and
result persistence. The runner can perform source validation, CPU reference
evaluation, optional hardware differential execution, and repeated timing for
registered kernel candidates.

## Video and model layers

The video layer provides NV12 frame contracts, an NV12-to-NCHW-FP32
preprocessing path, a frame ring, and a headless preprocessing runner. These
components establish queue/buffer ownership and pipeline contracts; they are
not yet a complete camera application.

The TinyLlama layer provides:

- checkpoint/configuration/artifact loading;
- a numerically transparent CPU reference runtime;
- FP32 RMSNorm, RoPE, embedding, SiLU-gated activation, residual add, and
  greedy sampling tasks;
- FP32 KV-cache storage;
- complete causal GQA prefill and incremental decode with persistent per-layer
  KV caches;
- per-output-channel INT8 weight quantization;
- native signed-`v8dot` W8A8 GEMM and GEMV paths with exact INT32 accumulation;
- fused W8A8 GEMM/dequantization and a fused FP32 SwiGLU candidate;
- persistent packed projection weights and reusable activation/accumulator
  workspaces;
- calibrated exact-shape dispatch through `CalibratedW8A8Linear`.

Only projections with at least one exact promoted shape are quantized and
packed during runtime construction. CPU-only fallback projections retain just
their checkpoint FP32 weights.

The winning prefill projection quantizes activations on the CPU and uses one
QPU launch for native-dot accumulation plus direct scaled FP32 stores. Shapes
that fail either the 1.05x stage gate or a measured one-layer model gate use
the optimized FP32 CPU checkpoint projection. Single-token decode candidates
are correct, but the complete measured decode paths regress and therefore are
not eligible for automatic placement.

## Benchmarking the current runtime

The reproducible runtime matrix is
[`examples/benchmark_qpu_xla_matrix.py`](examples/benchmark_qpu_xla_matrix.py).
It measures whole-operation wall time for:

- CPU-only matmul;
- QPU-only matmul;
- uncalibrated `AUTO` placement;
- explicit CPU/QPU row splits;
- CPU-only SDPA;
- the QPU GEMM attention core;
- mixed QPU-GEMM/CPU-softmax SDPA.

Run it on Raspberry Pi 5 hardware with:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run examples/benchmark_qpu_xla_matrix.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run examples/benchmark_qpu_xla_matrix.py \
  --size 512 --warmup 2 --repeat 7 --output fp32-matmul-attention-s512.json
```

Current FP32 results are shape- and machine-specific. The authoritative
inventory, result summaries, promotion evidence, and reproduction commands
are kept together in the
[`FP32 kernel-suite directory`](experiment_logs/20260819-qpu-xla-kernel-suite/README.md).
The compact per-shape view remains
[`KERNEL_BACKEND_LATENCY_MATRIX.md`](experiment_logs/20260819-qpu-xla-kernel-suite/KERNEL_BACKEND_LATENCY_MATRIX.md).

Packed model-kernel benchmarks are separate:

```bash
uv run examples/benchmark_qpu_xla_w8a8.py --case prefill-h1024-t64 --epilogue fused-qpu
uv run examples/benchmark_qpu_xla_tinyllama_runtime.py --case prefill-h1024-t64
uv run examples/benchmark_qpu_xla_tinyllama_runtime.py --case decode-h2048-c512
uv run examples/benchmark_qpu_xla_yolo_w8a8.py --case p3-1x1
uv run scripts/run_w8a8_evaluation.py --output-root experiment_logs/w8a8-matrix
```

The exhaustive runner launches every dense projection/epilogue and YOLO case
in an isolated subprocess with an automatically sized device arena. It uses
two warmups, seven retained measurements, and four OpenBLAS/OpenMP threads by
default, then writes `W8A8_EVALUATION_MATRIX.md`,
`W8A8_BACKEND_LATENCY_MATRIX.md`, `W8A8_KERNEL_SCOREBOARD.md`, and the
calibrated candidate registry.

On the current machine, fused prefill projections were bit-exact to the
dynamic W8A8 oracle and were compared with the fastest of NumPy/Torch FP32 and
INT32. Promoted projections measured 1.21x-1.68x at 64x1024,
1.09x-1.30x at 128x2048, and 3.10x-5.32x at 16x4096. The one-layer mixed
runtime measured 2.86x, 1.54x, and 1.16x versus the FP32 NumPy reference for
the 512-, 1024-, and 2048-hidden tuning cases. The 256-token holdout measured
0.83x, so its standalone SwiGLU win was demoted.

Persistent-cache decode is functionally complete and exact versus the
calibrated CPU oracle, but measured only 0.60x at cache 512, 0.48x at cache
2048, and 0.38x for the 3072-hidden/cache-4096 holdout. AUTO therefore uses
CPU for all currently measured decode shapes.

The measured YOLO GEMM-backed candidates remain experimental: p3 1x1 measured
0.13x Torch native FP32 convolution, p3 3x3 measured 0.064x, and p3 depthwise
measured 0.012x. These numbers demonstrate that packed GEMM correctness alone
does not solve spatial-convolution layout and dispatch overhead.

The legacy machine-specific merged archive is
`experiment_logs/20260819-qpu-xla-w8a8/llama-calibrated.candidates.json`. It
retains all 48 evaluated Llama stage records while exposing only 17 prefill
winners through `CandidateRegistry.supported`; applications pass this registry to
`CalibratedW8A8Linear` for exact-shape `AUTO` placement.

New W8A8 evaluation reports keep two independent promotion gates:

- `same-contract-win` is at least 1.05x faster than the fastest NumPy/Torch
  dynamic-W8A8 CPU implementation;
- `supported-win` is quality-safe and at least 1.05x faster than the fastest
  NumPy/OpenBLAS or Torch FP32 implementation.

Only `supported-win` is visible to AUTO. Exact-shape calibrated dispatch now
chooses the fastest QPU, row-hybrid, or output-hybrid record. The TinyLlama
runtime owns a second CPU queue so calibrated hybrids execute instead of being
rejected.

Benchmark categories must remain distinct:

- cold start includes driver, assembly, allocation, and setup;
- cached total includes repeated upload, execution, and readback;
- execute-only isolates the driver dispatch where possible;
- host preparation covers CPU lowering, packing, and padding;
- hybrid wall time includes the join after concurrent CPU/QPU work.

## Native llama.cpp Q4_0 evaluation path

The separate [`integrations/llama_cpp`](integrations/llama_cpp) bridge consumes
the pinned runtime's native 18-byte Q4_0 weight blocks and 34-byte Q8_0
activation blocks. Its prepared M=1 and M=4 operators retain selected weights,
programs, uniforms, and staging buffers across calls. CPU-only references and
the CPU side of output-column hybrids call the same exported optimized
`ggml-cpu` Q4_0×Q8_0 block-dot path as the pinned build.

The canonical raw manifests, diagnostics, case matrix, and generated report
are in
[`experiment_logs/20260819-llama-cpp-qpu`](experiment_logs/20260819-llama-cpp-qpu).
The current QPU-only and hybrid records are exact within the established FP32
tolerance, but lose to CPU on both the bounded drafter projection and its
262144-column output tensor. The sessions also used an `ondemand` CPU governor,
so their timings are explicitly rejected from promotion. No llama.cpp node is
automatically offloaded from those records.

The end-to-end runner uses isolated servers and exact token streams. It keeps
cold startup, context prefill, request wall time, prompt evaluation, decode,
and MTP-cycle measures separate. The generated full case file contains the 39
available Gemma CPU cases and records the absent Qwen model as a coverage gap.

## Correctness and maturity

The package has CPU/fake-backend contract tests and hardware-marked differential
tests for the supported QPU paths. Hardware tests require a usable
`/dev/dri/renderD128` and appropriate device permissions.

The current maturity is a systems prototype:

- packaged FP32/INT32 GEMM, pooling, attention-core, and convolution paths have
  explicit contracts and hardware differential coverage;
- INT32 operations must preserve their signed-24-bit restrictions;
- general convolution is still GEMM-backed and can be host-preparation-bound;
- hybrid partitioning supports calibrated exact-shape AUTO selection while
  retaining explicit `qpu_rows` and `qpu_outputs` overrides;
- the native llama.cpp path has real Gemma GGUF and tokenizer-backed diagnostic
  coverage, but lacks retained performance-governor sessions, Qwen model
  coverage, and an eligible automatic placement;
- the DSL and autotuner cover a constrained initial subset.

The older `examples/tiled*.py`, `examples/sgemm*.py`, and related scripts remain
useful legacy kernel experiments. They are not the canonical description of the
new runtime; see [`EXPERIMENTS_REGISTRY.md`](EXPERIMENTS_REGISTRY.md) for their
separate benchmark inventory.

## Next implementation priorities

1. Persist hybrid workspaces and prepared operands to remove repeated allocation
   and lowering from steady-state heterogeneous inference.
2. Add richer queue profiling for CPU duration, QPU duration, overlap, and join
   overhead.
3. Improve direct common-case convolution support, especially 3x3, while
   retaining the existing GEMM-backed correctness path.
4. Build direct 1x1 and general 3x3 W8A8 YOLO kernels; do not promote the
   current GEMM-backed candidates.
5. Optimize complete decode, especially GEMV dispatch/dequantization and
   CPU/QPU boundary costs; do not promote its current standalone stage wins.
6. Validate the calibrated runtime against a real Llama checkpoint/tokenizer,
   including layer fixtures, perplexity/task quality, and multi-layer memory.
7. Expand the DSL lowering and autotuning surface only behind CPU and hardware
   differential tests.
