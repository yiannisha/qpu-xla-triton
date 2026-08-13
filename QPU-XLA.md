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

- `vc7.copy_words`: word copy;
- `vc7.tiled_fp32_gemm`: tiled FP32 GEMM;
- `vc7.tiled_int32_gemm`: tiled INT32 GEMM using the VideoCore `smul24`
  arithmetic path;
- FP32 and INT32 min/max word kernels;
- FP32 and INT32 2D max-pool and average-pool kernels.

The internal GEMM tile contract is 16 output rows, 16 output columns, and a
reduction tile of 4 values. The INT32 kernels retain the signed-24-bit operand
contract imposed by `smul24`.

Programs and uniform storage are cached per hardware backend where the kernel
adapter supports it. This is separate from cold-start setup: assembly,
allocation, and program creation should not be charged to steady-state kernel
execution.

## Operator surface

The stable operator entry points are in
[`src/qpu_xla/ops`](src/qpu_xla/ops).

### Matmul and placement

`matmul()` supports CPU and tiled-QPU implementations. `plan_matmul()` exposes
the selected candidate, while `calibrate_matmul()` records whole-operation CPU
and QPU timings in a shape-specific `CostModel`.

`Placement.AUTO` currently selects between whole-operation CPU and whole-
operation QPU candidates. Its uncalibrated estimate is only a heuristic;
calibration is required for reliable machine-specific placement.

`hybrid_matmul()` is an explicit CPU/QPU split. It assigns a QPU-compatible
prefix of output rows to the tiled GEMM and the remaining rows to NumPy on a
separate queue. The output regions are disjoint and share the input storage,
so the CPU and QPU can overlap. Hybrid partitioning is not yet a scheduler
candidate and is not selected by `Placement.AUTO`.

### Convolution

`conv2d_fp32()` and `conv2d_int32()` are GEMM-backed NCHW convolution paths.
General spatial windows use `im2col`; the common 1x1, stride-1, no-padding,
unit-dilation case packs pixels directly into GEMM rows and avoids window
duplication. Neither path is a native direct-convolution microkernel.

`Conv2dInt32Plan` persists padded workspaces and the transformed weight matrix
for repeated fixed-shape INT32 inference. The current persistent plan is
INT32-only.

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
`copy()`, `minimum()`, and `maximum()` currently have queue-integrated CPU
reference implementations whose contracts are ready for QPU replacement.

## Scheduling and heterogeneous execution

[`src/qpu_xla/scheduler`](src/qpu_xla/scheduler) provides:

- `Placement.CPU`, `Placement.QPU`, and `Placement.AUTO`;
- capability checks;
- deterministic candidate selection;
- exact-shape median timing samples;
- JSON-serializable cost models.

The scheduler currently performs whole-operator placement. Explicit
`hybrid_matmul()` is the current heterogeneous split primitive. This is an
important distinction: the runtime can execute CPU and QPU work concurrently,
but it does not yet automatically search CPU/QPU partitions.

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
- per-output-channel INT8 weight quantization;
- an INT8-to-INT32 GEMM path that reuses the existing `smul24` QPU GEMM.

The direct quantized projection path still performs host-side quantization and
dequantization around the existing INT32 GEMM. A dedicated native quantized
QPU projection kernel is not yet implemented.

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
uv run examples/benchmark_qpu_xla_matrix.py
uv run examples/benchmark_qpu_xla_matrix.py --size 512 --warmup 2 --repeat 7
```

The current observed hybrid result is shape- and machine-specific. On the
tested machine, a 512x512 FP32 matmul with 128 QPU output rows and 384 CPU
output rows measured 4.207 ms versus 4.932 ms for the CPU-only qpu_xla path,
or about 1.17x faster. The QPU-only path was slower, so this is a heterogeneous
overlap result, not evidence that the FP32 QPU kernel alone beats the CPU.

Benchmark categories must remain distinct:

- cold start includes driver, assembly, allocation, and setup;
- cached total includes repeated upload, execution, and readback;
- execute-only isolates the driver dispatch where possible;
- host preparation covers CPU lowering, packing, and padding;
- hybrid wall time includes the join after concurrent CPU/QPU work.

## Correctness and maturity

The package has CPU/fake-backend contract tests and hardware-marked differential
tests for the supported QPU paths. Hardware tests require a usable
`/dev/dri/renderD128` and appropriate device permissions.

The current maturity is a systems prototype:

- packaged FP32/INT32 GEMM, pooling, attention-core, and convolution paths have
  explicit contracts and hardware differential coverage;
- INT32 operations must preserve their signed-24-bit restrictions;
- general convolution is still GEMM-backed and can be host-preparation-bound;
- hybrid partitioning is explicit rather than automatically scheduled;
- TinyLlama quantized projections still reuse INT32 infrastructure rather than
  a native quantized kernel;
- the DSL and autotuner cover a constrained initial subset.

The older `examples/tiled*.py`, `examples/sgemm*.py`, and related scripts remain
useful legacy kernel experiments. They are not the canonical description of the
new runtime; see [`EXPERIMENTS_REGISTRY.md`](EXPERIMENTS_REGISTRY.md) for their
separate benchmark inventory.

## Next implementation priorities

1. Add calibrated hybrid candidates to placement so the scheduler can choose
   CPU-only, QPU-only, or a measured CPU/QPU partition.
2. Persist hybrid workspaces and prepared operands to remove repeated allocation
   and lowering from steady-state heterogeneous inference.
3. Add richer queue profiling for CPU duration, QPU duration, overlap, and join
   overhead.
4. Improve direct common-case convolution support, especially 3x3, while
   retaining the existing GEMM-backed correctness path.
5. Implement and differentially validate a native quantized projection kernel
   before using it for TinyLlama performance claims.
6. Expand the DSL lowering and autotuning surface only behind CPU and hardware
   differential tests.
