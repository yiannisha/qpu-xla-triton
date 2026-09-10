# Technical Lessons from Building an ML Runtime on VideoCore VII

This is the technical retrospective for the repository from its initial
assembler and driver work through the project integration endpoint at merge
commit `11fb9b6`. It covers the low-level VideoCore VII experiments, the first
standalone ML kernels, the QPU-XLA runtime, TinyLlama and native `llama.cpp`
work, and the final SmolVLA merge. This document itself was added after that
technical-history endpoint.

The central conclusion is simple:

> The Raspberry Pi 5 QPU is useful as a selectively placed coprocessor for
> sufficiently large, regular, layout-compatible regions whose data and
> programs can stay prepared. It is not a blanket replacement for the
> Cortex-A76 CPU or its optimized numerical libraries.

That conclusion was not obvious from the first kernel benchmarks. It emerged
only after correctness testing, persistent execution, fair CPU comparisons,
native model integration, and full-request evaluation repeatedly changed the
placement decision.

Numbers in this document are observations from the checked-in machine-specific
experiments. They characterize the tested shapes and software environment, not
all Raspberry Pi 5 systems or all possible QPU kernels. “Projected” and
“estimated” results are labeled and are never presented as measured end-to-end
speedups.

## Compact field guide

### Workload fit

| Fit | Workload characteristics | Examples from this project |
|---|---|---|
| Good candidate | Large or moderately batched; regular contiguous access; enough independent rows or tiles; reusable weights and buffers; arithmetic maps directly to QPU instructions; one launch performs substantial work | Exact-stripe copy/min/max/residual; selected pooling shapes; fused bias+ReLU; selected FP32 and W8A8 prefill projections; selected prefill RMSNorm, softmax, and SwiGLU shapes |
| Conditional candidate | Some useful parallelism, but tails, staging, reductions, CPU competition, or synchronization can erase the gain; exact shape and partition must be calibrated | FP32 tiled GEMM; row reductions; hybrid row/output splits; attention stages; prepared 1x1 convolution; model operator islands |
| Poor match today | Tiny operations; single-token decode; frequent launches or graph splits; irregular or format-heavy reduction; repeated packing; im2col expansion; depthwise work; a faster optimized CPU node already exists | M=1 GEMV and MTP drafting; exact GGML Q4_K/Q6_K/Q8_0 linears; current fused attention; general and depthwise GEMM-backed convolution; small KV append, embedding, residual, and argmax |

### Before writing a kernel

Answer these questions in order:

1. **Does the workload matter?** Obtain a production graph profile and apply an
   Amdahl-style gate. A fast node that occupies little request time cannot
   produce a useful model result.
2. **Does its arithmetic match the machine?** Prefer native operations such as
   signed `v8dot` for four packed INT8 values. Do not silently change an FP32 or
   exact block-quantized contract to fit an instruction.
3. **Is there enough parallel work?** Count complete rows, tiles, and exact
   stripes, not just total scalar operations.
4. **Can the layout be consumed directly?** Include lowering, padding,
   quantization, cache synchronization, and output conversion in the design.
5. **Can state persist?** Programs, uniforms, weights, address metadata, and
   workspaces should be created once for repeated inference.
6. **What is the complete boundary?** Measure staging, submission, execution,
   synchronization, result handling, and the join—not only the QPU body.
7. **Which CPU is the real competitor?** Compare against the fastest relevant
   NumPy/OpenBLAS, Torch, or native GGML implementation using the production
   thread configuration.
8. **What is the failure contract?** Define numerical tolerances, untouched
   regions, fallback behavior, and recovery after asynchronous failures before
   enabling placement.

### Promotion ladder

A candidate advances only in this order:

1. ISA and buffer-safety test.
2. Hardware differential against an independent reference.
3. Persistent whole-operation win for the exact shape and layout.
4. Enclosing block or layer win.
5. Complete model/request win with semantic agreement.
6. Automatic placement, keyed by the exact supported shape and implementation
   hash.

A win at one rung is evidence for testing the next rung, not permission to skip
it.

## How the project evolved

### 1. The assembler and driver made hardware behavior executable

The 2025 history established instruction encoding and execution semantics
before attempting an ML stack. Tests covered ALU operations, branches,
conditions, labels, signals, register rotation and replication, TMU traffic,
uniform access, multi-thread submission, workgroups, and explicit dual issue.
The early `scopy` and `sgemm` examples then connected those semantics to real
memory and compute workloads.

Several small-looking fixes became durable systems lessons:

- Invalid rotate-to-broadcast forms were rejected rather than left as a kernel
  author convention (`83fced8`).
- FP32 operations with small immediates required encoding-specific tests
  (`e6f4566`).
- Compute-shader workgroup size was corrected to use local invocation geometry,
  not the product of submitted workgroup counts (`4de6e36`).
- Payload and workgroup inspection proved more reliable than reasoning from
  parameter names.

The lesson is broader than these bugs: undocumented or partially documented
accelerator behavior must be turned into minimal hardware tests. Compiler
guards should reject known-dangerous instruction forms early. Higher-level
kernels cannot compensate for uncertainty in the assembler, payload, or driver
launch contract.

The accompanying assembler refactors—encapsulating instructions, registers,
signals, labels, and conditions while reducing untyped `Any` usage—also proved
important. Kernel source benefits from a compact Python assembly DSL, but the
objects behind that DSL must enforce encoding invariants. A controlled `raw`
escape hatch is useful for new instruction research; it should not become the
normal way to bypass validation.

### 2. Standalone kernels exposed the difference between structure and maturity

From March through April 2026, the repository expanded from SGEMM/IGEMM into
packed INT16 GEMM, faster FP32 GEMM, min/max, pooling, convolution, MLP,
attention, and an end-to-end LeNet-style pipeline. The shared architectural
idea was tiled matrix multiplication plus persistent host-side executors.

This era produced useful abstractions, but the fresh legacy rerun also found
NaN/Inf or large-error paths in several optimized examples. The right lesson
was not to discard the architecture. It was to separate three claims that had
previously been too easy to conflate:

- a kernel has a plausible schedule;
- a persistent executor has a useful runtime shape;
- the numerical path is validated for production.

The public INT16 convolution wrapper illustrates the resulting policy. It
widens INT16 inputs and weights to INT32 and uses the validated INT32 kernel.
The packed assembly experiment remains in the source, but it is quarantined
because live hardware differentials produced large errors. Reduced bandwidth
is not a feature until exactness is established.

The maintained Conv2D contract is documented in [AGENTS.md](AGENTS.md), and the
legacy evolution and rerun are summarized in [REPORT.md](REPORT.md).

### 3. QPU-XLA shifted attention from kernels to boundaries

The August 2026 QPU-XLA work introduced host-mapped QPU-visible tensors,
queues/events, buffer access declarations, reusable kernel objects, persistent
plans, operator contracts, calibration, and explicit CPU/QPU placements. It is
a hand-written heterogeneous runtime prototype, not a complete XLA compiler.

This runtime changed what “QPU time” meant. Assembly, allocation, transformed
weights, and uniform storage could be retained. Host preparation, QPU execution,
and result handling could be measured separately. CPU and QPU could operate on
disjoint output regions through separate queues. Exact-shape calibration could
keep a correct-but-slow kernel available for experiments without selecting it
automatically.

The main contribution of this phase was therefore not a single instruction
sequence. It was making residency, dependencies, arithmetic contracts,
placement, and measurement first-class runtime concepts. See
[QPU-XLA.md](QPU-XLA.md) for the current architecture.

### 4. TinyLlama and W8A8 showed where native dot product helps

The packed W8A8 family aligned four signed INT8 values in every `uint32` word
with VideoCore VII’s signed `v8dot`, accumulating into INT32. Prepared linear
plans retained packed weights and activation/accumulator workspaces. A fused
store path applied row and column scales while writing FP32, avoiding a
standalone dequantization traversal.

Several prefill projections won, but decode did not. One-layer composition
also demoted candidates that looked good in isolation. This established two
important policies:

- `same-contract-win` is not enough if the deployable FP32 CPU path is faster;
- a standalone stage win is not enough if its enclosing layer regresses.

Only exact `supported-win` records enter automatic placement. The candidate
registry is consequently a measured allowlist, not a table of kernels that are
merely capable of executing.

### 5. Native llama.cpp integration made the CPU advantage concrete

The native bridge consumed pinned GGML Q4_0, Q4_K, Q6_K, Q8_0, and activation
formats rather than converting the model into a QPU-friendly synthetic layout.
It added persistent programs, weights, uniforms, staging buffers, failure
recovery, program hashes, graph telemetry, exact token comparisons, and
process-isolated evaluation.

The result was mostly negative, and highly informative. Exact quantized QPU
linears lost to CPU_REPACK; fused attention lost to the native CPU node; graph
splits around small operations caused severe regressions; and hybrid overlap
often moved work without shortening the critical path. Approximate column-W8
and a complete FFN-island screen became more promising as M increased, but
neither cleared the complete-request promotion gate.

The authoritative summaries are
[the native result report](experiment_logs/20260819-llama-cpp-qpu/RESULTS.md)
and
[the agentic-prefill implementation report](integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md).

### 6. SmolVLA made model fit, provenance, and memory part of correctness

The final merge added a complete 450M-parameter vision-language-action graph,
from three RGB observations through a 50-action flow chunk. It includes a
pinned upstream Torch oracle, strict checkpoint provenance, replay artifacts,
FP32 and dynamic-W8A8 modes, native QPU and hybrid stages, hierarchical
placement gates, and pre-driver memory planning.

This phase adds an essential final lesson: a model is not a useful accelerator
target merely because a few operators are supported. The exact topology must
fit with operating-system and camera headroom; its preprocessing, KV state,
action loop, and numerical semantics must be reproducible; and reduced test
models must never be presented as production performance. The current
production-size latency remains a benchmark deliverable, not a checked-in
claim. See [SMOLVLA-QPU.md](SMOLVLA-QPU.md).

## A practical mental model of this QPU

### Parallelism is geometric, not automatic

The project targets twelve VideoCore VII QPU cores. Kernel work is expressed
through compute workgroups and lane-oriented vector operations. The common
matrix schedules use 16-wide dimensions because one 16×16 output tile maps
naturally to the execution and register layout. The runtime’s `wgs_per_sg`
setting controls how submitted workgroups are scheduled; tested values include
24 and 48, with 192 screened for selected large-M kernels.

None of those values is a universal optimum. A workgroup setting changes the
number of independent streams, their memory pressure, and how much latency can
be hidden. The large-M column-W8 screen improved the isolated suffix boundary
by roughly 4–5% at 192 workgroups per supergroup, but that was insufficient to
move the full request from about 1.014× to the 1.05× promotion threshold.

The practical unit of parallelism is therefore not “number of tensor
elements.” It is complete independent work that can be assigned without
creating expensive tails, duplicated preparation, or cross-workgroup
coordination.

### The driver boundary is real

QPU-XLA queues are asynchronous from the application’s perspective, but the
underlying driver execution remains synchronous inside a queue worker. A QPU
launch carries fixed submission, synchronization, and cache-management costs.
Multiple tiny nodes remain multiple costly boundaries even if their total
arithmetic looks large on paper.

The MTP experiment made this visible. A conventional device backend inserted
scheduler boundaries around many M=1 nodes and produced 1,753 QPU dispatches;
the 64-token run slowed from 22.476 seconds to 50.845 seconds. An inline CPU
hook removed the graph-split overhead, but the actual K=256, N=2048 operator
still took about 350 microseconds on QPU versus about 25 microseconds in the
relevant one-thread CPU_REPACK configuration. The boundary was not the only
problem—the kernel itself was a poor match.

### Shared visibility is not free access

Host-mapped buffers let NumPy views and QPU addresses refer to the same
allocation. That removes a conceptual host/device-copy API and enables disjoint
CPU/QPU work. It does not eliminate:

- cache synchronization and DMA-BUF access fences;
- conversion into a kernel’s preferred layout;
- CPU quantization or deinterleaving;
- padding and address-table construction;
- output joins and fallback repair.

The native GEGLU path improved when cacheable DMA import replaced uncached BO
readback. Direct AArch64 NEON deinterleaving later reduced representative M=257
Q8_0x4 activation staging from roughly 1.3 ms to 0.12–0.16 ms. Both were real
improvements, but neither made the exact Q4 consumer competitive. Optimize
each boundary, then remeasure the whole dependency chain.

### Hardware access has separate privilege and reproducibility domains

Normal compute uses the DRM V3D render node and should require only the
appropriate device-group permissions. Direct register and performance-counter
experiments may depend on `/proc` address discovery or `/dev/mem` privileges;
the legacy GPU-clock rerun was blocked for that reason even while ordinary QPU
compute remained available.

Do not make privileged observability a hidden prerequisite for inference.
Record clocks, throttling, and driver state when available, but keep the normal
execution path on the supported DRM interface. A benchmark failure caused by
missing performance-counter access is operationally different from a kernel
correctness failure.

## Kernel design lessons

### 1. Build the numerical contract before the fast path

Every useful kernel needs an explicit statement of:

- input, accumulation, and output dtype;
- layout, alignment, and shape restrictions;
- overflow behavior and valid operand range;
- rounding, activation, masking, and tie-breaking semantics;
- which output bytes may be modified;
- the reference used for differential testing.

This matters especially for integer work. The INT32 GEMM uses `smul24`, so both
operands must fit the signed 24-bit range and the accumulation must fit INT32.
MLP and attention paths must validate intermediate hidden values and attention
scores, not only original inputs. W8A8 uses a different contract: signed
`v8dot` consumes four packed INT8 pairs and accumulates exactly into INT32, with
host rejection of unsafe worst-case sums.

FP32 and quantized model semantics require the same discipline. The fused
GEGLU-to-Q8 producer once appeared byte-exact on its retained seed, but an
M=272 adversarial differential found two one-code differences caused by a
one-ULP reciprocal variation near half-integer rounding boundaries. Its
dequantized maximum error was only 0.004749, yet the byte-exact claim was no
longer true. It was correctly reclassified as an approximate research
primitive.

### 2. Derive tiles from data movement and registers

The packaged FP32 and INT32 GEMMs use a logical 16 output rows × 16 output
columns × 4 reduction-values tile. The packed W8A8 path uses 16×16×16 because
four signed bytes travel per word and four packed words are consumed per
reduction iteration.

Those tile sizes encode several coupled decisions:

- sixteen output accumulators can remain in registers;
- A values can be reused across output columns;
- B values can be rotated/broadcast across lanes;
- TMU requests can be interleaved with multiply/add work;
- stores can be emitted in contiguous groups.

The fast SGEMM experiment extended this reasoning by computing two adjacent
16×16 output tiles while reusing A. That increases arithmetic reuse, but it
also increases register pressure and scheduling complexity. An optimization
is valid only when the resulting register allocation, load pipeline, and
numerical output remain correct.

Padding is an implementation tool, not free compute. Host helpers can hide
tile constraints from callers, but the additional bytes and preparation time
must still be charged to the candidate. Direct execution from already aligned
caller buffers was one reason later FP32 prefill projections improved.

### 3. Schedule TMU latency instead of waiting for it

The Tensor Memory Unit is central to these kernels. The effective pattern is:

1. issue future A/B or vector loads;
2. compute on previously returned values;
3. overlap address updates, rotations, and dual-issued arithmetic;
4. consume the next values only after useful independent instructions;
5. group stores into the widest validated contiguous transaction.

Early SGEMM improvements pretransformed strided input access so TMU loads
matched the kernel’s traversal. Later kernels batch scale requests, pipeline
packed quantized values, and use scalar fallbacks only when a vector contract
is not satisfied. Merely replacing scalar arithmetic with vector arithmetic is
not enough if each iteration immediately stalls on the next TMU result.

The exact Q4_0 path remains a warning: frequent per-block scale handling and
shared-memory traffic can dominate even when the integer reduction uses native
`v8dot`. Instruction selection does not repair an unfavorable data format.

### 4. Exact stripes are a performance feature

The strongest bandwidth-oriented kernels process four contiguous words per
lane and distribute exact stripes across available QPUs. This produced clear
wins for large copy, min/max, residual, ReLU, and pooling cases. Nearby shapes
can fall off a cliff when they lack enough exact vector stripes and select the
generic scalar path.

For example, min/max at length 4,194,048 used the vector path and measured
about 1.55×/1.52× versus NumPy. The nearby length 4,194,240 did not have enough
exact four-word QPU stripes, fell back to the scalar path, and measured about
0.91×. “Nearly the same number of elements” did not imply nearly the same
schedule.

Kernel planners should therefore expose alignment and exact-divisor capability
checks explicitly. The prepared pooling path improved when it selected the
largest QPU count up to twelve that exactly divided the vector stream instead
of supporting only a one-or-twelve-QPU choice.

### 5. Preserve alignment across lazy allocations

Vector TMU performance depended on 256-byte tensor-base alignment. The
low-level allocator is linear, so a small cached uniform allocation could
misalign every tensor created after it. The runtime reserves 256-byte uniform
blocks for vector FP32 stages and tiled GEMM even when the kernel consumes far
fewer words.

This is a systems lesson disguised as an allocator detail: persistent metadata
must respect the alignment contract of allocations that do not exist yet.
Alignment cannot be checked only at the first benchmark buffer.

### 6. Fuse traversals, not just function names

Fusion helps when it removes memory traffic, staging, or synchronization. The
fused bias+ReLU epilogue wins because it replaces two CPU passes with one QPU
traversal. The fused W8A8 GEMM/dequantize kernel applies row/column scales in
the native-dot store path, eliminating a separate accumulator traversal.

Fusion is less useful when it only combines launch syntax while the dominant
reduction and memory traffic remain. The persistent exact-Q4 prototype placed
two full projections and a global barrier in one CSD. It proved that private
uniform streams, callable projection phases, and global handoff worked, but it
improved two-projection execution by only about 1.036×. Launch collapse was not
the missing order-of-magnitude gain.

Likewise, the resident GEGLU-to-Q8 producer was locally faster at M=257, but
the following QPU Q4 down projection erased the win. A fused producer has
value only when its resident format feeds a competitive consumer.

### 7. Persistence is the default inference architecture

Repeated inference should retain:

- the driver/device and assembled program;
- uniform and launch metadata;
- transformed or packed weights;
- address tables and padding plans;
- activation, accumulator, and output workspaces;
- model KV caches and other long-lived state.

This progression began with `TiledMatmulExecutor` and continued through
persistent convolution/MLP/attention plans, prepared W8A8 linear operators,
the native llama.cpp bridge, and the complete SmolVLA runtime.

Persistence removes repeated setup, but it also creates engineering
obligations: memory budgeting, cache synchronization, stale-view protection,
source/program hashes, allocation reuse rules, and failure cleanup. It is a
runtime design, not merely a benchmark optimization.

## What the QPU did well

The successful cases share regularity, width, reuse, and a low boundary-to-work
ratio. The following table selects representative results from the maintained
[FP32 kernel suite](experiment_logs/20260819-qpu-xla-kernel-suite/README.md)
and [QPU-XLA status](QPU-XLA.md). It intentionally does not list every winning
shape.

| Family and exact shape | Winning implementation | Measured result | Lesson |
|---|---|---:|---|
| FP32 min/max, 4,194,048 values | Exact-stripe four-word TMU | 1.55× / 1.52× | Large contiguous vector streams amortize launch and map cleanly across QPUs |
| FP32 residual add, 4,194,048 values | Exact-stripe four-word TMU | 1.41× | Simple bandwidth work can win when the tensor is large enough |
| FP32 max-pool, `1x48x128x128` | Prepared multi-QPU pooling | 1.64× | Cached address metadata and exact QPU divisors matter |
| FP32 bias+ReLU, `16x262128` | Fused single pass | 1.92× | Removing a full memory traversal can matter more than extra arithmetic |
| FP32 down projection, `16x1536x512` | QPU-only tiled GEMM | 1.19× | Aligned prefill can provide enough work without staging |
| FP32 gate projection, `64x1024x2816` | 16 QPU rows plus NumPy tail | 1.22× | A small calibrated QPU prefix can complement a strong CPU path |
| FP32 SwiGLU, `256x2048` | Vector QPU path | 2.30× | Wide independent rows suit vector elementwise kernels |
| FP32 RMSNorm, `256x2048` | Vector row reduction | 2.04× | Reductions can win when there are enough regular rows |
| FP32 softmax, selected 256-token shape | QPU or high-row hybrid | 1.24× | Numerically stable reductions are viable at the right scale |

The packed W8A8 study provides a second positive family. Selected prefill
projections measured 1.21–1.68× at `64x1024`, 1.09–1.30× at `128x2048`, and
3.10–5.32× at `16x4096` against the fastest measured deployable FP32 CPU
reference. The corresponding one-layer mixed runtime measured 2.86×, 1.54×,
and 1.16× for the 512-, 1024-, and 2048-hidden tuning cases.

These results teach five things:

1. **Prefill is more promising than decode.** Multiple rows expose reusable,
   tile-shaped parallel work and amortize fixed costs.
2. **Width can compensate for small row count.** The `16x4096` projection cases
   provide enough output/reduction work to use the QPU effectively.
3. **Native arithmetic matters.** W8A8 aligns with `v8dot`; exact GGML formats
   require more unpacking and scale handling.
4. **CPU/QPU coexistence can beat either alone.** Several FP32 wins use only a
   16-row QPU prefix while NumPy computes the tail.
5. **Fusion and prepared state are part of the kernel.** The winning boundary
   includes cached operands and, where applicable, the fused epilogue.

The results do **not** establish that all large tensors or all prefill
projections should run on the QPU. The W8A8 `256x2048` holdout measured 0.83× at
the one-layer boundary despite a standalone SwiGLU win. That counterexample is
why exact-shape and enclosing-layer gates are permanent parts of placement.

## What the QPU did not do well

### Tiny and single-row work

Every tested one-token FP32 GEMV remained slower than both CPU choices. The
complete W8A8 decode runtime measured only 0.60× CPU at cache 512, 0.48× at
cache 2048, and 0.38× for the 3072-hidden/cache-4096 holdout. Exact MTP drafting
was even clearer: after graph-split overhead was removed, the relevant
one-thread CPU_REPACK node was still roughly fourteen times faster than the
QPU boundary.

Decode is hostile to this implementation because it combines:

- too little row parallelism;
- repeated small launches;
- per-token synchronization;
- dequantization or block-scale work;
- a highly optimized CPU implementation with hot model state.

Output-column partitioning can expose some parallelism for M=1, but it cannot
erase fixed costs when the complete QPU slice is slower than the CPU slice it
replaces.

### Native GGML block-quantized reductions

The exact native-format matrix is decisive:

| Candidate | Exact shape | QPU-only result versus CPU | Best measured non-CPU result |
|---|---:|---:|---:|
| Gemma drafter Q4_0 LM head | `1x256x262144` | 0.079× | 1/12-output hybrid still slower |
| Gemma drafter Q4_0 LM head | `4x256x262144` | 0.132× | 1/4-output hybrid still slower |
| Qwen Q4_K FFN gate | `4x2560x9216` | 0.060× | Hybrid still slower |
| Qwen Q6_K FFN down | `4x9216x2560` | 0.025× | Hybrid still slower |
| Qwen Q8_0 SSM output | `4x4096x2560` | 0.082× | Hybrid still slower |

The QPU-only ratios above are derived from the recorded CPU and QPU medians;
the source report retains the underlying times and error bounds.

The QPU results were numerically correct within the established FP32
tolerances. The loss came from the execution contract: low row count, complex
packed fields, per-block scales, unpacking, and memory traffic. Correct native
format support is valuable integration evidence, but it is not automatically
an efficient accelerator format.

### Current attention kernels

The first tiled attention example implemented only `(Q @ K.T) @ V`; it omitted
softmax, masking, and scale. The packaged SDPA path added those semantics but
kept softmax as a CPU host stage until a validated QPU implementation existed.
Later kernels implemented online softmax for the exact Gemma subset and batched
query rows and heads into one launch.

Neither approach beat the optimized CPU node. M=1 persistent attention at
context 4096 measured 8.561 ms versus 1.600 ms for CPU. The later one-launch
batched kernel measured between 0.328× and 0.588× CPU for tested M=65/257 and
contexts 512/4096. Flattening the launch geometry removed dispatch multiplicity
but exposed insufficient compute/memory throughput inside the kernel.

This is an important diagnostic distinction:

- if batching launches creates a win, dispatch was the bottleneck;
- if one batched launch still loses, the kernel’s data movement, reduction,
  or arithmetic schedule must change.

### Spatial convolution through GEMM

The current convolution path lowers NCHW to matrix multiplication. General
windows use `im2col`, which duplicates inputs, expands memory traffic, and adds
host preparation. Pointwise 1x1 avoids window duplication by packing pixels
directly into GEMM rows, but it is still a GEMM-backed pointwise path rather
than a direct spatial microkernel.

The packed W8A8 YOLO measurements show the limitation:

- p3 1x1: 0.13× Torch native FP32 convolution;
- p3 3x3: 0.064×;
- p3 depthwise: 0.012×.

The 1x1 result proves that avoiding window expansion is helpful but
insufficient. Layout transformation, tile padding, launch overhead, and the
quality of Torch’s native spatial kernel still dominate. For 3x3, duplicated
windows worsen the boundary. Depthwise convolution has little reduction reuse
per output channel and is especially poorly matched to the dense GEMM tile.

The GEMM-backed path should remain the correctness and fallback substrate.
Competitive convolution needs direct kernels specialized for common 1x1 and
3x3 layouts, with grouped/depthwise designs that do not manufacture dense work.

### Irregular small operators

Embedding, argmax, small residual, and KV append all have valid QPU kernels,
but the tested TinyLlama-sized shapes remained CPU-preferred. Address-table
gathers and first-index tie behavior can be implemented exactly; exactness does
not create enough work to amortize submission. These kernels are still useful
inside a future resident fused region or for much larger shapes, but they
should not be placed individually on current evidence.

## Heterogeneous execution lessons

### Partition along an ownership boundary

The runtime supports two useful partition axes:

- **rows**, when batches/tokens are independent and each side can produce a
  disjoint set of output rows;
- **output columns**, especially for single-row projection where row splitting
  is impossible.

Both partitions share read-only input state and assign non-overlapping output
regions to CPU and QPU. This avoids merge arithmetic and makes failure repair
well defined. The CPU and QPU must use separate queues if they are intended to
overlap.

Partition axes are not interchangeable. Grouped convolution requires aligned
output slices within each group, and depthwise output splits are explicitly
unsupported. A planner must understand the operator’s ownership semantics,
not just divide a dimension numerically.

### Balance the critical path, not the work count

The best partition is where CPU and QPU complete near the same time after all
boundary costs. Equal rows or equal FLOPs are usually wrong because the two
processors have different kernels and overheads.

The exact `ffn_up` overlap demonstrates the limit. At M=257 and a 1/4 QPU
output fraction, the complete QPU boundary took 40.66 ms per layer while useful
CPU work overlapped for 21.43 ms, exposing a 19.23 ms QPU tail. Across 35
layers that tail was about 0.673 seconds. Reducing the complete QPU boundary to
the 21.4 ms CPU overlap window—about a 1.9× improvement—was projected to yield
roughly a 1.06× request speedup.

At a smaller QPU fraction, more QPU work could be hidden but too little CPU
work was removed. At a larger fraction, the QPU became the critical path.
Partition search is therefore a latency-balancing problem under Amdahl’s law,
not an offload-percentage contest.

### Measure the join

A correct hybrid timing includes:

- input preparation for both partitions;
- cache/DMA synchronization;
- QPU submission and execution;
- simultaneous CPU work;
- the wait for the slower side;
- output synchronization or copy;
- the operator or graph join.

Reporting CPU duration plus QPU duration would double-count overlap. Reporting
only the QPU event would omit exposed tail and result handling. Wall-clock time
through the join is the placement metric; component timings explain it.

### Make fallback restore the full contract

The llama.cpp overlap path injects failures at allocation, submission, wait,
and hash-validation stages. If failure occurs after asynchronous QPU work owns
an output suffix, that suffix is recomputed on CPU before the following GEGLU
reads the combined result. Falling back by merely returning an error or leaving
the region untouched would corrupt the graph.

This is a general heterogeneous-runtime rule: once ownership has been split,
fallback must reconstruct the complete logical output, not just choose a new
backend for future calls.

### Calibration must be exact and conservative

Hybrid placement is excluded from `AUTO` without an exact-shape retained
record. The calibrated key needs shape, dtype, layout, partition axis,
partition size, implementation/source hash, and relevant numerical mode.

Interpolating from a nearby shape is unsafe. The min/max exact-stripe cliff,
the W8A8 256-token holdout regression, and different optimal FFN fractions at
M=65/129/257 all show that small shape changes can select a different schedule
or move the critical path.

## Arithmetic and quantization lessons

### Use the format the instruction wants—when the model contract permits it

Signed W8A8 is the best arithmetic match found in this project because four
signed bytes fit each word and `v8dot` performs the intended dot product
directly. Per-output-channel weight scales and per-row activation scales also
permit a simple fused FP32 epilogue.

By contrast, exact GGML block formats were designed around CPU storage and
dequantization kernels. Their fields, scales, and block sizes can be decoded on
the QPU, but the extra scalar and TMU work weakens the benefit of native dot
product. Keeping the native model format avoids an up-front conversion and
additional resident weights, while converting once to column-W8 simplifies
the kernel but changes arithmetic and consumes memory. Neither choice is free.

The placement decision must compare complete alternatives:

- native exact format with complex per-block work;
- persistent converted format with extra memory;
- dynamic activation quantization with per-request preparation;
- FP32 CPU execution with no quantization boundary.

### Exactness has layers

The project encountered at least four useful notions of correctness:

1. **Bitwise kernel contract**, such as exact INT32 accumulation or exact Q8
   bytes.
2. **Numerical operator tolerance**, such as FP32 maximum absolute error.
3. **Model-semantic agreement**, such as unchanged greedy token IDs or action
   cosine similarity.
4. **Task-quality agreement**, such as perplexity or multi-prompt behavior over
   a longer generation.

An approximate kernel can pass a single greedy-token case by margin, while a
small logit perturbation changes a different prompt later. The retained
column-W8 request results prove correct execution for the fixed cases; they do
not prove general quality equivalence. Approximate paths require explicit
logit, greedy divergence, perplexity/task, and long-generation gates before
deployment.

### Quantization can save memory without saving latency

For SmolVLA, the native planning estimates are:

| Contract | Artifact | QPU arena | Projected peak RSS |
|---|---:|---:|---:|
| FP32 | 1.677 GiB | 0.906 GiB | 3.259 GiB |
| Dynamic W8A8 | 0.687 GiB | 0.625 GiB | 1.989 GiB |

That reduction is operationally valuable on an 8 GiB system even if some
stages remain on CPU. It creates headroom for the operating system, camera
pipeline, and process overhead. But activation packing, scale application, and
QPU boundaries may still make a quantized kernel slower. Memory fit and latency
acceleration are independent acceptance criteria.

## Operator and model composition lessons

### A reusable GEMM substrate is necessary but not sufficient

Tiled GEMM enabled convolution, MLP, attention score/value products, linear
layers, and patch embeddings without writing a new reduction core each time.
It also made padding, persistent weights, differential testing, and hybrid
partitioning reusable.

The abstraction breaks down when an operator’s natural reuse differs from
dense GEMM. Depthwise convolution, single-row quantized decode, and online
attention each need specialized dataflows. A general substrate is the right
correctness baseline and development scaffold; hot cases still need direct
microkernels.

### Keep semantically different operators distinct

Names influence benchmarks and design decisions. The repository learned to
state precisely that:

- NumPy `im2col + dot` is a lowered-convolution baseline, not native
  convolution;
- the tiled attention core `(Q @ K.T) @ V` is not full scaled masked softmax
  attention;
- one-layer TinyLlama composition is not a complete model benchmark;
- a complete FFN island component screen is not a llama.cpp request result;
- QPU-XLA is a constrained runtime/DSL prototype, not a finished XLA compiler.

Accurate naming prevents a narrower success from being carried forward as a
larger claim.

### Optimize islands, not isolated nodes

The most promising later direction was a channel-partitioned complete FFN
island. CPU computed one channel slice while QPU computed the other through
gate, up, GEGLU-to-Q8, and down; only hidden-size FP32 partial results crossed
the join. At M=257 the component screen reached 1.154× on the narrow layer and
1.088× on the wide layer. Larger M screens produced weighted estimated
FFN-only speedups around 1.10–1.19×.

Those numbers were not promoted because the boundary was optimistic and did
not yet exist as the production llama.cpp hook. A real M=513 request using the
implemented `ffn_up` overlap reached only 1.0143× in its held-out pair.

The durable lesson is nevertheless useful: a good accelerator boundary should
contain several producer/consumer stages, keep a QPU-friendly intermediate
resident, and return only a compact result. This reduces launches and format
crossings while exposing enough independent work. It must still be integrated
and measured in the real graph.

### Profile the graph before choosing the next kernel

Static tensor inventories identify shapes, but they do not show frequency,
thread configuration, cache state, or critical-path share. The MTP study first
looked attractive against a four-thread CPU comparison because thread-team
startup dominated the tiny CPU node. Production drafting actually used one
thread, where CPU_REPACK was about fourteen times faster.

A production graph profile must establish:

- the exact executed node and tensor layouts;
- row count, context, and batch behavior at runtime;
- how many times the node appears per request;
- the actual CPU thread count and implementation;
- dependencies and work that can overlap;
- an upper bound on end-to-end benefit.

Only then is assembly work justified.

## Which models and workloads fit better

### Better candidates

The evidence favors models or subgraphs with:

- **prefill or batched tokens**, rather than only single-token decode;
- **wide dense projections** with K/N dimensions compatible with 16-wide
  tiling;
- **repeated fixed weights**, allowing conversion and residency once;
- **large regular elementwise/reduction stages**, especially when multiple CPU
  passes can be fused;
- **stable shapes**, so exact calibration records remain reusable;
- **operator islands**, where intermediate layouts stay on the QPU side;
- **enough total work but moderate total model size**, preserving system memory
  and avoiding paging.

TinyLlama-style prefill was useful for kernel calibration because it combines
moderate row counts with broad projection and activation dimensions. SmolVLA
is a better end-to-end systems baseline than a 3.5–4B VLA model because its
450M parameters and W8A8 artifact leave credible memory headroom on an 8 GiB
Pi while still exercising vision preprocessing, transformers, KV caches,
attention, and iterative action generation.

Vision preprocessing may also be attractive when resize, normalization,
gather, and patch extraction operate over substantial regular images and their
outputs feed resident projection stages. The final SmolVLA implementation
provides those native stages, but production checkpoint timing is still
required before declaring them wins.

### Worse candidates

Current evidence disfavors:

- **decode-dominated LLMs** where most expensive nodes have M=1;
- **speculative drafters composed of many tiny projections**;
- **models whose native quantized format requires heavy per-block unpacking and
  scaling**;
- **depthwise-heavy CNNs** mapped through dense GEMM;
- **dynamic shapes** that frequently miss exact tile and registry entries;
- **graphs with many CPU/QPU crossings** or host-only operations between every
  accelerated stage;
- **models that only barely fit**, because swap and memory pressure invalidate
  performance and reliability;
- **large models that fit only after quantization but still require impractical
  per-token compute**.

The SmolVLA analysis estimates that FP32 weights alone for a 3.5–4B
PaliGemma-class model exceed 13 GiB, before activations, KV state, QPU staging,
framework overhead, or the action expert. A 16 GiB device and aggressive
quantization might make the bytes fit, but does not make multi-billion-
parameter decode a sensible baseline. Start with a model whose complete
workload is credible, then scale.

### Model selection scorecard

Before adopting a model as an accelerator target, record:

| Dimension | Favorable signal | Warning signal |
|---|---|---|
| Memory | Planned peak leaves OS/runtime headroom | Relies on swap or near-total RAM occupancy |
| Shape | Stable aligned prefill batches | Mostly M=1 or changing tails |
| Arithmetic | FP32 regular tiles or signed W8A8 | Complex per-block decode dominates |
| Reuse | Weights and workspaces persist | Repack/reallocate every call |
| Graph | Few broad islands | Hundreds of tiny backend transitions |
| CPU baseline | Meaningful expensive node | Highly optimized CPU node already sub-millisecond |
| Quality | Independent oracle and replay | Only synthetic random tensors |
| Benefit | Node/block exceeds an Amdahl gate | Even infinite node speedup cannot move request 5% |

## Runtime architecture lessons

### Make ownership explicit

`Tensor`, `Buffer`, and `AccessMode` establish which byte ranges a task reads or
writes. Queue dependencies serialize conflicts and reject foreign-device
resources. Derived views become invalid when their allocation closes, and a
reused allocation does not revive stale views.

These contracts seem conservative for a small prototype, but heterogeneous
execution needs them. Without range ownership, two queues can race on a shared
mapping or a fallback can overwrite a valid partition. Correctness cannot
depend on the current scheduler accidentally being in order.

### Separate capability, policy, and mechanism

The repository now has three distinct questions:

- **Capability:** can this kernel execute the exact dtype, layout, and shape?
- **Mechanism:** how are buffers, queues, programs, and partitions used?
- **Policy:** should this exact candidate run automatically on this machine?

A hardware-differential-tested kernel may be capable but policy-disabled. A
forced QPU mode remains valuable for evaluation while `AUTO` uses CPU. Keeping
these layers separate allowed correct-but-slower GEMV, dequantization,
convolution, and llama.cpp kernels to remain available without creating a
performance regression for normal users.

### Fail early on impossible deployments

SmolVLA computes artifact, arena, and projected RSS budgets before opening the
VideoCore driver. It defaults to a 6 GiB process limit even on an 8 GiB system.
Checkpoint and replay loaders validate revision, tensor topology, shape, and
dtype before expensive initialization.

This is preferable to discovering an unsupported topology after allocating
half the model or entering swap. Memory and provenance are input contracts,
not post-benchmark observations.

### Process isolation can be part of the runtime benchmark

The upstream Torch policy, FP32 native artifact, W8A8 artifact, and QPU arena
cannot all coexist comfortably on the target machine. The SmolVLA benchmark
runs modes in fresh subprocesses. Native llama.cpp evaluation also uses fresh
servers and independent sessions.

Isolation reduces memory interference and makes cold-start accounting honest.
It also requires stronger provenance and serialized result records so that
outputs from different processes remain comparable.

## Correctness and validation lessons

### Use a ladder of independent references

The strongest validation combines:

1. a transparent scalar or NumPy formulation;
2. an optimized CPU implementation such as Torch or GGML;
3. hardware differential tests for the QPU kernel;
4. a complete graph replay;
5. an upstream model oracle when one exists.

Agreement between two implementations that share lowering or packing code can
preserve the same bug. Conv2D therefore compares im2col/GEMM with explicit
convolution and Torch. GGML kernels compare native block decoding with the
pinned CPU_REPACK path and guarded hardware outputs. SmolVLA pins LeRobot and
the checkpoint revision and records upstream actions into portable replays.

### Test shape boundaries and untouched regions

Random “happy path” matrices are insufficient. The hardware suites cover:

- multiple output tiles and non-tile logical shapes;
- every aligned partition of resident output ranges;
- invalid shapes, layouts, partitions, and operand bounds;
- first-index ties for argmax;
- negative integer average-pooling semantics;
- causal positions and KV-cache behavior;
- adversarial quantization rounding;
- canary or guarded buffers around outputs;
- repeated launches and context recovery after failure.

Partition tests must verify both computed values and preservation of the CPU-
owned region. Buffer safety is part of numerical correctness.

### Quarantine beats optimistic fallback

The project repeatedly kept experimental assembly available without routing
public or automatic execution through it:

- packed INT16 convolution remained quarantined;
- current YOLO W8A8 convolution stayed scheduler-invisible;
- llama.cpp quantized and attention kernels remained opt-in;
- column-W8 stayed research-only after failing end-to-end gates;
- GEGLU-to-Q8 was downgraded after an adversarial byte mismatch;
- persistent multi-phase Q4 was not exported as a candidate.

This preserves research value without weakening stable contracts. A
correctness reference should remain simple even when it is slower.

### Reproduce executed semantics, not configuration intent

SmolVLA exposed a subtle example: the pinned LeRobot `apply_rope` helper uses
its own default wavelength of 10,000 even though the underlying text config
contains 100,000. The native oracle follows the executed upstream helper.

Model ports must trace actual control flow and defaults. Configuration files,
paper descriptions, and framework abstractions may not match the precise
runtime semantics that determine outputs.

## Benchmarking lessons

### Timing categories answer different questions

Keep these measurements distinct:

| Category | Includes | Question answered |
|---|---|---|
| Kernel execute-only | Driver execution with prepared resident buffers | Is the instruction schedule itself competitive? |
| Host preparation | Lowering, padding, packing, address tables, quantization | Is data preparation the bottleneck? |
| Cached whole operation | Preparation as defined, synchronization, submission, execute, and result handling with persistent setup | Is repeated operator use worthwhile? |
| Hybrid wall time | Both processors through their join | Does concurrency shorten the critical path? |
| Cold start | Driver, assembly, allocation, loading, first full operation | Is one-shot use practical? |
| Block/layer/model | All enclosing work and boundaries | Does the accelerator improve the deployed workload? |

The earliest Conv2D benchmark charged driver creation, assembly, allocation,
im2col, padding, packing, upload, execution, and readback to QPU while CPU
baselines timed optimized computation. Splitting the categories corrected the
comparison. It did not guarantee a win; it made the reason for a loss visible.

### Benchmark the fastest relevant CPU path

CPU winners change by operation and shape. Torch beat OpenBLAS for 64×64 FP32
matmul, while OpenBLAS beat Torch at 512×512. NumPy is appropriate for some
elementwise references, Torch for native convolution and several activations,
and pinned GGML CPU_REPACK for native quantized llama.cpp nodes.

The CPU thread count must match production. Comparing MTP’s one-thread draft
node against a four-thread CPU path measured thread-team startup rather than
the real competitor. Conversely, using a single CPU thread to make a larger
QPU operator look attractive would also be misleading when production uses
four.

### Retain the environment, not only the median

Eligible performance sessions record or enforce:

- CPU governor and relevant clock state;
- temperature and current/historical throttling;
- process swap and system memory pressure;
- competing server processes;
- CPU/BLAS versions and thread settings;
- model, tokenizer, runtime, source, program, and case hashes;
- warmups, raw samples, run order, and per-process medians;
- actual QPU dispatch counts and fallbacks.

The first llama.cpp sessions ran with the `ondemand` governor, full zram, and a
pre-existing server. Their correctness data remained useful, but their timings
were rejected for promotion. Evidence quality is a property of the environment
and protocol, not just sample count.

### Use held-out shapes and sessions

Tuning and evaluation should not use the same samples. W8A8 manifests separate
tuning and holdout shapes. Agentic-prefill evaluation calibrates partitions in
separate processes, then uses fresh paired sessions with randomized CPU/QPU
order. One median is taken per independent session before bootstrap analysis.

Promotion requires at least a 1.05× median complete-boundary speedup and a
bootstrap 95% lower bound above 1.0, in addition to correctness and telemetry.
The threshold prevents tiny noisy differences from becoming permanent runtime
policy.

### Stop after a lower-bound failure

An expensive full-model campaign is inappropriate when a lower-bound operator
test proves the candidate cannot win. The MTP inline comparison, Qwen exact
quantized rows, and batched attention diagnostics all stopped further promotion
work. Likewise, a component screen below 1.05× does not become promising by
collecting more identical sessions; more samples narrow uncertainty but do not
create effect size.

Negative results preserve engineering time when they include enough telemetry
to identify whether the next change needs a better kernel, data boundary,
partition, or workload.

## Course corrections that became durable rules

| Initial interpretation or attempt | Evidence that changed it | Durable rule |
|---|---|---|
| `--num-qpus` was discussed as a larger hardware unit | Payload and rerun audit showed it means QPU cores | Define hardware terminology and inspect launch payloads |
| One-shot QPU Conv2D looked disproportionately slow | Setup, lowering, allocation, and I/O were mixed into one timing | Separate cold start, prep, cached total, and execute-only |
| Packed INT16 promised lower bandwidth | Live numerical errors remained large | Keep widened exact public path; quarantine packed assembly |
| Tiled Conv2D could be described as convolution acceleration | Implementation and Torch comparison showed im2col/GEMM behavior | Name lowering honestly and compare with native convolution |
| The attention core represented transformer attention | It omitted scale, mask, and softmax | State the exact operator contract |
| A stage win justified runtime placement | One-layer W8A8 holdout regressed | Require enclosing-block and model gates |
| Graph-backend GEGLU kernel win would help requests | 35 graph splits made the request roughly 1.85× slower | Measure and minimize graph boundaries |
| Removing graph splits would rescue tiny Q4 decode | Inline M=1 remained about 14× slower than CPU | Apply an operator lower-bound gate before server campaigns |
| Batching all attention rows/heads would amortize launch | One batched launch still measured 0.328–0.588× CPU | Distinguish launch overhead from kernel throughput |
| A byte-exact seed established exact GEGLU-to-Q8 | Adversarial rounding found one-code differences | Test numerical discontinuities and classify approximation explicitly |
| More QPU fraction meant more acceleration | Larger splits exposed QPU tails; smaller splits moved too little work | Tune against the critical path, not offload percentage |
| Two projections in one persistent CSD might unlock a large gain | Measured improvement was only about 1.036× | Estimate the maximum value of launch removal before building a superkernel |
| An FFN component estimate implied a model win | Real M=513 request reached only 1.0143× | Treat component estimates only as candidate-selection evidence |
| A model that fits after quantization is a suitable baseline | Large-model compute and runtime overhead remain impractical | Gate models separately on memory, latency, and full-graph reproducibility |

## Recommended workflow for future QPU work

### 1. Establish the deployment case

- Pin the model, runtime, CPU implementation, thread count, and exact workload.
- Capture a production graph profile rather than relying on tensor metadata.
- Compute memory headroom and fail before driver initialization if unsafe.
- Estimate the maximum request-level benefit from the target region.

### 2. Define contracts and references

- Write down dtype, layout, shapes, arithmetic, tolerance, and ownership.
- Build a transparent independent reference and retain the optimized CPU
  baseline separately.
- Include edge shapes, invalid inputs, canaries, repeated launches, and failure
  recovery in the test plan.

### 3. Design from the memory path inward

- Choose a tile that matches lanes, registers, and contiguous TMU traffic.
- Decide whether weights, metadata, and intermediates can remain resident.
- Account for tails and exact QPU striping before coding.
- Map arithmetic to native instructions without changing the contract.
- Schedule prefetch and independent arithmetic to hide TMU latency.

### 4. Measure progressively

- Validate hardware output first.
- Record execute-only and each preparation/synchronization phase.
- Compare persistent whole-operation wall time with the fastest CPU backend.
- If hybrid, sweep meaningful partitions and measure through the join.
- Stop when a lower bound cannot meet the promotion threshold.
- Advance winners to block, layer, and complete-model replays.

### 5. Promote conservatively

- Store exact-shape results with raw samples, environment, and hashes.
- Separate tuning from held-out evaluation.
- Require numerical, semantic, and task-quality gates appropriate to the
  arithmetic contract.
- Keep CPU fallback complete and tested.
- Expose correct-but-slower kernels only through explicit evaluation modes.

## Highest-value remaining directions

The evidence suggests the following order for new work:

1. **Direct convolution kernels.** Implement common 1x1 and 3x3 dataflows, then
   a depthwise/grouped design, without im2col duplication. Preserve the
   GEMM-backed path as the reference and fallback.
2. **Resident operator islands.** Pursue producer/consumer regions where a
   QPU-friendly intermediate stays resident and only a compact result crosses
   the join. Integrate the real graph boundary before claiming the component
   estimate.
3. **Exact-Q4 boundary improvement only against a numerical target.** For the
   measured M=257 1/4-output case, require complete QPU time at or below roughly
   21.4 ms before another full request campaign.
4. **Graph-profile-driven prefill families.** Prefer shapes with multiple rows,
   wide outputs, persistent weights, and at least 5% plausible request impact.
5. **Memory-efficient SmolVLA evaluation.** Run production replay measurements
   for FP32, W8A8, forced QPU/hybrid, and gated AUTO in isolated processes;
   retain upstream action metrics and full RGB-to-action timing.
6. **Richer queue profiling.** Report CPU work, QPU work, actual overlap,
   exposed tail, cache synchronization, and join overhead directly.
7. **DSL expansion behind differential tests.** Automate only instruction and
   scheduling patterns whose source forms, numerical contracts, and hardware
   behavior are already understood.

The current evidence does not support prioritizing more M=1 decode kernels,
another row-fraction sweep of the same exact-Q4 implementation, or additional
GEMM-backed YOLO tuning. Those paths need a changed dataflow or boundary, not
more samples.

## Evidence map

Use these documents as the maintained entry points rather than copying a
single result out of context:

- [QPU-XLA runtime and current status](QPU-XLA.md): runtime semantics,
  operators, current kernel inventory, and placement status.
- [Conv2D Kernel Guide](AGENTS.md): current lowering, dtype, tiling, timing, and
  packed-INT16 quarantine contracts.
- [Historical report and legacy rerun](REPORT.md): evolution from the inherited
  base and the first standalone ML kernels.
- [Experiments registry](EXPERIMENTS_REGISTRY.md): canonical commands and
  interpretation of benchmark scripts.
- [FP32 kernel suite](experiment_logs/20260819-qpu-xla-kernel-suite/README.md):
  exact-shape FP32 wins, losses, and promotion rules.
- [W8A8 scoreboard](experiment_logs/20260819-qpu-xla-w8a8/KERNEL_SCOREBOARD.md):
  retained packed-kernel candidates and inventory.
- [Native llama.cpp results](experiment_logs/20260819-llama-cpp-qpu/RESULTS.md):
  exact quantized and attention measurements plus end-to-end evidence rules.
- [Agentic-prefill implementation results](integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md):
  inline boundaries, column-W8, batched attention, FFN islands, MTP, and
  persistent multi-phase experiments.
- [Concurrent GEGLU result](experiment_logs/20260824-qpu-agentic-prefill/HYBRID_RESULTS.md)
  and [full-model `ffn_up` overlap result](experiment_logs/20260824-qpu-agentic-prefill/UP_OVERLAP_RESULTS.md):
  partition and Amdahl evidence.
- [Native SmolVLA baseline](SMOLVLA-QPU.md): model topology, provenance,
  numerical gates, memory planning, and end-to-end replay contract.

## Final takeaway

The project began by asking whether VideoCore VII could execute useful ML
kernels. It ended with a more precise question: **which exact regions of which
real workloads can the QPU shorten after every preparation, synchronization,
CPU competitor, numerical contract, and downstream dependency is counted?**

The answer is neither “none” nor “everything.” The QPU has demonstrated real
advantages for selected aligned prefill projections, wide vector stages,
fused memory passes, and large regular tensors. It has also demonstrated clear
limits for single-token decode, native block-quantized reductions, current
attention, and GEMM-backed spatial convolution.

The most reusable achievement is the method that separates those cases:
hardware tests first; design around data movement; persist all reusable state;
calibrate exact shapes; compare with the real CPU path; measure through the
join; validate the enclosing graph; and keep every unsupported or unproven path
off by default. That is how to use this QPU efficiently—and how to know when
not to use it.
