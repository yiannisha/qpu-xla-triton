# QPU-XLA Rewrite and Roadmap Implementation Plan

Date: 2026-08-12
Repository: <https://github.com/yiannisha/qpu-xla>
Reviewed commit: `5c13e9714caa9524242affb59bbd8c6d3141082d` (`main`)
Reviewed working-tree additions: ML validation tests, `sgemm_batched_small.py`,
batched-SGEMM edits, experiment reports, and this plan are uncommitted.
Roadmap source status: the referenced `QPU-XLA Roadmap.pdf` is **not present in
this checkout**. The named roadmap requirements below are retained as planning
inputs, but their wording, priority, and the claimed page mapping must be
checked against the original PDF before this plan is approved.

## Current implementation addendum — 2026-08-13

This file began as a pre-implementation rewrite plan. Its statements that no
`qpu_xla` package, queue/event layer, shared-memory contract, or DSL exists are
historical observations from the reviewed commit; they are no longer current.
The implemented runtime is documented canonically in
[`QPU-XLA.md`](QPU-XLA.md). The plan below remains useful as a record of the
desired architecture and unfinished roadmap work, but the status is now:

- `src/qpu_xla` exists as an importable runtime package;
- `Device`, host-mapped `Tensor`/`Buffer`, `Queue`, `Event`, and access modes
  are implemented;
- reusable copy, FP32/INT32 GEMM, min/max, and pooling kernel adapters exist;
- CPU, QPU, calibrated whole-operator placement, and explicit CPU/QPU hybrid
  row-split matmul exist;
- packaged convolution, attention, pooling, elementwise, MLP, video, DSL,
  autotuning, and TinyLlama building blocks exist with varying maturity;
- persistent INT32 convolution, MLP, and attention plans exist;
- hardware-marked differential tests and a qpu_xla configuration-matrix
  benchmark exist;
- automatic hybrid partition selection, native quantized projection kernels,
  complete FP32 packaged plans, and a general XLA backend remain unfinished.

The current 512x512 FP32 benchmark has one measured heterogeneous win: an
explicit split of 128 QPU output rows and 384 CPU rows completed in 4.207 ms,
versus 4.932 ms for the CPU-only qpu_xla path on the tested machine. This is a
shape-specific CPU/QPU overlap result, not a claim that QPU-only FP32 GEMM is
faster than the CPU.

## Executive decision

QPU-XLA should be built as a real installable runtime *on top of* the existing
`py-videocore7` hardware layer. Do not rewrite DRM/V3D submission or the
instruction encoder unless an upstream limitation is proven. Freeze the current
example scripts as behavioral and performance references, then replace their
orchestration with a clean `qpu_xla` package. `videocore7` remains the low-level
package and internal backend for this first release.

At the time this plan was written, the main rewrite was necessary because there
was no `qpu_xla` package. The original checkout contained `videocore7` and
`_videocore7` (assembler, driver, DRM/V3D plumbing), while the ML stack was
embedded in example scripts:
10,597 lines across 14 tracked scripts, or roughly 11,100 lines across 15
scripts when the current untracked batched-SGEMM example is included. The
roadmap features need shared abstractions that those scripts do not provide.

Recommended delivery envelope, assuming two full-time engineers, continuous access to at least two Raspberry Pi 5 test machines, and no need to build a true XLA compiler backend in this release:

- Foundation rewrite: 13-17 calendar weeks.
- Complete roadmap, including TinyLlama and camera demo: 27-35 calendar weeks total.
- Effort: approximately 50-65 engineer-weeks.
- First usable SDK milestone: end of week 10-12.
- First end-to-end TinyLlama milestone: end of week 24-28.

The name `qpu-xla` currently does not correspond to an XLA backend. Before implementation starts, record an architecture decision that this roadmap targets a standalone Python runtime and DSL. A real XLA custom-call/plugin backend should be a separate later project.

## 1. Historical evidence from the pre-runtime repository state

The evidence in this section describes the repository state inspected when the
rewrite plan was authored. Use the current implementation addendum above and
[`QPU-XLA.md`](QPU-XLA.md) for present-day status.

### Current strengths to preserve

- The repository has a real VideoCore VII hardware foundation: an assembler,
  DRM-backed buffer allocation/mapping, program loading, and synchronous
  compute-shader dispatch in `_videocore7`.
- ML-oriented kernels and executors exist for FP32/INT32 GEMM, packed INT16
  GEMM, min/max, pooling, GEMM-backed convolution, MLP, an unnormalized
  attention core, and a LeNet-style pipeline. They are valuable migration
  references, not yet a stable public operator library.
- Persistent executors already demonstrate the right performance principle: assemble kernels and allocate buffers once, then reuse dispatch metadata.
- The current benchmark scripts separate host preparation, cached total, and
  execute-only timing for several higher-level paths. Preserve that taxonomy.
- Integer range contracts for `smul24` are explicitly checked in several examples.
- Host-side differential tests exist for the new convolution, MLP, attention,
  and LeNet helpers. They establish useful reference behavior, but they do not
  validate the QPU results on hardware.

### Correctness baseline and immediate constraint

Do not treat all current example kernels as known-good. The checked-in
conv2d guide states that FP32 and INT32 convolution are good for its tested
benchmark shapes, but packed INT16 convolution is explicitly untrustworthy and
has produced large numerical errors. The experiment report also records live
numerical problems in several optimized/higher-level paths. The first rewrite
phase therefore has a correctness-triage gate: reproduce each issue with a
small hardware test, repair or quarantine it, and retain the failing case.
No performance claim or migration gate may use a path as a reference until its
hardware differential test passes.

### Current structural gaps

| Finding | Evidence | Consequence |
| --- | --- | --- |
| No QPU-XLA runtime API | There is no `src/qpu_xla/`; `src/videocore7/driver/__init__.py` re-exports low-level `Array`, `Driver`, and `Memory` | No ML runtime or operator library is importable as a supported API |
| Product code lives in examples | 14 tracked example scripts (10,597 lines); the current tree adds a 15th untracked script | Reuse, compatibility, testing, and ownership are unclear |
| Tight dynamic coupling | `tiledmlp.py`, `tiledattention.py`, and `tiledlenet5.py` load sibling example files with `importlib` | Operators are not stable modules and private symbols form hidden APIs |
| Dispatch is synchronous and direct | `Driver.execute(...)` submits then its `Dispatcher` context waits on the BO; executors call it directly | No queue, event, dependency, host task, overlap, cancellation, or profiling abstraction |
| Shared memory has no runtime contract | `Driver.alloc(...)` makes NumPy `Array` views into one driver-owned BO and exposes physical addresses | No ownership, lifetime, coherency, alias, slice, or race rules above the raw driver |
| Fixed manual scheduling | Tile sizes, workgroups, QPU counts, uniforms, and buffer sizes are embedded per script | No reusable scheduling policy or CPU/QPU partitioner |
| No compiler/DSL | Kernels use wildcard assembler imports and handwritten register-level QPU assembly | Roadmap's Triton-like interface and LLM kernel exploration cannot be built safely on the current surface |
| Incomplete operator validation | 14 tracked low-level tests exist; four current untracked ML tests import examples dynamically and largely test host helpers | There is no packaged-operator contract suite or hardware differential gate for every supported QPU path |
| No hardware CI | The workflow explicitly comments out `pytest`; it runs formatting, linting, and type checking only | Correctness and performance regressions can reach `main` |
| Benchmarks are mixed with demos | Timing, reference implementations, argument parsing, and kernels share files | Results are hard to reproduce and compare over time |

### What not to rewrite

1. **DRM/V3D device submission and BO mapping:** keep `py-videocore7` as an upstream dependency and wrap it behind an internal backend interface.
2. **Instruction encoding and low-level assembler:** use the upstream assembler as the first code-generation backend.
3. **Known-good kernel algorithms:** preserve the existing scripts on a `legacy-reference` tag/branch, port kernels one at a time, and prove equivalence. Avoid a blind line-by-line cleanup.
4. **Published baselines:** retain the paper's measurements as historical baselines, but do not treat them as regression thresholds until the benchmark harness captures hardware, clocks, temperature, and software versions.
5. **CPU reference implementations:** extract and retain them as differential-test oracles.

## 2. Target architecture and what must be rewritten

### 2.1 Package and API boundary

Add a real `qpu_xla` package with versioned public interfaces alongside the
existing `videocore7` package. The initial distribution may remain a
monorepo, but `qpu_xla` must depend on a narrow internal backend protocol,
not on the examples or on callers reaching into `_videocore7`. Do not rename
or break the existing low-level `videocore7` API as part of this work.

Example target layout:

```text
src/qpu_xla/
  device.py              # discovery, capabilities, lifetime, backend protocol
  memory.py              # Buffer, Tensor, views, mapping, allocator
  queue.py               # Queue, Event, dependencies, host tasks
  kernel.py              # Kernel, launch configuration, specialization
  compiler/
    dsl.py               # user-facing Triton-like primitives
    ir.py                # typed SSA-like kernel IR
    passes.py             # validation and optimization
    vc7.py                # lowering to py-videocore7 assembly
  kernels/               # supported reusable kernels
  ops/                   # tensor/operator API and CPU references
  scheduler/             # CPU/QPU placement and partitioning
  autotune/               # search spaces, runner, cache, result DB
  models/tinyllama/       # loader, quantization, execution plan
  video/                  # camera preprocessing pipeline
  benchmark/              # reproducible benchmark protocol
```

Public API v0 should expose only stable concepts: `Device`, `Buffer`/`Tensor`,
`Queue`, `Event`, `Kernel`, `jit`, and supported operators. Raw addresses and
upstream driver objects remain internal escape hatches. Version 0 must document
which pieces are experimental rather than implying that every legacy kernel is
production-ready.

### 2.2 Runtime, dispatch, and synchronization

Implement a queue/event runtime rather than exposing `Driver.execute`:

- `Queue.submit(kernel, args, grid, wait_for=...) -> Event`
- `Queue.host_task(fn, buffers, wait_for=...) -> Event`
- `Event.wait(timeout)`, status, timestamp/profiling data, and failure propagation
- in-order queue first; explicit dependency DAG and optional out-of-order execution later
- a dedicated submission thread so Python can perform CPU work while a QPU job is in flight
- deterministic shutdown, timeout handling, device-loss errors, and context manager semantics
- debug hazard tracking for overlapping CPU and QPU reads/writes

The initial backend can still use synchronous upstream dispatch inside the submission worker. This provides asynchronous application semantics without prematurely modifying the kernel driver.

### 2.3 Unified memory layer

Create an owned memory abstraction over DRM buffer objects:

- typed shape, strides, byte offset, alignment, and QPU address
- host NumPy view and QPU argument representation for the same allocation
- slices/views without losing the parent allocation lifetime
- explicit access intent: host/QPU read, write, or read-write
- map/unmap or acquire/release operations that attach dependency events
- pooled allocator with alignment and reuse; optional pinned staging buffers
- deterministic close and leak diagnostics
- bounds, dtype, alignment, and 32-bit address-range validation
- import/export hook for future DMA-BUF camera integration

Define one clear memory rule: concurrent access is allowed only for read/read or provably disjoint regions; all other overlap requires an event dependency. In debug mode, reject violations.

### 2.4 Triton-like Python kernel DSL and compiler

Build a deliberately small DSL for VideoCore VII rather than cloning all of Triton:

- `@qpu_xla.jit` kernel definition
- program IDs, 16-lane vectors, masked `load`/`store`, scalar/vector arithmetic, comparisons, select, reductions, and barriers/fences supported by the hardware
- compile-time constants and specialization by dtype, shape class, tile, and QPU count
- typed intermediate representation with source locations
- static verification for bounds masks, types, register pressure, uniform layout, unsupported control flow, and `smul24` operand contracts
- lowering to `py-videocore7` assembly, then program caching keyed by source, constants, backend version, and device capability
- compiler diagnostics that refer to DSL source rather than generated assembly
- an assembly escape hatch for expert kernels

MVP scope should be driven by four kernels: vector add/copy, reduction/min-max, tiled GEMM, and image color/normalization. Do not attempt general Python semantics, dynamic allocation, recursion, or arbitrary loops in v0.

### 2.5 Kernel and operator library

Move every supported operation out of `examples/` and into versioned modules. Each operation must have:

- shape/dtype/layout contract
- CPU reference implementation
- QPU implementations and specializations
- launch and memory metadata built through shared utilities
- numerical tolerance or exactness rule
- property tests and hardware differential tests
- benchmark cases with cold, cached, and execute-only measurements

Before porting, establish a status matrix for every candidate implementation:
host oracle, hardware differential coverage, supported shapes, dtype/range
contract, known failures, and benchmark provenance. Fix or quarantine known
failures first—specifically the packed INT16 conv2d path required by
`AGENTS.md`—instead of migrating a broken result behind a cleaner API.

Port in this order: copy/elementwise, min/max reductions, pooling, GEMM,
bias/activation, convolution lowering, MLP, attention core, and LeNet.
Deduplicate packing, tile rounding, dispatch metadata, range checking, and
benchmarking utilities during the port. Keep the present conv2d truth in the
contract: it is `im2col` plus tiled GEMM, not a native direct-convolution
kernel; FP32/INT32 use a 16x16x4 internal tile and packed INT16 uses 16x16x8.

### 2.6 CPU/QPU scheduler

Implement an operator plan and cost-based placement layer:

- kernel capability registry keyed by op, dtype, layout, shape constraints, and QPU count
- calibrated cost model for CPU time, QPU time, dispatch overhead, and any copy/packing cost
- whole-op placement first; split CPU/QPU tiles only after whole-op scheduling is stable
- hybrid partition rules for FP32 or small shapes where CPU is faster
- event-based dependencies so CPU and QPU work can overlap on disjoint shared-memory regions
- deterministic fallback to NumPy/NEON CPU implementations
- per-device calibration persisted with hardware/software fingerprint

This is the component that turns the proposed heterogeneous-kernel feature into a measurable one. It must never assume that QPU offload always makes a CPU kernel faster.

### 2.7 Test, benchmark, and release infrastructure

Rewrite validation as a product subsystem:

- CPU-only fake backend for API, scheduler, memory lifetime, and compiler tests on standard CI
- instruction/IR golden tests without QPU hardware
- Raspberry Pi 5 self-hosted CI for correctness on every merge and performance nightly; retain the existing generic CI for non-hardware tests
- randomized differential tests against NumPy/PyTorch
- timeout, invalid shape, alias, overlap, and device-loss tests
- benchmark manifests recording Pi model, firmware/kernel, CPU governor, QPU/CPU clocks, temperature, commit, dependency versions, warmup, and repetitions
- machine-readable JSON results plus human summaries
- regression policy: correctness blocks merges; performance alerts on a statistically defined threshold and blocks only after baselines stabilize

## 3. Roadmap requirement traceability

The source PDF is unavailable in this checkout. This table traces the named
requirements carried forward by the existing draft; replace its first column
with exact PDF quotations/page references during approval. It is deliberately
not evidence that the PDF makes the stated claims.

| Roadmap requirement to verify | Deliverable | Depends on | Definition of done |
| --- | --- | --- | --- |
| Parallel Dispatch API | Queue/Event/host-task API and submission worker | Device backend, memory access tracking | CPU task and QPU kernel overlap correctly; dependencies and errors are tested; trace shows overlap |
| Unified Memory Access API | Shared `Tensor`/`Buffer` with NumPy view and QPU address | Device backend | Same allocation is read/written from CPU and QPU without copies; lifetime and hazards are enforced |
| Triton-like Python Assembler | Small typed DSL, compiler IR, verifier, VC7 lowering | Kernel ABI, cache | Four reference kernels compile from DSL, match CPU results, and show actionable diagnostics |
| Unified Memory Kernels | Reusable kernels and hybrid partition templates | Memory, dispatch, scheduler | At least GEMM and video preprocessing split work safely across CPU/QPU and beat the best single-device plan for a documented case |
| LLM-Assisted Kernel Development | Constrained generation/evaluation/autotuning loop | DSL, tests, benchmark harness | A model can generate a candidate in the DSL, compile, verify, differential-test, benchmark, and record/reject it automatically |
| TinyLlama Demo | Quantized TinyLlama inference runtime with QPU acceleration | Operator library, scheduler | Fixed model/prompt generates expected tokens or logits within tolerance; reports load, prefill, and decode performance against CPU baseline |
| Video Preprocessing Demo | Live camera capture plus QPU preprocessing | Memory, dispatch, video kernels | Sustains target resolution/FPS, reports p50/p95 latency and CPU use, and displays/verifies output from a real camera |

## 4. Implementation sequence

### Phase 0 - Correctness triage, baseline, and contracts (weeks 1-3)

1. Resolve the original roadmap PDF and turn its exact requirements into versioned acceptance criteria.
2. Tag commit `5c13e97` as the legacy baseline; commit or otherwise version the currently useful snapshot-only tests and benchmark artifacts before relying on them.
3. Reproduce reported numerical failures using minimal hardware differential cases. Fix packed INT16 conv2d first; quarantine any other failing kernel behind an explicit unsupported status until it is fixed.
4. Write architecture decisions for standalone runtime vs XLA, upstream driver boundary, supported OS/hardware, Python version, error model, and API stability.
5. Select reference Raspberry Pi 5 configurations and configure two self-hosted runners: correctness and benchmark.
6. Extract CPU reference functions and create golden inputs/outputs for every current operation, including tile-boundary and invalid-range cases.
7. Capture cold, cached, execute-only, and end-to-end measurements with clock and thermal metadata.

Exit gate: reproducible baseline report; hardware differential status for every legacy kernel; packed INT16 conv2d is exact against explicit INT32 accumulation or is disabled from the supported surface. No path with an unresolved correctness failure is a performance baseline.

### Phase 1 - Device and memory foundation (weeks 4-7)

1. Add an internal backend protocol and `PyVideoCore7Backend`.
2. Implement `Device`, capability discovery, deterministic cleanup, and typed errors.
3. Implement allocator, `Buffer`, `Tensor`, views/slices, NumPy mapping, access modes, and debug hazard tracking.
4. Add a fake backend and exhaustive CPU-only lifetime/bounds/alias tests.
5. Prove zero-copy CPU/QPU access using copy, fill, and checksum kernels.

Exit gate: 1,000 repeated allocate/map/dispatch/free cycles without leaks or stale addresses; shared-array round trips pass on hardware.

### Phase 2 - Parallel dispatch (weeks 6-10, overlaps Phase 1)

1. Add in-order `Queue`, `Event`, submission thread, timeouts, and profiling timestamps.
2. Add host tasks with declared buffer access and event dependencies.
3. Add trace export showing CPU work, queue delay, QPU execution, and synchronization.
4. Test exception propagation, shutdown during work, timeouts, and invalid dependency graphs.
5. Demonstrate CPU/QPU overlap on independent tiles and compare against serial execution.

Exit gate: no data races under stress; overlap is visible in traces; the parallel path improves a documented mixed workload after overhead.

### Phase 3 - Kernel package migration (weeks 8-13)

1. Port existing assembly kernels into `qpu_xla.kernels` without semantic changes.
2. Replace per-example executors with `Kernel`, launch specs, shared metadata builders, and reusable operator plans.
3. Port exact integer contracts and CPU oracles.
4. Convert current examples into thin API demonstrations.
5. Compare every migrated kernel against legacy correctness and performance.

Exit gate: every *supported* operator is importable from the package; examples contain no runtime implementation; each migration has a hardware differential test; performance is within 5% of the validated baseline unless an explained measurement correction applies. Unsupported/quarantined kernels are not advertised as migrated.

### Phase 4 - DSL/compiler MVP (weeks 11-18)

1. Freeze the kernel ABI: argument encoding, addressing, grid semantics, lanes, uniforms, and specialization keys.
2. Implement AST capture, typed IR, verifier, source mapping, and VC7 lowering.
3. Implement masked memory operations, arithmetic, select, reductions, and compile-time loops/constants.
4. Add disk/in-memory kernel cache and disassembly/debug output.
5. Write copy/vector, min-max, GEMM, and image normalization kernels in the DSL.
6. Differential-test DSL kernels against CPU and assembly versions.

Exit gate: the four kernels compile from user code, fail safely on invalid programs, and reach at least 80% of the matching handwritten assembly performance. Close remaining gaps before expanding the language.

### Phase 5 - Operator planner and unified-memory kernels (weeks 16-22)

1. Add operator descriptions, capability registry, shape/layout normalization, and fallback.
2. Benchmark CPU and QPU variants to seed a per-device cost model.
3. Add whole-op placement, then disjoint-tile hybrid partitioning.
4. Add reusable patterns for producer/consumer overlap and CPU epilogues.
5. Validate GEMM/MLP, convolution, and video preprocessing hybrid plans.

Exit gate: scheduler always selects a correct implementation and is within 10% of the best measured static plan; at least one hybrid workload is faster than either CPU-only or QPU-only.

### Phase 6 - LLM-assisted kernel development (weeks 19-24)

1. Publish a machine-readable DSL reference, hardware constraints, kernel templates, and example corpus.
2. Build an isolated candidate runner with compile timeout, resource limits, deterministic seeds, and no direct device/file access from generated code.
3. Pipeline: propose -> parse -> verify -> compile -> CPU differential test -> QPU differential test -> benchmark -> persist result.
4. Add shape/dtype test generation, counterexample retention, and performance result database.
5. Add search over tiles, unrolling, memory order, QPU count, and hybrid split; begin with constrained edits to known-correct templates.
6. Require human review before generated code enters the maintained kernel library.

Exit gate: starting from a baseline DSL kernel, the system can independently find a correct variant, reproduce its score, and either improve performance or conclusively retain the baseline.

### Phase 7 - TinyLlama demo (weeks 20-29)

1. Fix the model target and artifact contract: TinyLlama 1.1B checkpoint, tokenizer revision, quantization format, prompt, expected output, and redistribution rules.
2. Implement model loader, tensor sharding/layout, memory-budget estimator, and weight preprocessing/cache.
3. Add missing operators: embedding lookup, quantized matvec/GEMM, RMSNorm, RoPE, causal attention, softmax, KV cache, SiLU-gated MLP, residual add, and sampling.
4. Start with a CPU-correct reference runtime and layer-by-layer logits fixtures.
5. Offload dense quantized projections first; keep numerically sensitive or low-intensity operations on CPU until the cost model justifies QPU execution.
6. Integrate event-driven CPU/QPU overlap for normalization/position work, QKV projections, attention, and sampling.
7. Optimize prefill and single-token decode separately. Cache compiled kernels and prepared weights.
8. Report model load time, memory peak, first-token latency, prefill tokens/s, decode tokens/s, CPU utilization, temperature, and output quality/correctness.

Exit gate: a clean setup command runs a fixed prompt end to end on Raspberry Pi 5; logits/tokens meet the agreed tolerance; the accelerated path improves at least one primary latency/throughput metric over the same runtime's CPU-only path without regressing the other beyond an agreed budget.

### Phase 8 - Live video preprocessing demo (weeks 22-27)

1. Define the camera pipeline and performance target, for example 1280x720 at 30 FPS to model-ready 224x224 NCHW.
2. Implement kernels for crop/resize, YUV-to-RGB, normalization/quantization, and NHWC-to-NCHW conversion; fuse where profitable.
3. Integrate `libcamera` through a small capture adapter. Start with mapped/copy input, then add DMA-BUF import only if the zero-copy benefit justifies backend work.
4. Use two or three ring buffers and event dependencies to overlap capture, CPU bookkeeping, QPU preprocessing, and consumer work.
5. Provide a headless correctness mode and a live preview/demo mode.
6. Measure frame drops, p50/p95 latency, throughput, CPU load, QPU time, and thermal behavior for at least ten minutes.

Exit gate: real-camera demo sustains the agreed FPS without unbounded queue growth; output matches a CPU reference within tolerance; setup and troubleshooting are documented.

### Phase 9 - Hardening and release (weeks 28-35)

1. Run long-duration stress, allocation churn, timeout, thermal-throttling, and recovery tests.
2. Stabilize API v0.1, error messages, compatibility policy, and examples.
3. Produce a reproducibility bundle for all headline numbers.
4. Add installation packaging with a pinned/tested upstream dependency range and lockfile.
5. Publish architecture, DSL, operator support, benchmark, TinyLlama, and camera-demo documentation.

Exit gate: clean-machine install and both demos pass; hardware CI is green; published numbers can be regenerated from versioned manifests.

## 5. Recommended milestones and staffing

| Milestone | Calendar target | Primary output | Suggested ownership |
| --- | ---: | --- | --- |
| M0 Baseline locked | Week 3 | Exact roadmap trace, corrected golden tests, and benchmark manifest | Both engineers |
| M1 Memory/runtime alpha | Week 7 | Device + unified memory | Runtime engineer |
| M2 Parallel API alpha | Week 10 | Queue/Event/host tasks | Runtime engineer |
| M3 Packaged legacy kernels | Week 13 | Importable, hardware-validated ops; thin examples | Kernel engineer |
| M4 DSL alpha | Week 18 | Four DSL kernels | Compiler/kernel engineer |
| M5 Hybrid scheduler | Week 22 | Cost-based CPU/QPU plans | Both engineers |
| M6 LLM search alpha | Week 24 | Safe generate/test/benchmark loop | Compiler engineer |
| M7 Video demo | Week 27 | Live preprocessing pipeline | Runtime engineer |
| M8 TinyLlama demo | Week 29 | End-to-end quantized inference | Kernel/model engineer |
| M9 v0.1 release | Weeks 31-35 | Tested SDK and reproducible demos | Both engineers |

Add a third engineer focused on model integration and benchmarking to reduce the critical path by roughly 4-6 weeks. Hardware access, not general coding capacity, is likely to become the bottleneck; maintain a separate benchmark Pi to avoid noisy CI results.

## 6. First two implementation sprints

### Sprint 1 (two weeks)

- Resolve and archive the authoritative roadmap PDF; update requirement IDs and page citations.
- Create `legacy-reference` tag and benchmark manifest from the reviewed commit.
- Commit the current snapshot-only tests/reports that are to become inputs, then add CPU oracles and fixtures for all current operators.
- Reproduce all known hardware numerical failures; repair packed INT16 conv2d or remove it from the public support matrix pending repair.
- Establish Raspberry Pi hardware smoke runner.
- Add internal backend protocol and fake backend.
- Record ADRs for standalone runtime scope, upstream boundary, memory rules, and public API.
- Draft `Device`, `Buffer`, `Tensor`, `Queue`, `Event`, and `Kernel` type signatures before implementation.

### Sprint 2 (two weeks)

- Implement device lifetime and allocator over `py-videocore7`.
- Implement host NumPy mapping, views/slices, and bounds/alignment checks.
- Port copy/fill/checksum kernels as hardware probes.
- Add leak, churn, close-order, and alias tests.
- Implement the submission worker and a minimal in-order queue.
- Emit the first Chrome-trace-compatible timeline for one host task plus one QPU dispatch.

## 7. Non-negotiable acceptance metrics

### Correctness

- Exact equality for integer kernels unless overflow behavior is explicitly part of the contract. In particular, the packed INT16 conv2d result must match an explicit INT32-accumulation reference before it is exposed as supported.
- Defined absolute/relative tolerances for FP32 by operation and shape.
- A QPU hardware differential test is required for every supported dtype/layout/shape family; CPU-only helper tests do not satisfy this requirement.
- Layer-by-layer TinyLlama fixtures, not only final generated text.
- Randomized shapes around tile boundaries and invalid-input tests.

### Reliability

- No leaks or use-after-close in 10,000 dispatch stress runs.
- Timeouts return control and leave the runtime in a defined state.
- Queue shutdown never abandons mapped memory silently.
- Camera pipeline runs for ten minutes without queue growth or frame-buffer reuse races.

### Performance

- Migrated handwritten kernels within 5% of legacy baselines.
- DSL reference kernels at least 80% of handwritten assembly before DSL scope expands.
- Scheduler within 10% of the best sampled static plan after calibration.
- Report cold, cached, execute-only, and full application timing separately.
- Performance comparisons use controlled clocks/temperature and multiple repetitions with raw results retained.

### Developer experience

- A new kernel can be expressed, compiled, tested, and benchmarked without touching driver internals.
- Compiler failures point to DSL source and state the violated constraint.
- Demos install and run from documented commands on a clean Pi.

## 8. Key risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Existing kernel has an unresolved numerical error | A clean API could formalize incorrect behavior | Reproduce a minimal hardware case, retain it as a regression test, fix it before migration, or quarantine/disable the path |
| Upstream driver only exposes blocking waits | Limits true asynchronous submission | Hide blocking call in a worker first; prototype fence/syncobj support only after profiling proves it is needed |
| CPU/QPU mapped-buffer coherency assumptions are wrong | Silent corruption | Document platform guarantees, add alternating CPU/QPU stress tests, and use explicit synchronization/access tracking |
| DSL becomes a full compiler project | Schedule overrun | Limit v0 to the operations needed by GEMM, reduction, and video; retain an assembly escape hatch |
| TinyLlama 1.1B exceeds useful memory/latency targets | Demo misses value | Quantize weights, stream/pack once, optimize decode separately, and keep a smaller model/configuration as a diagnostic fallback rather than changing the headline target silently |
| FP32 QPU work loses to CPU | Weak acceleration story | Use measured cost-based placement and focus QPU effort on dense integer work while overlapping CPU FP32/normalization tasks |
| Thermal and clock variance invalidate results | Unreliable claims | Record temperatures/clocks/governor, precondition devices, run repeated trials, and reserve a stable benchmark machine |
| LLM-generated kernels hang or corrupt memory | Device instability | Typed verifier, masked bounds, isolated runner, timeout, differential tests, and human merge gate |
| Camera zero-copy requires unsupported import paths | Video delay | Deliver copy-backed ring buffer first; make DMA-BUF import a measured optimization, not a demo prerequisite |
| Dependency from an unpinned Git branch changes behavior | Non-reproducible builds | Pin tested upstream commit/release and keep a compatibility matrix |

## 9. Decisions required before week 1 ends

1. Supply/confirm the authoritative roadmap PDF and approve its exact requirement wording and order.
2. Confirm that a standalone Python runtime, not an actual XLA backend, is the product for this roadmap.
3. Choose minimum supported Raspberry Pi OS/kernel/firmware and whether 4 GB boards are in scope.
4. Fix TinyLlama checkpoint, quantization, context length, and success metric.
5. Fix camera model, input format, target resolution/FPS, and model-ready output format.
6. Decide whether third-party LLM APIs may be used in the kernel-development loop or whether it must run locally.
7. Set performance regression thresholds and decide which benchmark claims block release.
8. Confirm licensing policy for generated kernels and continued dependence on GPL-licensed/upstream components where applicable.

## 10. Final recommendation

Start with the runtime/memory/dispatch rewrite and packaged migration of existing kernels. Those pieces provide immediate value and are prerequisites for every approved roadmap deliverable. Build the DSL narrowly against real kernels, then add the scheduler and LLM search loop. Treat TinyLlama as the integration test of the entire runtime and the camera demo as the latency/streaming test. Avoid beginning either demo directly on the current example-script architecture; doing so would create another large one-off executor that must later be rewritten.
