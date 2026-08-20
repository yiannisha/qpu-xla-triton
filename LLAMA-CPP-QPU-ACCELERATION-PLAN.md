# llama.cpp QPU and CPU/QPU Acceleration Goal

Date: 2026-08-19
Target hardware: Raspberry Pi 5, VideoCore VII, 8 GB RAM
Target runtime: `llama.cpp` build 10073 / commit `91d2fc387` first, followed by
the current upstream revision after the pinned baseline is reproducible

## Goal

Make the following `llama.cpp` inference workloads faster by using VideoCore
VII QPU kernels and calibrated CPU/QPU partitions implemented with the
low-level and `qpu_xla` technology in this repository:

- Gemma 4 E2B `UD-Q4_K_XL` with its Q4_0 MTP drafter;
- Qwen3.5 4B MTP `UD-Q4_K_XL`;
- plain single-token decode;
- MTP verification at its deployed batch width;
- text prompt processing;
- long-context attention;
- Gemma multimodal prompt processing where the model files are available.

The primary result is lower complete-request latency and lower MTP cycle time
at unchanged model semantics. Isolated kernel speedups are supporting evidence,
not completion.

Workload source:
<https://github.com/Mjrovai/EdgeML-with-Raspberry-Pi/blob/main/mtp-rasp/README.md>

Model sources:

- <https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF>
- <https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF>

## Completion Criteria

The goal is complete only when all of the following hold:

1. The pinned CPU-only `llama.cpp` configurations are reproducible with raw
   samples, exact model hashes, exact build hashes, thread counts, CPU/QPU
   clocks, governor, temperature, and throttling state retained in JSON.
2. At least one QPU-only or CPU/QPU candidate improves median end-to-end MTP
   cycle time or prompt latency by at least 1.05x over the fastest tuned native
   `llama.cpp` CPU configuration for the exact workload.
3. The lower bound of a bootstrap 95% confidence interval for the promoted
   end-to-end speedup is greater than 1.0 across at least five independent
   benchmark sessions.
4. Greedy inference is output-identical to the pinned CPU baseline. Quantized
   operator outputs and logits pass the differential contracts defined below.
5. The accelerated path does not swap, throttle, leak buffers, corrupt the KV
   cache or recurrent state, or require model-weight transfers during each
   token or MTP cycle.
6. Unsupported shapes, tensor types, alignments, contexts, and model revisions
   fall back to native `llama.cpp` without changing output.
7. The production hot path does not invoke Python per GGML node or per layer.
   Python may assemble QPU binaries, generate manifests, run tests, and
   orchestrate benchmarks outside timed inference.
8. Every retained result records CPU-only, QPU-only, best CPU/QPU partition,
   correctness, memory use, and whole-request timing. Kernel-only timing is
   never used by itself to promote a path.

## Fixed Baselines

### Gemma 4 E2B

Use the exact files selected by the workload article:

- base: `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`;
- draft: `mtp-gemma-4-E2B-it.gguf`;
- text MTP: `--spec-type draft-mtp --spec-draft-n-max 2`, three CPU threads,
  context 8192, flash attention enabled;
- plain decode: no MTP, sweep two through four CPU threads and retain the
  fastest exact configuration;
- MTP comparison: sweep `n=2` and `n=3` with two through four CPU threads;
- vision: run prompt and decode phases separately with the matching F16
  multimodal projector.

### Qwen3.5 4B

Use `Qwen3.5-4B-UD-Q4_K_XL.gguf` with its embedded MTP head:

- MTP: `--spec-type draft-mtp`, `--spec-draft-n-max 3`,
  `--spec-draft-n-min 0`, four CPU threads, context 8192, flash attention and
  unified KV enabled;
- plain decode: no MTP, sweep two through four CPU threads;
- verify-batch sweep: `n=0,1,2,3,7` so the accelerated path preserves the
  observed multiple-of-four behavior;
- context sweep: 0, 512, 2048, and 4096 populated tokens.

The published Qwen dimensions are initial benchmark targets:

- hidden dimension: 2560;
- FFN intermediate dimension: 9216;
- vocabulary/output dimension: 248320;
- 32 layers;
- 24 Gated DeltaNet layers;
- eight gated full-attention layers;
- 16 query heads, four KV heads, head dimension 256;
- RoPE dimension 64;
- DeltaNet convolution width 4.

Do not infer Gemma matrix dimensions or actual GGUF tensor types from the model
name. Extract them from the loaded GGUF and retain them in the workload
manifest.

## Non-Negotiable Design Rules

- Treat `UD-Q4_K_XL` as a model-level quantization selection, not a single
  storage type. Enumerate every actual GGML tensor type and accelerate only
  formats with measured cycle-time significance.
- Read quantized weights in their native GGUF block format. Do not expand a
  Q4 model into persistent FP32 or INT8 weights unless a measured candidate
  includes the added memory, bandwidth, and resident-size cost and still wins
  end to end.
- Reuse persistent QPU programs, uniforms, BOs, packed metadata, and selected
  weights across requests.
- Keep CPU and QPU outputs disjoint for hybrid candidates. Small-token linear
  operations split output columns; attention splits query heads; DeltaNet
  splits independent heads or channels; prompt operations may split token rows.
- Include host quantization, format conversion, dispatch, synchronization,
  and output handling in whole-operation timing.
- The native `llama.cpp` kernel for the exact tensor type and shape is the
  primary CPU baseline. NumPy, Torch, and scalar implementations are
  correctness references only.
- CPU and QPU share DRAM. Reject hybrid candidates whose simultaneous traffic
  loses to the fastest single backend even if their kernel-only measurements
  look favorable.
- Preserve cold-start and steady-state measurements separately. Scheduler
  promotion uses steady-state complete-operation and complete-request time.
- Apply exact-shape calibration. Do not generalize a win across batch width,
  context, tensor type, alignment, or model revision without a retained result.

## Required Repository Artifacts

Implement and retain the following:

- `integrations/llama_cpp/`: native C/C++ runtime bridge, GGML backend changes,
  reproducible patch/application scripts, and CMake integration;
- `src/qpu_xla/kernels/`: Python assembly sources and reference adapters for
  each new QPU kernel;
- `scripts/export_qpu_programs.py`: deterministic assembly-to-binary/header
  export with source hashes and uniform/launch metadata;
- `examples/benchmark_llama_cpp_qpu_ops.py`: exact-shape operator and hybrid
  matrix driven by extracted GGUF fixtures;
- `scripts/run_llama_cpp_qpu_evaluation.py`: isolated end-to-end benchmark
  runner for all fixed cases;
- `tests/`: CPU format tests, fake-backend lifecycle tests, hardware
  differential tests, hybrid-boundary tests, and end-to-end deterministic
  regressions;
- `experiment_logs/<run-date>-llama-cpp-qpu/`: manifests, raw JSON, generated
  matrix, environment metadata, and failure records;
- `LLAMA_CPP_QPU_MATRIX.md` in that experiment directory: generated compact
  table for CPU-only, QPU-only, best hybrid, end-to-end latency, cycle time,
  tokens/s, correctness, memory, and status.

Do not commit GGUF files, generated build trees, QPU BO dumps, or large tensor
fixtures. Record their hashes and generate bounded fixtures from the local
model files.

## Work Sequence

### 1. Reproduce and instrument the exact inference graph

1. Add an automated bootstrap that builds pinned `llama.cpp` with
   `-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON`, confirms Cortex-A76 NEON,
   dot-product, llamafile, and KleidiAI paths, and records the resolved build
   configuration.
2. Load each GGUF and emit a machine-readable manifest containing:
   - tensor name, shape, strides, GGML type, byte size, and alignment;
   - model architecture fields;
   - layer/operator ownership;
   - whether the tensor belongs to the base, MTP head, vision encoder, audio
     encoder, projector, embedding, or output head;
   - aggregate bytes and operation counts by tensor type and operator.
3. Instrument GGML graph execution for batch widths 1, 2, 3, 4, 5, and 8.
   Retain per-node and per-operator wall time without changing scheduling.
4. Record separate totals for:
   - draft execution;
   - target verification;
   - quantization/packing;
   - linear projections;
   - LM head;
   - normalization and elementwise stages;
   - attention;
   - DeltaNet convolution and scan;
   - sampling and speculative verification;
   - prompt processing;
   - vision/projector processing.
5. Compute an Amdahl upper bound for every operator family. Do not implement a
   candidate whose complete removal would improve the target cycle by less
   than 5%, unless the kernel is already supported with no new hot-path work.
6. Export deterministic bounded fixtures for the highest-cost exact nodes:
   native quantized weights, quantized or FP32 activations, CPU outputs, scales,
   shape/type metadata, and random seeds.

### 2. Add a native QPU runtime path suitable for llama.cpp

1. Export selected Python-assembled QPU programs as deterministic binary or C
   headers with source hash, required alignment, uniform layout, QPU count,
   and launch geometry.
2. Implement a minimal native VideoCore VII runtime under
   `integrations/llama_cpp/` that provides:
   - render-node discovery and capability checks;
   - V3D BO allocation, mapping, GPU address resolution, submission, wait, and
     timeout handling;
   - persistent program and uniform storage;
   - host/QPU-visible activation and output arenas;
   - reusable selected-weight BOs;
   - explicit cache/coherency boundaries matching the existing Python driver;
   - thread-safe per-context ownership and deterministic cleanup.
3. Add a C ABI for prepared operators. Preparation performs one-time model
   validation and selected-weight placement. Execution receives existing
   activation/output pointers, shape metadata, and a requested partition.
4. Add a GGML backend or narrowly scoped custom-operator integration that:
   - recognizes only manifest-supported tensor types and exact shapes;
   - substitutes QPU or hybrid nodes without graph-wide copies;
   - retains native CPU nodes for every unsupported operation;
   - allocates shared activation/output buffers once;
   - keeps the KV cache and recurrent state in their native owner unless an
     entire fused operation owns the update;
   - reports timed kernel, synchronization, and complete-node events.
5. Ensure the CPU side of a hybrid uses the same optimized GGML block-dot path
   as CPU-only execution on its disjoint output range. Do not replace it with a
   scalar or NumPy tail.
6. Add failure injection for device absence, allocation failure, submission
   failure, timeout, and invalid source hash. Every failure must cleanly retry
   or fall back before mutating model-visible output/state.

### 3. Opportunity A: native small-M quantized linear kernels

This is the first performance target because MTP creates a verification matrix
with three or four activation rows while reusing every weight block across
those rows.

Implement in this order, gated by the extracted tensor histogram:

1. `Q4_0 × matching GGML activation block → FP32`, specialized for `M=1` and
   `M=4`;
2. `Q4_K × matching GGML activation block → FP32`, specialized for `M=4`;
3. additional IQ4/UD-selected storage formats only when their measured nodes
   account for at least 5% of target cycle time;
4. generic `M=2/3/5/8` tails only after `M=4` is correct and competitive.

Kernel requirements:

- consume native GGUF blocks and per-block scales directly;
- unpack signed nibbles in QPU registers;
- reuse the existing signed `v8dot` patterns after widening packed nibbles to
  signed INT8 lanes;
- accumulate integer sub-block dots exactly before applying FP32 scales;
- reuse each weight block across all active rows;
- tile output columns across all 12 QPUs when exact striping permits and use a
  bounded tail path otherwise;
- write native GGML FP32 output layout directly;
- optionally fuse bias or output scaling only when it removes a measured node;
- provide QPU-only and output-column hybrid entry points;
- never write outside the selected output-column interval.

Required initial Qwen `M=4` shape coverage:

- `4 × 2560 × 9216` for FFN gate/up projections;
- `4 × 9216 × 2560` for the FFN down projection;
- `4 × 2560 × 2560` and exact attention-projection dimensions extracted from
  the graph;
- `4 × 2560 × 248320` for the tied embedding/LM output projection.

Benchmark QPU-only and every stable aligned output fraction from 1/12 through
11/12. Calibrate the fraction independently for each tensor type, shape, batch
width, and CPU thread count.

### 4. Opportunity B: large-vocabulary LM-head specialization

Treat the LM head separately from ordinary projections because both target
models have approximately 250K output columns.

1. Implement long contiguous output-column scheduling that amortizes one QPU
   launch across the full vocabulary.
2. Keep target-verification rows together so each vocabulary weight block is
   loaded once for all MTP positions.
3. Benchmark CPU-only, QPU-only, and CPU/QPU vocabulary partitions for the
   exact Gemma and Qwen heads.
4. Measure output-store traffic and sampling time separately.
5. Add an optional fused terminal reduction only for semantics that do not
   require full logits:
   - greedy argmax may return per-partition maxima and indices;
   - general sampling and speculative rejection retain full logits unless the
     fused implementation reproduces the exact required distributions.
6. For hybrid argmax, resolve equal maxima by the original first-index rule.

Promote an LM-head path only when the complete logits/sampling node wins. A
faster dot product that loses after output storage or sampling remains
experimental.

### 5. Opportunity C: Gemma Q4_0 MTP drafter

Use the separate Gemma MTP drafter as the first bounded model integration:

1. Extract all drafter tensor types and exact shapes from
   `mtp-gemma-4-E2B-it.gguf`.
2. Reuse the Q4_0 small-M kernel for every compatible drafter projection.
3. Persist only the drafter's selected weights in QPU BOs to bound memory
   duplication.
4. Benchmark QPU-only and output-column hybrid drafter execution at draft
   depths 2 and 3.
5. Measure the entire draft phase, target verification phase, accepted length,
   and complete MTP cycle separately.
6. Reject the drafter offload if it improves its own projections but fails to
   reduce complete cycle time by at least 5%.

### 6. Opportunity D: fused decode and verification attention

Do not reuse the current staged GEMM-plus-host-softmax implementation as the
production candidate. Implement a fused streaming kernel:

1. Consume query rows and the native KV-cache dtype/layout directly.
2. Apply required scale, causal/window bounds, GQA head mapping, and model RoPE
   semantics exactly as represented in the graph.
3. Stream K and V once per tile.
4. Maintain online maximum, normalization sum, and weighted value accumulator
   without materializing the complete score matrix.
5. Support one and four query rows.
6. Partition independent query heads between CPU and QPU for hybrid execution.
7. Preserve Gemma sliding-window/global-attention distinctions and Qwen's
   16-query/4-KV-head GQA mapping.

Required contexts: 0/empty, 512, 2048, and 4096. Add larger contexts only after
the 4096 path wins and memory behavior is stable.

Measure complete attention-node time, bytes read from the KV cache, dispatch
count, and end-to-end cycle impact. Promote only exact model/context records.

### 7. Opportunity E: Qwen Gated DeltaNet fused scan

Attempt this only when profiling confirms DeltaNet convolution/scan accounts
for at least 5% of the batch-four verification cycle after linear acceleration.

1. Port the exact `llama.cpp` Qwen3.5 DeltaNet equations and state layout into
   a scalar reference owned by this repository.
2. Fuse width-4 convolution, gates, recurrent-state update, and output
   projection preparation so no intermediate state is transferred between CPU
   and QPU within a token chunk.
3. Execute all four verification tokens in one launch while respecting the
   sequential recurrence inside each independent head/channel.
4. Split heads or channels, never the dependent token axis.
5. Keep CPU and QPU state regions disjoint during hybrid execution and merge
   only the final output view.
6. Differential-test the recurrent state after every token, not only the final
   layer output.
7. Test batch widths 1, 3, 4, 5, and 8 to preserve the incomplete-chunk
   behavior observed by the native CPU path.

### 8. Opportunity F: Gemma vision and large prompt processing

The multimodal prompt is a separate optimization target from decode.

1. Profile the vision encoder, projector, and subsequent 169-token-class
   prompt graph independently.
2. Extract actual FP16, FP32, and quantized tensor types and exact row counts.
3. First evaluate existing prepared FP32 GEMM, RMSNorm, softmax, SwiGLU,
   residual, copy, and calibrated row-hybrid kernels without changing their
   dtype contracts.
4. Add FP16 or native quantized projection kernels only for operators whose
   complete removal clears the 5% Amdahl gate.
5. For row-rich prompt operations, calibrate QPU token-row prefixes against
   the native CPU tail.
6. Fuse consecutive activation-only stages when separate QPU submissions or
   CPU/QPU-visible intermediates erase the measured kernel win.
7. Report image preprocessing, vision encoder, projector, text prompt, and
   decode latency separately.

Do not use the GEMM-backed `im2col` convolution path for vision convolutions
unless it beats the native CPU node end to end. If convolution is material,
implement a direct spatial kernel for the exact hot shape and retain the
Conv2D correctness/timing contracts in `AGENTS.md`.

### 9. Opportunity G: existing FP32 post-op kernels

Evaluate existing FP32 kernels only at exact graph shapes after the shared
native runtime path exists:

- RMSNorm;
- RoPE;
- SwiGLU;
- softmax;
- residual add;
- copy/KV append;
- embedding lookup;
- sampling/argmax.

The existing one-token results are expected to remain CPU-preferred. Do not
port them merely for coverage. Retain a candidate only when persistent-buffer
complete-node timing beats the native GGML implementation by 1.05x. Prefer a
fused layer or sampling kernel over several individually dispatched post-ops.

## Correctness Tests

### Quantized formats

For every supported GGML type:

1. Implement an independent scalar decoder and dot-product oracle from the
   format definition.
2. Test minimum/maximum codes, alternating signs, zero scales, very small and
   large finite scales, all-zero blocks, constant blocks, random blocks, and
   non-multiple tail dimensions.
3. Verify integer sub-block dot products exactly before dequantization.
4. Compare the scalar oracle, native `llama.cpp` kernel, QPU-only kernel, and
   every hybrid boundary.
5. Establish FP32 output tolerance from the scalar-oracle/native-CPU
   accumulation-order difference. Do not widen it to accommodate a QPU bug.
6. Record max absolute, max relative, mean absolute, p99 absolute, NaN count,
   Inf count, and the location/value of the worst element.

### Buffer and partition safety

- Test every supported alignment plus one invalid alignment on either side.
- Surround input, output, uniform, KV, and recurrent-state ranges with canary
  pages/words and verify them after every launch.
- Test zero-length CPU or QPU partitions, smallest aligned partitions, largest
  proper partitions, full-QPU, and all stable calibrated fractions.
- Verify CPU and QPU write sets are disjoint.
- Test repeated execution with aliased read-only input and disjoint output.
- Test timeout and fallback before and after program initialization.

### Model semantics

- Compare per-node outputs and final logits against pinned CPU-only
  `llama.cpp` on fixed fixtures.
- Run greedy MTP and non-MTP inference with fixed prompts and require identical
  token IDs, text bytes, token counts, and EOS position.
- For sampled inference, compare logits and speculative accept/reject decisions
  under fixed random streams. Do not require identical text when the upstream
  algorithm intentionally advances randomness differently after rejection.
- Verify Gemma sliding-window/global attention, Qwen GQA, Qwen DeltaNet state,
  KV-cache append, cache reuse, cache reset, and prompt-cache reuse.
- Run at least 100 consecutive generation cycles to detect state drift and
  lifetime errors.

## Performance Tests

### Environment

- Set the CPU governor to `performance`.
- Record CPU frequency, V3D frequency, thermal-zone temperature, throttling
  flags, RAM, swap, zram, kernel, firmware, model hashes, build hashes, compiler
  flags, CPU features, and thread affinity.
- Reject a session if throttling flags are nonzero or swap activity changes.
- Warm the model page cache before retained measurements.
- Run headless with no unrelated sustained workload.

### Operator matrix

For every candidate record:

- at least five warmups;
- at least 31 retained steady-state samples per session;
- five independent sessions;
- native CPU-only samples at every CPU thread count used by a hybrid;
- QPU execute-only samples;
- QPU complete-node samples;
- every stable aligned CPU/QPU partition;
- resident-weight preparation and cold-start samples retained separately;
- median, p05, p95, MAD, bootstrap confidence interval, throughput, and
  correctness metrics;
- host preparation, quantization, dispatch, wait, and output stages retained
  separately without excluding them from the complete-node total.

### End-to-end matrix

Run fixed prompts and seeds for:

| Model | Mode | Batch/depth | Contexts | Primary metric |
|---|---|---:|---|---|
| Gemma E2B | plain decode | 1 | 0, 512, 2048 | eval cycle and tokens/s |
| Gemma E2B | MTP | `n=2,3` | 0, 512, 2048 | accepted tokens per complete cycle |
| Gemma E2B | text prompt | deployed prompt sizes | empty cache | prompt latency |
| Gemma E2B | vision prompt | representative image, about 169 image tokens | empty cache | encoder/projector/prompt latency |
| Qwen3.5 4B | plain decode | 1 | 0, 512, 2048, 4096 | eval cycle and tokens/s |
| Qwen3.5 4B | MTP | `n=0,1,2,3,7` | 0, 512, 2048, 4096 | accepted tokens per complete cycle |
| Qwen3.5 4B | text prompt | deployed prompt sizes | empty cache | prompt latency |

For MTP, retain both reported tokens/s and the stable cycle measure:

```text
cycle_seconds = mean_accepted_tokens / generated_tokens_per_second
```

Measure at least 256 generated tokens for throughput cases and the fixed
40-token structured tool-call workload for application latency. Retain draft
acceptance by position so a speedup is not confused with a different sampled
workload.

## Promotion and Stop Rules

A candidate becomes an automatic exact-shape placement only when:

- source/binary hash matches the retained record;
- tensor type, dimensions, layout, model revision, batch width, context class,
  placement, and partition match exactly;
- all correctness and state tests pass;
- complete-node speedup is at least 1.05x over the fastest exact native CPU
  node;
- the complete end-to-end workload does not regress;
- the lower 95% confidence bound for end-to-end speedup exceeds 1.0;
- memory remains below the no-swap working-set limit;
- no full-layer or full-cycle composition regression exists.

Archive a correct-but-slower path as experimental and move to the next ranked
opportunity when either condition holds:

- the profiled operator family is below the 5% Amdahl gate;
- QPU-only plus all stable partitions fail to beat the exact CPU node after
  native-format input, persistent weights, and dispatch amortization are in
  place.

Do not compensate for a failed end-to-end result by lowering the 1.05x gate,
excluding dispatch/copies, changing sampling parameters, changing thread
counts only for the baseline, or reporting kernel-only time as inference time.

## Ranked Execution Order

1. Reproduce the pinned CPU baselines and emit exact GGUF/graph profiles.
2. Implement deterministic QPU program export and the minimal native runtime.
3. Implement Q4_0 `M=1/4` block dot and validate it on the Gemma drafter.
4. Implement the dominant `UD-Q4_K_XL` base-model storage formats for `M=4`.
5. Optimize the Qwen and Gemma large-vocabulary LM heads.
6. Calibrate output-column CPU/QPU splits for the winning quantized linears.
7. Integrate the winning nodes and rerun complete MTP cycles.
8. Implement fused long-context attention and head splits.
9. Implement fused Qwen DeltaNet only if it clears the post-linear Amdahl gate.
10. Evaluate Gemma vision/prompt FP32 kernels and add only required FP16/Q4
    kernels.
11. Evaluate remaining existing FP32 post-ops at exact graph shapes.
12. Regenerate the operator/end-to-end matrix and retain all wins, losses,
    coverage gaps, correctness data, and environment metadata.

## Final Deliverables

- Reproducible pinned and current-upstream `llama.cpp` builds with optional
  QPU backend support.
- Native-format QPU kernels with hardware differential coverage.
- Calibrated QPU-only and CPU/QPU exact-shape dispatch.
- Persistent selected-weight and activation-buffer plans with safe fallback.
- Complete Gemma and Qwen MTP/plain-decode/prompt benchmark logs.
- A generated latency matrix that distinguishes CPU-only, kernel-only,
  complete-node, QPU-only, hybrid, prompt, MTP-cycle, and request latency.
- A concise result document stating which operations win, which remain CPU,
  why the boundary occurs, and whether the accelerated complete workload beats
  the original 13.06 tok/s Gemma and 4.83 tok/s Qwen article configurations on
  the same hardware and measurement contract.
