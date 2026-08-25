# Native llama.cpp QPU integration

This directory owns the native VideoCore VII runtime used by the llama.cpp
acceleration work. The runtime and preload libraries build out of tree. Two
narrow, reproducible patches add opt-in weak registration, asynchronous
launch, and join hooks to the pinned `llama.cpp` CPU backend; they are stored
under `patches/`.

The runtime currently provides:

- V3D render-node discovery and compute-shader capability checks;
- BO allocation, mapping, GPU address resolution, synchronous execution plus
  separate asynchronous submit/wait, timeout, and deterministic cleanup;
- cacheable DMA-heap allocation, V3D PRIME import, and explicit DMA-BUF
  CPU/GPU coherency boundaries for low-cost inline input/output transfer;
- persistent program and uniform BOs;
- source and binary hash validation before program upload;
- prepared Q4_0, Q4_K, Q6_K, and Q8_0 C ABIs with persistent selected-weight,
  activation-staging, output-staging, program, and uniform BOs;
- per-context locking and failure injection for device, allocation,
  submission, wait, and source-hash failures.

There are two GGML integration boundaries:

- `libggml-qpu.so` is a conventional dynamic device backend used for direct
  correctness and partition experiments. Per-layer scheduler splits make its
  GEGLU path slower end to end, so it is not the retained placement.
- `libggml-qpu-inline.so` is a small preload library with no `libggml-cpu`
  dependency. It supports the earlier exact GEGLU experiment, the arbitrary-M
  `ffn_up` experiment, and an opt-in M=1 drafting experiment. The M=1 hook runs
  selected Q4_0 projections directly inside CPU_REPACK after llama.cpp creates
  the native Q8_0x4 activation, avoiding scheduler graph splits. The graph and
  model-visible tensors remain CPU-owned.

The arbitrary-M path supports an exact Q4_0 program plus research-only
per-column W8 and row/column W8A8 alternatives. Selected weights are prepared
once into persistent DMA-backed memory. Each invocation directly deinterleaves
the already-created CPU_REPACK Q8_0x4 activation into reusable cached DMA
memory, submits asynchronously, and copies only the QPU-owned F32 output
suffix from a reusable allocation. Input access, packing, synchronization,
submission/wait, overlap, output synchronization, and output copy are recorded
separately. The exact tiled 16x16 program applies every original Q4_0 and Q8_0
block scale; the W8 modes are explicitly labeled approximate and remain
disabled by default.

The integration also contains experimental fused Gemma attention programs for
M=1 and for a single launch spanning arbitrary query rows and eight query
heads. They consume native FP16 K/V/mask storage and perform online softmax
without a score matrix. Their current exact subset (256-wide heads, one KV
head, no ALiBi/softcap/sinks) is correctness-tested on hardware but is
substantially slower than the pinned GGML CPU node, so it is exported for
reproducibility and never selected automatically. A separate fused GEGLU
program emits GGML-compatible Q8_0 directly into resident memory for a
down-projection consumer. It is locally faster at M=257, but a deterministic
rounding-boundary differential found two value bytes off by one code, and the
current QPU down projection makes the complete chain slower. The producer is
therefore approximate, quarantined, and never selected automatically.

Build and test:

```sh
cmake -S integrations/llama_cpp -B build/llama-qpu-runtime \
  -DQPU_LLAMA_BUILD_HARDWARE_TESTS=ON \
  -DQPU_LLAMA_BUILD_BENCHMARK=ON \
  -DQPU_LLAMA_BUILD_GGML_BACKEND=ON \
  -DLLAMA_CPP_ROOT=/home/yiannis/side/llama.cpp
cmake --build build/llama-qpu-runtime --parallel 4
ctest --test-dir build/llama-qpu-runtime --output-on-failure
build/llama-qpu-runtime/qpu_llama_runtime_smoke
```

The hardware suite includes a shared native-format safety matrix for Q4_0 M=1,
Q4_0 M=4, Q4_K M=4, Q6_K M=4, and Q8_0 M=4. It checks guarded read-only
inputs and destinations, full-QPU and every aligned 16-column subrange in the
48-column test shape, invalid alignments on either side of the tile boundary,
undersized buffers, submission and wait failures, context recreation after a
timeout, and 100 repeated launches per format.

Operator benchmark retention independently enforces at least five warmups and
31 samples, the performance governor, zero and stable swap usage, zero current
throttling flags, and no pre-existing `llama-server`. Historical firmware
throttling bits are recorded but do not reject an otherwise clean run. The
agentic evaluator additionally cools below 60 C before every fresh CPU or QPU
process. `--quick` runs remain screening evidence rather than promotion runs.

The optional native operator and exact CPU_REPACK benchmark executables link to
the pinned llama.cpp `libggml-cpu` out of tree. The Python matrix driver extracts bounded
native fixtures directly from a retained GGUF manifest and compares CPU-only,
QPU-only, and all aligned output-column hybrids:

```sh
python examples/benchmark_llama_cpp_qpu_ops.py \
  experiment_logs/20260819-llama-cpp-qpu/gemma-mtp-gguf-manifest.json \
  --tensor blk.0.ffn_gate.weight \
  --output experiment_logs/20260819-llama-cpp-qpu/operator-session-1.json
```

Generated program headers and their launch manifest live in `generated/` and
are reproduced with:

```sh
python scripts/export_qpu_programs.py \
  --output-dir integrations/llama_cpp/generated --program all
```

The fused-attention diagnostic compares persistent QPU execution with the
exact standalone pinned GGML `FLASH_ATTN_EXT` node and an independent semantics
oracle:

```sh
python examples/benchmark_llama_cpp_qpu_attention.py \
  --output experiment_logs/20260819-llama-cpp-qpu/operator-attention.json
```

Apply the inline hooks to a clean pinned checkout with:

```sh
git -C /home/yiannis/side/llama.cpp apply \
  /home/yiannis/side/py-videocore7/integrations/llama_cpp/patches/0001-ggml-cpu-inline-geglu-hook.patch \
  /home/yiannis/side/py-videocore7/integrations/llama_cpp/patches/0002-ggml-cpu-inline-m1-q4-hook.patch
cmake --build /home/yiannis/side/llama.cpp/build --target llama-server
```

Q4_0 matmul remains disabled by default. Arbitrary-M execution is correct, but
the complete Q8 quantization, staging, QPU, and readback boundary is much
slower than pinned CPU_REPACK. The retained GEGLU policy is also opt-in and
shape-bounded; unsupported rows or thread counts continue through native CPU.

## Agentic incremental-prefill candidate

This is the current end-to-end QPU target. Gemma 4 E2B has 15 layers with
6144-wide FFNs and 20 with 12288-wide FFNs. A cached-prefix request adding 64,
128, or 256 tool-result tokens reaches `ffn_up` with observed M=65, 129, or
257. M includes the suffix tokens plus one graph/control token; it is measured
from QPU telemetry rather than assumed from the HTTP request size.

For every layer, the candidate assigns a calibrated suffix of `ffn_up` output
columns to QPU. Four GGML threads compute the disjoint CPU prefix and then the
independent `ffn_gate`; all threads enter the same GEGLU barrier, thread zero
waits for QPU when needed, and the node consumes the combined exact F32
projections. Decode and M<64 remain native CPU. Runtime failure recomputes the
owned suffix on CPU before the join.

Generate the exact-token cases and run the full calibration/held-out contract:

```sh
python scripts/generate_llama_cpp_qpu_agentic_cases.py \
  --suffixes 64,128,256 --prefixes 512,4096 --threads 4 \
  --output integrations/llama_cpp/eval/agentic_cases.json
python scripts/run_llama_cpp_qpu_agentic_eval.py \
  --calibration-sessions 3 --heldout-sessions 7 \
  --bootstrap-resamples 10000 \
  --output experiment_logs/20260824-qpu-agentic-prefill/agentic-up-full.json
```

Calibration uses separate fresh process pairs to choose among QPU output
fractions 1/16, 2/16, 3/16, and 4/16 for each suffix size. Held-out evidence
then uses fresh randomized CPU-first/candidate-first process pairs at cached
prefixes 512 and 4096. Every measured process receives the same native token
IDs and produces the same generated token IDs. A candidate request must attest
exactly 35 matching QPU dispatches (15 N=6144 and 20 N=12288), zero fallbacks,
program/source/binary hashes, exact shapes and partitions, all 35 resident
weights, current throttle state, swap state, temperature, peak RSS, and
per-process swap. Promotion requires a median post-tool request speedup of at
least 1.05x and bootstrap lower bound above 1.0 in every cell.

Clean-machine screening on 2026-08-25 exercised the real full model, all 35
layers, exact output tokens, zero fallbacks, persistent weights, cached DMA
staging, and both process orders. These two-pair screens are not the seven-pair
promotion campaign:

| Requested suffix | Observed M | Selected QPU fraction | Held-out median | Paired interval |
|---:|---:|---:|---:|---:|
| 64 | 65 | 0.0625 | 1.006x | 0.977-1.036x |
| 128 | 129 | 0.125 | 1.011x | 0.992-1.030x |
| 256 | 257 | 0.0625 | 0.986x | 0.975-0.997x |

No size meets the 1.05x gate. The M=129 median is only about 1.1%, is bounded
by a confidence interval that crosses 1.0, and cannot support an acceleration
claim. See `UP_OVERLAP_RESULTS.md` for the timing/Amdahl analysis and concrete
kernel target.

## Channel-partitioned FFN island

The full-island experiment partitions the FFN intermediate-channel dimension,
not one projection's output in isolation. CPU and QPU concurrently compute
disjoint `gate`, `up`, GEGLU, and `down` paths, and only their final hidden-size
F32 results are added. The QPU path keeps its GEGLU-to-Q8_0 intermediate on the
device, so staging, four dispatches, synchronization, final readback, and the
join are all charged to the candidate wall time. The CPU reference is a
persistent real GGML CPU_REPACK graph using the extracted Gemma Q4_0 weights.

Screening found useful local headroom once M is large enough. The best measured
fractions produced 1.10-1.19x weighted FFN-region estimates across the real
15 narrow plus 20 wide layer mix at M=257-2049. These are isolated-layer,
five-sample screens, not a full-model or promotion result; the implementation
therefore remains a benchmark boundary and is not selected by llama.cpp.

The island benchmark prepares the QPU-side Q8 input before its timed QPU chain,
so it is an optimistic component boundary unless an integration reuses the Q8
activation already owned by CPU_REPACK. A follow-up persistent-kernel probe
successfully ran two exact tiled Q4 projections plus a global barrier in one
CSD, with bitwise-identical results, but improved the exact M=513 narrow/wide
suffixes by only 1.036-1.037x. It remains an unexported experiment rather than a
production selection.

Reproduce the persistent-kernel screen with:

```sh
PYTHONPATH=. .venv/bin/python examples/benchmark_llama_cpp_qpu_persistent_q4.py \
  --rows 513 --phases 2 --baseline-wgs 192 --warmups 2 --samples 5 \
  --output experiment_logs/20260825-qpu-next/persistent-q4-two-phase-m513-screen.json
```

Reproduce the large-M screen with:

```sh
PYTHONPATH=. .venv/bin/python examples/benchmark_llama_cpp_qpu_ffn_island.py \
  experiment_logs/20260819-llama-cpp-qpu/gemma-base-gguf-manifest.json \
  --layer 0 --layer 15 --rows 513 --rows 1025 --rows 2049 \
  --fraction 0.125 --fraction 0.1875 --fraction 0.25 \
  --cpu-threads 4 --warmups 2 --samples 5 \
  --output experiment_logs/20260825-qpu-next/ffn-island-large-m-screen.json
```

## M=1 MTP drafting experiment

`GGML_QPU_M1_INLINE=1` enables exact M=1 Q4_0 execution only for tensor-name
substrings explicitly listed in `GGML_QPU_M1_TENSORS`. Registration happens
when CPU_REPACK receives the weight; each selected node reuses persistent QPU
weights and the already-quantized activation. Failure completes the submitted
work on CPU, and leaving the flag unset preserves native llama.cpp behavior.
The dynamic backend also supports `GGML_QPU_MAX_M`, which was used to prove
that the earlier device-backend MTP experiment executed only M=1 nodes.

The direct hook removes graph-split overhead and is correctness-tested, but it
does not change the throughput conclusion: a representative MTP
`ffn_gate` (K=256, N=2048) takes about 0.35 ms on QPU while the actual
one-thread CPU_REPACK drafter node takes about 0.025 ms. The QPU is roughly 14x
slower at this granularity. The earlier dynamic-backend server validation was
also 0.442x (50.845 s versus 22.476 s) and is retained as rejection evidence.
Accordingly the M=1 placement is implemented for reproducibility, disabled by
default, and not a viable speculative-decoding acceleration with this kernel.

## End-to-end evaluation

The CPU reference campaign keeps endpoint contracts separate. Decode and
exact-token prompt cases use the article's isolated `llama-server`
`/completion` surface. Structured application cases use
`/v1/chat/completions`, a fixed sensor-recording function schema, and a
40-token generation cap. The article reports a 40-token JSON tool call but does
not publish its exact prompt or tool schema, so these cases are deterministic
plan fixtures, not a claimed reproduction of that request.

The generator covers Gemma plain decode at two through four threads and
contexts 0/512/2048, MTP depths 2/3 over the same sweep, exact-token text
prompts, and the structured application workload at empty cache for every
thread/depth configuration. It adds the larger Qwen context/depth matrix only
when that model is locally available:

```sh
python scripts/generate_llama_cpp_evaluation_cases.py \
  --output experiment_logs/20260819-llama-cpp-qpu/full-evaluation-cases.json
python scripts/run_llama_cpp_qpu_evaluation.py \
  --case-file experiment_logs/20260819-llama-cpp-qpu/full-evaluation-cases.json \
  --case gemma-plain-t3-c0-decode \
  --samples 31 --session-id gemma-plain-session-1 \
  --output experiment_logs/20260819-llama-cpp-qpu/gemma-plain-session-1.json
```

Populated-context cases create an exact token prefix, prefill it outside the
timed request, and validate cache reuse. Prompt cases send an exact-length
native token-ID array and validate both evaluated and timed prompt counts.
Every server request asks llama.cpp to return raw generated token IDs; greedy
comparisons are reported only when those IDs are present. Structured tool
responses additionally require exactly one parsed call, the expected function
and arguments, a `tool_calls` stop reason, no assistant prose, and a generated
length within the fixed 40-token cap. The runner hashes each normalized case
and each distinct model, also hashes the model-visible workload independently
of thread/depth/placement tuning, keeps cold server startup separate from
request time, and retains prompt, decode, speculative acceptance, cycle,
response, raw server-log, process peak-RSS, and process-swap data. A retained
session requires the performance governor, zero system and per-process swap,
stable swap configuration, zero current throttling flags, no pre-existing
`llama-server`, and complete process memory evidence. Historical throttle bits
remain in the record. The CLI surface remains a diagnostic fallback because it
cannot currently provide the same per-process memory contract.

Generate the compact report from raw sessions with:

```sh
python scripts/generate_llama_cpp_qpu_matrix.py \
  --case-matrix experiment_logs/20260819-llama-cpp-qpu/full-evaluation-cases.json \
  --operator-session experiment_logs/20260819-llama-cpp-qpu/operator-smoke-m4-host-io.json \
  --end-to-end-session experiment_logs/20260819-llama-cpp-qpu/gemma-context-32-smoke-v2.json \
  --output experiment_logs/20260819-llama-cpp-qpu/LLAMA_CPP_QPU_MATRIX.md
```

The matrix applies promotion only to complete-request evidence. It selects the
fastest tuned CPU configuration for the exact workload, reduces each
configuration to one median per independent session, and requires at least five
retained CPU and five retained candidate sessions. A non-CPU case must carry a
stable `candidate_evidence` object containing the program-manifest path and
SHA-256, exported program name, source and binary SHA-256 values, plus a
nonempty `exact_shape` description. Before a run, the evaluator verifies those
values against the manifest entry and the source/binary bytes. Greedy token
semantics (and the fixed parsed tool call for application cases) must match.
The native server integration must also emit one JSON telemetry line prefixed
with `qpu_llama_candidate_json:` for each executed candidate, echoing the exact
program, hashes, shape, placement, partition, and a positive dispatch count;
the evaluator rejects a non-CPU label without matching execution telemetry.
Median speedup must be at least 1.05x, and the bootstrap 95% lower bound must
exceed 1.0. Operator-only wins are labeled provisional and cannot enable a
placement.

The older native Q4_0, Q4_K, Q6_K, Q8_0, fused-attention, and GEGLU row-hybrid
candidates remain slower than pinned `ggml-cpu`. The newer `ffn_up`
column-suffix candidate reaches the complete llama.cpp request and hides most
QPU time behind useful CPU work. A subsequent retained five-pair-per-cell
column-W8 campaign measured 1.000x at M=129 and 1.016x at M=257, with both
bootstrap intervals crossing 1.0; it also fails the 1.05x promotion gate.
Every QPU placement remains disabled by default, and no end-to-end acceleration
claim is made. See
[`IMPLEMENTATION_RESULTS.md`](eval/IMPLEMENTATION_RESULTS.md) for the five-path
2026-08-25 follow-up and retained end-to-end result,
[`UP_OVERLAP_RESULTS.md`](../../experiment_logs/20260824-qpu-agentic-prefill/UP_OVERLAP_RESULTS.md)
for the current full-model campaign,
[`HYBRID_RESULTS.md`](../../experiment_logs/20260824-qpu-agentic-prefill/HYBRID_RESULTS.md)
for the earlier concurrent row-partition campaign, and
[`LLAMA_CPP_QPU_MATRIX.md`](../../experiment_logs/20260819-llama-cpp-qpu/LLAMA_CPP_QPU_MATRIX.md)
for the full timing breakdown and
[`RESULTS.md`](../../experiment_logs/20260819-llama-cpp-qpu/RESULTS.md) for the
current boundary and next required evidence.
