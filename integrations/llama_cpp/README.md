# Native llama.cpp QPU integration

This directory owns the native VideoCore VII runtime used by the llama.cpp
acceleration work. It is intentionally built out of tree so the pinned
llama.cpp checkout stays reproducible and unmodified.

The runtime currently provides:

- V3D render-node discovery and compute-shader capability checks;
- BO allocation, mapping, GPU address resolution, submission, wait, timeout,
  and deterministic cleanup;
- persistent program and uniform BOs;
- source and binary hash validation before program upload;
- prepared Q4_0, Q4_K, Q6_K, and Q8_0 C ABIs with persistent selected-weight,
  activation-staging, output-staging, program, and uniform BOs;
- per-context locking and failure injection for device, allocation,
  submission, wait, and source-hash failures.

The prepared operators consume llama.cpp's native quantized weight and matching
activation blocks directly. Existing cacheable host activation
and output pointers remain the model-visible interface. Each call copies the
small activation into its persistent device BO, waits for the QPU to finish in
a private staging BO, then copies only the selected output-column interval to
the host destination. Those copies are included in complete-node timing.
The Q4_0 logical M=1 path currently expands into the validated four-row physical pipeline;
its expansion and extra device storage are also included in reported cost.

The integration also contains an experimental fused Gemma M=1 attention
program. It consumes native FP16 K/V/mask storage and performs online softmax
without a score matrix. Its current exact subset (256-wide heads, one KV head,
no ALiBi/softcap/sinks) is correctness-tested on hardware but is substantially
slower than the pinned GGML CPU node, so it is exported for reproducibility and
never selected automatically.

Build and test:

```sh
cmake -S integrations/llama_cpp -B build/llama-qpu-runtime \
  -DQPU_LLAMA_BUILD_HARDWARE_TESTS=ON \
  -DQPU_LLAMA_BUILD_BENCHMARK=ON \
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
31 samples, the performance governor, zero and stable swap usage, zero
throttling flags, and no pre-existing `llama-server`. `--quick` runs therefore
remain diagnostic even on an otherwise clean machine.

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

The generic runtime is not yet an automatic llama.cpp placement. Prepared
native-format operators and the narrow GGML integration must keep unsupported
nodes on the native CPU path and use a staging output until successful QPU
completion, so a failed submission cannot expose partial model state.

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
stable swap state, zero throttling flags, no pre-existing `llama-server`, and
complete process memory evidence. The CLI surface remains a diagnostic fallback
because it cannot currently provide the same per-process memory contract.

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

The current hardware diagnostics are correctness evidence, not retained
performance evidence, because the host governor was `ondemand`. The native
Q4_0, Q4_K, Q6_K, Q8_0, and fused-attention candidates were slower than pinned
`ggml-cpu` for every tested exact shape and stable partition, so no automatic
placement is enabled. See the generated
[`LLAMA_CPP_QPU_MATRIX.md`](../../experiment_logs/20260819-llama-cpp-qpu/LLAMA_CPP_QPU_MATRIX.md)
for the full timing breakdown and
[`RESULTS.md`](../../experiment_logs/20260819-llama-cpp-qpu/RESULTS.md) for the
current boundary and next required evidence.
