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
- a prepared Q4_0 C ABI with persistent selected-weight, activation-staging,
  output-staging, program, and uniform BOs;
- per-context locking and failure injection for device, allocation,
  submission, wait, and source-hash failures.

The prepared Q4_0 operator consumes llama.cpp's native 18-byte Q4_0 weight
blocks and 34-byte Q8_0 activation blocks. Existing cacheable host activation
and output pointers remain the model-visible interface. Each call copies the
small activation into its persistent device BO, waits for the QPU to finish in
a private staging BO, then copies only the selected output-column interval to
the host destination. Those copies are included in complete-node timing.
Logical M=1 currently expands into the validated four-row physical pipeline;
its expansion and extra device storage are also included in reported cost.

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

The optional `qpu_llama_q4_0_bench` executable links to the pinned
llama.cpp `libggml-cpu` out of tree. The Python matrix driver extracts bounded
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
  --output-dir integrations/llama_cpp/generated
```

The generic runtime is not yet an automatic llama.cpp placement. Prepared
native-format operators and the narrow GGML integration must keep unsupported
nodes on the native CPU path and use a staging output until successful QPU
completion, so a failed submission cannot expose partial model state.

## End-to-end evaluation

The CPU reference campaign uses the article's isolated `llama-server`
`/completion` surface. The case generator covers Gemma plain decode at two
through four threads and contexts 0/512/2048, MTP depths 2/3 over the same
sweep, and exact-token text prompts. It adds the larger Qwen context/depth
matrix only when that model is locally available:

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
The runner hashes each distinct model once per invocation, keeps cold server
startup separate from request time, and retains prompt, decode, speculative
acceptance, cycle, response, and raw server-log data.

Generate the compact report from raw sessions with:

```sh
python scripts/generate_llama_cpp_qpu_matrix.py \
  --case-matrix experiment_logs/20260819-llama-cpp-qpu/full-evaluation-cases.json \
  --operator-session experiment_logs/20260819-llama-cpp-qpu/operator-smoke-m4-host-io.json \
  --end-to-end-session experiment_logs/20260819-llama-cpp-qpu/gemma-context-32-smoke-v2.json \
  --output experiment_logs/20260819-llama-cpp-qpu/LLAMA_CPP_QPU_MATRIX.md
```

The current hardware diagnostics are correctness evidence, not retained
performance evidence, because the host governor was `ondemand`. The native
Q4_0 candidate was slower than pinned `ggml-cpu` for the tested drafter FFN
and 262144-column output tensor. Even its best measured half-output hybrid
reached only 0.430x of the CPU reference on the latter, so no automatic
placement is enabled. See the generated
[`LLAMA_CPP_QPU_MATRIX.md`](../../experiment_logs/20260819-llama-cpp-qpu/LLAMA_CPP_QPU_MATRIX.md)
for the exact timing breakdown and rejection status.
