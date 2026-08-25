# QPU agentic-prefill implementation and evaluation

Date: 2026-08-24

## Delivered path

The deployed candidate accelerates split FP32 GEGLU inside GGML's existing CPU
node. `libggml-qpu-inline.so` is preloaded, and the pinned llama.cpp CPU backend
calls its weak hook only when all of these are true:

- GEGLU has separate contiguous FP32 gate/up inputs and an FP32 output;
- the element count is a multiple of 768;
- GGML batch execution uses one thread;
- `GGML_QPU_CPU_INLINE_GEGLU=1`;
- M is present in `GGML_QPU_CPU_INLINE_ROWS`.

The production agentic policy uses M `17,33,129,257`, corresponding to exact
post-tool suffix targets `16,32,128,256`. Q4_0 matmul is explicitly disabled.
All other nodes and shapes remain on the native CPU path.

The QPU program implements GGML's exact FP16 lookup-table GEGLU semantics. A
cacheable DMA-heap buffer is imported into V3D with PRIME; DMA-BUF access fences
bound the two input copies and output copy. Program, lookup table, and the
largest scratch allocation persist for the process lifetime.

## Why this boundary

Three progressively more realistic boundaries were evaluated:

1. Arbitrary-M Q4_0 × Q8_0 was correct but approximately 0.06× CPU_REPACK for
   a representative Gemma FFN projection. It is disabled.
2. A conventional GGML QPU device backend ran exact GEGLU, but 35 per-layer
   graph splits made a diagnostic 21-token post-tool request about 1.85× slower.
3. The inline CPU hook removes graph splits. Cacheable DMA import then removes
   the expensive uncached BO readback that initially erased the kernel win.

## Sampled operator result

The 31-sample campaign alternates CPU/QPU order and includes the complete GGML
node boundary: DMA input copies and synchronization, QPU submission/wait, DMA
output copy, and backend-call overhead. Every tested output was bitwise equal.

For the real Gemma mix of 15 × N=6144 and 20 × N=12288 GEGLU nodes, one batch
thread produced these diagnostic results:

| M | Median speedup | Bootstrap 95% low | Phase |
|---:|---:|---:|---|
| 17 | 1.134× | 1.115× | held out |
| 33 | 1.113× | 1.094× | held out |
| 129 | 1.190× | 1.187× | calibration |
| 257 | 1.124× | 1.122× | calibration |

M=32, 64, 128, and 256 were correct but slower. Two- and four-thread CPU
baselines were also faster, which is why production placement requires one
batch thread and an exact row allowlist.

The campaign was not retained for a publication claim: CPU governors were
`ondemand` and a user-owned `llama-server` was already active. The evaluator
records both rejection reasons rather than silently accepting the timings.

## End-to-end diagnostic

For cached prefix 512 plus exact tool suffix 16, llama.cpp reused 507 prefix
tokens and replayed five. The timed prompt had 21 tokens, while GEGLU received
the expected M=17 suffix region. CPU and QPU generated the identical token ID
and text. The checked-in diagnostic pair measured:

- CPU prompt time: 1400.429 ms;
- bounded inline-QPU prompt time: 1445.608 ms;
- diagnostic speedup: 0.969× (candidate slower);
- verified QPU dispatches: 35 (one GEGLU per model layer).

An earlier non-paired dispatch check happened to move in the other direction;
the disagreement is exactly why neither one-sample result is a performance
claim. This pair is evidence that the placement executes correctly, not a
retained end-to-end win. End-to-end promotion still requires 31
samples per cell, at least five independent sessions, a performance governor,
zero swap/throttling, no competing server, identical token IDs, median speedup
of at least 1.05, and a bootstrap lower bound above 1.0.

## Reproduction

```sh
cmake --build build/llama-qpu-runtime --parallel 4
ctest --test-dir build/llama-qpu-runtime --output-on-failure

python scripts/run_llama_cpp_qpu_prefill_eval.py \
  --warmups 5 --samples 31 \
  --output experiment_logs/20260824-qpu-agentic-prefill/operator-eval.json

python scripts/generate_llama_cpp_qpu_agentic_cases.py \
  --output integrations/llama_cpp/eval/agentic_cases.json
python scripts/run_llama_cpp_qpu_evaluation.py \
  --case-file integrations/llama_cpp/eval/agentic_cases.json \
  --samples 31 --session-id agentic-prefill-1 \
  --output experiment_logs/20260824-qpu-agentic-prefill/end-to-end.json
```
