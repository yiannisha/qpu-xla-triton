# QPU agentic-prefill implementation and evaluation

Date: 2026-08-24

The original exact-GEGLU campaign is preserved below. The 2026-08-25 follow-up
at the end of this document evaluates the five subsequent implementation
directions and supersedes the earlier clean-machine screen for the current
placement decision.

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

## 2026-08-25 follow-up: five acceleration directions

No QPU placement is promoted by this follow-up. It does establish one useful
resident intermediate-format producer win, and it rules out the current
batched-attention and QPU down-projection implementations with direct
measurements.

### Implemented paths

1. The exact arbitrary-M Q4_0 path now batches TMU scale requests, exposes the
   workgroups-per-supergroup setting, and records input access, packing,
   synchronization, submission/wait, overlap, output synchronization, and
   output-copy time separately.
2. CPU_REPACK Q8_0x4 activation staging now uses a direct AArch64 NEON
   deinterleave. Representative M=257 staging fell from roughly 1.3 ms to
   0.12-0.16 ms, so staging is no longer the dominant exact-Q4 cost.
3. Two explicitly approximate alternatives were added: persistent per-column
   W8 weights with native CPU_REPACK Q8_0x4 activations (`column-w8`), and a
   row/column W8A8 mode. The latter remains unattractive because per-request row
   packing costs about 5 ms at M=257. Neither mode is enabled by default.
4. Gemma attention can now flatten all query rows and eight query heads into a
   single QPU launch for the supported 256-wide, one-KV-head subset.
5. A fused QPU GEGLU producer emits GGML-compatible Q8_0 scale/value bytes
   directly into a resident buffer, which a tiled Q4_0 down-projection consumes
   without host access to the intermediate.

### Retained end-to-end column-W8 result

The full-model campaign used a performance governor, disabled swap, cooled
below 60 C before every fresh process, randomized CPU/candidate order, and ran
two calibration pairs for each fraction followed by five held-out pairs for
each selected cell. Calibration selected a 1/8 QPU output-column suffix for
both requested suffix sizes.

All 18 CPU/candidate pairs were token-identical, process-valid,
measurement-valid, and retained. Every candidate process attested exactly 35
QPU dispatches (15 N=6144 plus 20 N=12288), all 35 resident weights, and zero
fallbacks. That is 630 verified QPU layer dispatches in total. The model,
server, plugin, case/spec, exported program, and result-record SHA-256 values
all revalidate against the stored record.

| Requested tool suffix | Observed M | CPU median | Candidate median | Request speedup | Bootstrap 95% interval |
|---:|---:|---:|---:|---:|---:|
| 128 | 129 | 3019.758 ms | 3021.002 ms | 0.9996x | 0.9466-1.0236x |
| 256 | 257 | 5603.841 ms | 5515.050 ms | 1.0161x | 0.6587-1.1098x |

The M=257 observations include a valid 0.659x slow outlier. It is retained:
the process started at the required clock and thermal state, reported no
current throttling, produced the correct token, and completed all 35 expected
QPU dispatches without fallback. Removing it post hoc would invalidate the
paired evaluation. Neither cell reaches the 1.05x median gate, and both
intervals include 1.0, so `promotion.passed` is false. Median total memory
overhead was about 127.4 MB (121.5 MiB) at M=129 and 130.3 MB (124.3 MiB) at
M=257.

This evidence demonstrates correct execution for the fixed greedy cases, not
general quality equivalence of approximate weights. Any future deployment of
`column-w8` also needs a multi-prompt perplexity/task-quality and long
generation divergence gate.

Raw evidence:
`experiment_logs/20260825-qpu-next/agentic-column-w8-retained.json`.

### Batched attention result

Each row below is an 11-sample persistent-buffer diagnostic. The QPU result
passes both the exact standalone GGML node comparison and an independent
oracle, but it loses decisively to three-thread GGML CPU execution.

| KV context | Query rows (M) | CPU node | One QPU launch | Speedup |
|---:|---:|---:|---:|---:|
| 512 | 65 | 15.581 ms | 26.521 ms | 0.588x |
| 512 | 257 | 35.752 ms | 109.117 ms | 0.328x |
| 4096 | 65 | 124.670 ms | 220.574 ms | 0.565x |
| 4096 | 257 | 288.954 ms | 861.595 ms | 0.335x |

Flattening rows and heads removes launch multiplicity, but the current QPU
online-softmax implementation has insufficient per-launch throughput. It is
not a candidate for llama.cpp graph selection.

Raw evidence:
`experiment_logs/20260825-qpu-next/attention-batched-retained.json`.

### Fused GEGLU-to-down result

The fused producer emits byte-exact Q8_0 scales and values. Its M=129 result is
approximately break-even and its M=257 result is a real 1.139x local win over
four-thread CPU GEGLU plus Q8_0 quantization. The following QPU Q4_0 down
projection erases that gain.

| M | CPU GEGLU+Q8 | QPU fused producer | Producer speedup | CPU FFN region | QPU fused chain | Region speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 65 | 1.564 ms | 1.697 ms | 0.921x | 4.555 ms | 21.104 ms | 0.216x |
| 129 | 3.157 ms | 3.005 ms | 1.050x | 9.542 ms | 37.582 ms | 0.254x |
| 257 | 6.343 ms | 5.568 ms | 1.139x | 18.904 ms | 70.637 ms | 0.268x |

The useful primitive is therefore the resident GEGLU-to-Q8_0 producer, not the
current all-QPU FFN chain. It becomes actionable only if its output can feed a
faster consumer (for example a CPU down projection with a genuinely low-cost
shared-buffer boundary, or a substantially rewritten QPU down kernel).

Raw evidence:
`experiment_logs/20260825-qpu-next/fused-ffn-retained.json`.

### Verification and decision

The complete Python suite passes (371 tests), the native integration suite
passes (17 tests), and all 16 generated QPU programs export successfully. The
decision remains:

- keep every experimental placement opt-in and disabled by default;
- reject current batched QPU attention and the current QPU down projection;
- retain the direct-staging changes because they remove avoidable overhead;
- retain `column-w8` as research-only because its full-model median is too
  small and unstable for promotion;
- treat resident GEGLU-to-Q8_0 as the only newly measured local acceleration,
  subject to finding a faster downstream consumer and then rerunning the same
  held-out end-to-end gate.

Reproduce the three principal records with:

```sh
PYTHONPATH=. .venv/bin/python scripts/run_llama_cpp_qpu_agentic_eval.py \
  --calibration-prefix 512 --heldout-prefixes 512 \
  --suffixes 128,256 --fractions 0.125,0.25 \
  --calibration-sessions 2 --heldout-sessions 5 \
  --bootstrap-resamples 5000 --weight-mode column-w8 --wgs 24 \
  --output experiment_logs/20260825-qpu-next/agentic-column-w8-retained.json

PYTHONPATH=. .venv/bin/python examples/benchmark_llama_cpp_qpu_attention.py \
  --context 512 --context 4096 --query-rows 65 --query-rows 257 \
  --heads 8 --warmups 3 --samples 11 \
  --output experiment_logs/20260825-qpu-next/attention-batched-retained.json

PYTHONPATH=. .venv/bin/python examples/benchmark_llama_cpp_qpu_fused_ffn.py \
  --rows 65 --rows 129 --rows 257 --warmups 5 --samples 11 \
  --output experiment_logs/20260825-qpu-next/fused-ffn-retained.json
```
