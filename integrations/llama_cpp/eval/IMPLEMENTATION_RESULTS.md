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

The retained benchmark seed emitted byte-exact Q8_0 scales and values. Its
M=129 result is approximately break-even and its M=257 result is a 1.139x
local win over four-thread CPU GEGLU plus Q8_0 quantization. The following QPU
Q4_0 down projection erases that gain.

| M | CPU GEGLU+Q8 | QPU fused producer | Producer speedup | CPU FFN region | QPU fused chain | Region speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 65 | 1.564 ms | 1.697 ms | 0.921x | 4.555 ms | 21.104 ms | 0.216x |
| 129 | 3.157 ms | 3.005 ms | 1.050x | 9.542 ms | 37.582 ms | 0.254x |
| 257 | 6.343 ms | 5.568 ms | 1.139x | 18.904 ms | 70.637 ms | 0.268x |

A later deterministic differential found that this exactness does not
generalize: at M=272, N=6144, two quantized value bytes differed from the CPU
reference by one integer code. The dequantized output max absolute error was
0.004749. The QPU reciprocal used in scale formation is usually exact but can
differ by one ULP, which changes rounding at half-integer boundaries. Two
attempted corrections either left the mismatch in place or moved it to other
elements. The source was restored to the previously validated implementation.

The producer is therefore a useful approximate resident-format primitive, not
a byte-exact production boundary. It remains quarantined until a hardware
differential passes across adversarial rounding cases. Even after that, it is
actionable only if its output can feed a faster consumer (for example a CPU
down projection with a genuinely low-cost shared-buffer boundary, or a
substantially rewritten QPU down kernel).

Raw evidence:
`experiment_logs/20260825-qpu-next/fused-ffn-retained.json`.

### Verification and decision

The complete Python suite passes (376 tests), the native integration suite
passes (18 tests, including 12 hardware tests), and all 16 generated QPU
programs export successfully with a manifest byte-identical to the checked-in
manifest. The decision remains:

- keep every experimental placement opt-in and disabled by default;
- reject current batched QPU attention and the current QPU down projection;
- retain the direct-staging changes because they remove avoidable overhead;
- retain `column-w8` as research-only because its full-model median is too
  small and unstable for promotion;
- retain resident GEGLU-to-Q8_0 only as an approximate research primitive; its
  local acceleration does not override the later byte-exactness counterexample
  or the need for a faster downstream consumer.

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

## 2026-08-25 follow-up: FFN island, large M, and MTP drafting

The next three recommended strategies were implemented in order. No new
placement is promoted: the FFN-island result is promising but remains a
component screen, while the M=1 drafting result is decisively negative.

### 1. Channel-partitioned complete FFN island

`qpu_llama_cpu_ffn_bench` provides a persistent, exact GGML CPU_REPACK
reference for `Q4 gate + Q4 up + GEGLU + Q4 down`. The Python island driver
extracts the same weights from the Gemma GGUF and partitions the intermediate
channels. CPU computes one complete channel slice while QPU concurrently runs
the other complete slice, including its resident GEGLU-to-Q8_0 boundary and
down projection. Only the two hidden-size F32 partial results cross the join.

This is an optimistic component boundary rather than a production hook. The
driver prepares the QPU-side Q8 activation before `qpu_chain` timing, while the
CPU reference performs its own activation quantization. A real integration
may be able to reuse CPU_REPACK's existing Q8 input, but that ownership and the
gate/down hooks are not implemented in the pinned external llama.cpp tree.

At M=257, the best narrow-layer screen was 1.154x at a 3/16 QPU fraction. The
wide layer reached 1.088x at 1/8; neither value is a promotion measurement.
The reconstructed output passed `7e-3 + 5e-4 * abs(reference)` tolerance. The
looser absolute term, compared with a single node, accounts for the changed
F32 accumulation grouping when two independently accumulated down-projection
partials are added; mean absolute errors remained about 2e-4.

Raw evidence:
`experiment_logs/20260825-qpu-next/ffn-island-m257-screen.json`.

### 2. Large-M screen

The same complete boundary was screened at M=513, 1025, and 2049. The table
selects the best measured fraction per representative layer. Each cell used
two warmups and five retained samples.

| M | Layer type | Best QPU fraction | CPU FFN | Candidate wall | Speedup |
|---:|---|---:|---:|---:|---:|
| 513 | 6144-wide | 3/16 | 138.32 ms | 98.29 ms | 1.407x |
| 513 | 12288-wide | 3/16 | 204.93 ms | 186.01 ms | 1.102x |
| 1025 | 6144-wide | 1/8 | 195.93 ms | 184.35 ms | 1.063x |
| 1025 | 12288-wide | 3/16 | 393.68 ms | 352.49 ms | 1.117x |
| 2049 | 6144-wide | 3/16 | 396.58 ms | 353.03 ms | 1.123x |
| 2049 | 12288-wide | 1/8 | 875.68 ms | 780.87 ms | 1.121x |

Weighting the two representative layer timings by Gemma's 15 narrow and 20
wide layers gives estimated FFN-only speedups of 1.188x, 1.102x, and 1.122x
for M=513, 1025, and 2049 respectively. This arithmetic is useful for choosing
the next integration target, but it excludes attention, graph scheduling, and
the rest of the request. It must not be reported as llama.cpp end-to-end
speedup.

Raw evidence:
`experiment_logs/20260825-qpu-next/ffn-island-large-m-screen.json`.

### 3. Large-M full-model boundary

The component estimate was followed by a real llama.cpp M=513 screen of the
implemented `ffn_up` overlap. The default physical batch size also accelerated
the untimed prefix population, producing 70 dispatches at M=508/M=512 instead
of the required 35 dispatches at M=513. The request remained token-identical,
but telemetry correctly rejected it. The evaluator now writes all completed
calibration pairs and summaries before raising a selection failure.

An exact control uses a 32-token cached prefix and `--ubatch-size 1024`, keeping
the prefix below the QPU row floor and the 512-token suffix in one M=513 graph
boundary. Both the calibration and fresh held-out pair were token-identical,
environment-valid, and attested exactly 35 M=513 dispatches with zero
fallbacks. At the selected 3/16 output-column fraction:

| Phase | CPU request | Candidate request | Speedup |
|---|---:|---:|---:|
| calibration | 9090.719 ms | 8871.950 ms | 1.0247x |
| held out | 9053.511 ms | 8925.507 ms | 1.0143x |

Resident DMA memory was 262,564,400 bytes and held-out total measured overhead
was 262,711,856 bytes. This is only a one-pair calibration/held-out screen, but
the median is sufficiently below the 1.05x gate that it does not advance to a
five-session promotion campaign.

A five-sample launch-geometry diagnostic then compared WGS 24 with WGS 192 on
the exact M=513 suffix. WGS 192 improved complete QPU suffix time by 1.041x for
N=6144 and 1.048x for N=12288. Inspection found no removable inner-loop work:
the kernel already uses native `v8dot` reduction and applies the per-column
scale once in its epilogue. Because the QPU and CPU partitions already finish
near one another, this isolated improvement cannot plausibly raise the full
request from 1.014x to 1.05x. The manifest default is unchanged.

Raw evidence:
`experiment_logs/20260825-qpu-next/agentic-column-w8-m513-failure.json`,
`experiment_logs/20260825-qpu-next/agentic-column-w8-m513-ubatch-screen.json`,
and `experiment_logs/20260825-qpu-next/column-w8-m513-wgs-screen.json`.

### 4. QPU-backed MTP/speculative drafting

The conventional GGML device backend gained an upper M bound so an evaluation
can require `M=1..1`. It did execute the selected draft-model Q4_0 operations,
but created scheduler boundaries around many tiny nodes: a 64-token validation
took 50.845 s versus 22.476 s without QPU, or 0.442x. Telemetry recorded 1,753
exact M=1 dispatches with the expected program hashes.

A second implementation removes that structural overhead. Patch 0002 adds
weak hooks directly in CPU_REPACK. The preload library registers selected
Q4_0 weights once, consumes the Q8_0x4 activation already created by GGML,
runs the exact M=1 kernel, and writes the CPU-owned output. Tensor selection is
an explicit allowlist, token embedding is excluded, failure falls back to CPU,
and the path is disabled by default. The native smoke test checks a real GGML
graph, dispatch evidence, and numerical agreement.

The lower-bound operator comparison rejects the strategy even after removing
the graph split: for K=256, N=2048, one-thread CPU_REPACK is about 25 us while
full QPU execution is about 350 us. The four-thread comparison initially made
QPU look attractive because thread-team startup dominated the tiny CPU node;
that is not the drafter configuration. MTP uses one draft thread here, so the
relevant CPU baseline is approximately 14x faster. This direct result makes a
larger server campaign inappropriate as acceleration evidence, though the MTP
case matrix and inline implementation are retained for reproducibility.

Raw evidence:
`experiment_logs/20260825-qpu-next/mtp-qpu-small-validation-2.json`,
`experiment_logs/20260825-qpu-next/mtp-m1-operator-screen.json`, and
`experiment_logs/20260825-qpu-next/mtp-inline-fixture-screen.json`.

### 5. Persistent multi-phase exact Q4 prototype

A final architectural probe tested whether a fixed QPU thread team and global
barriers could collapse multiple tiled projection submissions into one CSD.
Hardware tests now cover a two-barrier global-memory handoff across the full
48-thread threading configuration and private uniform streams across the
stable 24-thread configuration. The exact tiled Q4_0 by Q8_0 kernel was then
ported to 24 persistent task streams with a callable projection subroutine and
a global barrier between phases.

One CSD successfully executed two complete exact projections at M=513. Both
outputs were bitwise identical to two invocations of the existing 470-word
tiled kernel. Five alternating samples at the exact 3/16 Gemma suffix sizes
measured:

| Full output width | QPU suffix | Two conventional CSDs | One persistent CSD | Speedup |
|---:|---:|---:|---:|---:|
| 6144 | 1152 | 48.176 ms | 46.502 ms | 1.0360x |
| 12288 | 2304 | 96.187 ms | 92.776 ms | 1.0368x |

The result proves that an in-dispatch projection/barrier/projection sequence
is feasible, but also shows that launch collapse is not the missing order-of-
magnitude improvement: exact reduction and shared-memory traffic dominate.
The gain is too small to justify implementing a four-phase
gate/up/GEGLU/down superkernel as a promotion candidate, especially because
the current full-model M=513 boundary is only 1.014x. The prototype remains
experimental and is not exported or selected automatically.

The remaining Qwen M=4 projection family was not rerun: retained exact-shape
records already put QPU-only Q4_K gate, Q6_K down, and Q8_0 SSM-out at roughly
0.060x, 0.025x, and 0.082x CPU respectively. A 1.04x persistent scheduling
gain cannot change those placement decisions.

Raw evidence:
`experiment_logs/20260825-qpu-next/persistent-q4-two-phase-m513-screen.json`.

Reproduce the persistent-kernel screen with:

```sh
PYTHONPATH=. .venv/bin/python examples/benchmark_llama_cpp_qpu_persistent_q4.py \
  --rows 513 --phases 2 --baseline-wgs 192 --warmups 2 --samples 5 \
  --output experiment_logs/20260825-qpu-next/persistent-q4-two-phase-m513-screen.json
```
