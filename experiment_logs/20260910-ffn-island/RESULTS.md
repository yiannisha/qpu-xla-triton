# Integrated llama.cpp QPU FFN-island results

Date: 2026-09-10

## Outcome

The complete channel-partitioned FFN island is now integrated into the actual
llama.cpp CPU_REPACK graph. The final current-binary campaign produces a
statistically positive reduction in full-model prompt latency at M=129 and
M=257. It also produces a smaller, marginally positive reduction in complete
post-tool request latency for a cached 128-token tool result. This is the first
retained end-to-end result in this project where the QPU executes model work
and lowers llama.cpp latency.

The strongest target-workflow result is:

- 512 cached tokens plus a 128-token tool-result suffix;
- physical FFN M=129;
- complete request wall time: 2947.869 ms CPU_REPACK versus 2908.097 ms QPU
  island;
- speedup: 1.0137x, bootstrap 95% interval 1.0005-1.0458;
- five of five raw greedy token sequences and response bytes identical;
- all candidate processes executed all 35 FFN islands with no fallback.

The confidence bound is only just above 1.0 and the median remains far below
the project's conservative 1.05x automatic-promotion threshold. The
integration is therefore an opt-in research path, not an automatic placement.

## Integrated execution boundary

For every eligible layer, the first gate/up node launches a persistent worker
that owns a suffix of the intermediate channels:

1. QPU Q4_0 x Q8_0 gate suffix.
2. QPU Q4_0 x Q8_0 up suffix.
3. QPU lookup-table GEGLU and direct Q8_0 quantization, without host access to
   the intermediate.
4. QPU Q4_0 x Q8_0 down suffix, producing a hidden-size F32 partial.

At the same time CPU_REPACK computes the disjoint channel prefix through gate,
up, GEGLU, and the down reduction. The integration starts from CPU_REPACK's
existing Q8_0x4 activation and joins only the final hidden-size partial. There
is no scheduler backend split and no new F32 activation quantization.

The down-projection change is essential: CPU_REPACK quantizes only the prefix,
then invokes one repacked output-row group at a time with reduced K. This keeps
the original full-K stride between repacked weight groups while skipping the
QPU-owned reduction blocks.

Raw Q4 weights are packed once during CPU_REPACK model loading. For the tested
Gemma model, a 1/8 partition reports 206,807,040 resident QPU-weight bytes. The
agentic runs measured median peak RSS of 3.905 GiB for CPU and 4.102 GiB for the
candidate, a 0.197 GiB increase.

## Correctness and failure behavior

The dedicated hardware smoke executes one four-dispatch QPU island and then
injects a gate-stage failure. The fallback recomputes the QPU-owned suffix on
CPU, and the two synthetic partials are bitwise identical. The test also checks
all four restriction hooks, dispatch/island counters, join, and repeated-use
lifecycle.

Online verification was then enabled on real Gemma activations at M=65. It ran
70 layer instances (one llama-bench warmup plus one measured pass) and compared
each QPU down partial against the scalar Q4_0 x Q8_0 suffix reference:

- maximum absolute error across all layers: 0.000906914473;
- largest layer mean absolute error: 0.00000116930986;
- correctness gate: passed;
- QPU fallback count: zero.

A separate real-model failure-injection run failed layer 0 at the gate in both
warmup and measured passes. Exactly those two islands used CPU fallback; the
other 68 islands completed normally. The execution contract remained valid.

## Full-model prompt processing

The final campaign used llama.cpp commit `91d2fc387529940230555abd297a8b5e99737d3f`,
the Gemma 4 E2B Q4_K_XL model with Q4_0 FFN tensors, four CPU threads, five
separately cooled CPU processes and five separately cooled QPU processes per
shape, randomized process order, performance governor, zero swap use, and zero
current throttling flags. Each process performed its own untimed warmup and one
measured prompt. Values below use the fixed 1/8 policy, rather than selecting a
fraction after seeing the confirmation data.

| Logical prompt M | QPU physical M | CPU_REPACK median | QPU median | Speedup | Bootstrap 95% interval |
|---:|---:|---:|---:|---:|---:|
| 129 | 129 | 1964.232 ms | 1908.430 ms | 1.0292x | 1.0176-1.0574 |
| 257 | 257 | 4099.156 ms | 3983.224 ms | 1.0291x | 1.0146-1.0553 |

Both intervals are entirely above 1.0. Across the five candidate processes per
shape, the evaluator observed 350 island events, 1,400 completed QPU
dispatches, all 35 layers, all four CPU restrictions, 105 resident
gate/up/down weights per process, and zero fallback.

The first confirmation used five repetitions inside one process per
configuration. It reported 1.0500x at M=129 and 1.0358x at M=257, but a
current-binary repeat of that method reported only 1.0106x and 1.0077x, with
both intervals crossing 1.0. Per-pass latency drift showed those repetitions
were not independent. Those artifacts remain tracked, but the table above
supersedes them with randomized independent-process evidence.

The earlier calibration also retained 0.140625 and 0.15625 fractions. It shows
the expected balance behavior: once the QPU branch exceeds the CPU prefix,
exposed wait erases the saved CPU work. The fixed 1/8 policy is the simplest
supported choice over the useful M=129-257 region. Earlier M=65 evidence was
inconclusive, while physical M=512 was neutral.

## Cached agentic tool-result requests

Each sample launched a fresh llama-server. It populated an exact 512-token
prefix outside the timed request, then submitted an exact 128-token tool result
and generated one greedy token. CPU and QPU process order was randomized.
The prefix was larger than the island's maximum M, so only the uncached suffix
used QPU. Every context-population/cache-reuse check passed.

| Tool suffix | Physical M | CPU prompt | QPU prompt | Prompt speedup | Complete request speedup (95% interval) |
|---:|---:|---:|---:|---:|---:|
| 128 | 129 | 2946.410 ms | 2906.548 ms | 1.0137x | 1.0137x (1.0005-1.0458) |

All five CPU/QPU pairs returned identical raw greedy token IDs, content, stop
reason, and tool-call structure. The 128-token result is marginally
statistically positive. A separate earlier three-pair 64-token result measured
1.0088x with an interval of 0.9844-1.0466 and is not distinguishable from noise.

## Evidence and reproduction

Primary retained artifacts for the exact current preload library:

- `current-binary-independent-processes.json`: five independent CPU and five
  independent QPU processes at each of M=129 and M=257, with randomized order,
  complete telemetry, provenance, and bootstrap intervals.
- `agentic-current-binary-5.json`: ten fresh server processes, exact requests and
  responses, token semantics, cache validation, server logs, per-process
  memory/swap, all island events, and paired intervals.
- `current-binary-confirmation.json`: current-binary online numerical
  verification, failure injection, and the superseded repeated-in-process
  latency diagnostic.

Earlier and screening artifacts:

- `full-island-confirmation.json`: initial repeated-in-process campaign, all
  per-layer events, numerical verification, failure injection, and intervals.
- `agentic-confirmation.json`: initial three-pair runs at suffixes 64 and 128.
- `agentic-current-binary.json`: three-pair current-binary validation screen.
- `full-island-screen.json`: rejected coarse screen; zram swap was in use.
- `full-island-tuning-screen.json`: calibration-only fraction screen.
- `agentic-screen.json`: one-pair validation screen.
- `agentic-interrupted.json`: partial four-process checkpoint from an
  interrupted run; retained for provenance and excluded from every comparison.

The retained model SHA-256 is
`e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889`.
The final evaluated preload-library SHA-256 is
`2674660110c5c2c3c4c5dc375330d58d9464600723660af987da9f9972817430`.
The earlier campaign's library SHA-256 was
`7ab457d2b1439ae0b4ff733af70214f19e6fd3c2289afee71311f983206266fc`;
the later binary adds lifecycle cleanup and stricter active-tensor matching.
The exported linear and GEGLU programs, source hashes, and binary hashes are
embedded in every candidate event and revalidated against
`integrations/llama_cpp/generated/manifest.json` by both evaluators.

Build and smoke:

```sh
cmake --build build/llama-qpu-runtime --parallel 4
ctest --test-dir build/llama-qpu-runtime --output-on-failure
```

Full-model and agentic runs:

```sh
python scripts/run_llama_cpp_qpu_ffn_island_eval.py \
  --rows 129,257 --fractions 0.125 --repetitions 5 \
  --skip-verification --skip-fallback \
  --output experiment_logs/20260910-ffn-island/current-binary-independent-processes.json
python scripts/run_llama_cpp_qpu_ffn_island_agentic_eval.py \
  --prefix 512 --suffixes 128 --fraction 0.125 --samples 5 \
  --output experiment_logs/20260910-ffn-island/agentic-current-binary-5.json
```

The three-patch stack under `integrations/llama_cpp/patches/` applies cleanly
to the pinned commit and reproduces the modified `ops.cpp` and `repack.cpp`
byte for byte.
