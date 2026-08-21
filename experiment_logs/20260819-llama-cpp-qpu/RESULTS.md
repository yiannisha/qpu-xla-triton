# llama.cpp QPU acceleration results

This is the current evidence summary for the pinned out-of-tree integration.
No automatic QPU placement is enabled.

## Reproducibility boundary

- Pinned `llama.cpp`: `91d2fc387529940230555abd297a8b5e99737d3f`
  (build 10073, Release, native CPU and llamafile enabled, KleidiAI disabled).
- Gemma base SHA-256:
  `e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889`.
- Gemma MTP drafter SHA-256:
  `586f2460b909008640981ec34060aa864e03c144fbabfb3173c4335087e4aae0`.
- Qwen base-with-MTP SHA-256:
  `d2bfbee4de17c74e6308a4dc750be4c2271f92ae2df1e28f5c5afa0dcdb6fccc`.
- Nine QPU programs are deterministically exported with source, binary,
  uniform, and launch hashes in `integrations/llama_cpp/generated/manifest.json`.

The current sessions are diagnostic rather than retained performance evidence:
the CPU governor was `ondemand`, zram was full, and a pre-existing Qwen
`llama-server` was active. Correctness differentials remain useful; timing rows
cannot promote a placement.

The implementation itself passes `354` Python tests and all `11` native CTest
targets. The seven VideoCore VII hardware tests include numerical smokes plus a
cross-format safety matrix for Q4_0 M=1/M=4, Q4_K, Q6_K, and Q8_0. That matrix
checks guarded buffers, every aligned partition of a 48-column resident range,
invalid ranges, failure staging, context recovery, and 100 repeated launches
per format. Focused Ruff, mypy, and whitespace checks also pass.

## Native quantized linears

All candidate outputs pass the exact pinned `llama.cpp` CPU_REPACK comparison.
Every QPU-only path and every tested output-column partition is slower than the
fastest exact CPU node.

| Candidate | Exact shape M×K×N | CPU ms | QPU ms | Best non-CPU ms | QPU max abs error | Result |
|---|---:|---:|---:|---:|---:|---|
| Gemma drafter Q4_0 LM head | 1×256×262144 | 2.981 | 37.762 | 3.990 at 1/12 QPU | 4.77e-7 | archive |
| Gemma drafter Q4_0 LM head | 4×256×262144 | 5.232 | 39.703 | 12.857 at 1/4 QPU | 4.77e-7 | archive |
| Qwen Q4_K FFN gate | 4×2560×9216 | 1.174 | 19.648 | 5.028 at 1/6 QPU | 8.34e-7 | archive |
| Qwen Q6_K FFN down | 4×9216×2560 | 1.811 | 73.232 | 15.245 at 0.08125 QPU | 4.77e-7 | archive |
| Qwen Q8_0 SSM output | 4×4096×2560 | 0.839 | 10.192 | 3.091 at 0.08125 QPU | 4.77e-7 | archive |

The exact captured Gemma drafter LM-head replay separately proves that the
production graph output is reproduced: CPU_REPACK is bitwise identical to the
capture, the full QPU output has maximum absolute error `5.72e-6`, and greedy
argmax is unchanged. That replay is correctness-only because its graph callback
serializes execution.

## Fused attention prototype

The experimental M=1 kernel consumes FP32 query heads plus native FP16 K, V,
and mask tensors, performs online softmax, and emits FP32 without materializing
scores. It currently supports the exact Gemma local-attention subset with
256-wide heads, one KV head, no ALiBi, no softcap, and no sinks.

| KV rows | Exact pinned GGML CPU ms | QPU persistent execute ms | Speedup | Max abs error |
|---:|---:|---:|---:|---:|
| 256 | 0.064 | 0.674 | 0.094x | 2.13e-4 |
| 512 | 0.120 | 1.201 | 0.100x | 7.25e-5 |
| 2048 | 0.542 | 4.356 | 0.124x | 7.43e-5 |
| 4096 | 1.600 | 8.561 | 0.187x | 7.45e-5 |

At 4096 rows, executing only one QPU query head still takes about 8.53 ms,
while the complete eight-head CPU node takes about 1.74 ms in the separate
head-boundary diagnostic. Therefore every nonzero head-partition hybrid has a
best-case lower bound worse than CPU-only, even assuming perfect CPU/QPU
overlap and zero merge cost. This prototype is archived and is not a placement
candidate.

The graph profiler now records all source slots and the exact 64-byte GGML op
parameter block. Pending capture cases cover Gemma M=1/M=4 at contexts
0/512/2048/4096 for both the 256-wide sliding-window node and the 512-wide
global-attention node. They must be run after the external server is gone and
the machine satisfies the retained environment contract.

## End-to-end status

The generated latency matrix contains 113 native-operator rows, four attention
diagnostic rows, four earlier end-to-end smoke rows, and 150 planned end-to-end
cases. The plan matrix comprises 99 decode cases, 24 exact-token prompt cases,
and 27 fixed-schema application cases with a 40-token cap. It contains zero
provisional operator wins, zero promoted end-to-end configurations, and zero
retained end-to-end rows.

The report generator now treats any complete-node operator win as provisional.
Its promotion gate selects the fastest tuned CPU configuration for an exact
model-visible workload, reduces samples to one median per independent session,
and requires five retained sessions on both sides, complete model/server/QPU
hash and exact-shape evidence, peak-RSS plus zero per-process swap, identical
greedy semantics, and native telemetry proving that the exact candidate binary
actually dispatched. It then requires at least 1.05x median complete-request
speedup and a bootstrap 95% lower bound above 1.0. There are currently zero
non-CPU end-to-end configurations to evaluate through that gate.

The structured application fixture uses `/v1/chat/completions` and validates
the exact parsed function call. It is not represented as an article
reproduction because the article gives the output length but not the request
prompt or tool schema. The four older smoke records also predate raw-token
request capture, so their matrix rows intentionally make no exact greedy-token
claim. No application case was run while the external Qwen server and full
zram made the machine ineligible for retained measurement.

Consequently the work does not yet establish an accelerated complete workload
or a same-contract improvement over the article's 13.06 tok/s Gemma and
4.83 tok/s Qwen configurations. Those claims require isolated, performance-
governor, no-swap multi-session runs plus an actual winning node placement.

## Current boundary and next evidence

The tested native-format compute is correct, but the Cortex-A76 implementations
are far enough ahead that dispatch amortization and CPU/QPU partitioning do not
recover a win. The next implementation should be chosen from a production graph
profile rather than another static tensor histogram. In priority order:

1. run the pending Gemma long-context local/global captures and all Qwen graph
   profiles in a clean environment;
2. apply the 5% Amdahl gate to Qwen DeltaNet and Gemma vision/prompt families;
3. implement only a family that clears that gate and has a credible complete-
   node bandwidth/dispatch lower bound;
4. rerun the full fixed end-to-end matrix only after such a node passes its
   exact CPU comparison.

The retained pinned baseline, five independent promotion sessions, production
Qwen graph profile, Gemma captured attention replays, and current-upstream
follow-up remain intentionally unclaimed. The Gemma vision path additionally
requires the absent matching multimodal projector and a representative source
image. These are completion requirements or explicit coverage gaps, not
implementation test failures.
