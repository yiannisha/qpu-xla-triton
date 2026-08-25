# QPU `ffn_up` overlap: full-model result

## Decision

The implemented path is correct, reaches the real Gemma 4 E2B llama.cpp
prefill, dispatches QPU work in all 35 layers, and overlaps most of that work
with useful four-thread CPU work. It does **not** yet demonstrate lower
end-to-end latency. It remains opt-in and disabled by default.

The retained screening runs used the `performance` governor, zero swap, no
competing `llama-server`, cooldown to at most 60 C before every fresh process,
and zero current throttle flags. Historical firmware flags were recorded. CPU
and candidate order was randomized within each pair. Every candidate request
reported 15 N=6144 and 20 N=12288 dispatches, zero fallbacks, the expected
program and hashes, the observed M, and identical output token IDs.

| Requested tool suffix | Observed `ffn_up` M | Calibrated QPU output fraction | Held-out request speedup | Pair interval | Decision |
|---:|---:|---:|---:|---:|:---|
| 64 | 65 | 0.0625 | 1.006x | 0.977-1.036x | inconclusive, below 1.05x |
| 128 | 129 | 0.125 | 1.011x | 0.992-1.030x | inconclusive, below 1.05x |
| 256 | 257 | 0.0625 | 0.986x | 0.975-0.997x | slower |

These are two-pair screens, not the predeclared seven-pair promotion campaign.
Their purpose is to reject an unpromising kernel cheaply. A larger campaign
cannot turn the observed effect into the required 1.05x gain; it can only
narrow the uncertainty around it.

Raw evidence:

- `agentic-up-quick-retained.json`
- `agentic-up-m129-screen.json`
- `agentic-up-m257-screen.json`

## What was implemented

The work pursued the four selected directions as one complete path:

1. An arbitrary-M, tiled 16x16 QPU kernel consumes CPU_REPACK Q8_0x4
   activations and weights converted once from native Q4_0. It unpacks weight
   quants once per tile, uses integer dot-product instructions, and applies
   both original per-block scales. Hardware tests compare it with the pinned
   CPU_REPACK operator across two activation blocks and both x4/x8 layouts.
2. `ffn_up` is split by output columns. QPU owns a calibrated suffix while all
   four CPU threads compute the disjoint prefix and then the independent
   `ffn_gate`. All workers share one barrier, and thread zero joins QPU before
   split-F32 GEGLU consumes the combined projections.
3. Each selected layer's converted weight suffix remains resident in imported
   DMA memory. Activation and output DMA allocations are cached and resized
   only when necessary. At the selected fractions, resident weight storage is
   73,321,520 bytes (1/16) or 131,105,840 bytes (2/16) across 35 layers.
4. A dedicated evaluator creates exact cached-prefix/post-tool requests,
   calibrates fractions in separate processes, evaluates fresh held-out pairs
   in randomized order, bootstraps pair speedups, verifies tokens and dispatch
   telemetry, and records thermal, throttle, swap, RSS, hashes, and raw logs.

Failure injection and fallback tests cover allocation, submission, wait, and
hash failures. On an asynchronous completion failure, the QPU-owned suffix is
recomputed on CPU before the GEGLU join.

## Why the present kernel misses end-to-end latency

Persistent DMA eliminates per-layer weight conversion and allocation, and the
hybrid schedule hides a large part of QPU execution. It does not make the exact
QPU matmul faster than four-thread CPU_REPACK. The complete QPU duration and
CPU overlap observed during calibration were:

| M | QPU fraction | Complete QPU boundary per layer | CPU work overlapped before wait | Exposed QPU tail |
|---:|---:|---:|---:|---:|
| 65 | 0.0625 | 7.44 ms | 6.97 ms | 0.48 ms |
| 129 | 0.125 | 13.75 ms | 12.68 ms | 1.06 ms |
| 257 | 0.0625 | 31.85 ms | 26.71 ms | 5.14 ms |
| 257 | 0.2500 | 40.66 ms | 21.43 ms | 19.23 ms |

The exposed tail is paid in every layer. Increasing the QPU fraction reduces
CPU work but quickly makes QPU the critical path. Decreasing it hides QPU work
but offloads too little of one FFN projection to move a request by 5%. This is
the Amdahl limit of the current kernel, not a measurement-boundary artifact.
The input conversion alone was about 1.3 ms per M=257 layer, and exact Q4_0
requires frequent per-block scale handling inside the kernel.

The older W8A8 experiment is not an existing escape hatch: at M=16, K=N=512,
its QPU kernel median was about 0.282 ms versus about 0.205 ms for the fastest
measured CPU FP32 reference, and its complete QPU boundary was about 0.479 ms.
Likewise, the existing exact fused-attention prototype is substantially slower
than the CPU node. Moving another current kernel into llama.cpp would therefore
increase latency.

## Concrete target for an actual gain

The M=257, 1/4-output calibration supplies a useful go/no-go target. That
candidate took about 5.814 s versus 5.476 s for CPU. Its 19.23 ms exposed QPU
tail across 35 layers is about 0.673 s. Reducing the 1/4-output complete QPU
boundary from 40.66 ms to no more than the 21.43 ms CPU-overlap window—roughly
a 1.9x kernel/boundary improvement—would make the QPU work fully hidden. Holding
the rest of the measured schedule fixed projects roughly a 1.06x request
speedup. This is an engineering target, not a measured claim.

The next implementation should therefore optimize against that threshold
before another expensive full-model campaign:

- pipeline Q4_0/Q8_0 scale loads and integer tiles to remove TMU wait bubbles;
- reduce scalar per-block scale work, input conversion, and cache-sync cost;
- retain the 1/4 output split and require complete QPU time <=21.4 ms at M=257
  in the 6144/12288 layer mix;
- only then run the default 3-pair calibration and 7-pair held-out matrix at
  prefixes 512/4096 and suffixes 64/128/256.

If exact Q4_0 cannot reach that threshold, a separately labeled approximate
path could convert weights once to per-column W8 and use per-row activation
scales. That may remove much of the exact per-block scaling cost, but it changes
model arithmetic and must pass logit-error, greedy-token agreement, and task
quality evaluation before latency promotion. The result is not impossible in
principle; it is impossible to claim from the current exact kernel's measured
critical path.
