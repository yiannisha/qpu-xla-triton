# Concurrent CPU/QPU GEGLU result

The asynchronous row-hybrid implementation is bitwise correct, but it did not
accelerate the four-thread GGML reference. It is disabled by default and did
not advance to end-to-end promotion testing.

## Evaluated boundary

The candidate submits the leading GEGLU rows to VideoCore VII, computes the
remaining rows concurrently on all four GGML CPU threads, waits for QPU, copies
the disjoint QPU output rows back, and joins before the GGML node returns. The
timing includes cached-DMA input copies, submission, QPU work, CPU work, wait,
output copy, and the node join.

Calibration tested QPU row counts 1, 5, and 9 where valid. It selected one QPU
row for every M. Held-out evaluation then used five fresh processes, 64 warmups
and 31 retained alternating CPU/candidate samples per cell. Each process was
reduced to the Gemma 4 E2B region containing 15 N=6144 nodes and 20 N=12288
nodes. Promotion required bitwise equality, median speedup at least 1.05x, and
a paired-session bootstrap 95% lower bound above 1.0.

| M | QPU rows | Median speedup | Bootstrap 95% interval | Result |
|---:|---:|---:|---:|:---|
| 17 | 1 | 0.980x | 0.701-1.070x | no stable win |
| 33 | 1 | 0.951x | 0.935-1.022x | no stable win |
| 129 | 1 | 0.964x | 0.918-0.981x | slower |
| 257 | 1 | 0.971x | 0.967-1.072x | no stable win |

All held-out cells were bitwise exact and had exact positive QPU dispatch
accounting. The raw records, commands, samples, hashes, calibration data, and
session-level medians are in `hybrid-operator-eval.json`.

## Retention and decision

The campaign was diagnostic because the CPU governor was `ondemand`, zram swap
had 2,147,467,264 bytes in use, and a pre-existing Qwen `llama-server` was
active. Those conditions increase variance, but they do not rescue the
candidate: the median is below 1.0 at every M, and no M approaches the 1.05x
promotion threshold across sessions.

End-to-end agentic-prefill evaluation was intentionally skipped. Running it
after an operator-gate failure would turn whole-request noise into a misleading
acceleration claim.

## What would have to change

The present hybrid can only hide QPU work behind the CPU tail. One QPU row is
the only competitive split because it removes the `M mod 4 = 1` CPU load
imbalance; larger splits make QPU the critical path. A credible next candidate
therefore needs a different data boundary, not another row-count sweep:

- allocate relevant GGML activations from CPU-cacheable DMA-BUF storage so QPU
  can read and write them without per-node staging copies;
- reduce the fixed V3D dispatch/kernel cost enough that more than one row can be
  balanced against four CPU threads; or
- fuse a producer/GEGLU/consumer region so one dispatch and one synchronization
  replace several GGML node boundaries.

Those are material runtime or graph-layout changes. Until one is implemented
and passes this same held-out gate, the QPU path remains an experiment rather
than an inference policy.
