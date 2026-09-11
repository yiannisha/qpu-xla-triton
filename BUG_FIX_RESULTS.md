# Corrected llama.cpp QPU FFN-island results

Date: 2026-09-11

Canonical report: `BUG_FIX_RESULTS.md`

## Verdict

The source-row-stride bug in the CPU half of the channel-partitioned FFN has
been fixed. With that repair in place, the QPU FFN island still provides a
small but statistically supported end-to-end gain at physical M=129. It does
not establish a gain at M=257.

The two primary, swap-disabled results are:

| Workload | CPU_REPACK median | QPU-island median | Saved | Speedup | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Full prompt, M=129 | 2116.016 ms | 2045.965 ms | 70.051 ms | 1.0342x | 1.0090-1.1104 |
| Full prompt, M=257 | 4451.305 ms | 4420.986 ms | 30.319 ms | 1.0069x | 0.9651-1.0168 |
| Cached 512 + tool suffix 128, prompt timer | 3194.305 ms | 3077.670 ms | 116.635 ms | 1.0379x | 1.0241-1.0521 |
| Cached 512 + tool suffix 128, request wall | 3195.809 ms | 3079.077 ms | 116.733 ms | 1.0379x | 1.0241-1.0520 |

The M=129 reductions are 3.31% for `llama-bench` prompt processing and
3.65% for the complete cached request. Every one of the 15 agentic process
pairs favored the QPU candidate. M=257's interval crosses 1.0, so its 0.68%
median reduction is not distinguishable from noise.

This is correctness-preserving acceleration under the tests below, not
bitwise equivalence to CPU_REPACK. The result remains below the project's
conservative 1.05x automatic-promotion threshold, so the path stays opt-in.

## Repair

The faulty implementation quantized the 5,376-channel CPU prefix before the
down projection but also used 5,376 as the distance between F32 source rows.
The actual rows remain 6,144 values apart. Rows two through four of every
four-row group therefore read incorrect channels.

`0005-ggml-cpu-stride-aware-repack-quantizer.patch` separates the number of
columns to quantize from the source-row stride. The ARM 4x4 and 4x8 paths and
their generic, x86, and RISC-V interfaces now receive both values. This avoids
an extra compacting copy while preserving the original 6,144-value row
spacing. The correction is commit `3c7b3fb740bbf2dc522318dd4cf75b5ba4e3d88a`.

## Correctness evidence

Both retained evaluators ran the same three mandatory preflights before
collecting timing samples.

1. The stride regression compares strided and compact Q8_0 quantization byte
   for byte for both Q8_0x4 interleavings. It covers a small guarded case and
   the production `columns=5376, row_stride=6144` shape over multiple four-row
   groups. It also proves the former shortened-stride call produces different
   bytes.
2. The complete synthetic graph compares native CPU_REPACK with the joined
   CPU-prefix/QPU-suffix gate, up, GEGLU, and down path. Maximum/mean absolute
   error was 0.0015990/0.0001596 at M=129 and
   0.0016147/0.0001574 at M=257. Injected QPU failure recovery matched the
   candidate reference, and the test observed eight dispatches across four
   islands.
3. The real Gemma model check compared all 262,144 next-token probabilities
   for three deterministic prompts at M=129 and M=257. All six argmax IDs
   matched. Across the six cases, maximum probability error was 1.438e-5,
   maximum total variation was 1.805e-5, and maximum KL divergence was
   7.843e-7, all below the fixed gates of 1e-4, 1e-4, and 1e-5. It observed
   the expected 210 islands and 840 QPU dispatches.

The real-model path is not raw-logit identical. The largest retained raw-logit
absolute difference was 0.6361 and the largest shift-centered difference was
0.3495 after accumulated quantized execution through 35 layers. Those values,
along with the much smaller probability-space differences, are retained in
the JSON rather than hidden behind the pass/fail result.

The agentic campaign adds an end-to-end semantic check. All 15 CPU/QPU pairs
returned identical greedy output, and every cache-population check passed.
The 15 QPU processes covered all 35 layers, 525 total islands, 2,100 total QPU
dispatches, all four CPU restrictions, 105 resident weights per process, and
zero fallback. Median peak RSS was 3.905 GiB for CPU_REPACK and 4.102 GiB for
the candidate, an increase of 201.5 MiB.

## Measurement design

The model is Gemma 4 E2B Q4_K_XL with selected Q4_0 FFN tensors. Its SHA-256
is `e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889`.
The evaluated preload-library SHA-256 is
`2674660110c5c2c3c4c5dc375330d58d9464600723660af987da9f9972817430`.
The pinned llama.cpp revision is
`91d2fc387529940230555abd297a8b5e99737d3f`.

Both campaigns used four GGML threads, QPU fraction 0.125, WGS 24, performance
CPU governors, a 65 C start gate, and randomized process order. The fraction
was fixed before the corrected confirmation rather than selected from these
results.

The full-prompt campaign used five independent CPU processes and five
independent candidate processes at each M, with one measured prompt per
process. Its interval is an independent two-sample bootstrap with 10,000
resamples. Candidate telemetry recorded 350 islands and 1,400 completed QPU
dispatches per M across warmup and measurement, with all 35 layers and zero
fallback.

The agentic campaign used 15 independent process pairs. Each fresh server
first populated an exact 512-token prefix outside the compared request, then
processed an exact 128-token tool-result suffix and generated one token. The
observed physical FFN M was 129. Its interval is a paired bootstrap over the
15 process-index pairs with 10,000 resamples.

Zram was disabled for both primary campaigns and restored afterward. Both
artifacts record zero swap use and unchanged `pswpin`/`pswpout` counters, zero
current throttling flags, and no competing workload at their guarded
boundaries. This matters: repeated model launches otherwise caused measurable
swap activity and materially changed the observed effect.

## Primary artifacts

- `corrected-independent-processes-v7.json`: retained 20-process prompt
  campaign, all preflight output, raw samples, per-island telemetry, hashes,
  environment, and independent bootstrap intervals.
- `corrected-agentic-15-v2.json`: retained 30-process cached-tool-suffix
  campaign, exact requests/responses, cache checks, process memory, all island
  events, hashes, environment, and paired intervals.

The prompt artifact records repository commit `3c7b3fb`; the agentic artifact
records later repository commit `f74e80e`, which contains `3c7b3fb` as an
ancestor. The preload library, model, generated programs, and all three smoke
binaries have identical hashes across the two retained campaigns. The
`llama-bench` and `llama-server` hashes are recorded separately in their
respective artifacts.

## Excluded and superseded artifacts

- The original files under `experiment_logs/20260910-ffn-island/` measured
  the incorrect shortened-stride binary and cannot support acceleration.
- `corrected-independent-processes.json` completed with the fix but used the
  old unavailable/nonzero-swap check and was rejected.
- `corrected-independent-processes-v4.json` completed but observed 85
  swap-in pages and was rejected by the then-exact I/O rule.
- `corrected-independent-processes-v5.json` observed 75 swap-ins and 86
  swap-outs and was rejected.
- `corrected-independent-processes-v6.json` began after clearing zram, but
  model-process churn caused 2,232 swap-ins and 5,306 swap-outs; it was
  rejected. This run demonstrates why residual host activity cannot be
  ignored for a 1-4% effect.
- `corrected-agentic-5.json` is a retained five-pair exploratory run whose
  0.9920x interval crossed 1.0. It preceded the guard for generic `pytest` and
  `mypy` contention and is superseded by the guarded 15-pair confirmation.
- `corrected-agentic-15.json` is a 16-placement checkpoint from a campaign
  aborted when another session started `mypy`; it has no final comparison.
- `corrected-independent-processes-retained.json`,
  `corrected-independent-processes-v2.json`, and
  `real-model-validation.json` are interrupted partial checkpoints and are
  excluded from every interval.

No result was discarded based on whether its speedup was favorable. Exclusion
is determined by correctness, completion, swap, throttling, and competing-work
contracts recorded in each artifact.

## Reproduction

The evaluators automatically run the stride, complete-graph, and real-model
preflights. On this 8 GiB host, disable zram during measurement to prevent the
kernel from swapping idle sessions while repeatedly loading the model:

```sh
sudo swapoff /dev/zram0

python scripts/run_llama_cpp_qpu_ffn_island_eval.py \
  --rows 129,257 --fractions 0.125 --repetitions 5 \
  --skip-verification --skip-fallback \
  --output experiment_logs/20260910-ffn-island-stride-fix/corrected-independent-processes-v7.json

python scripts/run_llama_cpp_qpu_ffn_island_agentic_eval.py \
  --prefix 512 --suffixes 128 --fraction 0.125 --samples 15 \
  --output experiment_logs/20260910-ffn-island-stride-fix/corrected-agentic-15-v2.json

sudo swapon /dev/zram0
```

The QPU must be exclusively owned and unrelated CPU-heavy tests/indexers must
remain stopped for the full campaigns.
