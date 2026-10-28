# Completed rebuttal experiments: TODOs 1 and 2

28 October 2026. Both experiments in `coa.md` completed. TODO 1 collected all 20 instrumented processes. TODO 2 completed its 24-process exploratory screen after all three fractions passed full-model numerical validation. No new undervoltage or swap I/O occurred. All timed candidate processes passed the recorded all-layer dispatch contracts, with zero fallback.

**Principal findings:** CPU-only FFNs occupy a median 67.69% of the instrumented prompt at M=129 and 66.51% at M=257. The complete FFN's measured critical path is predominantly the CPU branch with preparation and join costs, rather than an exposed QPU tail. None of the six uninstrumented screening configurations reaches a median complete-prompt speedup of 1.05. The screen does not justify a new automatic policy or an optimality claim for one eighth.

## Configuration and boundaries

The model file's SHA-256 is `e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889`, matching the retained Gemma 4 E2B Q4_K_XL artifact with Q4_0 FFN tensors. The llama.cpp base is `91d2fc387529940230555abd297a8b5e99737d3f`, with the bundled integration and stride-aware patches. There are 35 FFNs: 15 of intermediate width 6144 and 20 of width 12288. All timings use four CPU threads, WGS=24, persistent weights/programs/workspaces, a performance governor, disabled swap and a 65 C start gate. Physical logical row counts are 129/257; padded QPU row counts are 144/272. Model loading and resident packing are outside the timed prompt after each process's warmup.

The repository base is `f2cf2be`. During execution, the host competing-workload guard incorrectly matched its own explicit `--llama-bench` and smoke-path arguments. The fix excludes only the evaluator and its ancestors from pgrep matches; unrelated workload matches remain errors. Raw unfiltered matches remain in environment records. The original failed timing launch and the successful resumed launch are both preserved. No numerical gate, native program, or partition arithmetic changed. See [guard-fix.patch](guard-fix.patch), [runtime source snapshots](source/llama_cpp_common.py), and [portable tests](portable-tests.log): 19 tests passed.

## TODO 1: integrated breakdown

Five independent CPU/candidate process pairs were run at each row count, with randomized order within pairs. Both placements used the isolated timestamped CPU backend; the candidate additionally used the QPU plugin and fixed one-eighth channel share. These are instrumented diagnostics, not replacement uninstrumented headline measurements.

The table reports **medians of process-level sums for each width class**, in milliseconds. The narrow class contains 15 FFNs and the wide class 20. Each process sum uses only the final measured 35-layer pass; the warmup pass is excluded.

| M | FFN width | CPU-only FFN wall | Candidate prep | CPU branch | QPU path | Merge | Candidate FFN wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 129 | 6144 | 389.708 | 50.187 | 332.545 | 256.629 | 3.887 | 392.172 |
| 129 | 12288 | 1030.373 | 67.158 | 920.085 | 697.484 | 6.623 | 995.394 |
| 257 | 6144 | 758.521 | 100.953 | 682.601 | 503.067 | 9.974 | 807.040 |
| 257 | 12288 | 2019.602 | 134.358 | 1876.300 | 1268.847 | 13.774 | 2037.470 |

Preparation includes the first projection's CPU activation quantization and QPU preparation up to its ready boundary. The CPU branch spans its start through the pre-wait completion barrier. The QPU path spans readiness through observed worker completion, including scheduling, dispatch, device work, synchronization and output handling. Merge brackets the hidden-size partial-output addition and its barrier. Handoff, CPU-start scheduling gaps, and post-merge work are retained in the raw timeline/summary rather than silently assigned to arithmetic.

**CPU and QPU branches overlap. Do not add the branch columns.** Absolute finish timestamps determine the critical path. Independently reported column medians also need not add to a wall-time median. The summarizer checks the decomposition on each actual layer interval before aggregation.

| M | CPU prompt median, ms | CPU-only all-FFN median, ms | CPU-only FFN share | Candidate prompt median, ms | Candidate all-FFN median, ms | Diagnostic speedup [paired bootstrap 95% CI] |
|---:|---:|---:|---:|---:|---:|---|
| 129 | 2059.174 | 1420.608 | 67.69% | 2034.474 | 1385.692 | 1.01214 [0.98603, 1.06628] |
| 257 | 4186.305 | 2784.264 | 66.51% | 4252.559 | 2834.646 | 0.98442 [0.95082, 0.99546] |

Shares are medians of each process's FFN sum divided by its own complete prompt sample. They are not ratios of the two displayed medians. Intervals use 10,000 resamples preserving these five diagnostic process pairs. The M=129 interval includes parity; all five M=257 diagnostic pairs favor CPU-only. These measurements do not reproduce the manuscript's historical 3.31% and 0.68% reductions and must not be substituted for them without identifying the different instrumented campaign.

### Timestamp-based interpretation

At M=129 the CPU finishes last in 172/175 measured FFNs; the QPU finishes last in three narrow FFNs and zero wide FFNs. At M=257 the CPU finishes last in all 175 measured FFNs. The median process-level exposed QPU tail is zero in every width class. A long QPU tail from the paper's older projection-overlap experiment therefore does not describe the dominant critical path of this measured complete FFN.

Across all 35 FFNs, preparation medians increase from 116.707 to 236.591 ms and merge medians from 10.510 to 25.659 ms. CPU-branch medians increase from 1254.348 to 2555.561 ms; QPU-path medians increase from 960.492 to 1760.914 ms. The measured path remains CPU-dominated as M increases. Within matched diagnostic pairs, the median candidate-minus-CPU FFN difference is +3.387 ms for the narrow class and -30.716 ms for the wide class at M=129, versus +62.002 and +21.524 ms at M=257. These paired differences are computed before taking medians and should not be reconstructed by subtracting unrelated column medians.

This local evidence identifies CPU-side completion plus increasing preparation/join costs as the observed size-sensitive path. It does not isolate memory contention versus CPU kernel behavior, or prove the cause of the precise historical request reductions. Separate mechanism ablations are outside TODOs 1 and 2.

Evidence: [full process record](raw/todo1-2-todo1-breakdown.json.gz), [per-process JSON](raw/todo1-2-todo1-summary.json.gz), [per-process CSV](todo1-summary.csv), [derived critical-path analysis](raw/todo1-2-analysis.json.gz).

## TODO 2: full executed calibration screen

The historical audit first recovered 24 integrated comparisons. Both historical multi-fraction screens predate the CPU-stride correction and had rejected swap conditions; they cannot establish the corrected split optimum. Historical corrected confirmation fixed one eighth before evaluation. See [historical grid](historical-calibration.csv) and [audit/provenance](raw/todo1-2-historical-calibration.json.gz).

The new exploratory grid uses three independent processes per CPU/candidate configuration at each M. CPU samples are reused as the matching baseline for the three fractions at a shape. Each process warms up and contributes one complete-prompt timing. Numerical preflights separately evaluate six full-vocabulary model cases for every fraction. Every fraction passed; none was omitted for a numerical failure.

| M | QPU fraction | CPU median, ms | Candidate median, ms | Speedup | Bootstrap 95% CI | Numerical / execution status |
|---:|---:|---:|---:|---:|---|---|
| 129 | 1/16 | 2209.615 | 2250.048 | 0.98203 | [0.95762, 1.00314] | Pass / pass |
| 129 | 1/8 | 2209.615 | 2193.990 | 1.00712 | [0.96892, 1.06911] | Pass / pass |
| 129 | 3/16 | 2209.615 | 2172.981 | 1.01686 | [0.97299, 1.02668] | Pass / pass |
| 257 | 1/16 | 4446.638 | 4772.363 | 0.93175 | [0.87670, 0.93907] | Pass / pass |
| 257 | 1/8 | 4446.638 | 4455.623 | 0.99798 | [0.93828, 1.05388] | Pass / pass |
| 257 | 3/16 | 4446.638 | 4511.789 | 0.98556 | [0.92071, 0.99453] | Pass / pass |

Intervals use 10,000 independent two-sample bootstrap resamples, as recorded by the screen evaluator; three processes per group provide limited uncertainty resolution. With a 95% interval entirely above/below parity as the descriptive rule, there are **0 wins, 4 inconclusive comparisons, and 2 losses**. Four of six point estimates are below one. All six numerical/execution contracts passed; no numerical failure is hidden.

The best point estimate at M=129 is 3/16; at M=257 it is 1/8. This screen neither establishes an optimum nor independently confirms either selection. No point estimate reaches 1.05. The applicable promotion test thresholds the ratio of median request latencies at >=1.05 and requires a bootstrap lower bound >1.0 plus the remaining gates; a confidence interval that merely extends above 1.05 is insufficient. No new policy is promoted and no threshold is lowered.

Actual QPU/CPU intermediate columns are 384/5760, 768/5376, and 1152/4992 for width 6144, and 768/11520, 1536/10752, and 2304/9984 for width 12288. All boundaries are 32 aligned. Candidate telemetry attests physical M=129/257, all 35 layers, expected program identities, four dispatches per island, all CPU restrictions, and zero fallback. The screen records 1260 islands and 5040 dispatches including warmups, or 630 islands and 2520 dispatches in measured passes.

Evidence: [complete grid JSON](raw/todo1-2-todo2-grid.json.gz), [grid CSV](todo2-grid.csv), [raw timing and environment record](raw/todo1-2-todo2-screen.json.gz), [fraction validation and initial/resumed attempts](raw/todo1-2-todo2-screen-attempt.json.gz).

## Numerical and environment validation

Both Q8 interleavings match the compact reference byte for byte at width/stride 64/96 and production 5376/6144. Joined synthetic FFN checks pass at M=17,129,257, with maximum absolute errors 0, 0.00159901381 and 0.00161474943; injected recovery matches the candidate exactly. The six-case one-eighth full-model preflight passes before TODO 1.

| Fraction | Model cases passed | Maximum probability error | Total variation | KL divergence | Maximum raw-logit absolute error |
|---:|---:|---:|---:|---:|---:|
| 1/16 | 6/6 | 3.66183e-5 | 3.88592e-5 | 2.61528e-6 | 0.631973 |
| 1/8 | 6/6 | 1.43795e-5 | 1.80487e-5 | 7.84266e-7 | 0.636106 |
| 3/16 | 6/6 | 2.16953e-5 | 2.39264e-5 | 1.06547e-6 | 0.485402 |

All 18 fraction-specific argmax comparisons match. Probability, total-variation and KL limits remain 1e-4, 1e-4 and 1e-5; the raw-logit column is reported separately rather than presented as an exact computation. These are model-output checks, not a new MMLU or long-generation evaluation.

Existing environment retention checks pass for both campaigns: performance governor during timing, zero swap usage, unchanged swap-I/O counters, no reported current throttle flags or competing workloads at guarded boundaries, and no new undervoltage in live monitors. Governors and original zram swap were restored afterward. TODO 1 attests 700 candidate islands / 2800 QPU dispatches including warmups, with zero fallback.

Two **V3D MMU errors** were captured: one during the initial full-vocabulary preflight and one during the resumed screening stage. Both logs report `MMU error from client PTB (1) ... pte invalid`. Processes completed; the reported numerical tests and dispatch contracts passed. The logs do not establish which context caused the errors or demonstrate sustained driver reliability. Preserve this stability qualification alongside the timing and numerical results. The timestamps and paths are listed in [analysis.json](raw/todo1-2-analysis.json.gz); raw logs are retained.

## Reproduction and rebuttal use

The commands and build/model checks are in [run.py](run.py) and [verified-build-model.json](raw/todo1-2-verified-build-model.json.gz). The first timing launch's self-match failure is recorded in [campaign.json](raw/todo1-2-campaign.json.gz); [resume-todo2.py](resume-todo2.py) reuses the successful fraction preflights and records the resumed timing source and controls. See `coa.md` for standalone commands, and [SHA256SUMS](SHA256SUMS) for the local evidence inventory. No anonymous artifact upload or manuscript replacement is implied.

Suggested TODO 1 wording: “We profiled M=129/257 at a fixed one-eighth split. CPU-only FFN shares are 67.69/66.51%. Median all-layer prep/CPU-branch/QPU-path/merge times are 116.707/1254.348/960.492/10.510 ms and 236.591/2555.561/1760.914/25.659 ms. Timestamped overlap shows CPU completion dominates 172/175 and 175/175 measured FFNs. These instrumented diagnostics identify the measured critical path; they do not replace the original request measurements.”

Suggested TODO 2 wording: “The new exploratory screen covers two row counts and three fractions (1/16,1/8,3/16), with six passing numerical/execution configurations. The complete grid includes four inconclusive comparisons and two losses under the stated interval rule; no median clears 1.05x. One eighth was fixed before historical corrected confirmation and is not claimed optimal. No new selection is promoted without independent confirmation.”

The original working-directory paths and inventory above describe the retained run. This source-package copy links to compressed records where applicable. `SHA256SUMS` is the original local inventory; use `provenance.json` to verify the files actually distributed here.
