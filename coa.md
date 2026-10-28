# Course of action: rebuttal experiments 1 and 2

This handoff covers **TODO 1: integrated FFN breakdown** and **TODO 2: integrated calibration and split selection** in the author response. These experiments completed on 28 October after the power issue was fixed. Their retained results, subsequent TODO 3–9 diagnostics, failed attempts, and scope qualifications are in [the experiment report](experiment_logs/20261028-rebuttal/REPORT.md). Follow the commands below to generate new measurements on an actively cooled Raspberry Pi 5 with reliable power and sufficient RAM for the same model. Historical results are not placement evidence for a changed implementation.

## 1. Prepare the identical model and runtime

Use Gemma 4 E2B Q4_K_XL with Q4_0 FFN tensors, the pinned llama.cpp revision, four CPU threads, WGS=24, and the existing arithmetic and numerical gates. These scripts deliberately validate the tested model's 35 layers: 15 FFNs of width 6144 and 20 of width 12288. A different model is a different experiment.

From this repository root, set absolute paths for your machine:

```bash
export LLAMA_ROOT=/absolute/path/to/llama.cpp
export GEMMA_MODEL=/absolute/path/to/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
export FFN_RESULTS=/absolute/path/to/new-results-directory
mkdir -p "$FFN_RESULTS"
uv sync --locked --group dev
```

The recorded model SHA-256 is `e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889`. Verify the acquired file with `sha256sum "$GEMMA_MODEL"` and retain the result. Model weights are not supplied by these scripts.

Use a dedicated llama.cpp checkout at `91d2fc387529940230555abd297a8b5e99737d3f`. Apply the bundled patches in order, checking each first. Do not reapply them to an already patched checkout:

```bash
git -C "$LLAMA_ROOT" rev-parse HEAD
for patch in integrations/llama_cpp/patches/000{1,2,3,4,5}-*.patch; do
  git -C "$LLAMA_ROOT" apply --check "$PWD/$patch" || break
  git -C "$LLAMA_ROOT" apply "$PWD/$patch" || break
done
git -C "$LLAMA_ROOT" status --short
```

Read [the integration README](integrations/llama_cpp/README.md) for patch details. The stride-aware patch 0005 is mandatory. The profiled CPU builder currently requires the **Unix Makefiles** generator and Linux GCC-compatible tools; it reuses existing objects from that build. The following baseline settings match the retained configuration:

```bash
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CPU_REPACK=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_ALL_VARIANTS=OFF
cmake --build "$LLAMA_ROOT/build" --parallel 4 --target llama-bench

cmake -S integrations/llama_cpp -B build/llama-qpu-profile \
  -DLLAMA_CPP_ROOT="$LLAMA_ROOT" \
  -DLLAMA_CPP_BUILD_DIR="$LLAMA_ROOT/build" \
  -DQPU_LLAMA_BUILD_GGML_BACKEND=ON \
  -DQPU_LLAMA_BUILD_HARDWARE_TESTS=ON -DQPU_LLAMA_BUILD_BENCHMARK=ON
cmake --build build/llama-qpu-profile --parallel 4 --target \
  ggml-qpu-inline qpu_repack_stride_smoke \
  qpu_ffn_island_graph_smoke qpu_ffn_island_model_smoke

uv run scripts/build_llama_cpp_ffn_profile.py \
  --llama-root "$LLAMA_ROOT" --output build/ffn-profile-cpu
```

`build_llama_cpp_ffn_profile.py` generates timestamped **copies** of two CPU source files and links an isolated `libggml-cpu.so.0`, leaving the retained sources, objects and libraries untouched. It records original/generated source hashes, compile/link commands and the resulting library hash in `build/ffn-profile-cpu/build.json`. For a nondefault llama.cpp build directory, pass `--llama-build` here and an explicit `--llama-bench` to the runners.

## 2. Check correctness before timing

These commands run on the colleague's Pi, not the original unstable board:

```bash
build/llama-qpu-profile/qpu_repack_stride_smoke \
  > "$FFN_RESULTS/stride.log" 2>&1

LD_LIBRARY_PATH="$PWD/build/ffn-profile-cpu" GGML_QPU_FFN_PROFILE=1 \
  build/llama-qpu-profile/qpu_ffn_island_graph_smoke \
  > "$FFN_RESULTS/joined-graph.log" 2>&1

LD_LIBRARY_PATH="$PWD/build/ffn-profile-cpu" \
  build/llama-qpu-profile/qpu_ffn_island_model_smoke "$GEMMA_MODEL" 0.125 \
  > "$FFN_RESULTS/full-vocabulary.log" 2>&1
```

Check each exit status before continuing. The full-model test compares three deterministic prompts at each M=129 and M=257 across the complete vocabulary. Its fixed gates are maximum probability error <=1e-4, total variation <=1e-4, KL divergence <=1e-5, finite outputs and matching argmax IDs; raw-logit errors are reported separately. Do not relax a gate to accept a run. The optional fraction argument is new; omission retains 1/8.

## 3. Machine controls

The runners target Linux/Raspberry Pi OS with systemd and `dev-zram0.swap`. They temporarily set all CPU frequency policies to `performance`, mask that swap unit, and disable active swap. They restore governors and swap after normal completion or an ordinary error. They need `sudo` access for these actions and kernel monitoring; run `sudo -v` and check `sudo -n true` immediately before a campaign. A long campaign may outlast a sudo timestamp: arrange appropriate existing noninteractive lab access for these exact commands.

Stop other inference servers and CPU-intensive jobs. Do not run the two experiments concurrently. Each process has a 65 C start gate. `dmesg` access is checked during capture; new undervoltage events abort by default and are saved in the kernel log. `--ignore-undervoltage` records events without aborting, for exploratory diagnosis only. Such data must not be represented as controlled latency evidence.

A hard power failure cannot execute cleanup. After a reboot, inspect the saved JSON and machine state before retrying. Always use a **new output directory**; existing nonempty experiment directories are rejected. Completed process records and logs are saved incrementally; the scripts do not automatically resume an interrupted statistical campaign.

## 4. TODO 1: integrated FFN breakdown

```bash
sudo -v
uv run scripts/run_llama_cpp_ffn_breakdown.py \
  --llama-root "$LLAMA_ROOT" --model "$GEMMA_MODEL" \
  --pairs 5 --output "$FFN_RESULTS/todo1"

uv run scripts/summarize_llama_cpp_ffn_breakdown.py \
  "$FFN_RESULTS/todo1/breakdown.json" --output "$FFN_RESULTS/todo1-summary.json"
```

This launches 20 independent processes: five CPU/candidate pairs at each M=129,257, with a fixed 1/8 share. Each process warms up once and times one complete prompt. Both placements load the isolated profiled CPU backend; only the candidate loads the QPU plugin. The original uninstrumented executable/library is preserved for headline confirmation.

For every layer, the logs include region entry/exit, initial preparation, CPU branch start/finish, four QPU submission/observed-completion intervals, worker completion and merge start/finish. The CPU-only region starts before the first projection's quantization and finishes after the down projection's completion barrier. The QPU interval includes host scheduling, queueing, device execution, observation and output handling; it does not isolate arithmetic or launch overhead.

The summarizer removes the first 35 warmup spans and aggregates the final measured pass **within each process**. JSON and CSV report the two width classes, measured FFN wall sums, preparation, CPU branch, QPU path, merge and handoff. CPU-only FFN share is that process's sum of FFN wall durations divided by that process's complete prompt sample. Final tables use medians of process totals/shares. CPU and QPU intervals overlap: the critical path uses absolute finish timestamps, not the sum of branch times. Do not substitute Fig. 5's older projection-overlap path or a sum of per-layer medians.

Report a compact table for each M/width class with CPU-only FFN, prep, CPU branch, QPU path, merge, FFN wall and CPU-only request share. This is an **instrumented diagnostic**, not an uninstrumented headline speedup. Explain the size effect only if the collected critical-path data support it. Failed/missing processes remain visible and cannot be counted as completed pairs.

## 5. TODO 2: recover first, then screen if needed

Recover the checked-in historical records without hardware:

```bash
uv run scripts/audit_llama_cpp_ffn_calibration.py \
  --output "$FFN_RESULTS/historical-calibration.json"
```

This exports all 24 historical integrated comparisons and the corrected fixed-policy confirmation records, with identities and exclusions. The two multi-fraction campaigns predate the CPU-stride fix and were rejected for swap activity. Matching QPU telemetry or a suffix-only check does not validate the erroneous CPU prefix. Corrected confirmation used a fixed 1/8; the available records do not establish it as an optimum. Historical prepared FFN speedups exclude input preparation and cannot fill this gap.

The new corrected **exploratory** screen validates 1/16,1/8,3/16 separately on full-model outputs before timing accepted fractions:

```bash
sudo -v
uv run scripts/run_llama_cpp_ffn_split_screen.py \
  --llama-root "$LLAMA_ROOT" --model "$GEMMA_MODEL" \
  --repetitions 3 --output "$FFN_RESULTS/todo2"

uv run scripts/summarize_llama_cpp_ffn_split_screen.py \
  "$FFN_RESULTS/todo2" --output "$FFN_RESULTS/todo2-grid.json"
```

If all fractions pass, timing uses three independent processes for each baseline/candidate configuration at both row counts (24 processes total). The inner evaluator reruns its mandatory preflights; it uses the same explicit model, benchmark, generated programs and plugin build. Telemetry checks physical M, all 35 layers, complete channel coverage, program hashes, four dispatches per island and zero unexpected fallbacks. Numerical rejections are kept; they are not silently turned into timed configurations. The exported six-cell grid also shows configurations never reached after an interruption.

Report the **entire** grid, including losses, numerical rejection/error states, missing cells, environment retention and uncertainty. A screen is not independent confirmation. Any newly selected policy requires a separate request-level confirmation campaign. The fixed FFN hook is opt-in; the related placement gate uses the ratio of median request latencies >=1.05 and bootstrap lower bound >1.0 with correctness/environment/memory checks. It does not require the lower bound to exceed 1.05. Do not claim a prepared-region win or a favorable screen automatically clears that gate.

## 6. Return package and scope

Return the complete results directory, `build/ffn-profile-cpu/build.json`, source revisions/diffs, model hash and the exact commands. Keep stdout, stderr and kernel logs, all process samples, validation outcomes, telemetry, rejected runs and machine-state records. Exit 2 means incomplete/failed execution; inspect `completed`, `power_clean`, numerical outcomes and `retention` even when a child returned zero. Local filesystem paths may identify a colleague, so anonymize copies before an anonymous release without altering measured values or source/program identities.

Only verified values may replace the rebuttal TODOs. If model profiling remains incomplete, retain TODO 1's fallback. For TODO 2, explain that 1/8 was fixed before corrected confirmation and do not call it optimal without valid selection evidence. These procedures prepare measurements and documentation; they do not upload an artifact or replace a submitted manuscript.

Portable checks, safe to run without QPU execution:

```bash
uv run pytest -q tests/test_llama_cpp_ffn_breakdown.py
uv run ruff check scripts/*ffn*breakdown.py scripts/*ffn*split_screen.py \
  scripts/build_llama_cpp_ffn_profile.py scripts/audit_llama_cpp_ffn_calibration.py \
  tests/test_llama_cpp_ffn_breakdown.py
```

## 7. Remaining rebuttal diagnostics (TODOs 3–9)

Read [the completed report](experiment_logs/20261028-rebuttal/REPORT.md) first.
The commands below produce new evidence; they do not reconstruct the exact
historical numbers. Use empty output directories. Run hardware commands
sequentially, without builds or other CPU-intensive work in parallel. The
control wrapper records the exact command, environment and live kernel stream,
restores governors/swap, and returns exit 2 for a failed process, new voltage
event or GPU error. Check `retained_for_latency`, not just process completion.

### Matched FFN ablations (TODO 3)

Build separate libraries to preserve the retained baseline:

```bash
uv run scripts/build_llama_cpp_ffn_profile.py \
  --llama-root "$LLAMA_ROOT" --output build/ffn-ablation-cpu

cmake -S integrations/llama_cpp -B build/llama-qpu-ablation \
  -DLLAMA_CPP_ROOT="$LLAMA_ROOT" \
  -DLLAMA_CPP_BUILD_DIR="$LLAMA_ROOT/build" \
  -DQPU_LLAMA_BUILD_GGML_BACKEND=ON \
  -DQPU_LLAMA_BUILD_HARDWARE_TESTS=ON -DQPU_LLAMA_BUILD_BENCHMARK=ON
cmake --build build/llama-qpu-ablation --parallel 4 --target \
  ggml-qpu-inline qpu_ffn_island_graph_smoke qpu_ffn_island_model_smoke \
  qpu_llama_cpu_repack_bench

export MORE_RESULTS=/absolute/path/to/new-remaining-results
mkdir -p "$MORE_RESULTS"
uv run scripts/run_llama_cpp_ffn_ablation.py \
  --llama-root "$LLAMA_ROOT" --model "$GEMMA_MODEL" \
  --rounds 5 --output "$MORE_RESULTS/todo3"
```

The four modes are `full`, `compact`, `materialized`, and `serialized`.
`GGML_QPU_FFN_ABLATION` is an opt-in diagnostic switch, unset in ordinary
execution. Every mode passes six full-model checks and must match the full
mode's complete candidate-logit file hashes before any timing. Five randomized
process blocks follow; fixed M=129, fraction=1/8, four threads and WGS=24.
Compaction uses the correct slice and matching quantizer; materialization
round-trips existing Q8 bytes and scales without re-quantization. The materialized
variant includes an extra CPU-access handoff. Serialization waits for the same
QPU branch before allowing the CPU branch. These are matched counterfactuals,
not separately additive contributions.

### Production MTP node (TODO 4)

Supply the retained real Q4_0 MTP weight and F32 activation fixture for
M=1, K=256, N=2048. Captured `.bin` payloads are external assets and are not
committed. Synthetic matrices would define a different diagnostic. On the
evaluation machine the inputs were in
`experiment_logs/20260825-qpu-next/mtp-inline-fixture/`, named
`blk-0-ffn_gate-weight.q4_0.bin` and `blk-0-ffn_gate-weight.m1.f32.bin`.

```bash
export MTP_WEIGHTS=/absolute/path/to/production.q4_0.bin
export MTP_ACTIVATION=/absolute/path/to/production.m1.f32.bin
uv run scripts/run_rebuttal_diagnostics.py \
  --output "$MORE_RESULTS/todo4-control" --timeout 300 -- \
  "$PWD/.venv/bin/python" scripts/benchmark_rebuttal_mtp.py \
  --binary build/llama-qpu-ablation/qpu_llama_cpu_repack_bench \
  --plugin build/llama-qpu-ablation/libggml-qpu-inline.so \
  --weights "$MTP_WEIGHTS" --activation "$MTP_ACTIVATION" \
  --pairs 5 --samples 31 --output "$MORE_RESULTS/todo4"
```

The CPU_REPACK benchmark uses one thread, persistent graph state and the actual
inline M=1 hook on candidate runs. Its `mode` JSON field describes the enclosing
CPU_REPACK graph, including when the hook executes; verify candidate telemetry.
Separate submit/wait/sync fields come from existing host timestamps. Device
unpacking, arithmetic and memory behavior remain unresolved.

### Convolution, fused CPU control and ResNet (TODOs 5, 6(b), 7)

The operator worker requires the existing optional Torch/Torchvision evaluation
dependencies (`uv sync --locked --group dev --extra evaluation`). Each accepted
convolution case runs in its own controlled process:

```bash
uv run scripts/run_rebuttal_diagnostics.py \
  --output "$MORE_RESULTS/todo5-pointwise-control" -- \
  "$PWD/.venv/bin/python" scripts/benchmark_rebuttal_operators.py conv \
  --conv-case pointwise --warmup 3 --repeat 11 \
  --output "$MORE_RESULTS/todo5-pointwise.json"

uv run scripts/run_rebuttal_diagnostics.py \
  --output "$MORE_RESULTS/todo5-dense-control" -- \
  "$PWD/.venv/bin/python" scripts/benchmark_rebuttal_operators.py conv \
  --conv-case dense-3x3 --warmup 3 --repeat 11 \
  --output "$MORE_RESULTS/todo5-dense.json"

uv run scripts/run_rebuttal_diagnostics.py \
  --output "$MORE_RESULTS/todo6-control" -- \
  "$PWD/.venv/bin/python" scripts/benchmark_rebuttal_operators.py epilogue \
  --warmup 3 --repeat 31 --output "$MORE_RESULTS/todo6.json"
```

The depthwise case is implemented as `--conv-case depthwise-3x3`, but the
completed campaign exposed GPU resets and a numerical failure. Its timings
were rejected. Do not use an earlier completed attempt as accepted evidence
or silently retry until one succeeds. The current depthwise reduction also
differs from the historical general lowered path.

ResNet requires an existing verified artifact and an actual image from the
quality-study manifest. If those assets are unavailable, omit this experiment:

```bash
export RESNET_ARTIFACT=/absolute/path/to/resnet18-native
export RESNET_IMAGE=/absolute/path/to/manifest-image.jpg
uv run scripts/run_rebuttal_diagnostics.py \
  --output "$MORE_RESULTS/todo7-control" -- \
  "$PWD/.venv/bin/python" scripts/benchmark_rebuttal_operators.py resnet \
  --artifact "$RESNET_ARTIFACT" --image "$RESNET_IMAGE" \
  --warmup 2 --repeat 7 --output "$MORE_RESULTS/todo7.json"
```

The runtime receives V1-preprocessed 1x3x224x224 input, even when the source
dataset images are 128 pixels. Timings exclude image preprocessing and include
the complete inference graph to logits. This is one-input steady-state timing,
not a new accuracy campaign. The available optimized fused CPU control is
`aten::_add_relu.out`. No new optimized CPU W8A8 runtime is constructed for
TODO 6(a).

### Audits and documentation (TODOs 8–10)

```bash
uv run scripts/audit_rebuttal_fp32.py --output "$MORE_RESULTS/todo8-9.json"
```

This requires the retained FP32 suite logs and git history. It exports the
documented 112-configuration grid and the selector evidence; it does not infer
a denominator from the separate stage sweep. If historical logs are absent,
use the archived completed audit for the historical result and explicitly
report that recovery on the new clone was unavailable.

The remaining-result summarizer takes the exact layout of the completed
campaign, including its rejected-attempt records:

```bash
uv run scripts/summarize_rebuttal_remaining.py \
  benchmark_runs/20261028-remaining --output /path/to/summary.json
```

Do not manufacture rejected-attempt records to satisfy that campaign-specific
summarizer on a different run. Raw per-command records remain usable directly.
The portable checks are:

```bash
uv run python -m pytest -q tests/test_llama_cpp_ffn_breakdown.py \
  tests/test_llama_cpp_qpu_ffn_island_eval.py tests/test_llama_cpp_process_guards.py
```

[REBUTTAL_NOTES.md](REBUTTAL_NOTES.md) records prior-work attribution and claim
corrections. Unavailable submission-specific and anonymous-upload tasks remain
unclaimed. Preparing these repository files is distinct from uploading an
anonymous artifact or replacing a submitted manuscript.
