# Reproducing the paper

Start with [INSTALL.md](INSTALL.md). The exact 26-page author-supplied
manuscript has SHA256
`3bcc96f58f3b8b500fc8e89ff7c2b8e25bb5226dbf2a946159c7b75f587b6ad2`.
[artifact/PAPER.md](artifact/PAPER.md) maps all 39 tables and five figures to
their evidence and records manuscript corrections. The PDF remains an external
author-supplied input; the result checker does not require it or a PDF parser.

Regenerate the main results from retained measurements without hardware or
external models:

```sh
uv run --locked python scripts/reproduce_paper.py --output build/paper-analysis
```

This writes `paper-data.json`, CSV tables, complete FP32 operator-grid records,
`agreement.csv`, `agreement.json` and `PAPER-RESULTS.md`, alongside the original
headline/bootstrap and rebuttal analyses. Expected values are transcribed from
the identified PDF; calculated values come from archived measurements. Checks
use the precision printed in the paper and exit nonzero on disagreement.
Figures 4 and 5 can also be exported as PDF plots with `--plots` (requires
`matplotlib`, available through the locked `evaluation` extra). Their appearance
can differ from the manuscript while the measured values agree. Figures 1–3
are design diagrams, not empirical plots. Static memory estimates, recorded
quality confidence intervals and COCO AP are explicitly preserved report views.
The checker does not claim to recompute every supplementary table cell.

For fresh operator results (Tables 1/8 and Figure 4), the original
[kernel-suite commands](experiment_logs/20260819-qpu-xla-kernel-suite/README.md#reproduction)
provide full arguments. Write to new paths, for example:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run --locked python \
  examples/benchmark_qpu_xla_fp32_linear.py --case prefill-h1024-t64 \
  --warmup 2 --repeat 7 --output build/projections-new.json
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run --locked python \
  examples/benchmark_qpu_xla_fp32_minmax.py --length 4194048 \
  --warmup 3 --repeat 15 --output build/minmax-new.json
```

Repeat the min/max command at length 4194240 for the other Figure 4 point.
Historical operator timings used ondemand governors; keep that distinction
from the controlled FFN campaign. For Table 2, after native preparation:

```sh
uv run --locked python examples/benchmark_llama_cpp_qpu_fused_ffn.py \
  --rows 65 --rows 129 --rows 257 --warmups 5 --samples 11 \
  --cpu-geglu-benchmark "$BUILD_ROOT/llama-qpu-runtime/qpu_ggml_geglu_bench" \
  --cpu-down-benchmark "$BUILD_ROOT/llama-qpu-runtime/qpu_llama_cpu_repack_bench" \
  --qpu-plugin "$BUILD_ROOT/llama-qpu-runtime/libggml-qpu.so" \
  --output build/producer-new.json
```

This is the prepared producer/down boundary; input preparation is outside
it. The native preparer now builds both additional targets required here.
Use the controlled wrapper described below for retained timing and kernel logs.

For fresh measurements, prepare the pinned native build and verified assets,
then export `LLAMA_ROOT`, `BUILD_ROOT` and `GEMMA_MODEL` as documented in the
artifact guide. Inspect the commands before executing:

```sh
uv run --locked python scripts/artifact.py run paper --output build/paper-new
uv run --locked python scripts/artifact.py run paper --output build/paper-new --execute
```

The full paper profile uses five independent processes per placement at
M=129/257 and 15 cached-request CPU/candidate pairs, preserving the statistical
design in §4 and Tables 4/30. Stride, joined-graph and full-vocabulary preflights
are mandatory. The outer monitor rejects latency on a V3D error even if the
child completes numerical checks. A reduced functional profile is also supplied
for installation checks; it does not confirm the small performance effects.

Use `run quality --only mmlu` or the named ResNet/YOLO stages after acquiring
the corresponding inputs. See [artifact/profiles.json](artifact/profiles.json)
for all arguments and stage IDs. SmolVLA is an explicit diagnostic because the
reported replay misses its FP32 gates and suffered hangs/resets; completed
resumed actions establish neither sustained reliability nor robot task success.

Run the October rebuttal using `run rebuttal` or selected `--only` stages.
It adds critical-path, split-grid, ablation, MTP, convolution preparation,
fused-CPU and complete-ResNet controls. These remain separate from the paper's
historical headline campaigns; the supplied manuscript does not contain the
October numerical tables. All rejected attempts and qualifications stay in
the package.

Controlled timings require an idle board, cooling, performance governors,
disabled swap and existing lab access for the runners' host controls. Run
stages sequentially, preserve every attempt, and use a fresh output directory.
Do not relax numerical or kernel-log gates to obtain a retained result.
Runtime, seed, sample count, model, representation and timing boundary must
accompany any comparison. New measurements need not match exact milliseconds;
assess the paper's conclusions with its numerical gates and confidence
intervals. No blanket timing tolerance is asserted.

[artifact/EXTENDING.md](artifact/EXTENDING.md) gives a custom workload example.
[artifact/REVIEWER_ACCESS.md](artifact/REVIEWER_ACCESS.md) describes independent
reviewer setup and returned records. Authors' checks prepare the artifact for
independent evaluation; they cannot award either badge.
