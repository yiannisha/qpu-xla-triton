# QPU-XLA research artifact

This is the local artifact for the VideoCore VII runtime and its native
`llama.cpp` integration, including the October 2026 rebuttal. Publication to an
archival service is intentionally deferred. No DOI, badge, independent result
validation, or final-manuscript alignment is claimed.

Associated paper title supplied by the author: **Practical: Building and
Evaluating a Native CPU-QPU Runtime for Edge AI Inference on Raspberry Pi 5**.

## Start here

Use Linux and Python 3.12 or later. On a Raspberry Pi, use 64-bit Raspberry Pi
OS and an actively cooled Pi 5; the historical model experiments used 8 GiB
RAM, reliable power, four CPU threads, and WGS=24. Ordinary PCs can verify
evidence, analyze retained results, and run the CPU/fake-backend check.

From the repository or extracted release root:

```sh
uv sync --locked --group dev
uv run --locked python scripts/artifact.py smoke --output build/artifact-smoke
uv run --locked python scripts/artifact.py analyze --output build/artifact-analysis
```

The basic test returns `ok: true`, verifies evidence checksums, and reports
maximum absolute errors for FP32 and INT32 matmul. The chosen integral inputs
normally give zero error; the FP32 gate is `atol=rtol=3e-4`, while INT32 is
exact. The fake backend exercises contracts without attesting QPU execution.
Analysis regenerates the corrected September headline medians/intervals, the
October rebuttal summaries and full 112-case projection grid,
recomputes inexpensive numerical metrics from retained model outputs, and
writes `RESULTS.md` and JSON. It neither reruns hardware measurements nor
recomputes expensive quality confidence intervals or COCO AP; the original
quality reports remain separate, explicitly labeled views.

The local source/evidence package is about 210 MB unpacked before dependencies.
Allow several GB for the locked Torch environment on ordinary disk, not a
small RAM-backed `/tmp`; model builds and datasets
need additional space. [ASSETS.md](ASSETS.md) distinguishes these costs. See
[VALIDATION.md](VALIDATION.md) for actual preparation/check durations measured
during this release; complete campaign durations remain unmeasured here.

## Inventory and evidence

| Path | Purpose |
|---|---|
| `src/`, `examples/`, `tests/` | Runtime, kernels, examples, differential tests |
| `integrations/llama_cpp/` | Native integration, five ordered patches, program binaries/headers, launch manifest |
| `experiment_logs/` | Historical raw results and reports, including excluded campaigns |
| `experiment_logs/20261028-rebuttal/` | TODOs 1–9, raw compressed records, provenance, checksums and rejection qualifications |
| `artifact/evidence/` | Selected completed model-quality outputs, reports, sample manifests, small MTP activation, source provenance |
| [claims.json](claims.json) | Stable claim IDs, retained evidence, fresh-run profile, scope and pending manuscript locations |
| [assets.json](assets.json), [profiles.json](profiles.json) | External input identities and configurable argument-array workflows |
| [LIMITATIONS.md](LIMITATIONS.md) | Numerical contracts, rejected evidence and unsupported claims |
| [LICENSES.md](LICENSES.md) | Existing component notices and external-asset boundaries |

Original reports are preserved byte for byte and may name their original
working paths. `artifact/evidence/provenance.json` maps selected payloads to
their distributed paths; `artifact/evidence/manifests/` contains the sample
and converted-model manifests used by the fresh-run profiles.

The corrected September headline evidence is in `BUG_FIX_RESULTS.md`. The
October instrumented breakdown and exploratory screen are distinct campaigns.
Keep both, including all losses, inconclusive cells, rejected attempts, and
machine-state qualifications. Their results are not interchangeable.

## Pi preparation and basic test

Install `git`, a C/C++ compiler, `make`, `cmake`, Python development headers,
and `uv`. Install optional model evaluators only when needed:

```sh
uv sync --locked --group dev --extra evaluation
uv run --locked python scripts/artifact.py check --hardware --output build/pi-environment.json
```

The current low-level driver opens
`/dev/dri/by-path/platform-1002000000.v3d-card`; also retain access to
`/dev/dri/renderD128`. Check the host's `video`/`render` groups. Containers
cannot supply the V3D kernel driver. [Dockerfile](Dockerfile) is an optional
ARM64 environment recipe; see its limitations below.

Use a dedicated `llama.cpp` checkout, not a checkout containing unrelated edits:

```sh
git clone https://github.com/ggml-org/llama.cpp.git /path/to/llama.cpp
git -C /path/to/llama.cpp checkout 91d2fc387529940230555abd297a8b5e99737d3f
export LLAMA_ROOT=/path/to/llama.cpp
export BUILD_ROOT="$PWD/build/artifact-native"
uv run --locked python scripts/prepare_artifact_native.py \
  --llama-root "$LLAMA_ROOT" --build-root "$BUILD_ROOT" --rebuttal
uv run --locked python scripts/artifact.py smoke --hardware \
  --native-build "$BUILD_ROOT/llama-qpu-runtime" --output build/pi-smoke
```

Preparation checks the base revision, checks/applies bundled patches in order,
builds the ordinary integration, and with `--rebuttal` builds the separate
profile/ablation libraries. It does not reset source trees. A patch conflict is
an error; return to a dedicated clean checkout rather than forcing application.
`BUILD_ROOT` also selects those integration/profile outputs in the experiment
profiles. Keeping it separate preserves any historical compiled binaries.
The Pi smoke adds public widened INT16 convolution with exact INT32
accumulation, the native stride regression, and a synthetic joined FFN graph.
It is a bounded basic check, not full-model validation or a timing campaign.

## Fresh experiments

Acquire and verify [external assets](ASSETS.md), then export paths as needed:

```sh
export GEMMA_MODEL=/models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
export RESNET_ARTIFACT=/models/resnet18-native
export RESNET_IMAGE=/datasets/imagenet/0000/manifest-image.jpg
uv run --locked python scripts/artifact.py run paper --output build/paper-run
uv run --locked python scripts/artifact.py run rebuttal --output build/rebuttal-run
```

These commands print a plan. Add `--execute` to run it, or select stages with
`--only todo1,todo2`. Always choose a new output directory. The `smoke` profile
is portable; `paper` runs corrected headline prompt and cached-request
experiments; `rebuttal` runs TODOs 1–9; `quality` runs the fixed qualification
workloads. Missing environment variables fail before execution. `--only`
allows a workload to run without acquiring unrelated assets.

For a bounded end-to-end check after a clean native build, use the supplied
functional profile, which runs M=129 with two independent CPU/candidate
processes and two cached-request pairs:

```sh
uv run --locked python scripts/artifact.py run paper \
  --config artifact/profiles-functional.json --output build/functional-run --execute
```

This preserves the numerical/dispatch/environment gates but has fewer
processes than the full paper campaign. It verifies exercisability; it does
not independently confirm small performance gains or promote a placement.
The clean-build follow-through passed all numerical/dispatch checks, but a
new MMU event rejected its prompt latency; the separately exercised cached
request passed with clean kernel retention. Both records are retained in
`artifact/evidence/clean-build/`; see `VALIDATION.md` before interpreting them.

Hardware stages run sequentially. Before controlled timing, establish the
documented lab access with `sudo -v`; runners need noninteractive access for
governors, swap control, and kernel monitoring. They record and restore host
state. Stop unrelated inference, compilation and CPU-heavy work. A child exit
of zero does not replace numerical, dispatch, kernel-log, or retention checks.
Do not mix build time, setup, steady-state execution, or instrumented diagnostics.

Fresh measurements need not equal historical medians. Use fixed numerical
gates and the recorded statistical design; require the paper's qualitative
claims to survive variation. Do not invent a blanket percentage tolerance for
small gains. The automatic placement rule remains median speedup >=1.05 and
bootstrap lower bound >1.0 plus all other gates; it is not a badge criterion.

## Local release

```sh
uv run --locked python scripts/artifact.py verify
uv run --locked python scripts/artifact.py build --output dist/qpu-xla-artifact.tar.gz
```

The builder reads [release-files.json](release-files.json), an explicit list;
it never recursively copies the working directory. Models, arbitrary
`benchmark_runs/`, `.git`, environments, local configuration and incidental
files are excluded. Add intended new release files to the inventory explicitly.
The release includes `SHA256SUMS`, `artifact/RELEASE.json` with per-file hashes,
and version metadata for installation without Git. The archive has a sidecar
checksum. Identical inputs produce identical archive bytes.

Extract into a separate directory on ordinary disk, run `sha256sum -c SHA256SUMS`, install using
`uv sync --locked --group dev`, then run the portable smoke and analysis there.
The artifact-specific package version is a content identifier; it is not an
upstream software release number. A later correction creates a new package
and should be entered in [CLARIFICATIONS.md](CLARIFICATIONS.md).

## Independent evaluation and reuse

An independent evaluator should follow the extracted-package instructions,
record their environment, and start with smoke, then selected main results.
Retained-log analysis checks analysis reproducibility; fresh hardware execution
checks the empirical claims. Authors' successful runs do not establish an
independently awarded badge.

For a new workload, copy `profiles.json` to `artifact/local-profiles.json`, add
a stage using a list of arguments, and pass `--config`. Change shape/thread/
fraction arguments in a copied stage, identify new inputs by hash, and write
to a new output directory. Treat changed models/representations as new
experiments. An example is in [EXTENDING.md](EXTENDING.md); runtime contracts
and kernel extension guidance remain in `QPU-XLA.md` and `LESSONS_LEARNED.md`.

Before submission, fill in [APPENDIX.md](APPENDIX.md) from the accepted
manuscript and complete the claim/figure map. [APPENDIX.tex](APPENDIX.tex) is an
insertable draft; check its two-page fit in the actual manuscript. Coordinate
hardware using the concrete [reviewer-access plan](REVIEWER_ACCESS.md).
CGO's appendix limit is two pages
before References, using the linked ASPLOS template. The current CGO process is
single-blind; historical notes about anonymous uploads do not establish a CGO
requirement. See the [CGO requirements](https://2027.cgo.org/track/cgo-2027-artifact-evaluation#submission-requirements).

The container recipe has not been built unless `VALIDATION.md` records a build.
For hardware runs, prefer the measured native host setup. Mount both DRM paths
and supply their numeric host groups; optional system-control timing runners
also need host services and narrowly scoped existing lab access that the
container does not configure. Do not interpret a portable container run as
hardware or timing validation.
