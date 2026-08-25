# Native SmolVLA CPU/QPU Baseline

This repository uses the 450M-parameter SmolVLA checkpoint as its first
end-to-end, π0-style vision-language-action baseline. The implementation runs
the complete graph from recorded RGB observations through the 50-action flow
chunk; it is not a collection of disconnected kernel tests.

The authoritative comparison is the upstream LeRobot PyTorch policy on CPU in
FP32. It is always reported as `upstream_torch_cpu_fp32`, at these immutable
revisions:

- LeRobot: `8b256a6c0d4769c3cc3e7e98f04940126398a391`
- `lerobot/smolvla_base`: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`

The native artifact loader and replay loader reject different provenance. The
upstream runner also rejects a different LeRobot checkout and validates the
loaded policy's complete state-dictionary topology before timing it.

## Why SmolVLA is the baseline

The production topology has 450,046,176 parameters. The native memory planner
calculates these pre-driver budgets for the exact checkpoint:

| Contract | Artifact | QPU arena | Projected peak RSS |
| --- | ---: | ---: | ---: |
| FP32 | 1.677 GiB | 0.906 GiB | 3.259 GiB |
| dynamic W8A8 | 0.687 GiB | 0.625 GiB | 1.989 GiB |

That leaves useful operating-system and camera/runtime headroom on an 8 GiB Pi
5. The planner defaults to a stricter 6 GiB process limit and fails before it
opens the VideoCore driver if a topology exceeds the limit.

A π0.5/PaliGemma-class model around 3.5–4B parameters is not a sound baseline
for this device. FP32 weights alone exceed 13 GiB, before activations, KV state,
QPU staging, PyTorch overhead, and the action expert. A carefully quantized
version may fit a 16 GiB device, but memory fit would not make its multi-billion
parameter CPU/QPU decode latency practical. Treat that as a later deployment
experiment after the SmolVLA full-replay measurements, not as the regression
baseline.

## Implemented graph

The default checkpoint uses three 256×256 RGB observations. Each is bilinearly
resized with top/left padding to 512×512, normalized to `[-1, 1]`, encoded by a
12-layer 768-wide SigLIP vision transformer, pixel-shuffled from 1024 patches to
64 tokens, and projected into the 960-wide VLM. Image tokens, 48 language
tokens, and one state token form a 241-token prefix.

The runtime fills the 16-layer VLM KV cache once. A 720-wide, 16-layer action
expert then performs ten Euler flow steps over a 50×32 padded action tensor,
alternating self-attention and cached-prefix cross-attention. The returned
logical action shape is 50×6.

One upstream detail is intentionally encoded rather than inferred from the
underlying SmolVLM config: the pinned LeRobot `apply_rope` helper executes with
its own 10,000 default wavelength. The native oracle uses 10,000 as well, even
though the source SmolVLM text config contains 100,000.

## Real heterogeneous stages

Forced QPU and hybrid modes execute native VideoCore programs. They do not
rename NumPy execution as QPU work. Hybrid plans submit disjoint CPU and QPU
partitions to separate queues and join their events.

| Graph stage | QPU implementation | Hybrid partition |
| --- | --- | --- |
| RGB resize and normalization | address-table bilinear FP32 kernel | output rows |
| Patch extraction | address-table gather kernel | patch rows |
| FP32 projections/patch embedding | padded tiled FP32 GEMM plus affine | output rows, or output columns for M=1 |
| W8A8 projections/patch embedding | dynamic row quantization, signed `v8dot` GEMM/GEMV, FP32 scales and affine | output rows or columns |
| LayerNorm and RMSNorm | FP32 reduction/affine kernels | token rows |
| GELU-tanh, SiLU, SwiGLU | FP32 elementwise kernels | token rows |
| Residual and scale | FP32 elementwise/affine kernels | token rows |
| Token embedding | FP32 address-table lookup | token rows |
| Split-half RoPE | precomputed tables plus FP32 QPU rotation | token/head rows |
| Grouped-query attention | persistent QPU GEMM → softmax → GEMM plan | query heads |
| Pixel shuffle | address-table gather kernel | output tokens |

Mask construction, sinusoidal time-table construction, the tiny Euler update,
and final action slicing remain on CPU. Attention mask application is a host
task between QPU score GEMM and QPU softmax. These are explicit boundaries,
not missing model stages.

When a tensor has only one indivisible QPU tile, forced hybrid mode uses that
tile on the QPU. Production projection and attention shapes are large enough
for the normal concurrent partitions; the fallback exists for exact tiny-shape
differential tests.

## Numerical modes and gates

The benchmark supports:

- `upstream_torch_cpu_fp32`
- `native_cpu_fp32`
- `qpu_fp32`
- `hybrid_fp32`
- `native_cpu_w8a8`
- `qpu_w8a8`
- `hybrid_w8a8`
- `auto_fp32`
- `auto_w8a8`

FP32 actions must match upstream with NRMSE at most `1e-3` and cosine
similarity at least `0.9999`. QPU W8A8 must match the dynamic-W8A8 CPU contract
within `1e-4` maximum absolute error. W8A8 versus FP32 must have NRMSE at most
`0.10` and cosine similarity at least `0.99`.

AUTO is CPU-safe. A stage is promoted only when the candidate registry contains
matching `supported-win` records for its exact stage shape, its enclosing block,
and the complete model shape/replay, all for the same QPU or hybrid placement.
A full-model win alone is deliberately insufficient.

## Build native artifacts

Download `config.json` and `model.safetensors` from the pinned SmolVLA model
revision. Convert the checkpoint twice; conversion streams one source tensor at
a time and the runtime memory-maps the result.

```console
.venv/bin/python examples/convert_smolvla_checkpoint.py \
  --checkpoint /models/smolvla/model.safetensors \
  --policy-config /models/smolvla/config.json \
  --output /models/smolvla-native-fp32 \
  --numerics fp32

.venv/bin/python examples/convert_smolvla_checkpoint.py \
  --checkpoint /models/smolvla/model.safetensors \
  --policy-config /models/smolvla/config.json \
  --output /models/smolvla-native-w8a8 \
  --numerics w8a8
```

The W8A8 artifact quantizes eligible projection and patch weights per output
channel. Token embeddings, positional embeddings, LM head, biases, norms, and
scales remain FP32. Activations are dynamically quantized per input row.

## Deterministic replay and upstream oracle

A replay is a non-pickled NPZ containing recorded uint8 HWC RGB images, camera
presence masks, fixed-length token IDs and language masks, normalized state,
and the exact initial FP32 flow noise. `SmolVLAReplay.save` and
`SmolVLAReplay.load` enforce all shapes, dtypes, and pinned revisions.

After installing the pinned LeRobot checkout with its `smolvla` optional
dependencies, attach authoritative upstream actions to a replay:

```console
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/record_smolvla_upstream.py \
  --artifact /models/smolvla-native-fp32 \
  --input replay-inputs.npz \
  --output replay-with-upstream-actions.npz \
  --lerobot-checkout /src/lerobot-pinned
```

Recorded upstream actions make correctness replays portable. They do not
replace a live upstream run when measuring upstream performance.

## Run the end-to-end benchmark

Each mode runs in a fresh subprocess so the upstream Torch policy, FP32 native
artifact, W8A8 artifact, and QPU arena never coexist on an 8 GiB device.

```console
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python examples/benchmark_qpu_xla_smolvla.py \
  --fp32-artifact /models/smolvla-native-fp32 \
  --w8a8-artifact /models/smolvla-native-w8a8 \
  --replay replay-with-upstream-actions.npz \
  --lerobot-checkout /src/lerobot-pinned \
  --warmup 1 \
  --repeat 3 \
  --output smolvla-qpu-benchmark.json
```

Cold start is artifact/model load, allocation and assembly, plus the first
complete action chunk. Steady state reuses the runtime and includes the entire
RGB-to-action path, including transfers. The JSON also retains prefix/denoise
times, QPU event/category totals, memory budgets, complete action arrays,
quality gates, and speedup over `upstream_torch_cpu_fp32`.

## Validation status

CPU tests cover the complete graph, strict checkpoint conversion, W8A8 oracle,
replay round trips, process-isolated benchmarking, and hierarchical AUTO
gating. VideoCore hardware differential tests cover every new stage and the
complete graph in forced QPU/hybrid FP32 and W8A8 modes using a reduced but
topologically complete model.

The production checkpoint and a recorded robot replay are intentionally not
stored in this repository. Therefore production-size latency and upstream
action metrics must come from the JSON benchmark above; reduced-shape test
results must never be presented as production performance.
