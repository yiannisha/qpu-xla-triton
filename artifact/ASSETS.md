# External inputs and preparation

The release retains numerical outputs and selection/provenance manifests,
not model weights or source images/videos. `assets.json` records input hashes
and upstream locations. Obtain external assets under their original terms;
the artifact does not remove upstream access conditions or grant new licenses.

Copy `local-assets.example.json` to ignored `artifact/local-assets.json` and
replace paths. Verify only the inputs needed for the selected experiment:

```sh
uv run --locked python scripts/artifact.py assets \
  --config artifact/local-assets.json --only gemma,mtp-model
```

The command hashes the actual files; a matching filename is insufficient.
Omit `--only` to check all six file assets. A missing or mismatched asset gives
`ok: false` and exit 2. Dataset preparation and identities are described below.

## Gemma and production MTP

The pinned model source is
[Unsloth Gemma QAT revision 66a399f](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/tree/66a399f68ddd113b06dff02fca9523e55465d11d).
Acquire `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` (2.62 GB) and
`mtp-gemma-4-E2B-it.gguf` (59.2 MB) from that revision using the URLs in
`assets.json`. Their full SHA256 values are authoritative. The model card
and retained GGUF manifests describe the quantized representation; substituting
a differently quantized GGUF is a different experiment.

Prepare the exact M=1 K=256 N=2048 production MTP diagnostic fixture:

```sh
uv run --locked python scripts/artifact.py prepare-mtp \
  --model /models/mtp-gemma-4-E2B-it.gguf --output build/artifact-mtp
```

This verifies the whole model, extracts the native `blk.0.ffn_gate.weight`
range identified in the retained GGUF manifest, and checks it against the
measured weight SHA256. The 1024-byte retained activation is copied and
verified against the October record. The output is 294912 bytes of Q4_0
weights plus the activation and provenance. No synthetic substitute is used.
Set `MTP_FIXTURE` if you choose a location other than `build/artifact-mtp`.

## MMLU

Download the 6.9 MB `mmlu-test.bin` from the pinned validation-dataset URL in
`assets.json`. Its hash matches the original campaign. The historical source
commit was not recorded; the currently pinned commit has been checked to
contain the exact retained file hash. Run the file check before evaluation.
The fixed task selection is llama.cpp's seed-1 permutation of 1000 tasks.
Export `MMLU_DATASET` and use the `quality:mmlu` stage. Native preparation
applies patch 0004 and builds `llama-perplexity`.

## ResNet-18 and the ImageNet derivative

Install optional evaluator dependencies and convert Torchvision's fixed V1
weights using the existing content-addressed converter:

```sh
uv sync --locked --group dev --extra evaluation
uv run --locked qpu-model-quality convert-resnet18 --output /models/resnet18-native
```

The retained manifest in `evidence/manifests/resnet18-artifact.json` records the
expected converted state hash, preprocessing and source. Export
`RESNET_ARTIFACT` to the converted directory.

The study used `benjamin-paine/imagenet-1k-128x128` at revision
`dd3adfd6ecfb3ef3ae26205733bf5d9d5410643b`, not official full-resolution
ImageNet. Its validation Parquet is 217 MB. The selected ImageFolder files
occupy about 36 MB. The isolated preparation script has its own locked
PyArrow dependency, so it is not added to the runtime:

```sh
uv run --locked --script scripts/prepare_imagenet_samples.py \
  --download --parquet /datasets/imagenet-validation.parquet \
  --output /datasets/imagenet-128-selected
```

The script verifies the Parquet hash, writes only the 5000 selected original
encoded image files under the manifest's names, and checks every reconstructed
image against the retained content hashes. Source drift fails rather than regenerating
a new favorable subset. Export `IMAGENET_ROOT` to that ImageFolder.
The quality profile passes the original sample manifest explicitly. For TODO7,
set `RESNET_IMAGE` to an actual image under that root; V1 preprocessing produces
224x224 model input. Different valid images define a new one-input timing.

## YOLOv8n and COCO

Acquire the `yolov8n.pt` checkpoint and COCO val2017 images/annotations from
the upstream URLs in `assets.json`. Verify the checkpoint and the extracted
`instances_val2017.json` hashes. The annotations hash applies to the JSON,
not its distribution ZIP. Images require roughly 0.8 GB; annotations and
extraction require additional space.

Export `YOLO_CHECKPOINT`, `COCO_IMAGES` (the extracted val2017 directory), and
`COCO_ANNOTATIONS`. The quality profile passes the retained seed-20260911
1000-image manifest, image size 640, and the established evaluator settings.
The source annotations determine image identities; the retained COCO manifest
does not hash each JPEG. Original numerical predictions are archived for
offline inspection. Recomputing AP from predictions also needs the external
annotations and optional `pycocotools`; use
`qpu_xla.quality.coco.coco_bbox_metrics` with the manifest's image IDs.

## SmolVLA

Use `lerobot/smolvla_base` revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205` and LeRobot checkout
`8b256a6c0d4769c3cc3e7e98f04940126398a391`. Follow the conversion commands
in `SMOLVLA-QPU.md` and the replay-building/upstream-recording commands in
`MODEL-QUALITY.md`. The replay source is `lerobot/svla_so100_pickplace` v3.0;
the retained manifest records ten selected episodes and camera mapping.
Retain the exact language tokens, state, masks, image preprocessing, and flow
noise. Check the final replay hash from `assets.json`; the earlier inputs-only
replay has a different hash and is not the upstream-action oracle.

Export `SMOLVLA_ARTIFACT` and `SMOLVLA_REPLAY` only for an explicitly selected
`quality --only smolvla` diagnostic. This path misses its FP32 quality gates
and experienced driver hangs. It is excluded from the default fresh-run
profile. Its completed actions and report remain available for offline checks.
Budget multiple GB for checkpoints, upstream dependencies and replay/video
data; this preparation has not been timed as part of the local release.

## Resource planning

The portable basic test and archived analysis need no model download or QPU.
Install the dependency environment on ordinary disk. A fresh install into a
4 GiB RAM-backed `/tmp` exhausted space while copying the locked Torch package;
keeping the environment on the same filesystem as the uv cache also permits
hardlink reuse. Source-only extraction does not account for dependency space.
Allow several GB for dependencies, at least 3 GB for the Gemma inputs, and
additional room for native builds and logs. Quality studies add image and robot
data. Fresh model timing requires an otherwise idle, reliably powered and
cooled Pi with enough free RAM to keep swap disabled; the historical host had
8 GiB. Full qualification workloads may take hours and known unstable paths
are not silently retried. Record actual preparation/run costs for the reviewer
machine rather than presenting an unmeasured estimate as a guarantee.

The clean-build follow-through measured about 7.4 min for native preparation,
roughly 290 MB for the checkout/build and integration outputs (excluding the
shared Git object store), and about 2.1 min for each bounded main FFN stage,
including full numerical preflights. The largest model process reached 4.41 GB
RSS. These measurements used cached model inputs and a local upstream object
store; cold download/cache costs are additional. See `VALIDATION.md` for the
prompt-stage kernel rejection and the separately accepted cached-request run.

For scheduling only, extrapolating the bounded process costs suggests roughly
10–20 min for the full `paper` profile on a similar idle, cooled host with
cached assets, if its gates pass. This is an estimate, not a measured full
campaign duration. Cold downloads, cooling waits and fault investigation add
time; a failed gate stops the run rather than triggering automatic retries.
