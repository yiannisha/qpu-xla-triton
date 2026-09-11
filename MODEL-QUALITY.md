# QPU Model-Quality Qualification

This suite measures task-output differences between an authoritative CPU path
and the corresponding QPU or CPU/QPU implementation. It does not turn those
differences into a pass/fail threshold. `valid_run` only attests that the same
samples were processed, outputs were finite, required QPU/CPU work occurred,
and no eligible operation silently fell back.

The fixed qualification workloads are:

- Gemma: 1,000 deterministically selected MMLU questions;
- ResNet-18: 5,000 ImageNet validation images, five from every class;
- YOLOv8n: a fixed seeded random 1,000-image COCO val2017 subset;
- SmolVLA: 100 observations, ten from each of ten episodes.

Install the optional evaluator dependencies with:

```console
uv sync --extra evaluation
```

The evaluator writes a `qpu-model-quality-v1` `report.json`, a compact
`README.md`, and task-specific raw outputs. Dataset and model payloads remain
external; the report records their SHA-256 identities.

## Vision execution modes

`ResNet18Runtime` and `YoloV8Runtime` replace every learned Torch `Conv2d` in
the canonical model graph with a prepared QPU-XLA adapter. The rest of each
canonical graph remains intact, including ResNet residual/pooling operations
and YOLO C2f/SPPF/FPN/PAN, DFL decoding, coordinate restoration, and NMS.

Supported modes are `cpu_fp32`, `qpu_fp32`, `hybrid_fp32`, `cpu_w8a8`,
`qpu_w8a8`, and `hybrid_w8a8`. Prepared weights and device buffers persist for
the runtime lifetime. Pointwise 1x1 convolution bypasses window expansion;
spatial convolution currently uses vectorized lowering into the validated
tiled GEMM kernels. W8A8 uses signed INT8 operands, exact INT32 accumulation,
dynamic per-row activation scales, and per-output-channel weight scales.

Hybrid convolution divides output rows into disjoint 16-aligned CPU and QPU
regions. Telemetry records eligible layers, calls, dispatched groups, and the
number of rows assigned to each processor.

## Gemma MMLU-1000

Apply the per-task output patch to the already pinned llama.cpp checkout after
the FFN-island patches, then rebuild `llama-perplexity`:

```console
git -C /home/yiannis/side/llama.cpp apply \
  /home/yiannis/side/py-videocore7/integrations/llama_cpp/patches/0004-per-item-multiple-choice-jsonl.patch
cmake --build /home/yiannis/side/llama.cpp/build --target llama-perplexity --parallel 4
```

Use the pinned `mmlu-test.bin` from
`ikawrakow/validation-datasets-for-llama.cpp` and run:

```console
uv run qpu-model-quality mmlu \
  --binary /home/yiannis/side/llama.cpp/build/bin/llama-perplexity \
  --model /path/to/gemma.gguf \
  --dataset /path/to/mmlu-test.bin \
  --plugin /path/to/libggml-qpu-ffn-island.so \
  --resume \
  --output experiment_logs/model-quality/gemma-mmlu-1000
```

The runner starts independent CPU and hybrid processes with four CPU threads,
uses llama.cpp's seed-1 task permutation, and compares answer accuracy,
directional answer flips, exact agreement, and normalized option log
probabilities. A hybrid report is invalid if FFN-island telemetry contains no
QPU dispatch.

## ResNet-18 ImageNet-5000

Create the non-pickled, content-addressed artifact from Torchvision's
`IMAGENET1K_V1` weights:

```console
uv run qpu-model-quality convert-resnet18 \
  --output benchmark_runs/resnet18-imagenet1k-v1
```

The ImageNet validation root must be in ImageFolder form with exactly 1,000
class directories. The first run creates a deterministic manifest containing
five hash-ranked paths per class; retain that manifest for every mode.

```console
uv run qpu-model-quality resnet18 \
  --artifact benchmark_runs/resnet18-imagenet1k-v1 \
  --dataset /datasets/imagenet/val \
  --mode hybrid_w8a8 \
  --resume \
  --output experiment_logs/model-quality/resnet18-hybrid-w8a8
```

Run the command once per candidate mode with the same `--manifest`. Reports
contain top-1/top-5 accuracy and candidate-minus-CPU deltas, direction of
top-1 flips, exact prediction agreement, per-class deltas, class-stratified
paired confidence intervals, and full-logit numerical diagnostics.
`--resume` checkpoints logits and placement telemetry every ten images. After
the first completed mode, pass its `logits.npz` through `--baseline-logits` to
reuse the byte-identical Torchvision baseline in later modes.

## YOLOv8n COCO val2017

The checkpoint is local and its checksum may be required with
`--checkpoint-sha256`. Validation fixes image size 640, confidence 0.001, NMS
IoU 0.7, `max_det=300`, and class-aware NMS for both paths.

```console
uv run qpu-model-quality yolov8n \
  --checkpoint /models/yolov8n.pt \
  --checkpoint-sha256 SHA256 \
  --images /datasets/coco/val2017 \
  --annotations /datasets/coco/annotations/instances_val2017.json \
  --samples 1000 \
  --sample-seed 20260911 \
  --mode hybrid_w8a8 \
  --resume \
  --output experiment_logs/model-quality/yolov8n-hybrid-w8a8
```

The first run writes `sample_manifest.json` in the output directory. Resumed
runs reload that exact ordered subset; `--manifest` can share it across modes.
The report uses authoritative `pycocotools` bbox AP50:95/AP50/AP75,
small/medium/large AP, and AR. It also contains per-category AP deltas,
same-image/same-class box agreement at IoU 0.5, and a true paired image
bootstrap. The qualification default is 10,000 bootstrap replicates; use
`--bootstrap-replicates 0` only for a smoke run.
`--resume` stores one durable JSONL record per image. Later modes can reuse a
completed first run's `cpu_predictions.json` with `--baseline-predictions`.

## SmolVLA replay-100

Create a replay from a compatible LeRobot dataset using the pinned checkout
and checkpoint processors. The default is `lerobot/svla_so100_pickplace`, whose
state and action dimensions match this artifact. Source cameras must be mapped
explicitly; unavailable artifact views are zero-filled and masked out. Images
are aspect-preserving letterboxed to the artifact input shape. State/action
schema mismatches still stop the build.

```console
uv run examples/build_smolvla_replay.py \
  --artifact benchmark_runs/smolvla/native-fp32 \
  --lerobot-checkout benchmark_runs/smolvla/lerobot-pinned \
  --checkpoint benchmark_runs/smolvla/checkpoint \
  --image-map observation.images.top=observation.images.camera1 \
  --image-map observation.images.wrist=observation.images.camera2 \
  --skip-upstream-actions \
  --output benchmark_runs/smolvla/replay-100-inputs.npz
uv run examples/record_smolvla_upstream.py \
  --artifact benchmark_runs/smolvla/native-fp32 \
  --input benchmark_runs/smolvla/replay-100-inputs.npz \
  --output benchmark_runs/smolvla/replay-100.npz \
  --lerobot-checkout benchmark_runs/smolvla/lerobot-pinned \
  --upstream-checkpoint benchmark_runs/smolvla/checkpoint \
  --resume
```

Then run each native CPU, QPU, and hybrid FP32/W8A8 mode:

```console
uv run qpu-model-quality smolvla \
  --fp32-artifact benchmark_runs/smolvla/native-fp32 \
  --w8a8-artifact benchmark_runs/smolvla/native-w8a8 \
  --replay benchmark_runs/smolvla/replay-100.npz \
  --mode hybrid_w8a8 \
  --resume \
  --output experiment_logs/model-quality/smolvla-hybrid-w8a8
```

Reports cover the complete 50x6 action chunk: max/mean/p99 absolute error,
per-action MAE/RMSE, normalized RMSE, cosine and sign agreement, worst
observation, and an episode-clustered confidence interval.
`--resume` checkpoints each completed candidate action chunk and cumulative
QPU/CPU dispatch telemetry.

## Batch orchestration and tests

`qpu-model-quality all --config runs.json` accepts a `runs` list whose entries
contain a `command` and its argument strings. Each workload remains a fresh
process and writes its own report directory.

CPU/fake-backend verification:

```console
PYTHONPATH=. uv run pytest -q -m 'not hardware' \
  tests/test_qpu_xla_quality.py \
  tests/test_qpu_xla_quality_optional.py \
  tests/test_qpu_xla_vision_runtime.py \
  tests/test_qpu_xla_vision_optional.py \
  tests/test_qpu_xla_smolvla.py \
  tests/test_qpu_xla_conv2d_w8a8.py
```

Hardware convolution and complete-model tests require the VideoCore VII render
node:

```console
PYTHONPATH=. uv run pytest -q -m hardware \
  tests/test_qpu_xla_vision_runtime.py \
  tests/test_qpu_xla_vision_optional.py \
  tests/test_qpu_xla_conv2d_w8a8.py \
  tests/test_qpu_xla_smolvla.py
```
