# Model Correctness Report

Status as of 2026-09-11. This report covers task-quality and numerical
correctness; it does not make latency or throughput claims. Commands and metric
definitions are documented in [MODEL-QUALITY.md](MODEL-QUALITY.md).

## Results

| Workload | Baseline | Candidate | Paired result | Status |
| --- | ---: | ---: | --- | --- |
| Gemma MMLU, 1,000 questions | 36.2% | QPU/CPU hybrid 36.3% | +0.1 pp; 95% CI [-0.2, +0.4] pp | No measurable loss |
| ResNet-18 ImageNet-5k, QPU FP32 | top-1 53.68%; top-5 78.72% | top-1 53.68%; top-5 78.72% | zero top-1/top-5 flips | Pass |
| ResNet-18 ImageNet-5k, hybrid FP32 | top-1 53.68%; top-5 78.72% | top-1 53.68%; top-5 78.72% | zero top-1/top-5 flips | Pass |
| ResNet-18 ImageNet-5k, QPU W8A8 | top-1 53.68%; top-5 78.72% | top-1 52.76%; top-5 77.86% | top-1 -0.92 pp, 95% CI [-1.44, -0.38] pp | Regression |
| SmolVLA replay-100, QPU FP32 | upstream Torch FP32 | native QPU FP32 | normalized RMSE 1.56%; cosine 0.999881 | Numerically close; stability not qualified |

The MMLU candidate agreed with the CPU answer on 99.2% of questions. It changed
one CPU-correct answer to wrong and two CPU-wrong answers to correct, across
16,625 QPU/CPU hybrid dispatches with no fallback.

Both ResNet-18 FP32 placements processed 5,000 images and 100,000 QPU
convolutions without fallback. Forced QPU had a maximum absolute logit error of
`3.15e-5`; hybrid had `3.43e-5`. The W8A8 run also completed 100,000 QPU
convolutions without fallback, but its accuracy confidence interval excludes
zero. The current quantized model therefore cannot support a no-quality-loss
claim against FP32.

The SmolVLA replay contains ten deterministic observations from each of ten
episodes and compares complete 50x6 action chunks. Across 30,000 action values,
forced-QPU FP32 produced mean absolute error 0.00441, p99 absolute error 0.174,
maximum absolute error 0.716, and 99.91% sign agreement. All committed outputs
were finite and covered 1,293,300 QPU dispatches. Several long-run V3D
hangs/resets produced rejected non-finite attempts before checkpointed resumes,
so forced-QPU operational stability is not qualified even though the completed
replay is numerically close.

YOLOv8n COCO val2017 is still in progress. The corrected square-input run has a
durable checkpoint at 936 of 5,000 images, so no COCO AP/AR conclusion is
reported yet.

## Method notes

- MMLU uses a deterministic 1,000-question selection and paired answer/log-probability output.
- ImageNet uses 5,000 validation images, five per class, from the public 128x128 derivative because official ImageNet access was unavailable. CPU and QPU see identical decoded samples.
- COCO uses all 5,000 val2017 images, standard bbox AP/AR, per-category AP, prediction agreement, and a paired image bootstrap.
- SmolVLA uses fixed observations, language tokens, state, image masks, and flow noise so action comparisons are paired.
- Downloaded datasets, model weights, raw logits, predictions, and action arrays are intentionally not committed.

## Current conclusion

FP32 QPU execution has no observed task-quality loss for the completed MMLU and
ResNet-18 qualifications. The present ResNet-18 W8A8 configuration has a
measurable accuracy regression, and forced-QPU SmolVLA has a long-run stability
issue. Those two paths must not be described as no-regression candidates until
they are corrected and requalified. COCO remains pending.
