# Rebuttal evidence and scope corrections

This document records repository work for the October 2026 rebuttal. It is not
evidence that a revised manuscript, anonymous archive, or ACM artifact package
has been uploaded. Exact HotCRP uploads and manuscript/BibTeX sources were not
available locally; those checks are omitted at the authors' request.

## Contribution and prior work

The FFN partition equation and the benefit of keeping a producer and consumer
on the same processor are established ideas. Describe the contribution as the
native VideoCore VII integration and its evaluation with the production Arm
CPU_REPACK implementation: logical width versus physical stride, existing
activation representations, a resident GEGLU-to-Q8 intermediate, and explicit
ownership and numerical checks.

- [Megatron-LM](https://arxiv.org/abs/1909.08053) provides intra-layer model
  parallelism, including splitting the first FFN projection across intermediate
  channels and summing contributions after the second projection.
- [PowerInfer](https://arxiv.org/abs/2312.12456) places hot neurons on GPU and
  computes cold neurons on CPU, using activation locality and sparse operators.
  Do not imply that CPU/GPU neuron execution is new here.
- [CoDL](https://chrisplus.me/assets/pdf/mobisys22-CoDL.pdf) combines CPU/GPU
  operator partitioning, compatible data sharing, and operator chains to reduce
  sharing overhead. Credit its operator-chain locality precedent.
- [Toward a Small ML Runtime Stack for Raspberry Pi 5 QPUs,
  arXiv:2606.09905](https://arxiv.org/abs/2606.09905) presents the earlier native
  QPU substrate, tiled matrix operations, GEMM-backed convolution, persistent
  executors, and integer kernels. Distinguish that substrate from the later
  model-runtime integration and production CPU_REPACK comparison. Cite this
  public work in the third person in anonymous material.

## Required manuscript qualifications

- The exploratory 2.30x stage result is SwiGLU on 256x5632 values, CPU 4.190 ms
  versus QPU 1.821 ms, maximum absolute error 2.86e-6. Bias-plus-ReLU's 1.920x
  is a separate operation and compares against NumPy's two passes. The new fused
  CPU control must be reported separately.
- The submitted complete-request gains apply to the tested M=129 configuration.
  M=257's interval spans parity. The FFN remains opt-in below the 1.05x automatic
  placement threshold. A new exploratory split screen is not independent
  confirmation of a newly chosen policy.
- SmolVLA replay has 1.56% NRMSE and cosine 0.999881, outside both predefined
  FP32 acceptance gates. Completion required resumes after V3D hangs. Replace
  broad claims of comparable action outputs with this explicit qualification.
  Software failure recovery does not establish recovery from a driver hang.
- Task-output coverage and static memory estimates do not establish complete
  model latency, energy savings, sustained reliability, or robot task success.
  State the boundary and limited input coverage of any new vision timing.
- Historical FP32/W8A8 comparisons use different numerical representations.
  Numerical CPU oracles are not necessarily competitive CPU performance controls.

## Historical projection denominator and vector selector

The FP32 suite README's matrix regeneration command uses `*-fp32-linear.json`.
Those three retained files contain 112 projection configurations: eight M=16
QPU cases, 32 M=64 QPU/hybrid cases, and 72 single-row decode QPU/hybrid cases.
Exactly ten have median speedup above one; all ten satisfy the recorded
`supported-win` rule (maximum error <=1e-3 and median speedup >=1.05). The other
102 are slower than the selected CPU reference. These are descriptive median
classifications, without confidence intervals. Projection roles are separate
observations even when dimensions repeat. Separate per-role reruns are excluded
from this documented campaign, and the 240 transformer-stage candidates are a
different denominator.

The min/max selector uses 64-value vector stripes and chooses the largest
divisor of the stripe count up to twelve workers. It selects that vector path
only if the worker count is at least six. At 4,194,048 elements, twelve workers
qualify. At 4,194,240 elements, the largest divisor is five, so the current code
uses scalar execution despite divisibility by 64. Six is an implementation
cutoff; retained evidence does not establish a hardware requirement or optimum.
No selector behavior was changed for the rebuttal.

Reproduce these audits with:

```bash
python scripts/audit_rebuttal_fp32.py --output /path/to/projection-selector-audit.json
```

## Completed measurements

TODOs 1 and 2 are recorded in
[the archived report](experiment_logs/20261028-rebuttal/TODO1-2-REPORT.md) with raw process, numerical,
environment, and kernel logs. Instrumented component diagnostics are separate
from uninstrumented request measurements. The split screen has six shape/split
cells, including all losses and inconclusive outcomes; none establishes a
validated complete-request improvement above the promotion threshold.

Further experiment commands are in [coa.md](coa.md), with completed results in
[the TODO 3–9 report](experiment_logs/20261028-rebuttal/REPORT.md). Never fill a
response's conditional passage from an unfinished or rejected run.

## Unavailable work

The exact submitted main/supplement PDFs, manuscript source, and anonymous
artifact checkout/package were not available. No claim is made that submission
page numbers were independently verified or that an anonymous package was
updated. The same-contract optimized CPU W8A8 performance control is omitted:
the available lowered INT32 references are numerical controls, and a new CPU
quantized runtime is outside this rebuttal experiment scope.
