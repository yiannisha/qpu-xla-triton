# Scope and limitations

This package preserves evidence, including negative results. `valid_run` in
quality reports attests completion, finite outputs and dispatch integrity; it
does not mean a predefined task-quality gate passed.

- The runtime is a prototype, not a general XLA compiler. Hybrid partitioning
  is explicit, with exact-shape evidence required for automatic placement.
- General dense 3x3 convolution is vectorized im2col plus GEMM. Pointwise 1x1
  avoids window duplication but remains GEMM-backed. Neither is a direct
  spatial QPU microkernel.
- Public INT16 convolution widens to INT32 and accumulates exactly. The packed
  INT16 assembly experiment is quarantined. INT32 `smul24` inputs retain the
  signed 24-bit range contract. No artifact work changes these paths.
- Historical NumPy convolution is im2col plus dot; Torch native conv2d is the
  optimized CPU convolution control. Keep prep, cached transfer/execute/readback,
  execute-only, and prep-plus-cached totals separate. Label cold start versus
  steady state. QPU queue events include host submission/wait and are not
  isolated device arithmetic.
- Corrected M=129 headline FFN gains are small and below automatic promotion;
  M=257 is inconclusive. The October diagnostics use instrumentation and do
  not replace September uninstrumented measurements. The new six-cell split
  screen produces no policy eligible for promotion. One eighth is not claimed
  optimal.
- October TODO1/2 logs retain two V3D MMU events. The accepted TODO3–9 runs
  have their own environment/retention records; do not transfer that status
  to every campaign. The two depthwise attempts reset the GPU, with a numerical
  failure in the second attempt, and their timings are rejected. They remain
  distributed in the original compressed records.
- The November clean-build follow-through also observed a new MMU event within
  the real-model numerical preflight. All six cases and the prompt dispatch
  contracts passed, but the outer kernel monitor rejected the entire prompt
  campaign's latency. Its cause remains unisolated. A separately executed
  two-pair cached-request stage passed numerical/execution/environment gates
  without new kernel errors; neither outcome establishes sustained stability
  or independently confirms the historical small performance gain.
- MTP inline QPU, matched fused bias+ReLU, and complete one-input ResNet QPU
  execution are slower than their measured CPU controls. A same-contract
  optimized CPU W8A8 performance control remains unavailable. Numerical INT32
  oracles are not replacements for that performance baseline.
- ResNet W8A8 loses 0.92 percentage points top-1 on the fixed sample, with an
  interval excluding zero. ResNet FP32 accuracy uses the public 128-pixel
  ImageNet derivative; preprocessing makes 224x224 model inputs. It is not a
  measurement on the official full-resolution ImageNet distribution.
- SmolVLA FP32 replay gives NRMSE 1.56% and cosine 0.999881, missing both
  predefined FP32 gates. It completed through checkpointed resumes after
  driver hangs/resets. Finite retained outputs and software fallback tests
  do not demonstrate driver-hang recovery, sustained reliability, robot task
  success, energy savings, or complete VLA latency. Fresh SmolVLA is disabled
  in the default quality profile and available only by explicit stage choice.
- COCO uses the retained seed-20260911 random 1000-image subset. Its original
  10000-replicate bootstrap was omitted (estimated 83 hours). The matching
  fresh-run profile keeps that omission visible; it is not a complete bootstrap
  qualification.
- Full-vocabulary checks compare probabilities/TV/KL with fixed gates; the
  joined quantized FFN is not raw-logit identical to CPU_REPACK. Always retain
  raw-logit max errors alongside probability metrics.
- External models, datasets and upstream runtimes retain their own terms.
  The software container does not replace Raspberry Pi hardware or its host
  kernel driver. Not every external asset is needed for smoke or offline analysis.
- The accepted manuscript, exact figure/table numbering, final paper metadata,
  independent evaluator reports, archival DOI and publisher badge award are
  not present. Their completion must be recorded separately.

See `REBUTTAL_NOTES.md`, `BUG_FIX_RESULTS.md`, `CORRECTNESS_REPORT.md`, and the
October reports for full experimental boundaries and exclusions. Attribution
to the earlier QPU substrate and prior CPU/GPU partitioning work remains part
of the manuscript alignment task.
