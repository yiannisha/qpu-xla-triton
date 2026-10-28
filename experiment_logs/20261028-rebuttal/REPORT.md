# Rebuttal experiments, 28 October 2026

TODOs 3, 4, 7, 8, and 9 are complete. TODO 5 has accepted pointwise and
dense-3x3 measurements; depthwise attempts are rejected. TODO 6(b), the fused
CPU control, is complete; the optimized same-contract CPU W8A8 control is
omitted. Repository attribution and scope notes for TODO 10 are prepared in
[REBUTTAL_NOTES.md](../../REBUTTAL_NOTES.md). Submitted PDFs, manuscript sources,
and anonymous-package uploads were unavailable and are not represented as done.

The earlier TODO 1/2 campaign is in [TODO1-2-REPORT.md](TODO1-2-REPORT.md).
Machine-readable new results are in [summary.json](summary.json). Compressed
raw records, including failed attempts, are in [raw/](raw/); [provenance.json](provenance.json)
maps their paths and uncompressed SHA-256 hashes. Decompress with `gzip -dc`.
Models, images, captured tensor payloads, and full-vocabulary dump arrays are
external assets, excluded from this package.

New TODO 3–9 hardware runs used four CPU threads unless stated otherwise,
performance governors, disabled swap, a 65 C start gate, and live kernel logs.
Each wrapper restored the original governors and zram state. Accepted runs
have no new voltage events or GPU error messages. This is a finite diagnostic
campaign, not sustained reliability or energy evidence. The rejected convolution
attempts contain three GPU resets despite the repaired power supply.

## TODO 3: matched FFN mechanisms

Gemma E2B Q4_K_XL, physical M=129, QPU channel fraction 1/8, WGS=24, production
CPU_REPACK. Five randomized blocks each launch one independent process per
variant, with a warmup and one timed 129-token prompt. All variants use the
same isolated CPU library and QPU plugin. CPU timestamps are disabled during
timing. Existing telemetry plus the same ablation record is enabled for all
variants, so these are diagnostics, separate from the submitted headline runs.

| Variant | Median prompt ms | Full / variant | Paired bootstrap 95% interval |
|---|---:|---:|---|
| Full design | 2255.720 | 1.0000 | reference |
| Compact CPU slice before identical quantizer | 2225.862 | 1.0134 | [0.9594, 1.0458] |
| Materialize and restore exact QPU intermediate | 2253.957 | 1.0008 | [0.9846, 1.0382] |
| Serialize otherwise identical branches | 2974.337 | 0.7584 | [0.7338, 0.7690] |

Intervals use 10,000 paired block resamples. The compact-copy and
materialization intervals span parity: this screen does not establish separate
positive gains from avoiding either operation. Serialization increases the
median by 31.86%, supporting the value of concurrency for this configuration.
Marginal effects must not be added as independent contributions.

Compaction copies every CPU-owned intermediate channel into reused thread-local
scratch, then invokes the same quantizer with the compact stride. The full
variant uses the correct original physical stride. Neither variant uses the
historical shortened-stride bug.

Materialization copies the existing Q8 bytes and block scales into persistent
host scratch and restores them before down projection, with CPU-access
synchronization around each transfer. It includes the extra host handoff and
does not re-quantize. The copied intermediate footprint is 6,842,880 bytes per
timed prompt across 35 layers (each byte is copied out and back). Median summed
materialization time is 7.765 ms, which does not imply the same request penalty
because it can overlap CPU execution.

All four variants pass six full-model numerical checks; every variant's six
complete 262,144-element candidate logit arrays have the same hashes as the
full variant. All channels and weights remain present. Timed telemetry verifies
1,400 FFN islands and 5,600 dispatches, without fallback or kernel errors.

## TODO 4: actual production MTP node

The retained real Q4_0 weight and F32 activation fixture is replayed through a
CPU_REPACK graph at M=1, K=256, N=2048, one CPU thread. The candidate uses the
actual inline M=1 hook after CPU activation preparation. Five process pairs,
five warmups and 31 samples per process; aggregation is the median of process
medians, with paired process bootstrapping.

| Boundary | Median microseconds |
|---|---:|
| CPU_REPACK graph | 25.037 |
| Inline-QPU graph | 353.534 |
| Hook input access | 0.481 |
| Hook activation packing | 0.093 |
| Hook input synchronization | 0.649 |
| Submission call | 4.741 |
| Wait until observed completion | 337.387 |
| Output synchronization | 1.926 |
| Output copy | 1.056 |
| Complete hook | 346.331 |

Input access/packing/synchronization together are reported separately from the
aggregate input-copy interval (1.222 us). Submission plus wait is also available
as an aggregate (342.146 us). Independently rounded medians do not sum exactly.
The graph boundary additionally includes CPU preparation, hook lookup and
telemetry printing. Do not label its difference from hook time as pure
quantization cost.

CPU/QPU graph ratio is 0.07082, 95% interval [0.07048, 0.07155]. All five paired
outputs pass the fixed numerical check; maximum absolute error is 3.5763e-7.
Telemetry verifies 180 dispatches including warmups, without fallback or kernel
errors. The observable wait dominates this path, but device format unpacking,
memory access, arithmetic and completion observation remain unresolved. No
empty-dispatch subtraction is used.

## TODO 5: current W8A8 convolution breakdown

Prepared, steady-state mapped NCHW input to restored NCHW output, including
dynamic lowering, quantization, packing, submission and synchronization. Fixed
weight preparation and allocation are outside timing. The CPU reference is
native Torch FP32 convolution, so this comparison changes representation.
Each accepted case has three warmups and eleven retained paired samples in one
process. All QPU invocations are checked outside the timed interval.

| Case | CPU ms | QPU wall ms | Layout/window ms | Total prepare ms | Prepare remainder ms | QPU event ms | Output restore ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pointwise | 4.482 | 21.209 | 0.007 | 14.757 | 14.750 | 2.535 | 3.283 |
| Dense 3x3 | 7.830 | 48.823 | 16.481 | 38.102 | 20.784 | 6.681 | 4.367 |

Shapes are (1,128,80,80) with (64,128,1,1) weights, and (1,64,80,80) with
(64,64,3,3) weights, respectively. Dense 3x3 uses stride 1, padding 1.
Pointwise packs pixels without duplicating windows; it is GEMM-backed, not a
direct spatial QPU kernel.

Preparation remainder includes quantization, packing and host bookkeeping,
not just one isolated operation. The QPU event includes host submission/wait,
not device-only arithmetic. Layout wrappers and queue events use host clocks.
Component medians are not additive; queue and caller overhead also enter wall
time. Preparation dominates the accepted cases. Dataflow remedies remain
prospective.

Preflight maximum error against the matching W8A8 contract is zero in both
cases; each repeated invocation passes atol=1e-4, rtol=1e-5. Maximum preflight
differences from FP32 are 0.5242 and 1.3214, respectively, documenting the
representation change. Ratios are 0.2113 and 0.1604; their within-process
bootstrap intervals are [0.2011,0.2773] and [0.1250,0.2240]. These intervals
describe sample variation, not independent process replication.

Depthwise (1,128,80,80), 128 channelwise 3x3 filters, padding 1, groups=128
was attempted in the all-case runs. The first run reported completed outputs
but its kernel log contains two GPU resets. The second contains another reset
and fails numerical validation (maximum violating error 19.6850). Both attempts
are preserved and rejected for latency. There is no accepted new depthwise
measurement. The current specialized nine-term depthwise reduction also differs
from the historical lowered implementation; do not transfer old timing claims.

## TODO 6(b): fused CPU bias-plus-ReLU

Shape 16x262128, persistent FP32 source/bias and preallocated output on both
processors. The existing CPU implementation is
`aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out)`.
It fuses addition and ReLU rather than traversing NumPy arrays twice.

Three warmups and 31 randomized CPU/QPU sample pairs: CPU **4.908 ms**, QPU
**5.732 ms**, ratio **0.8563** (within-process 95% interval [0.8287,0.8734]).
Both preflight outputs match the reference exactly. No kernel errors occurred.
This control does not support a QPU win against fused CPU execution. The old
1.920x result remains specifically a comparison against NumPy's two passes.

TODO 6(a) is omitted: available same-contract lowered INT32 implementations are
numerical references, not established optimized CPU W8A8 performance baselines.

## TODO 7: complete ResNet-18 FP32 timing

Pinned Torchvision IMAGENET1K_V1 artifact and one image from the existing
deterministic sample manifest. The source image is from the 128-pixel dataset
derivative; the existing V1 preprocessing produces **1x3x224x224** model input.
This distinction must be retained when describing the quality study's inputs.

Complete preprocessed-input-to-1000-logit execution includes all convolution
calls, surrounding CPU graph operations and output handling. It excludes image
decode/preprocessing and persistent plan construction for both paths.

Two warmups, seven paired steady-state samples: CPU **87.047 ms**, QPU
**344.726 ms**, ratio **0.2525** (within-process 95% interval [0.2364,0.2639]).
Preflight maximum absolute logit error is 1.1921e-5 and argmax is unchanged;
every timed QPU output passes the fixed tolerance. All 140 timed convolution
dispatches execute without fallback or kernel error. This is one-image timing,
not a new task-accuracy evaluation. It establishes a slowdown, not a vision
latency gain. No VLA timing or robot-success claim is added.

## TODOs 8 and 9: lightweight audits

The documented historical projection campaign contains **112 configurations**:
eight M=16 QPU cases, 32 M=64 QPU/hybrid cases and 72 decode QPU/hybrid cases.
It has **10 supported wins and 102 slower cases**. All 112 satisfy the recorded
maximum-error gate of 1e-3. The promotion rule also requires median complete
speedup >=1.05 over the faster measured NumPy/OpenBLAS or Torch reference.
There is no CI-based historical inconclusive category. Separate projection
roles count separately; per-role reruns and the 240 transformer-stage sweep
are excluded from this denominator. Full grid and file hashes are archived.

Six QPUs is an implementation cutoff, without evidence of a hardware requirement
or optimum. Both min/max lengths are divisible by 64. The first has a twelve-way
stripe decomposition; the second's largest divisor up to twelve is five and
fails the current >=6 selector. No selector was changed and no five-QPU
performance claim is made.

## Rebuttal-ready passages

**R4Q3 / TODO 3:** Matched all-channel variants measure full/compact/materialized/
serialized prompts at 2255.720/2225.862/2253.957/2974.337 ms. All four produce
bitwise-identical candidate logits on six full-vocabulary checks. Compaction
and materialization intervals span parity; serialization is slower. We credit
the combined design and the measured concurrency effect, without assigning
independent positive gains to reuse or residency.

**R4Q2 / TODO 4:** On the real production MTP fixture, CPU_REPACK/inline-QPU
graph times are 25.037/353.534 us. Input preparation/submission/wait/output
components are recorded; observed wait is 337.387 us. Device format handling
and arithmetic remain unisolated. Maximum output error is 3.58e-7.

**R1Q1 / TODO 5:** New pointwise/dense-3x3 preparation/QPU-event/output times are
14.757/2.535/3.283 and 38.102/6.681/4.367 ms; preparation dominates these
current W8A8 paths. Depthwise attempts reset the GPU and one fails validation,
so no new accepted depthwise latency is reported. Remedies remain prospective.

**R4C5(b) / TODO 6:** A matched fused Torch CPU bias-plus-ReLU takes 4.908 ms
versus 5.732 ms on QPU, with exact preflight agreement. The earlier 1.920x
comparison used NumPy's two passes. A suitable optimized same-contract CPU
W8A8 performance control remains unavailable.

**R2Q3 / TODO 7:** Added ResNet-18 FP32 timing is CPU/QPU 87.047/344.726 ms on
one repeated matched input, including the full model from preprocessed input
to logits. Numerical checks pass and all timed dispatches execute. This
measures a slowdown; it does not establish a vision or VLA latency gain.

**R4C5(c) / TODO 8:** The documented projection campaign has 112 configurations,
10 median-rule wins and 102 slower cases; the full grid and selection rule are
recorded. The ten paper observations are this campaign's selected wins.

**R3C2 / TODO 9:** Six is an implementation cutoff, not an established hardware
requirement or optimum; the neighboring vector length permits five workers.

**TODO 10 / opening:** Repository attribution, scope corrections and reproduction
records are prepared. Exact submitted PDFs, manuscript/BibTeX edits and anonymous
package publication are unverified; do not claim those actions are complete.
