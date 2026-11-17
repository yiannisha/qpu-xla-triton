# Proposed manuscript qualifications

These are proposed insertions for the author-supplied PDF identified in
`manuscript.json`, not a claim that the PDF has been edited. Each addition
states its measurement boundary. Retained raw records and fresh-run profiles
are mapped in `claims.json`; full reports remain authoritative for details.

**Abstract/introduction: action-output scope.**
“On 100 matched SmolVLA observations, the completed QPU FP32 replay has 1.56%
normalized RMSE and cosine similarity 0.999881 against upstream actions. These
values miss our predefined FP32 gates, and completion required checkpointed
resumes after V3D hangs. We report numerical proximity with these limitations;
we do not establish sustained reliability or closed-loop robot task success.”

**§3.4.3: partition selection.**
“The corrected complete-FFN confirmation fixed the QPU intermediate-channel
share at one eighth before evaluation. A later exploratory screen covers
M=129/257 and fractions 1/16, 1/8 and 3/16. All six numerical/execution contracts
pass, but the interval rule yields four inconclusive comparisons and two
losses; no median reaches 1.05×. This does not establish an optimal share or
promote a newly selected policy.”

**§5.2: complete FFN versus Figure 5’s projection boundary.**
“Figure 5 describes an earlier projection-only overlap experiment. Separate
October complete-FFN diagnostics find CPU completion last in 172/175 measured
FFNs at M=129 and all 175 at M=257. CPU-only FFNs occupy median prompt shares
of 67.69% and 66.51%. These instrumented diagnostics identify the observed
critical path and do not replace the uninstrumented headline campaign. Two
MMU messages in the associated diagnostic/screening logs remain a stability
qualification; numerical and dispatch passes do not establish reliability.”

**§5.2: mechanism controls.**
“At M=129, five matched process blocks compare the full design with compact
CPU slicing, intermediate materialization, and serialized branches. Ratios
of full-design to variant latency are 1.0134 [0.9594, 1.0458], 1.0008
[0.9846, 1.0382], and 0.7584 [0.7338, 0.7690], respectively. Only serialization
shows a clear penalty in this screen. Avoided copy/materialization effects are
not separately established positive gains, and these marginal effects must
not be added.”

**MTP baseline qualification alongside Table 34.**
“Table 34’s OpenBLAS-target request screen differs from the later production
CPU_REPACK node replay. The latter measures M=1, K=256, N=2048 at 25.037 µs
for CPU_REPACK versus 353.534 µs for inline QPU, a ratio of 0.07082
[0.07048, 0.07155]. Maximum error is 3.5763e-7 against the matching contract.
Observed host wait dominates the hook boundary; it does not isolate device
arithmetic or format-unpacking time.”

**§5.1: convolution and the epilogue baseline.**
“Prepared pointwise and dense-3×3 W8A8 convolution spend 14.757 of 21.209 ms
and 38.102 of 48.823 ms, respectively, in host preparation. The Torch FP32
comparison changes representation. Pointwise avoids window duplication but
is GEMM-backed; dense 3×3 uses im2col. New depthwise attempts are rejected for
kernel resets/numerical failure. Table 13’s bias+ReLU comparison uses NumPy’s
two passes; the matched fused Torch control is faster than QPU. An optimized
same-contract CPU W8A8 performance control remains unavailable.”

**§5.3: complete vision latency scope.**
“The October ResNet experiment measures one preprocessed 1×3×224×224 input
through complete logits and finds the QPU path slower with passing FP32
logit checks. This adds a bounded model-latency result; it does not add task
accuracy, energy, or production VLA latency evidence.”

**Appendix B/Table 22: rejected machine conditions.**
“The isolated GEGLU numbers retain their calibration/held-out identities.
Their source campaign was not retained for a publication performance claim
because governors were ondemand and a competing server was active. They are
diagnostic operator observations, not retained complete-request acceleration.”

**Artifact description and Related Work.**
Replace the blank p.12 template with `APPENDIX.tex` and distinguish the CGO
source-and-evidence release from the earlier anonymous source-only snapshot.
Include the upstream assembler/driver and llama.cpp attribution. Integrate the
Megatron-LM, PowerInfer, CoDL and prior QPU-substrate scope corrections already
specified with primary links in `REBUTTAL_NOTES.md`; choose citation keys in
the actual bibliography rather than inventing entries in the PDF.

**New author reproduction.**
Report the full November run separately from September confirmation. Its
M=129 prompt ratio is 1.0167 [0.9843, 1.0567], so the fresh attempt does not
confirm the historical statistically positive gain. M=257 is 1.0032
[0.9667, 1.0255], still inconclusive. Preserve the earlier MMU-rejected reduced
attempt. The 15-pair cached-request ratio is 0.9919 [0.9726, 1.0170], which does not
confirm the historical gain. Its interval does not overlap the historical
[1.0241, 1.0520] interval. Full controls are recorded in `VALIDATION.md`;
neither run is independent validation or proof of a resolved driver fault.
