# Official manuscript and evidence map

Paper: **Practical: Building and Evaluating a Native CPU–QPU Runtime for Edge AI Inference on Raspberry Pi 5**.

Authors: Yiannis Hadjiyianni, Kun Ouyang, Linyuan Han, Panagiotis Michelakis, Yijing Wang, Hao Wu, Rongqiang Zhao, Dimitrios Stamoulis, Jie Liu.

Author-supplied `QPUs_CGO2027.pdf`: 26 pages, SHA256
`3bcc96f58f3b8b500fc8e89ff7c2b8e25bb5226dbf2a946159c7b75f587b6ad2`.
The input PDF is preserved locally and excluded from the source release inventory.
Page numbers below refer to this exact version. Later manuscript revisions need a new mapping.

## Reproduction entry points

Follow [INSTALL.md](../INSTALL.md) and [HOWTO.md](../HOWTO.md).
`scripts/reproduce_paper.py --output build/paper-analysis` regenerates the main
measurement data, checks printed values and exports the full FP32 operator grid.
Use `--plots` for PDF renderings of Figures 4/5. Static estimates, preserved
quality intervals and COCO AP remain explicitly labeled report views.
The map covers all tables; this does not assert automated recomputation of every
supplemental cell. The JSON map gives exact evidence paths for each entry.

| Table | Page | Content | Workflow and limits |
|---|---:|---|---|
| 1 | 8 | Selected FP32 projections | paper-results:projections. Main complete-operator steady-state subset; all ten selected wins are in Table 8. |
| 2 | 8 | Prepared GEGLU producer and down projection | paper-results:producer. Prepared inputs; not complete FFN latency. |
| 3 | 9 | Column-W8 complete requests | paper-results:column-w8. M=129/257 held-out five pairs; M=513 one-pair screen, no interval. |
| 4 | 9 | Corrected integrated FFN | paper-results:headline; paper:prompt,agentic. Main evidence. Five independent processes per placement/shape; 15 request pairs. Original shortened-stride results invalid. |
| 5 | 9 | Static SmolVLA memory | retained static estimate; examples/benchmark_qpu_xla_smolvla.py. Planner estimate, not observed RSS. Requires separately acquired model for new plan. |
| 6 | 10 | Task quality | paper-results:quality; quality profile. Raw classification metrics recomputed; preserved confidence intervals and COCO AP labeled report views. |
| 7 | 15 | Source entry points | manual source inventory. Documentation table, no measurement. |
| 8 | 15 | All ten selected FP32 projections | paper-results:projections. Selection rule and all 112 cases retained; CPU reference varies by case. |
| 9 | 15 | Stage sweep counts | paper-results:operator-grid. 240 stage configurations; not independent process samples. |
| 10 | 16 | Selected FP32 stages | paper-results:operator-grid. Selected stage wins and slower RoPE case; preceding table indexes source records. |
| 11 | 16 | Vectors and min/max | paper-results:operator-grid,figure4. Exact stripe cutoff, not hardware worker minimum. |
| 12 | 16 | FP32 pooling | paper-results:operator-grid. Prepared whole-operation boundaries. |
| 13 | 16 | FP32 epilogues | paper-results:operator-grid. NumPy two-pass bias+ReLU reference; rebuttal fused Torch control is separate. |
| 14 | 16 | Square matmul | paper-results:operator-grid. Also contains staged SDPA; whole-operation, not raw kernel speed. |
| 15 | 16 | Historical W8A8 projection ranges | paper-results:operator-grid. FP32 CPU reference changes representation; not optimized same-contract W8A8 comparison. |
| 16 | 17 | W8A8 M16 H4096 projections | paper-results:operator-grid. Historical isolated operator evidence. |
| 17 | 17 | One-layer runtime | paper-results:operator-grid. Layer boundary; holdout/decode losses demote isolated wins. |
| 18 | 18 | Historical W8A8 convolution | paper-results:operator-grid. Torch FP32 baseline; distinct from runtime-summary and October prepared timing campaigns. |
| 19 | 18 | Early native quantized nodes | raw JSON; examples/benchmark_llama_cpp_qpu_ops.py. Ondemand/zram diagnostic; hybrid CPU generic block dot, not CPU_REPACK. |
| 20 | 18 | Early native attention | raw JSON; examples/benchmark_llama_cpp_qpu_attention.py. M=1 early screen, not complete-model attention claim. |
| 21 | 18 | Batched native attention | raw JSON; examples/benchmark_llama_cpp_qpu_attention.py. Three CPU threads, eleven post-warmup samples. |
| 22 | 18 | Isolated GEGLU | source report; integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md. One CPU thread; calibration and held-out statuses differ. Original report says ondemand governor and competing server rejected the campaign for a publication performance claim; paper should retain that qualification. |
| 23 | 18 | Row-partitioned GEGLU | source report; integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md. Five processes per configuration; no confirmed end-to-end gain. |
| 24 | 18 | Native ffn_up request screens | raw JSON; scripts/run_llama_cpp_qpu_agentic_eval.py. Two-pair held-out screens, no policy promotion. |
| 25 | 18 | Projection overlap | paper-results:figure5. Median per-layer observable timings; overlapping durations are not additive. |
| 26 | 19 | Prepared FFN larger M | raw JSON; integrations/llama_cpp/README.md. Input quantization outside boundary; numerical failures retained. |
| 27 | 19 | Prepared FFN split sweep | raw JSON; integrations/llama_cpp/README.md. Do not accept faster rows that fail numerical gates. |
| 28 | 19 | Two projections per QPU submission | raw JSON; integrations/llama_cpp/README.md. Prepared operator change, not complete-request result. |
| 29 | 20 | Column-W8 complete requests | paper-results:column-w8. Expanded Table 3, includes single-pair calibration. |
| 30 | 20 | Corrected complete FFN | paper-results:headline; paper:prompt,agentic. Expanded Table 4 including cached prompt boundary. |
| 31 | 20 | Full-vocabulary preflight | raw JSON; native qpu_ffn_island_model_smoke. Probability, TV, KL and argmax gates; raw logits have no absolute-error gate. |
| 32 | 21 | OpenBLAS vs CPU_REPACK prompt | raw JSON; experiment_logs/20260909-openblas/RESULTS.md. Three post-warmup samples per prompt, CPU_REPACK is production baseline. |
| 33 | 21 | OpenBLAS FFN splits | raw JSON; experiment_logs/20260909-openblas/RESULTS.md. All 24 partitions fail the F32-activation numerical contract; no valid acceleration. |
| 34 | 22 | OpenBLAS target MTP | raw JSON; experiment_logs/20260909-openblas/RESULTS.md. One request per configuration; exact greedy output, QPU slower. |
| 35 | 22 | Quality sample selection | assets; quality profile. Fixed identities and pairing required. |
| 36 | 22 | Detailed task quality | paper-results:quality; quality profile. Expanded Table 6; no COCO bootstrap in reported campaign. |
| 37 | 23 | SmolVLA operators | manual implementation inventory. Supported partition axes, not measured end-to-end speedups. |
| 38 | 23 | SmolVLA action agreement | paper-results:quality; quality:smolvla (explicit diagnostic). Completed 100-observation outputs only; FP32 gates fail; hangs/resumes retained. |
| 39 | 23 | Static SmolVLA memory | retained static estimate. Same as Table 5; not measured peak RSS. |

| Figure | Page | Kind | Evidence |
|---|---:|---|---|
| 1 | 2 | design | Native runtime architecture; `src/qpu_xla`, `integrations/llama_cpp` |
| 2 | 4 | design | FP32 and W8A8 matrix dataflows; `src/qpu_xla/kernels`, `src/qpu_xla/ops` |
| 3 | 6 | design | Native language-model execution scopes; `integrations/llama_cpp/backend` |
| 4 | 8 | empirical | Exact stripe selector; `experiment_logs/20260819-qpu-xla-kernel-suite/fp32-minmax-n4194048.json`, `experiment_logs/20260819-qpu-xla-kernel-suite/fp32-minmax-n4194240.json` |
| 5 | 9 | empirical | M257 quarter-output projection overlap; `experiment_logs/20260824-qpu-agentic-prefill/agentic-up-m257-screen.json` |

## Corrections for the manuscript source

Concrete proposed wording is in [MANUSCRIPT-EDITS.md](MANUSCRIPT-EDITS.md).
The supplied PDF is the official input, not an edited output. These items need
author integration into its LaTeX source; the artifact does not silently rewrite it.

1. Replace the blank obligatory artifact template on p.12 with [APPENDIX.tex](APPENDIX.tex). Keep it before References, verify the two-page limit, and resolve duplicate Appendix A numbering in the supplement. Source and a TeX toolchain were not supplied, so integration and page fit are unverified.
2. Table 7 and Appendix H promise `INSTALL.md` and `HOWTO.md`; these now exist and lead to the exercised installation and result workflows. Appendix H.2’s render-node wording is incomplete for the current driver: the V3D card path also needs access.
3. The abstract/introduction’s broad action-agreement wording should carry §5.3 and Appendix F.2’s qualification: SmolVLA NRMSE=1.56% and cosine=0.999881 fail the stated FP32 gates; the completed replay followed hangs/resets and does not establish reliability or robot success.
4. §3.4.3’s calibration wording should distinguish the fixed 1/8 share in corrected confirmation from exploratory selection. October TODO2 does not prove optimality or promote a split. Figure 5/Table 25 show earlier projection-only overlap, not the October complete-FFN critical path.
5. Supplement A describes an anonymized source-only snapshot without logs/arrays. The CGO evaluation release described here includes retained raw evidence and component provenance. Identify that distribution explicitly rather than claiming the two packages are identical.
6. Table 22’s isolated GEGLU numbers have calibration/held-out labels, but their source report also says the campaign was rejected for a publication performance claim because governors were ondemand and a competing server was active. Retain that qualification when interpreting the operator ratios.
7. Integrate the attribution corrections in `REBUTTAL_NOTES.md` (TODO10) into Related Work and references. The supplied PDF does not establish that those corrections have been made. Preserve upstream assembler/driver and native llama.cpp attribution.
8. The October critical-path, split, ablation, CPU_REPACK MTP, preparation, fused Torch and complete-ResNet measurements are supplemental rebuttal evidence, not numerical tables already present in this PDF. Their exclusions—including MMU events, rejected depthwise, and missing optimized same-contract CPU W8A8 control—must travel with the results.

## Badge boundaries

The repo supplies installation, integrity checks, raw evidence, result regeneration,
fresh profiles, reuse instructions and a reviewer access plan. An independent
reviewer must still evaluate functionality/reusability and reproduce the main
conclusions. A successful author run cannot award either badge. New hardware
errors remain failures even if logits agree; do not rerun until a clean attempt
and discard earlier ones. See [VALIDATION.md](VALIDATION.md) and the bounded [MMU investigation](MMU-INVESTIGATION.md).

Public archival deposit and the newer-code license decision remain deferred
under the author’s instructions. No DOI, independent validation, filled PDF
appendix or publication is claimed.
