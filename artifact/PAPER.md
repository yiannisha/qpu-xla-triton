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
Use `--plots` for PDF renderings of Figures 4/5. Static memory plans are
recomputed; optional pinned annotations enable COCO AP recomputation. Preserved
quality confidence intervals remain report views. Default/optional checks: 117/123.
The map covers all tables; this does not assert automated recomputation of every
supplemental cell. The JSON map gives exact evidence paths for each entry.

| Table | Page | Content | Workflow and limits |
|---|---:|---|---|
| 1 | 8 | Selected FP32 projections | paper-results:projections. Main complete-operator steady-state subset; all ten selected wins are in Table 8. |
| 2 | 8 | Prepared GEGLU producer and down projection | paper-results:producer. Prepared inputs; not complete FFN latency. |
| 3 | 9 | Column-W8 complete requests | paper-results:column-w8. M=129/257 held-out five pairs; M=513 one-pair screen, no interval. |
| 4 | 9 | Corrected integrated FFN | paper-results:headline; paper:prompt,agentic. Main evidence. Five independent processes per placement/shape; 15 request pairs. Original shortened-stride results invalid. |
| 5 | 9 | Static SmolVLA memory | paper-results:memory. Recomputed static planner estimate without weights; not observed RSS. |
| 6 | 10 | Task quality | paper-results:quality; quality profile. Raw classification metrics recomputed; optional pinned annotations recompute COCO AP. Confidence intervals remain preserved views. |
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
| 39 | 23 | Static SmolVLA memory | paper-results:memory. Same recomputed plan as Table 5; not measured peak RSS. |

| Figure | Page | Kind | Evidence |
|---|---:|---|---|
| 1 | 2 | design | Native runtime architecture; `src/qpu_xla`, `integrations/llama_cpp` |
| 2 | 4 | design | FP32 and W8A8 matrix dataflows; `src/qpu_xla/kernels`, `src/qpu_xla/ops` |
| 3 | 6 | design | Native language-model execution scopes; `integrations/llama_cpp/backend` |
| 4 | 8 | empirical | Exact stripe selector; `experiment_logs/20260819-qpu-xla-kernel-suite/fp32-minmax-n4194048.json`, `experiment_logs/20260819-qpu-xla-kernel-suite/fp32-minmax-n4194240.json` |
| 5 | 9 | empirical | M257 quarter-output projection overlap; `experiment_logs/20260824-qpu-agentic-prefill/agentic-up-m257-screen.json` |

## Corrections for the manuscript source

The supplied source and assets are now integrated in `main.tex`/`main.bib`.
`manuscript-ae.patch` shows the edits and `manuscript-revision.json` records the
local draft identity and actual two-page AE fit. The original PDF remains
unchanged; all page/table references above refer to it. Revised supplement
letters continue B–I after AE Appendix A.

The source now includes the completed AE, qualified SmolVLA/action and partition
claims, source/evidence distribution description, prior-substrate attribution,
Megatron-LM/PowerInfer/CoDL references and a concise rebuttal/corrections section.
That section preserves rejected conditions, slower controls, the missing CPU
W8A8 control and the November bounds corrections. Historical numerical tables
remain unchanged. [MANUSCRIPT-EDITS.md](MANUSCRIPT-EDITS.md) retains fuller wording
and the current author reproduction qualifications for review.

## Badge boundaries

The repo supplies installation, integrity checks, raw evidence, result regeneration,
fresh profiles, reuse instructions and a reviewer access plan. An independent
reviewer must still evaluate functionality/reusability and reproduce the main
conclusions. A successful author run cannot award either badge. New hardware
errors remain failures even if logits agree; do not rerun until a clean attempt
and discard earlier ones. See [VALIDATION.md](VALIDATION.md) and the bounded [MMU investigation](MMU-INVESTIGATION.md).

Public archival deposit and the newer-code license decision remain deferred
under the author’s instructions. No DOI, independent validation, badge award or publication is claimed.
A filled local PDF review draft has been built; no submission was performed.
