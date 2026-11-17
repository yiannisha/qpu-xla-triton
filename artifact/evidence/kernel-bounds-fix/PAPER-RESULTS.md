# Manuscript result agreement

Printed-value checks of main numerical results and Figures 4/5; all 39 tables are indexed, not all supplemental values are independently recomputed. Historical data analysis, not new empirical or independent validation.

Checks: 123; mismatches: 0.

Static memory estimates are recomputed from topology. Quality intervals remain preserved views; COCO AP is recomputed only when --coco-annotations is supplied.

| Item | Page | Result | Evidence/workflow |
|---|---:|---|---|
| Table 1 | 8 | Selected FP32 projections | paper-results:projections |
| Table 2 | 8 | Prepared GEGLU producer and down projection | paper-results:producer |
| Table 3 | 9 | Column-W8 complete requests | paper-results:column-w8 |
| Table 4 | 9 | Corrected integrated FFN | paper-results:headline; paper:prompt,agentic |
| Table 5 | 9 | Static SmolVLA memory | paper-results:memory |
| Table 6 | 10 | Task quality | paper-results:quality; quality profile |
| Table 7 | 15 | Source entry points | manual source inventory |
| Table 8 | 15 | All ten selected FP32 projections | paper-results:projections |
| Table 9 | 15 | Stage sweep counts | paper-results:operator-grid |
| Table 10 | 16 | Selected FP32 stages | paper-results:operator-grid |
| Table 11 | 16 | Vectors and min/max | paper-results:operator-grid,figure4 |
| Table 12 | 16 | FP32 pooling | paper-results:operator-grid |
| Table 13 | 16 | FP32 epilogues | paper-results:operator-grid |
| Table 14 | 16 | Square matmul | paper-results:operator-grid |
| Table 15 | 16 | Historical W8A8 projection ranges | paper-results:operator-grid |
| Table 16 | 17 | W8A8 M16 H4096 projections | paper-results:operator-grid |
| Table 17 | 17 | One-layer runtime | paper-results:operator-grid |
| Table 18 | 18 | Historical W8A8 convolution | paper-results:operator-grid |
| Table 19 | 18 | Early native quantized nodes | raw JSON; examples/benchmark_llama_cpp_qpu_ops.py |
| Table 20 | 18 | Early native attention | raw JSON; examples/benchmark_llama_cpp_qpu_attention.py |
| Table 21 | 18 | Batched native attention | raw JSON; examples/benchmark_llama_cpp_qpu_attention.py |
| Table 22 | 18 | Isolated GEGLU | source report; integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md |
| Table 23 | 18 | Row-partitioned GEGLU | source report; integrations/llama_cpp/eval/IMPLEMENTATION_RESULTS.md |
| Table 24 | 18 | Native ffn_up request screens | raw JSON; scripts/run_llama_cpp_qpu_agentic_eval.py |
| Table 25 | 18 | Projection overlap | paper-results:figure5 |
| Table 26 | 19 | Prepared FFN larger M | raw JSON; integrations/llama_cpp/README.md |
| Table 27 | 19 | Prepared FFN split sweep | raw JSON; integrations/llama_cpp/README.md |
| Table 28 | 19 | Two projections per QPU submission | raw JSON; integrations/llama_cpp/README.md |
| Table 29 | 20 | Column-W8 complete requests | paper-results:column-w8 |
| Table 30 | 20 | Corrected complete FFN | paper-results:headline; paper:prompt,agentic |
| Table 31 | 20 | Full-vocabulary preflight | raw JSON; native qpu_ffn_island_model_smoke |
| Table 32 | 21 | OpenBLAS vs CPU_REPACK prompt | raw JSON; experiment_logs/20260909-openblas/RESULTS.md |
| Table 33 | 21 | OpenBLAS FFN splits | raw JSON; experiment_logs/20260909-openblas/RESULTS.md |
| Table 34 | 22 | OpenBLAS target MTP | raw JSON; experiment_logs/20260909-openblas/RESULTS.md |
| Table 35 | 22 | Quality sample selection | assets; quality profile |
| Table 36 | 22 | Detailed task quality | paper-results:quality; quality profile |
| Table 37 | 23 | SmolVLA operators | manual implementation inventory |
| Table 38 | 23 | SmolVLA action agreement | paper-results:quality; quality:smolvla (explicit diagnostic) |
| Table 39 | 23 | Static SmolVLA memory | paper-results:memory |
