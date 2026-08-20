# llama.cpp serialized graph profiles

Generated: 2026-08-20T01:29:48+00:00

These profiles use llama.cpp's public eval callback. It synchronizes after every node, so the rankings are diagnostic and cannot promote a QPU placement or establish a production Amdahl gate.

## Profile 1: batch 1, context 0

- Model: `/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`
- Threads: 2
- Serialized decode median: 339.397 ms
- Node records: 1496

| Family | Median ms | Serialized share | Elimination bound | Diagnostic 5% gate |
|---|---:|---:|---:|---|
| normalization | 166.715 | 0.523 | 2.098x | True |
| ffn_gate_up_linear | 51.485 | 0.162 | 1.193x | True |
| lm_head | 24.791 | 0.078 | 1.084x | True |
| ffn_down_linear | 24.354 | 0.076 | 1.083x | True |
| embedding | 22.325 | 0.070 | 1.075x | True |
| rope | 8.772 | 0.028 | 1.028x | False |
| attention_linear | 7.320 | 0.023 | 1.024x | False |
| swiglu | 5.496 | 0.017 | 1.018x | False |
| elementwise | 3.931 | 0.012 | 1.012x | False |
| auxiliary_linear | 1.850 | 0.006 | 1.006x | False |
| per_layer_projection | 0.617 | 0.002 | 1.002x | False |
| attention_core | 0.589 | 0.002 | 1.002x | False |
| layout | 0.280 | 0.001 | 1.001x | False |
| kv_or_state_update | 0.080 | 0.000 | 1.000x | False |
