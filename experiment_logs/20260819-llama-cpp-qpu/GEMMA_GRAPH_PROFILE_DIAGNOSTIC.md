# llama.cpp serialized graph profiles

Generated: 2026-08-20T01:32:36+00:00

These profiles use llama.cpp's public eval callback. It synchronizes after every node, so the rankings are diagnostic and cannot promote a QPU placement or establish a production Amdahl gate.

## Profile 1: batch 1, context 0

- Model: `/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`
- Threads: 2
- Serialized decode median: 135.461 ms
- Node records: 4488

| Family | Median ms | Serialized share | Elimination bound | Diagnostic 5% gate |
|---|---:|---:|---:|---|
| ffn_gate_up_linear | 50.472 | 0.431 | 1.758x | True |
| ffn_down_linear | 23.916 | 0.204 | 1.257x | True |
| lm_head | 23.340 | 0.200 | 1.249x | True |
| attention_linear | 6.918 | 0.059 | 1.063x | True |
| elementwise | 3.737 | 0.031 | 1.032x | False |
| swiglu | 2.827 | 0.024 | 1.025x | False |
| normalization | 2.521 | 0.021 | 1.022x | False |
| auxiliary_linear | 1.656 | 0.014 | 1.014x | False |
| per_layer_projection | 0.642 | 0.005 | 1.006x | False |
| attention_core | 0.503 | 0.004 | 1.004x | False |
| rope | 0.319 | 0.003 | 1.003x | False |
| layout | 0.260 | 0.002 | 1.002x | False |
| kv_or_state_update | 0.078 | 0.001 | 1.001x | False |
| embedding | 0.022 | 0.000 | 1.000x | False |

## Profile 2: batch 4, context 0

- Model: `/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`
- Threads: 3
- Serialized decode median: 171.123 ms
- Node records: 4488

| Family | Median ms | Serialized share | Elimination bound | Diagnostic 5% gate |
|---|---:|---:|---:|---|
| ffn_gate_up_linear | 57.571 | 0.384 | 1.624x | True |
| lm_head | 28.327 | 0.191 | 1.236x | True |
| ffn_down_linear | 28.248 | 0.187 | 1.230x | True |
| elementwise | 12.594 | 0.084 | 1.092x | True |
| attention_linear | 9.225 | 0.061 | 1.065x | True |
| swiglu | 4.563 | 0.030 | 1.031x | False |
| normalization | 3.800 | 0.025 | 1.026x | False |
| auxiliary_linear | 2.268 | 0.015 | 1.015x | False |
| attention_core | 1.154 | 0.008 | 1.008x | False |
| per_layer_projection | 0.696 | 0.005 | 1.005x | False |
| rope | 0.520 | 0.003 | 1.003x | False |
| layout | 0.279 | 0.002 | 1.002x | False |
| kv_or_state_update | 0.179 | 0.001 | 1.001x | False |
| embedding | 0.043 | 0.000 | 1.000x | False |
