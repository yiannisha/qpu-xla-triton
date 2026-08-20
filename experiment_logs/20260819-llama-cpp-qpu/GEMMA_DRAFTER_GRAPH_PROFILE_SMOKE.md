# llama.cpp serialized graph profiles

Generated: 2026-08-20T01:39:16+00:00

These profiles use llama.cpp's public eval callback. It synchronizes after every node, so the rankings are diagnostic and cannot promote a QPU placement or establish a production Amdahl gate.

## Profile 1: batch 1, context 0

- Model: `/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/mtp-gemma-4-E2B-it.gguf`
- Threads: 3
- Serialized decode median: 123.500 ms
- Node records: 126

| Family | Median ms | Serialized share | Elimination bound | Diagnostic 5% gate |
|---|---:|---:|---:|---|
| normalization | 92.940 | 0.763 | 4.228x | True |
| rope | 23.933 | 0.197 | 1.245x | True |
| lm_head | 3.845 | 0.032 | 1.033x | False |
| ffn_gate_up_linear | 0.319 | 0.003 | 1.003x | False |
| ffn_down_linear | 0.162 | 0.001 | 1.001x | False |
| swiglu | 0.141 | 0.001 | 1.001x | False |
| attention_linear | 0.127 | 0.001 | 1.001x | False |
| other_linear | 0.109 | 0.001 | 1.001x | False |
| elementwise | 0.048 | 0.000 | 1.000x | False |
| attention_core | 0.044 | 0.000 | 1.000x | False |
| other | 0.036 | 0.000 | 1.000x | False |
| layout | 0.018 | 0.000 | 1.000x | False |
| embedding | 0.011 | 0.000 | 1.000x | False |
