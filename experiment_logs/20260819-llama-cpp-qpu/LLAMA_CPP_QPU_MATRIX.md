# llama.cpp QPU latency matrix

Generated: 2026-08-20T01:19:58+00:00

Only rows whose session is retained, correctness passes, complete-node speedup is at least 1.05x, and the 95% CI lower bound exceeds 1.0 may be promoted. Diagnostic rows are never automatic placements.

## Native Q4_0 operator matrix

| Tensor | M×K×N | Placement | CPU threads | QPU share | Complete ms | Quant ms | Input copy ms | Submit+wait ms | Output copy ms | Speedup (95% low) | Max abs err | Resident MiB | Retained | Session | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `blk.0.ffn_gate.weight` | 1×256×2048 | cpu | 3 | 0.000 | 0.029 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | smoke-m1-host-io | baseline |
| `blk.0.ffn_gate.weight` | 1×256×2048 | qpu | 3 | 1.000 | 0.350 | 0.000 | 0.000 | 0.344 | 0.005 | 0.083 (0.082) | 4.77e-07 | 0.314 | False | smoke-m1-host-io | experimental |
| `blk.0.ffn_gate.weight` | 1×256×2048 | hybrid | 3 | 0.500 | 0.198 | 0.000 | 0.000 | 0.193 | 0.003 | 0.146 (0.143) | 4.17e-07 | 0.157 | False | smoke-m1-host-io | experimental |
| `blk.0.ffn_gate.weight` | 4×256×2048 | cpu | 3 | 0.000 | 0.094 | 0.001 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | smoke-m4-host-io | baseline |
| `blk.0.ffn_gate.weight` | 4×256×2048 | qpu | 3 | 1.000 | 0.365 | 0.001 | 0.000 | 0.344 | 0.020 | 0.257 (0.253) | 4.77e-07 | 0.314 | False | smoke-m4-host-io | experimental |
| `blk.0.ffn_gate.weight` | 4×256×2048 | hybrid | 3 | 0.500 | 0.251 | 0.001 | 0.000 | 0.238 | 0.010 | 0.373 (0.366) | 4.77e-07 | 0.157 | False | smoke-m4-host-io | experimental |
| `token_embd.weight` | 4×256×262144 | cpu | 3 | 0.000 | 13.037 | 0.003 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | drafter-lm-head-m4-smoke | baseline |
| `token_embd.weight` | 4×256×262144 | qpu | 3 | 1.000 | 39.722 | 0.002 | 0.000 | 37.121 | 2.591 | 0.328 (0.295) | 4.77e-07 | 40.001 | False | drafter-lm-head-m4-smoke | experimental |
| `token_embd.weight` | 4×256×262144 | hybrid | 3 | 0.500 | 30.288 | 0.003 | 0.002 | 28.947 | 1.305 | 0.430 (0.300) | 4.77e-07 | 20.001 | False | drafter-lm-head-m4-smoke | experimental |

## End-to-end matrix

| Case | Mode | Placement | Threads | Depth | Context target | Prompt target | Startup s | Prefill s | Request s | Prompt ms | Prompt tok/s | Decode tok/s | Mean accepted | Cycle s | Greedy identical | Session | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| gemma-plain-t3-40-token-throughput-smoke | plain | cpu-only | 3 | — | 0 | — | 3.523 | — | 4.684 | 296.386 | 33.740 | 9.119 | — | — | True | server-throughput-smoke-v2 | diagnostic |
| gemma-mtp-n2-t3-40-token-throughput-smoke | mtp | cpu-only | 3 | 2 | 0 | — | 5.791 | — | 5.497 | 299.372 | 33.403 | 7.699 | 1.73 | 0.225 | True | server-throughput-smoke-v2 | diagnostic |
| gemma-plain-context-32-smoke | plain | cpu-only | 2 | — | 32 | — | 3.532 | 0.874 | 0.363 | 362.116 | 27.615 | — | — | — | True | context-32-smoke-v2 | diagnostic |
| gemma-prompt-p10-t2 | prompt | cpu-only | 2 | — | 0 | 10 | 3.530 | — | 0.372 | 371.063 | 26.950 | — | — | — | — | exact-prompt-10-smoke | diagnostic |

## Coverage and retention

- Operator rows: 9.
- End-to-end rows: 4.
- CPU baseline operator rows: 3.
- Promoted operator rows: 0.
- Retained end-to-end rows: 0.
- Planned end-to-end cases: 39.
- Missing model/workload combinations remain coverage gaps; absence is not a CPU or QPU win.
- Coverage gap `qwen3.5-4b-mtp`: Qwen3.5 MTP model was not supplied or is unavailable.
