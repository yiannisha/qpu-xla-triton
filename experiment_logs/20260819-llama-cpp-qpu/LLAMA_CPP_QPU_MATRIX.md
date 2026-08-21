# llama.cpp QPU latency matrix

Generated: 2026-08-20T04:00:43+00:00

Native operator wins are provisional evidence only. Promotion requires a complete-request comparison against the fastest tuned CPU configuration, identical greedy semantics, exact candidate hashes, at least five retained independent sessions per side, median speedup of at least 1.05x, and a bootstrap 95% CI lower bound above 1.0. Diagnostic rows are never automatic placements.

## Native quantized operator matrix

| Tensor | Type | M×K×N | Placement | CPU threads | QPU share | Complete ms | Quant ms | Input copy ms | Submit+wait ms | Output copy ms | Speedup (95% low) | Max abs err | Resident MiB | Retained | Session | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `token_embd.weight` | Q4_0 | 1×256×262144 | cpu | 2 | 0.000 | 2.981 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | baseline |
| `token_embd.weight` | Q4_0 | 1×256×262144 | cpu | 3 | 0.000 | 3.144 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | baseline |
| `token_embd.weight` | Q4_0 | 1×256×262144 | cpu | 4 | 0.000 | 3.263 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | baseline |
| `token_embd.weight` | Q4_0 | 1×256×262144 | qpu | 2 | 1.000 | 37.762 | 0.000 | 0.000 | 37.115 | 0.650 | 0.079 (0.078) | 4.77e-07 | 40.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.083 | 3.990 | 0.001 | 0.000 | 3.930 | 0.057 | 0.747 (0.609) | 2.38e-07 | 3.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.167 | 7.060 | 0.001 | 0.000 | 6.949 | 0.108 | 0.422 (0.304) | 3.58e-07 | 6.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.250 | 10.153 | 0.001 | 0.000 | 9.987 | 0.163 | 0.294 (0.290) | 3.58e-07 | 10.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.333 | 13.154 | 0.001 | 0.000 | 12.933 | 0.216 | 0.227 (0.224) | 3.58e-07 | 13.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.417 | 16.239 | 0.001 | 0.000 | 15.960 | 0.279 | 0.184 (0.181) | 3.58e-07 | 16.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.500 | 20.596 | 0.001 | 0.000 | 20.267 | 0.325 | 0.145 (0.140) | 3.58e-07 | 20.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.583 | 22.379 | 0.001 | 0.000 | 21.997 | 0.379 | 0.133 (0.132) | 3.58e-07 | 23.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.667 | 25.496 | 0.001 | 0.000 | 25.045 | 0.449 | 0.117 (0.115) | 3.58e-07 | 26.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.750 | 28.591 | 0.001 | 0.000 | 28.101 | 0.487 | 0.104 (0.103) | 3.58e-07 | 30.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.833 | 31.605 | 0.001 | 0.000 | 31.032 | 0.570 | 0.094 (0.093) | 3.58e-07 | 33.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 2 | 0.917 | 34.666 | 0.001 | 0.000 | 34.059 | 0.595 | 0.086 (0.082) | 3.58e-07 | 36.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.083 | 6.080 | 0.001 | 0.000 | 6.020 | 0.054 | 0.490 (0.481) | 2.38e-07 | 3.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.167 | 7.932 | 0.001 | 0.000 | 7.818 | 0.109 | 0.376 (0.332) | 3.58e-07 | 6.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.250 | 11.994 | 0.001 | 0.000 | 11.826 | 0.163 | 0.249 (0.245) | 3.58e-07 | 10.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.333 | 14.775 | 0.001 | 0.000 | 14.559 | 0.216 | 0.202 (0.199) | 3.58e-07 | 13.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.417 | 16.957 | 0.001 | 0.000 | 16.680 | 0.270 | 0.176 (0.168) | 3.58e-07 | 16.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.500 | 20.599 | 0.001 | 0.000 | 20.268 | 0.325 | 0.145 (0.143) | 3.58e-07 | 20.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.583 | 23.393 | 0.001 | 0.000 | 23.009 | 0.379 | 0.127 (0.126) | 3.58e-07 | 23.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.667 | 25.852 | 0.001 | 0.000 | 25.421 | 0.433 | 0.115 (0.114) | 3.58e-07 | 26.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.750 | 29.320 | 0.001 | 0.001 | 28.745 | 0.567 | 0.102 (0.100) | 3.58e-07 | 30.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.833 | 31.817 | 0.001 | 0.000 | 31.273 | 0.540 | 0.094 (0.093) | 3.58e-07 | 33.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 3 | 0.917 | 34.836 | 0.001 | 0.000 | 34.214 | 0.617 | 0.086 (0.085) | 3.58e-07 | 36.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.083 | 6.362 | 0.001 | 0.000 | 6.297 | 0.057 | 0.469 (0.463) | 2.38e-07 | 3.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.167 | 9.168 | 0.001 | 0.000 | 9.054 | 0.108 | 0.325 (0.321) | 3.58e-07 | 6.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.250 | 12.175 | 0.001 | 0.000 | 11.965 | 0.203 | 0.245 (0.242) | 3.58e-07 | 10.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.333 | 14.912 | 0.001 | 0.000 | 14.688 | 0.217 | 0.200 (0.197) | 3.58e-07 | 13.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.417 | 17.713 | 0.001 | 0.000 | 17.426 | 0.280 | 0.168 (0.166) | 3.58e-07 | 16.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.500 | 20.251 | 0.001 | 0.000 | 19.923 | 0.323 | 0.147 (0.145) | 3.58e-07 | 20.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.583 | 23.405 | 0.001 | 0.000 | 23.021 | 0.378 | 0.127 (0.126) | 3.58e-07 | 23.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.667 | 25.752 | 0.001 | 0.000 | 25.314 | 0.431 | 0.116 (0.114) | 3.58e-07 | 26.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.750 | 29.007 | 0.001 | 0.000 | 28.505 | 0.488 | 0.103 (0.101) | 3.58e-07 | 30.001 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.833 | 31.894 | 0.001 | 0.000 | 31.318 | 0.570 | 0.093 (0.092) | 3.58e-07 | 33.334 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 1×256×262144 | hybrid | 4 | 0.917 | 34.863 | 0.001 | 0.000 | 34.258 | 0.595 | 0.085 (0.084) | 3.58e-07 | 36.669 | False | gemma-drafter-lm-head-m1-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | cpu | 2 | 0.000 | 5.444 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | baseline |
| `token_embd.weight` | Q4_0 | 4×256×262144 | cpu | 3 | 0.000 | 5.232 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | baseline |
| `token_embd.weight` | Q4_0 | 4×256×262144 | cpu | 4 | 0.000 | 5.400 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | baseline |
| `token_embd.weight` | Q4_0 | 4×256×262144 | qpu | 3 | 1.000 | 39.703 | 0.001 | 0.000 | 37.112 | 2.587 | 0.132 (0.130) | 4.77e-07 | 40.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.083 | 15.604 | 0.003 | 0.001 | 4.051 | 0.462 | 0.335 (0.331) | 4.77e-07 | 3.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.167 | 14.741 | 0.003 | 0.000 | 7.799 | 0.936 | 0.355 (0.330) | 4.77e-07 | 6.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.250 | 12.857 | 0.003 | 0.000 | 11.369 | 1.460 | 0.407 (0.388) | 4.77e-07 | 10.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.333 | 15.658 | 0.002 | 0.000 | 14.765 | 0.880 | 0.334 (0.329) | 4.77e-07 | 13.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.417 | 18.537 | 0.002 | 0.000 | 17.441 | 1.075 | 0.282 (0.278) | 4.77e-07 | 16.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.500 | 21.243 | 0.003 | 0.000 | 19.955 | 1.292 | 0.246 (0.241) | 4.77e-07 | 20.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.583 | 24.336 | 0.003 | 0.000 | 22.818 | 1.510 | 0.215 (0.212) | 4.77e-07 | 23.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.667 | 27.569 | 0.002 | 0.000 | 25.798 | 1.747 | 0.190 (0.187) | 4.77e-07 | 26.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.750 | 30.603 | 0.002 | 0.000 | 28.640 | 1.953 | 0.171 (0.169) | 4.77e-07 | 30.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.833 | 34.209 | 0.002 | 0.000 | 32.016 | 2.180 | 0.153 (0.137) | 4.77e-07 | 33.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 2 | 0.917 | 36.587 | 0.003 | 0.000 | 34.187 | 2.387 | 0.143 (0.139) | 4.77e-07 | 36.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.083 | 14.853 | 0.003 | 0.001 | 8.252 | 0.558 | 0.352 (0.308) | 4.77e-07 | 3.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.167 | 12.923 | 0.003 | 0.000 | 12.483 | 0.425 | 0.405 (0.393) | 4.77e-07 | 6.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.250 | 17.108 | 0.003 | 0.000 | 16.462 | 0.646 | 0.306 (0.300) | 4.77e-07 | 10.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.333 | 19.861 | 0.002 | 0.000 | 18.975 | 0.873 | 0.263 (0.259) | 4.77e-07 | 13.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.417 | 21.047 | 0.002 | 0.001 | 19.958 | 1.083 | 0.249 (0.245) | 4.77e-07 | 16.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.500 | 23.791 | 0.003 | 0.000 | 22.484 | 1.299 | 0.220 (0.211) | 4.77e-07 | 20.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.583 | 26.758 | 0.003 | 0.000 | 25.222 | 1.529 | 0.196 (0.186) | 4.77e-07 | 23.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.667 | 30.418 | 0.002 | 0.000 | 28.680 | 1.731 | 0.172 (0.170) | 4.77e-07 | 26.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.750 | 33.020 | 0.002 | 0.000 | 31.082 | 1.941 | 0.158 (0.157) | 4.77e-07 | 30.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.833 | 34.988 | 0.002 | 0.000 | 32.805 | 2.161 | 0.150 (0.148) | 4.77e-07 | 33.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 3 | 0.917 | 36.711 | 0.003 | 0.000 | 34.178 | 2.531 | 0.143 (0.138) | 4.77e-07 | 36.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.083 | 13.041 | 0.002 | 0.001 | 9.609 | 0.798 | 0.401 (0.377) | 4.77e-07 | 3.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.167 | 14.283 | 0.002 | 0.000 | 13.833 | 0.432 | 0.366 (0.357) | 4.77e-07 | 6.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.250 | 16.816 | 0.002 | 0.000 | 16.157 | 0.640 | 0.311 (0.306) | 4.77e-07 | 10.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.333 | 19.432 | 0.002 | 0.000 | 18.547 | 0.872 | 0.269 (0.256) | 4.77e-07 | 13.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.417 | 22.939 | 0.003 | 0.000 | 21.786 | 1.147 | 0.228 (0.221) | 4.77e-07 | 16.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.500 | 24.210 | 0.003 | 0.000 | 22.766 | 1.296 | 0.216 (0.200) | 4.77e-07 | 20.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.583 | 26.833 | 0.002 | 0.000 | 25.311 | 1.515 | 0.195 (0.187) | 4.77e-07 | 23.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.667 | 30.166 | 0.003 | 0.000 | 28.157 | 1.763 | 0.173 (0.170) | 4.77e-07 | 26.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.750 | 31.332 | 0.003 | 0.000 | 29.294 | 1.942 | 0.167 (0.162) | 4.77e-07 | 30.001 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.833 | 35.164 | 0.003 | 0.001 | 32.938 | 2.221 | 0.149 (0.146) | 4.77e-07 | 33.334 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `token_embd.weight` | Q4_0 | 4×256×262144 | hybrid | 4 | 0.917 | 37.314 | 0.002 | 0.000 | 34.770 | 2.534 | 0.140 (0.139) | 4.77e-07 | 36.669 | False | gemma-drafter-lm-head-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | cpu | 4 | 0.000 | 1.174 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | baseline |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | qpu | 4 | 1.000 | 19.648 | 0.022 | 0.001 | 19.517 | 0.108 | 0.060 (0.054) | 8.34e-07 | 12.808 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.083 | 6.105 | 0.023 | 0.002 | 2.377 | 0.012 | 0.192 (0.127) | 4.77e-07 | 1.078 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.167 | 5.028 | 0.015 | 0.002 | 4.990 | 0.015 | 0.234 (0.231) | 7.15e-07 | 2.144 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.250 | 7.168 | 0.016 | 0.001 | 7.122 | 0.023 | 0.164 (0.163) | 7.15e-07 | 3.210 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.333 | 7.891 | 0.015 | 0.002 | 7.839 | 0.030 | 0.149 (0.145) | 7.15e-07 | 4.277 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.417 | 9.643 | 0.015 | 0.001 | 9.585 | 0.038 | 0.122 (0.121) | 7.15e-07 | 5.343 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.500 | 11.698 | 0.014 | 0.001 | 11.631 | 0.046 | 0.100 (0.099) | 7.15e-07 | 6.410 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.583 | 12.980 | 0.015 | 0.001 | 12.907 | 0.053 | 0.090 (0.090) | 7.15e-07 | 7.476 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.667 | 14.272 | 0.014 | 0.001 | 14.190 | 0.062 | 0.082 (0.082) | 7.15e-07 | 8.542 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.750 | 15.477 | 0.016 | 0.001 | 15.387 | 0.068 | 0.076 (0.075) | 7.15e-07 | 9.609 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.833 | 16.569 | 0.015 | 0.001 | 16.473 | 0.075 | 0.071 (0.070) | 7.15e-07 | 10.675 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_gate.weight` | Q4_K | 4×2560×9216 | hybrid | 4 | 0.917 | 17.979 | 0.015 | 0.001 | 17.878 | 0.082 | 0.065 (0.065) | 7.15e-07 | 11.742 | False | qwen35-ffn-gate-q4-k-m4-compact-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | cpu | 4 | 0.000 | 1.811 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | baseline |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | qpu | 4 | 1.000 | 73.232 | 0.052 | 0.003 | 73.150 | 0.025 | 0.025 (0.023) | 4.77e-07 | 18.536 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.081 | 15.245 | 0.052 | 0.006 | 15.177 | 0.003 | 0.119 (0.117) | 4.77e-07 | 1.543 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.169 | 19.911 | 0.056 | 0.004 | 19.841 | 0.005 | 0.091 (0.090) | 4.77e-07 | 3.161 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.250 | 23.547 | 0.053 | 0.008 | 23.474 | 0.007 | 0.077 (0.076) | 4.77e-07 | 4.664 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.331 | 27.609 | 0.056 | 0.005 | 27.534 | 0.009 | 0.066 (0.065) | 4.77e-07 | 6.167 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.419 | 33.884 | 0.052 | 0.006 | 33.809 | 0.011 | 0.053 (0.053) | 4.77e-07 | 7.785 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.500 | 39.647 | 0.056 | 0.007 | 39.562 | 0.013 | 0.046 (0.045) | 4.77e-07 | 9.288 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.581 | 49.180 | 0.055 | 0.006 | 49.097 | 0.015 | 0.037 (0.036) | 4.77e-07 | 10.791 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.669 | 52.885 | 0.053 | 0.006 | 52.776 | 0.017 | 0.034 (0.034) | 4.77e-07 | 12.409 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.750 | 57.144 | 0.055 | 0.005 | 57.058 | 0.019 | 0.032 (0.031) | 4.77e-07 | 13.912 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.831 | 61.305 | 0.054 | 0.006 | 61.217 | 0.021 | 0.030 (0.029) | 4.77e-07 | 15.415 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ffn_down.weight` | Q6_K | 4×9216×2560 | hybrid | 4 | 0.919 | 67.440 | 0.056 | 0.006 | 67.347 | 0.023 | 0.027 (0.026) | 4.77e-07 | 17.033 | False | qwen35-ffn-down-q6-k-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | cpu | 4 | 0.000 | 0.839 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.000 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | baseline |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | qpu | 4 | 1.000 | 10.192 | 0.017 | 0.002 | 10.147 | 0.025 | 0.082 (0.078) | 4.77e-07 | 10.681 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.081 | 3.091 | 0.018 | 0.003 | 2.242 | 0.008 | 0.271 (0.258) | 1.3e-06 | 0.883 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.169 | 3.609 | 0.019 | 0.002 | 3.576 | 0.005 | 0.233 (0.184) | 1.3e-06 | 1.816 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.250 | 4.451 | 0.018 | 0.002 | 4.416 | 0.007 | 0.189 (0.174) | 1.3e-06 | 2.683 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.331 | 4.172 | 0.018 | 0.002 | 4.136 | 0.009 | 0.201 (0.191) | 1.3e-06 | 3.549 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.419 | 6.361 | 0.018 | 0.003 | 6.324 | 0.011 | 0.132 (0.126) | 1.3e-06 | 4.482 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.500 | 6.621 | 0.018 | 0.003 | 6.580 | 0.013 | 0.127 (0.121) | 1.3e-06 | 5.349 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.581 | 6.431 | 0.018 | 0.003 | 6.389 | 0.015 | 0.130 (0.124) | 1.3e-06 | 6.215 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.669 | 7.030 | 0.018 | 0.003 | 6.988 | 0.017 | 0.119 (0.113) | 1.3e-06 | 7.148 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.750 | 7.986 | 0.018 | 0.002 | 7.940 | 0.019 | 0.105 (0.100) | 1.3e-06 | 8.015 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.831 | 8.867 | 0.018 | 0.003 | 8.820 | 0.021 | 0.095 (0.090) | 1.3e-06 | 8.881 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |
| `blk.0.ssm_out.weight` | Q8_0 | 4×4096×2560 | hybrid | 4 | 0.919 | 9.241 | 0.017 | 0.001 | 9.197 | 0.023 | 0.091 (0.086) | 1.3e-06 | 9.814 | False | qwen35-ssm-out-q8-0-m4-exact-all-partitions-diagnostic | experimental-nonproduction-cpu-prefix |

## Fused attention diagnostics

These rows use the exact pinned GGML CPU node on deterministic synthetic tensors. They remain nonpromotable until replayed on captured model nodes in an isolated session.

| M | Q heads / KV heads | KV rows | Head dim | CPU ms | QPU ms | Speedup | Max abs err | Input MiB | Dispatches | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 / 1 | 256 | 256 | 0.064 | 0.674 | 0.094x | 0.000213 | 0.258 | 1 | experimental |
| 1 | 8 / 1 | 512 | 256 | 0.120 | 1.201 | 0.100x | 7.25e-05 | 0.509 | 1 | experimental |
| 1 | 8 / 1 | 2048 | 256 | 0.542 | 4.356 | 0.124x | 7.43e-05 | 2.012 | 1 | experimental |
| 1 | 8 / 1 | 4096 | 256 | 1.600 | 8.561 | 0.187x | 7.45e-05 | 4.016 | 1 | experimental |

## End-to-end matrix

| Case | Mode | Workload | Endpoint | Placement | Threads | Depth | Context target | Prompt target | Startup s | Prefill s | Request s | Prompt ms | Prompt tok/s | Decode tokens | Decode tok/s | Mean accepted | Cycle s | Greedy identical | Tool valid | QPU dispatches | Peak RSS MiB | Process swap KiB | Session | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|
| gemma-plain-t3-40-token-throughput-smoke | plain | decode | completion | cpu-only | 3 | — | 0 | — | 3.523 | — | 4.684 | 296.386 | 33.740 | 40 | 9.119 | — | — | — | — | — | — | — | server-throughput-smoke-v2 | diagnostic |
| gemma-mtp-n2-t3-40-token-throughput-smoke | mtp | decode | completion | cpu-only | 3 | 2 | 0 | — | 5.791 | — | 5.497 | 299.372 | 33.403 | 40 | 7.699 | 1.73 | 0.225 | — | — | — | — | — | server-throughput-smoke-v2 | diagnostic |
| gemma-plain-context-32-smoke | plain | decode | completion | cpu-only | 2 | — | 32 | — | 3.532 | 0.874 | 0.363 | 362.116 | 27.615 | — | — | — | — | — | — | — | — | — | context-32-smoke-v2 | diagnostic |
| gemma-prompt-p10-t2 | prompt | prompt | completion | cpu-only | 2 | — | 0 | 10 | 3.530 | — | 0.372 | 371.063 | 26.950 | — | — | — | — | — | — | — | — | — | exact-prompt-10-smoke | diagnostic |

## End-to-end promotion gate

Each timing value entering this gate is first reduced to one median per independent session. Candidate rows cannot be promoted from operator timing alone.

| Model | Mode | Workload | Context | Prompt | Metric | Baseline sessions | Candidate sessions | Speedup (95% CI) | Greedy exact | Evidence exact | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|
| — | — | — | — | — | — | 0 | 0 | — | — | — | no non-CPU sessions |

## Coverage and retention

- Operator rows: 113.
- End-to-end rows: 4.
- Attention diagnostic rows: 4.
- CPU baseline operator rows: 9.
- Provisional operator wins: 0.
- Retained end-to-end rows: 0.
- Promoted end-to-end configurations: 0.
- Planned end-to-end cases: 150.
- Planned workload counts: decode=99, prompt=24, structured-tool-call=27.
- Missing model/workload combinations remain coverage gaps; absence is not a CPU or QPU win.
- Coverage gap `gemma-4-e2b-vision`: multimodal projector/vision GGUF or representative image is unavailable; the local text GGUF manifest contains no vision tensors.
