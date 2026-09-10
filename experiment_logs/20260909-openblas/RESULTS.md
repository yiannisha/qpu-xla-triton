# llama.cpp OpenBLAS/QPU results

Date: 2026-09-09

## Configuration

The comparison uses llama.cpp commit `91d2fc387` and the Gemma 4 E2B Q4_0
model. A separate shared-library build was configured with:

```sh
cmake -S /home/yiannis/side/llama.cpp \
  -B /home/yiannis/side/llama.cpp/build-openblas-qpu \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_CPU_REPACK=ON
```

The build found `/usr/lib/aarch64-linux-gnu/libopenblas.so` (OpenBLAS 0.3.29),
`ldd` confirms that `libggml-blas.so` loads `libopenblas.so.0`, and
`llama-server --list-devices` reports `BLAS: OpenBLAS`. All full-model BLAS
measurements use `--device BLAS --n-gpu-layers 99`, four llama.cpp threads,
and four OpenBLAS threads. The performance governor was enabled and swap was
disabled for the full-model and MTP measurements.

## Full-model prompt processing

The table reports the median of three `llama-bench` samples after its normal
warmup. Both builds use the same model, commit, batch settings, and four CPU
threads. The CPU build has BLAS disabled and uses CPU_REPACK; the OpenBLAS
build assigns all model layers to the BLAS device.

| Prompt M | CPU_REPACK | OpenBLAS | CPU_REPACK / OpenBLAS throughput |
|---:|---:|---:|---:|
| 33 | 482.435 ms (68.40 tok/s) | 7609.083 ms (4.34 tok/s) | 15.77x |
| 257 | 4184.725 ms (61.41 tok/s) | 16321.143 ms (15.75 tok/s) | 3.90x |
| 513 | 9048.111 ms (56.70 tok/s) | 27416.964 ms (18.71 tok/s) | 3.03x |
| 1025 | 18655.695 ms (54.94 tok/s) | 56552.947 ms (18.12 tok/s) | 3.03x |

There is no OpenBLAS crossover in the tested agentic-prefill range. A verbose
M=33 run assigned every layer to BLAS and reported 555 graph splits for the
33-token graph. The backend converts supported Q4 weights to F32 for GEMM;
quantized CPU_REPACK avoids that conversion and the repeated CPU/BLAS
boundaries.

Raw evidence:

- `llama-bench-cpu-repack.json`
- `llama-bench-openblas.json`

## Channel-partitioned FFN island

The OpenBLAS-aware harness requires and records this placement:

- gate, up, and down Q4 matmuls: OpenBLAS;
- GEGLU: llama.cpp CPU;
- QPU suffix: Q4 x Q8 gate/up, resident GEGLU-to-Q8, and Q4 x Q8 down;
- final join: hidden-size F32 addition.

The fastest measured partition in every tested cell was a 1/4 QPU channel
suffix:

| M | FFN width | OpenBLAS full FFN | Hybrid wall | Apparent speedup |
|---:|---:|---:|---:|---:|
| 257 | 6144 | 215.215 ms | 179.026 ms | 1.202x |
| 257 | 12288 | 419.355 ms | 354.771 ms | 1.182x |
| 513 | 6144 | 358.513 ms | 291.714 ms | 1.229x |
| 513 | 12288 | 681.829 ms | 560.303 ms | 1.217x |
| 1025 | 6144 | 675.515 ms | 555.487 ms | 1.216x |
| 1025 | 12288 | 1352.142 ms | 1059.679 ms | 1.276x |
| 2049 | 6144 | 1423.885 ms | 1032.172 ms | 1.380x |
| 2049 | 12288 | 2624.891 ms | 1973.678 ms | 1.330x |

None is a valid acceleration result. All 24 partitions failed the unchanged
numerical gate. OpenBLAS computes dequantized-Q4 x F32, whereas the current QPU
kernel computes Q4 x Q8. Mixing those semantics across FFN channels produced
mean absolute output errors of roughly 0.10-0.14 and maxima of 1.85-5.95. The
apparent speedup only shows that replacing part of this slow OpenBLAS path can
reduce time; it does not establish a drop-in equivalent hybrid.

OpenBLAS is also slower than CPU_REPACK on these isolated full FFNs. For
example, the earlier same-shape CPU_REPACK M=257 medians were 74.830 ms and
101.233 ms, versus 215.215 ms and 419.355 ms here.

Raw evidence: `ffn-island-openblas-screen.json`.

## M=1 MTP drafting

OpenBLAS requires M, N, and K to be at least 32 for `MUL_MAT`, so M=1 draft
nodes cannot run on OpenBLAS. The target model was assigned to the BLAS device;
the MTP model was deliberately kept on CPU_REPACK so the direct QPU hook could
register its weights. Each row is one isolated 32-token server request and is
screening evidence, not a promotion campaign.

| Placement | Request | Verified QPU dispatches | Result |
|---|---:|---:|---|
| OpenBLAS target, plain decode | 4368.999 ms | 0 | reference |
| OpenBLAS target, CPU MTP depth 2 | 4364.544 ms | 0 | 1.001x vs plain |
| OpenBLAS target, QPU MTP depth 2 | 4726.496 ms | 884 | 0.923x vs CPU MTP |

All three cases emitted identical greedy token IDs. The QPU case passed exact
program/hash/shape/partition telemetry validation and reported no CPU fallback.
It is nevertheless 8.3% slower than CPU MTP. The direct M=1 smoke separately
passed its numerical and dispatch checks (`m1_passed=true`, QPU complete time
about 0.360 ms); the combined historical GEGLU/M=1 smoke executable still
returns failure because its unrelated GEGLU test requires bitwise identity and
observed a `4.57e-5` maximum difference.

Raw evidence: `mtp-openblas-screen.json`.

## Decision

There is no valid result in which the QPU work beats OpenBLAS end to end:

- the isolated FFN hybrids are faster than OpenBLAS but numerically
  incompatible with its F32-activation semantics;
- the exact M=1 QPU MTP path is slower than CPU drafting;
- OpenBLAS itself is 3.0-15.8x slower than llama.cpp CPU_REPACK on full-model
  prompt processing in the tested range.

CPU_REPACK remains the correct production baseline for this quantized model.
Any future QPU claim should continue to compare against it, not OpenBLAS.
