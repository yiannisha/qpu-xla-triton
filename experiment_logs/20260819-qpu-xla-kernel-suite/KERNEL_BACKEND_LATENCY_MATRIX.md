# Kernel Backend Latency Matrix

Current packaged qpu_xla kernel-suite results. Median steady-state whole-operation latency in milliseconds. CPU measurements use four threads. NumPy is linked to OpenBLAS 0.3.28; non-matrix NumPy operations do not necessarily invoke BLAS. QPU-only and CPU/QPU columns include submission and synchronization unless explicitly labeled kernel-only.

The FP32 table shows the fastest measured hybrid partition while the exhaustive per-partition records remain in [FP32_EVALUATION_MATRIX.md](FP32_EVALUATION_MATRIX.md).

## FP32 backend matrix

| Case | Operation | QPU kernel | Shape | NumPy/OpenBLAS ms | Torch ms | qpu_xla CPU ms | QPU-only ms | Best CPU/QPU ms | Best split | Fastest | Best acceleration | Promotion | Max error |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| decode-h2048-c2048 | rms-norm | `vc7.rms_norm_fp32` | `1×2048` | 0.020 | 0.031 | - | 0.203 | - | - | NumPy/OpenBLAS (0.020 ms) | 0.100× | not promoted | 4.77e-07 |
| decode-h2048-c2048 | rope | `vc7.rope_fp32` | `32×64` | 0.014 | 0.023 | - | 0.185 | 0.225 | rows 28/32 QPU | NumPy/OpenBLAS (0.014 ms) | 0.076× | not promoted | 0 |
| decode-h2048-c2048 | softmax | `vc7.softmax_fp32` | `32×2048` | 0.661 | 0.071 | - | 0.234 | 0.356 | rows 28/32 QPU | Torch (0.071 ms) | 0.304× | not promoted | 1.49e-07 |
| decode-h2048-c2048 | swiglu | `vc7.swiglu_fp32` | `1×5632` | 0.060 | 0.027 | - | 0.197 | - | - | Torch (0.027 ms) | 0.139× | not promoted | 9.54e-07 |
| decode-h2048-c512 | down | `vc7.fp32_gemv` | `1×5632×2048` | 3.801 | 21.078 | - | 12.866 | 16.060 | output-columns 1792/2048 QPU | NumPy/OpenBLAS (3.801 ms) | 0.295× | not promoted | 0.000717 |
| decode-h2048-c512 | gate | `vc7.fp32_gemv` | `1×2048×5632` | 3.907 | 24.657 | - | 12.637 | 13.024 | output-columns 4928/5632 QPU | NumPy/OpenBLAS (3.907 ms) | 0.309× | not promoted | 0.000275 |
| decode-h2048-c512 | k | `vc7.fp32_gemv` | `1×2048×256` | 0.105 | 0.939 | - | 1.111 | 1.321 | output-columns 224/256 QPU | NumPy/OpenBLAS (0.105 ms) | 0.094× | not promoted | 0.00016 |
| decode-h2048-c512 | lm_head | `vc7.fp32_gemv` | `1×2048×32000` | 21.769 | 142.743 | - | 72.848 | 76.820 | output-columns 28000/32000 QPU | NumPy/OpenBLAS (21.769 ms) | 0.299× | not promoted | 0.000366 |
| decode-h2048-c512 | o | `vc7.fp32_gemv` | `1×2048×2048` | 1.336 | 9.833 | - | 4.549 | 4.905 | output-columns 1792/2048 QPU | NumPy/OpenBLAS (1.336 ms) | 0.294× | not promoted | 0.000259 |
| decode-h2048-c512 | q | `vc7.fp32_gemv` | `1×2048×2048` | 1.333 | 9.851 | - | 4.583 | 5.961 | output-columns 1536/2048 QPU | NumPy/OpenBLAS (1.333 ms) | 0.291× | not promoted | 0.000259 |
| decode-h2048-c512 | rms-norm | `vc7.rms_norm_fp32` | `1×2048` | 0.020 | 0.033 | - | 0.186 | - | - | NumPy/OpenBLAS (0.020 ms) | 0.107× | not promoted | 4.77e-07 |
| decode-h2048-c512 | rope | `vc7.rope_fp32` | `32×64` | 0.014 | 0.026 | - | 0.174 | 0.221 | rows 28/32 QPU | NumPy/OpenBLAS (0.014 ms) | 0.080× | not promoted | 0 |
| decode-h2048-c512 | softmax | `vc7.softmax_fp32` | `32×512` | 0.186 | 0.024 | - | 0.167 | 0.284 | rows 28/32 QPU | Torch (0.024 ms) | 0.143× | not promoted | 5.96e-08 |
| decode-h2048-c512 | swiglu | `vc7.swiglu_fp32` | `1×5632` | 0.062 | 0.028 | - | 0.186 | - | - | Torch (0.028 ms) | 0.150× | not promoted | 9.54e-07 |
| decode-h2048-c512 | up | `vc7.fp32_gemv` | `1×2048×5632` | 3.798 | 23.778 | - | 12.647 | 12.990 | output-columns 4928/5632 QPU | NumPy/OpenBLAS (3.798 ms) | 0.300× | not promoted | 0.000214 |
| decode-h2048-c512 | v | `vc7.fp32_gemv` | `1×2048×256` | 0.101 | 0.954 | - | 1.125 | 1.358 | output-columns 224/256 QPU | NumPy/OpenBLAS (0.101 ms) | 0.089× | not promoted | 0.000206 |
| decode-h3072-c4096 | rms-norm | `vc7.rms_norm_fp32` | `1×3072` | 0.021 | 0.041 | - | 0.215 | - | - | NumPy/OpenBLAS (0.021 ms) | 0.099× | not promoted | 9.54e-07 |
| decode-h3072-c4096 | rope | `vc7.rope_fp32` | `24×128` | 0.025 | 0.025 | - | 0.188 | 0.233 | rows 21/24 QPU | NumPy/OpenBLAS (0.025 ms) | 0.133× | not promoted | 0 |
| decode-h3072-c4096 | softmax | `vc7.softmax_fp32` | `24×4096` | 0.977 | 0.103 | - | 0.317 | 0.393 | rows 21/24 QPU | Torch (0.103 ms) | 0.324× | not promoted | 8.94e-08 |
| decode-h3072-c4096 | swiglu | `vc7.swiglu_fp32` | `1×8192` | 0.083 | 0.039 | - | 0.197 | - | - | Torch (0.039 ms) | 0.196× | not promoted | 1.91e-06 |
| fp32-copy-n4194048 | copy | `vc7.copy_words` | `4194048` | 3.465 | 4.481 | - | 3.000 | - | - | QPU (3.000 ms) | 1.155× | supported-win | 0 |
| fp32-epilogue-r128-c5632 | bias | `vc7.bias_fp32` | `128×5632` | 0.600 | 0.497 | - | 0.783 | - | - | Torch (0.497 ms) | 0.635× | not promoted | 0 |
| fp32-epilogue-r128-c5632 | bias_relu | `vc7.bias_relu_fp32` | `128×5632` | 1.020 | 1.161 | - | 0.812 | - | - | QPU (0.812 ms) | 1.256× | supported-win | 0 |
| fp32-epilogue-r128-c5632 | relu | `vc7.relu_fp32` | `128×5632` | 0.458 | 0.603 | - | 0.580 | - | - | NumPy/OpenBLAS (0.458 ms) | 0.790× | not promoted | 0 |
| fp32-epilogue-r16-c262128 | bias | `vc7.bias_fp32` | `16×262128` | 6.273 | 5.371 | - | 5.255 | - | - | QPU (5.255 ms) | 1.022× | not promoted | 0 |
| fp32-epilogue-r16-c262128 | bias_relu | `vc7.bias_relu_fp32` | `16×262128` | 10.048 | 11.032 | - | 5.234 | - | - | QPU (5.234 ms) | 1.920× | supported-win | 0 |
| fp32-epilogue-r16-c262128 | relu | `vc7.relu_fp32` | `16×262128` | 3.425 | 4.600 | - | 2.808 | - | - | QPU (2.808 ms) | 1.220× | supported-win | 0 |
| fp32-epilogue-r256-c8192 | bias | `vc7.bias_fp32` | `256×8192` | 1.724 | 1.929 | - | 2.268 | - | - | NumPy/OpenBLAS (1.724 ms) | 0.760× | not promoted | 0 |
| fp32-epilogue-r256-c8192 | bias_relu | `vc7.bias_relu_fp32` | `256×8192` | 3.162 | 4.345 | - | 2.285 | - | - | QPU (2.285 ms) | 1.383× | supported-win | 0 |
| fp32-epilogue-r256-c8192 | relu | `vc7.relu_fp32` | `256×8192` | 1.344 | 1.877 | - | 1.562 | - | - | NumPy/OpenBLAS (1.344 ms) | 0.860× | not promoted | 0 |
| fp32-epilogue-r64-c32000 | bias | `vc7.bias_fp32` | `64×32000` | 1.422 | 1.862 | - | 1.740 | - | - | NumPy/OpenBLAS (1.422 ms) | 0.817× | not promoted | 0 |
| fp32-epilogue-r64-c32000 | bias_relu | `vc7.bias_relu_fp32` | `64×32000` | 2.956 | 4.264 | - | 1.768 | - | - | QPU (1.768 ms) | 1.672× | supported-win | 0 |
| fp32-epilogue-r64-c32000 | relu | `vc7.relu_fp32` | `64×32000` | 1.303 | 1.890 | - | 1.349 | - | - | NumPy/OpenBLAS (1.303 ms) | 0.966× | not promoted | 0 |
| fp32-minmax-n4194048 | maximum | `vc7.maximum_words` | `4194048` | 6.748 | 7.199 | - | 4.447 | - | - | QPU (4.447 ms) | 1.517× | supported-win | 0 |
| fp32-minmax-n4194048 | minimum | `vc7.minimum_words` | `4194048` | 6.808 | 7.149 | - | 4.392 | - | - | QPU (4.392 ms) | 1.550× | supported-win | 0 |
| fp32-minmax-n4194240 | maximum | `vc7.maximum_words` | `4194240` | 6.693 | 7.153 | - | 7.362 | - | - | NumPy/OpenBLAS (6.693 ms) | 0.909× | not promoted | 0 |
| fp32-minmax-n4194240 | minimum | `vc7.minimum_words` | `4194240` | 6.708 | 7.179 | - | 7.385 | - | - | NumPy/OpenBLAS (6.708 ms) | 0.908× | not promoted | 0 |
| fp32-residual-n4194048 | residual_add | `vc7.residual_add_fp32` | `4194048` | 6.590 | 7.434 | - | 4.675 | - | - | QPU (4.675 ms) | 1.410× | supported-win | 0 |
| pool2d-n1-c32-h144-w144 | avgpool2d | `vc7.avgpool2d_fp32` | `1×32×144×144 → 1×32×72×72` | 1.178 | 0.548 | - | 0.870 | - | - | Torch (0.548 ms) | 0.630× | not promoted | 0 |
| pool2d-n1-c32-h144-w144 | maxpool2d | `vc7.maxpool2d_fp32` | `1×32×144×144 → 1×32×72×72` | 1.146 | 1.398 | - | 0.866 | - | - | QPU (0.866 ms) | 1.323× | supported-win | 0 |
| pool2d-n1-c48-h128-w128 | avgpool2d | `vc7.avgpool2d_fp32` | `1×48×128×128 → 1×48×64×64` | 1.742 | 1.955 | - | 1.059 | - | - | QPU (1.059 ms) | 1.645× | supported-win | 0 |
| pool2d-n1-c48-h128-w128 | maxpool2d | `vc7.maxpool2d_fp32` | `1×48×128×128 → 1×48×64×64` | 1.706 | 3.593 | - | 1.041 | - | - | QPU (1.041 ms) | 1.640× | supported-win | 0 |
| pool2d-n1-c64-h112-w112 | avgpool2d | `vc7.avgpool2d_fp32` | `1×64×112×112 → 1×64×56×56` | 1.533 | 0.697 | - | 1.280 | - | - | Torch (0.697 ms) | 0.545× | not promoted | 0 |
| pool2d-n1-c64-h112-w112 | maxpool2d | `vc7.maxpool2d_fp32` | `1×64×112×112 → 1×64×56×56` | 1.397 | 1.713 | - | 1.281 | - | - | QPU (1.281 ms) | 1.091× | supported-win | 0 |
| prefill-h1024-t64 | down | `vc7.tiled_fp32_gemm` | `64×2816×1024` | 10.112 | 7.052 | - | 16.932 | 6.165 | rows 16/64 QPU | CPU/QPU (6.165 ms) | 1.144× | supported-win | 0.000534 |
| prefill-h1024-t64 | gate | `vc7.tiled_fp32_gemm` | `64×1024×2816` | 9.591 | 7.398 | - | 16.715 | 6.083 | rows 16/64 QPU | CPU/QPU (6.083 ms) | 1.216× | supported-win | 0.000175 |
| prefill-h1024-t64 | k | `vc7.tiled_fp32_gemm` | `64×1024×256` | 0.407 | 3.093 | - | 1.878 | 0.924 | rows 16/64 QPU | NumPy/OpenBLAS (0.407 ms) | 0.441× | not promoted | 0.000191 |
| prefill-h1024-t64 | lm_head | `vc7.tiled_fp32_gemm` | `64×1024×32000` | 108.795 | 81.400 | - | 186.905 | 66.777 | rows 16/64 QPU | CPU/QPU (66.777 ms) | 1.219× | supported-win | 0.000229 |
| prefill-h1024-t64 | o | `vc7.tiled_fp32_gemm` | `64×1024×1024` | 3.344 | 4.760 | - | 6.360 | 2.484 | rows 16/64 QPU | CPU/QPU (2.484 ms) | 1.346× | supported-win | 0.000206 |
| prefill-h1024-t64 | q | `vc7.tiled_fp32_gemm` | `64×1024×1024` | 3.149 | 2.988 | - | 6.378 | 2.502 | rows 16/64 QPU | CPU/QPU (2.502 ms) | 1.194× | supported-win | 0.000183 |
| prefill-h1024-t64 | rms-norm | `vc7.rms_norm_fp32` | `64×1024` | 0.133 | 0.143 | - | 0.229 | 0.295 | rows 56/64 QPU | NumPy/OpenBLAS (0.133 ms) | 0.583× | not promoted | 9.54e-07 |
| prefill-h1024-t64 | rope | `vc7.rope_fp32` | `1024×64` | 0.343 | 0.083 | - | 0.269 | 0.340 | rows 896/1024 QPU | Torch (0.083 ms) | 0.307× | not promoted | 4.77e-07 |
| prefill-h1024-t64 | softmax | `vc7.softmax_fp32` | `1024×64` | 0.741 | 0.162 | - | 0.252 | 0.345 | rows 896/1024 QPU | Torch (0.162 ms) | 0.645× | not promoted | 2.38e-07 |
| prefill-h1024-t64 | swiglu | `vc7.swiglu_fp32` | `64×2816` | 1.880 | 0.369 | - | 0.361 | 0.704 | rows 56/64 QPU | QPU (0.361 ms) | 1.023× | not promoted | 1.91e-06 |
| prefill-h1024-t64 | up | `vc7.tiled_fp32_gemm` | `64×1024×2816` | 9.700 | 7.572 | - | 16.764 | 6.002 | rows 16/64 QPU | CPU/QPU (6.002 ms) | 1.261× | supported-win | 0.000172 |
| prefill-h1024-t64 | v | `vc7.tiled_fp32_gemm` | `64×1024×256` | 0.408 | 3.115 | - | 1.881 | 0.920 | rows 16/64 QPU | NumPy/OpenBLAS (0.408 ms) | 0.443× | not promoted | 0.000175 |
| prefill-h2048-t128 | rms-norm | `vc7.rms_norm_fp32` | `128×2048` | 0.525 | 0.827 | - | 0.423 | 0.590 | rows 112/128 QPU | QPU (0.423 ms) | 1.243× | supported-win | 9.54e-07 |
| prefill-h2048-t128 | rope | `vc7.rope_fp32` | `4096×64` | 1.745 | 0.350 | - | 0.584 | 0.663 | rows 3584/4096 QPU | Torch (0.350 ms) | 0.598× | not promoted | 4.77e-07 |
| prefill-h2048-t128 | softmax | `vc7.softmax_fp32` | `4096×128` | 5.668 | 0.641 | - | 0.869 | 1.184 | rows 3584/4096 QPU | Torch (0.641 ms) | 0.738× | not promoted | 2.38e-07 |
| prefill-h2048-t128 | swiglu | `vc7.swiglu_fp32` | `128×5632` | 9.366 | 1.935 | - | 0.944 | 1.200 | rows 112/128 QPU | QPU (0.944 ms) | 2.050× | supported-win | 1.91e-06 |
| prefill-h2048-t256 | rms-norm | `vc7.rms_norm_fp32` | `256×2048` | 1.264 | 1.816 | - | 0.619 | 0.964 | rows 224/256 QPU | QPU (0.619 ms) | 2.040× | supported-win | 1.43e-06 |
| prefill-h2048-t256 | rope | `vc7.rope_fp32` | `8192×64` | 3.880 | 0.769 | - | 0.958 | 1.050 | rows 7168/8192 QPU | Torch (0.769 ms) | 0.803× | not promoted | 4.77e-07 |
| prefill-h2048-t256 | softmax | `vc7.softmax_fp32` | `8192×256` | 22.814 | 3.523 | - | 2.841 | 3.076 | rows 7168/8192 QPU | QPU (2.841 ms) | 1.240× | supported-win | 2.98e-07 |
| prefill-h2048-t256 | swiglu | `vc7.swiglu_fp32` | `256×5632` | 19.375 | 4.190 | - | 1.821 | 2.033 | rows 224/256 QPU | QPU (1.821 ms) | 2.300× | supported-win | 2.86e-06 |
| prefill-h4096-t16 | rms-norm | `vc7.rms_norm_fp32` | `16×4096` | 0.130 | 0.147 | - | 0.232 | 0.318 | rows 14/16 QPU | NumPy/OpenBLAS (0.130 ms) | 0.561× | not promoted | 1.43e-06 |
| prefill-h4096-t16 | rope | `vc7.rope_fp32` | `512×128` | 0.320 | 0.084 | - | 0.258 | 0.340 | rows 448/512 QPU | Torch (0.084 ms) | 0.325× | not promoted | 2.38e-07 |
| prefill-h4096-t16 | softmax | `vc7.softmax_fp32` | `512×16` | 0.127 | 0.043 | - | 0.176 | 0.276 | rows 448/512 QPU | Torch (0.043 ms) | 0.246× | not promoted | 1.79e-07 |
| prefill-h4096-t16 | swiglu | `vc7.swiglu_fp32` | `16×11008` | 1.858 | 0.361 | - | 0.351 | 0.774 | rows 14/16 QPU | QPU (0.351 ms) | 1.029× | not promoted | 1.91e-06 |
| prefill-h512-t16 | argmax | `vc7.argmax_fp32` | `16×32000` | 0.202 | 0.607 | - | 0.587 | 0.715 | rows 14/16 QPU | NumPy/OpenBLAS (0.202 ms) | 0.344× | not promoted | 0 |
| prefill-h512-t16 | down | `vc7.tiled_fp32_gemm` | `16×1536×512` | 1.750 | 3.559 | - | 1.472 | - | - | QPU (1.472 ms) | 1.189× | supported-win | 0.000221 |
| prefill-h512-t16 | embedding | `vc7.embedding_lookup_fp32` | `16×512` | 0.004 | 0.013 | - | 0.180 | 0.262 | tokens 4/16 QPU | NumPy/OpenBLAS (0.004 ms) | 0.022× | not promoted | 0 |
| prefill-h512-t16 | gate | `vc7.tiled_fp32_gemm` | `16×512×1536` | 1.516 | 3.742 | - | 1.357 | - | - | QPU (1.357 ms) | 1.117× | supported-win | 8.39e-05 |
| prefill-h512-t16 | k | `vc7.tiled_fp32_gemm` | `16×512×64` | 0.025 | 1.047 | - | 0.339 | - | - | NumPy/OpenBLAS (0.025 ms) | 0.075× | not promoted | 5.34e-05 |
| prefill-h512-t16 | kv_append | `vc7.copy_words` | `16×64` | 0.003 | 0.007 | - | 0.293 | 0.387 | tokens 4/16 QPU | NumPy/OpenBLAS (0.003 ms) | 0.009× | not promoted | 0 |
| prefill-h512-t16 | lm_head | `vc7.tiled_fp32_gemm` | `16×512×32000` | 38.467 | 32.774 | - | 24.166 | - | - | QPU (24.166 ms) | 1.356× | supported-win | 0.000122 |
| prefill-h512-t16 | o | `vc7.tiled_fp32_gemm` | `16×512×512` | 0.210 | 2.246 | - | 0.627 | - | - | NumPy/OpenBLAS (0.210 ms) | 0.336× | not promoted | 6.87e-05 |
| prefill-h512-t16 | q | `vc7.tiled_fp32_gemm` | `16×512×512` | 0.235 | 1.418 | - | 0.636 | - | - | NumPy/OpenBLAS (0.235 ms) | 0.370× | not promoted | 9.16e-05 |
| prefill-h512-t16 | residual_add | `vc7.residual_add_fp32` | `16×512` | 0.003 | 0.006 | - | 0.175 | 0.276 | rows 4/16 QPU | NumPy/OpenBLAS (0.003 ms) | 0.017× | not promoted | 0 |
| prefill-h512-t16 | rms-norm | `vc7.rms_norm_fp32` | `16×512` | 0.034 | 0.037 | - | 0.179 | 0.241 | rows 12/16 QPU | NumPy/OpenBLAS (0.034 ms) | 0.187× | not promoted | 9.54e-07 |
| prefill-h512-t16 | rope | `vc7.rope_fp32` | `128×64` | 0.042 | 0.029 | - | 0.183 | 0.261 | rows 96/128 QPU | Torch (0.029 ms) | 0.158× | not promoted | 2.38e-07 |
| prefill-h512-t16 | softmax | `vc7.softmax_fp32` | `128×16` | 0.043 | 0.012 | - | 0.156 | 0.259 | rows 40/128 QPU | Torch (0.012 ms) | 0.080× | not promoted | 1.19e-07 |
| prefill-h512-t16 | swiglu | `vc7.swiglu_fp32` | `16×1536` | 0.247 | 0.109 | - | 0.198 | 0.272 | rows 14/16 QPU | Torch (0.109 ms) | 0.551× | not promoted | 9.54e-07 |
| prefill-h512-t16 | up | `vc7.tiled_fp32_gemm` | `16×512×1536` | 1.505 | 1.613 | - | 1.342 | - | - | QPU (1.342 ms) | 1.121× | supported-win | 7.63e-05 |
| prefill-h512-t16 | v | `vc7.tiled_fp32_gemm` | `16×512×64` | 0.026 | 1.048 | - | 0.338 | - | - | NumPy/OpenBLAS (0.026 ms) | 0.075× | not promoted | 5.34e-05 |
| s512 | matmul | `vc7.tiled_fp32_gemm` | `512×512×512` | 3.886 | 6.151 | 5.009 | 12.745 | 4.294 | CPU+QPU (160/512 QPU rows) | NumPy/OpenBLAS (3.886 ms) | 0.905× | not promoted | 0.000107 |
| s512 | scaled_dot_product_attention | `vc7.tiled_fp32_gemm + vc7.softmax_fp32` | `512×512` | 11.449 | 12.743 | - | 37.886 | 44.033 | CPU+QPU SDPA (448 QPU rows) | NumPy/OpenBLAS (11.449 ms) | 0.302× | not promoted | 1.07e-06 |
| s64 | matmul | `vc7.tiled_fp32_gemm` | `64×64×64` | 0.030 | 0.017 | 0.206 | 0.240 | 0.324 | CPU+QPU (48/64 QPU rows) | Torch (0.017 ms) | 0.070× | not promoted | 5.72e-06 |
| s64 | scaled_dot_product_attention | `vc7.tiled_fp32_gemm + vc7.softmax_fp32` | `64×64` | 0.135 | 0.086 | - | 1.117 | 1.754 | CPU+QPU SDPA (48 QPU rows) | Torch (0.086 ms) | 0.077× | not promoted | 4.77e-07 |

## W8A8 backend matrix

Dense W8A8 reports both the same-quantized-contract CPU path and the deployable FP32 CPU alternative.

| Case | Operation/kernel | Shape | NumPy W8A8 ms | Torch W8A8 ms | NumPy FP32 ms | Torch FP32 ms | QPU-only ms | QPU kernel-only ms | Best CPU/QPU ms | Best split | Fastest deployable backend |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| prefill-h512-t16 | hidden linear / `vc7.tiled_w8a8_gemm_dequantize` | `16×512×512` | 4.632 | 1.660 | 0.212 | 2.563 | 0.488 | 0.282 | 0.644 | outputs 192/512 QPU | NumPy/OpenBLAS FP32 |

| Case | Operation/kernel | Shape | NumPy dynamic W8A8 ms | Torch native FP32 ms | QPU prepared total ms | QPU execute-event ms | QPU host-prep-event ms | Fastest backend | Max error |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|

## Packaged kernels without current comparable latency

| Kernel | Dtype | Current gap |
|---|---|---|
| `vc7.avgpool2d_int32` | INT32 | pooling benchmark not included in the current rerun |
| `vc7.maxpool2d_int32` | INT32 | pooling benchmark not included in the current rerun |
| `vc7.bias_int32` | INT32 | standalone epilogue not included |
| `vc7.bias_relu_int32` | INT32 | standalone epilogue not included |
| `vc7.relu_int32` | INT32 | standalone epilogue not included |
| `vc7.tiled_int32_gemm` | INT32 | legacy result was not rerun in this backend matrix |
| `vc7.w8a8_dequantize` | W8A8→FP32 | only measured fused into dense GEMM |
| `vc7.w8a8_gemv` | W8A8→INT32 | decode W8A8 was not rerun in the current matrix |
