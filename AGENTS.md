# Conv2D Kernel Guide

This document describes the current `conv2d` implementation in
[examples/tiledconv2d.py](/home/yiannis/side/py-videocore7/examples/tiledconv2d.py)
and should be treated as the starting point for further `conv2d` work.

## Current Design

The current `conv2d` path is not a direct spatial convolution kernel. It lowers
`NCHW` convolution to GEMM:

1. `im2col` lowers the input tensor to a 2-D matrix.
2. `OIHW` weights are reshaped into a GEMM weight matrix.
3. The host pads the GEMM dimensions to the QPU tile constraints.
4. A tiled 16x16 QPU GEMM-style kernel runs on the lowered matrices.
5. The result is reshaped back to `NCHW`.

For the common `1x1`, stride-1, padding-0, dilation-1 case, the packaged
`qpu_xla` convolution operators bypass the general spatial-window `im2col`
loop. They pack NCHW pixels directly into the GEMM row layout, so no kernel
window duplication occurs. This is still a GEMM-backed pointwise-convolution
path, not a native direct-convolution QPU microkernel.

This means the current kernel is best understood as a tiled GEMM-backed
`conv2d`, not a native direct-convolution implementation.

## Implemented Entry Points

The current public host-side entry points are:

- `tiledconv2d_fp32`
- `tiledconv2d_int32`
- `tiledconv2d_int16`

The current QPU kernels are:

- `qpu_tiledconv2d_fp32`
- `qpu_tiledconv2d_int32`
- `qpu_tiledconv2d_int16_packed`

The implementation also includes reusable helpers for:

- `im2col_nchw`
- `numpy_conv2d_nchw`
- `torch_conv2d_nchw`
- `TiledMatmulExecutor`

## Dtype Contracts

### `fp32`

- Input dtype: `float32`
- Weight dtype: `float32`
- Output dtype: `float32`
- Correctness status: good for the tested benchmark shapes

### `int32`

- Input dtype: `int32`
- Weight dtype: `int32`
- Output dtype: `int32`
- Arithmetic uses `smul24`
- All input values must fit the signed 24-bit range
- Correctness status: good for the tested benchmark shapes

### `int16`

- Input dtype: `int16`
- Weight dtype: `int16`
- The public path widens inputs and weights to `int32` on the host, then uses
  the validated `int32` microkernel.
- Output dtype: `int32`
- Correctness status: exact for the current tested benchmark shape

Important: `qpu_tiledconv2d_int16_packed` remains in the file as an experimental
assembly entry point. It packs `int16` pairs into `int32` words, widens packed
halves, and accumulates into `int32`, but it has produced large numerical
errors in live benchmarking. It is quarantined and is not used by the public
`tiledconv2d_int16` entry point until it has a passing hardware differential
test.

## Internal Tile Assumptions

The host pads dimensions to satisfy the microkernel tile shape:

- `p` tile: 16
- `r` tile: 16
- `q` tile: 4 for `fp32` and `int32`
- `q` tile: 4 for the public widened `int16` path
- `q` tile: 8 only for the quarantined packed `int16` experiment

The host helpers hide this from callers by padding automatically, but future
kernel work should preserve explicit awareness of these internal constraints.

## What The Benchmarks Mean

The benchmark output now prints multiple timing categories. These must not be
interpreted as the same thing.

### `numpy`

- This is a NumPy baseline implemented as `im2col + dot`
- It is not a native convolution operator
- It is useful as a vectorized CPU lowered-conv baseline

### `torch native conv2d`

- This is `torch.nn.functional.conv2d`
- This is the real optimized CPU conv baseline in the current benchmark

### `QPU host prep`

- CPU-side lowering and preparation only
- Includes `im2col`, weight reshaping, padding, and any host-side packing
- Does not include QPU execution

### `QPU cached total`

- Repeated steady-state timing of:
  - upload
  - execute
  - readback
- Uses a persistent `Driver`, compiled program, uniforms, and device buffers
- Excludes one-time setup costs such as driver creation and program assembly

### `QPU execute only`

- Repeated timing of only `drv.execute(...)`
- Uses already allocated and already uploaded device buffers
- This is the closest metric to raw kernel speed

### `QPU prep+cached total`

- `QPU host prep + QPU cached total`
- This is the practical steady-state end-to-end cost when setup is reused
- This is not a cold-start single-call benchmark

## Cold-Start vs Steady-State

This distinction matters.

### Cold-start end-to-end

Includes:

- driver creation
- program assembly/upload
- buffer allocation
- host prep
- upload
- execute
- readback

The current benchmark does not print this directly.

### Steady-state end-to-end

Includes:

- host prep
- upload
- execute
- readback

Excludes:

- driver creation
- program assembly
- buffer allocation

`QPU prep+cached total` is the current steady-state end-to-end metric.

## How To Interpret Results

Use these rules:

- Compare against `QPU execute only` to judge the kernel itself.
- Compare against `QPU cached total` to judge repeated-inference throughput.
- Compare against `QPU prep+cached total` to judge steady-state end-to-end use.

If `QPU execute only` is bad, the kernel is the bottleneck.

If `QPU execute only` is decent but `QPU cached total` is bad, upload/readback
is the bottleneck.

If `QPU cached total` is decent but `QPU prep+cached total` is bad, host-side
lowering and packing are the bottleneck.

## Current Performance Conclusions

These conclusions are only valid for the tested shapes and current machine.
They must not be generalized without broader benchmarking.

### `fp32`

Observed behavior:

- `QPU execute only` is roughly competitive with the current NumPy lowered-conv
  baseline for the tested shape
- Torch native `conv2d` is still substantially faster

Interpretation:

- The current `fp32` kernel is not obviously broken
- The current direct competitor is Torch, and Torch is still ahead

### `int32`

Observed behavior for the tested benchmark:

- `QPU execute only` beats both NumPy and Torch
- `QPU cached total` also beats both NumPy and Torch
- `QPU prep+cached total` also beats both NumPy and Torch
- correctness was exact for the tested benchmark

Interpretation:

- For the tested `int32` shape, the current QPU path is better than both NumPy
  and Torch in steady-state end-to-end terms
- This does not prove it is better for all shapes
- This does not prove it is better in cold-start single-call usage

### `int16`

Observed behavior:

- The public widened path is exact for the tested benchmark shape.
- It does not currently receive the bandwidth benefit intended by the packed
  kernel.

Interpretation:

- Keep the public widened path as the correctness reference.
- Do not optimize or re-enable the packed kernel before its exact hardware
  differential test passes.

## Why QPU Looked Slow In The First Benchmark

The original benchmark charged too much host work to the QPU path:

- host-side `im2col`
- host-side padding
- host-side int16 packing
- `Driver` creation
- program assembly/upload
- buffer allocation
- upload
- execute
- readback

Torch and NumPy were only timing their optimized CPU compute paths.

That made the original QPU comparison unfair. The persistent executor and
split timings were added to fix that.

## Current Architectural Limitations

The current implementation still has important performance limits:

- It uses `im2col`, which duplicates data and increases memory traffic.
- It is built on a naive GEMM-style kernel, not the best available microkernel
  pattern in the repo.
- It is not a direct convolution kernel specialized for common shapes.
- Small shapes are sensitive to upload/readback and prep overhead.

## Implementation Priorities

Future work should proceed in this order.

1. Keep public `int16` correctness covered; fix the quarantined packed kernel
   before attempting to use it for performance.
2. Keep benchmark categories separate. Do not collapse setup and execute timing.
3. Reuse prepared weights and device buffers whenever possible.
4. Reuse the persistent executor for repeated inference workloads.
5. Port the conv lowering path to a faster GEMM microkernel if available.
6. Replace `im2col` with direct-convolution kernels for hot cases such as:
   - `3x3`
   - `1x1`
   - common stride/padding combinations

## Rules For Further Conv2D Work

- Do not compare QPU timings against Torch unless the benchmark states whether
  the result is cold-start or steady-state.
- Do not call the NumPy baseline a native convolution baseline.
- Do not treat the packed `int16` kernel as production-ready until correctness
  is demonstrated. The public widened `int16` wrapper must retain its exact
  `int32`-accumulation contract.
- Always report max error for numerical paths.
- For `int32`, preserve the signed 24-bit input contract unless the kernel is
  rewritten to remove the `smul24` restriction.
- If a new kernel changes internal tile sizes, update the host padding helpers
  and this document together.

## Validation Checklist

For any future change to the `conv2d` path:

1. Verify `fp32` correctness against NumPy and Torch.
2. Verify `int32` correctness against NumPy and Torch.
3. Verify the public `int16` path against an explicit `int32` accumulation
   reference, and run the same check separately before re-enabling packed
   `int16` assembly.
4. Report:
   - NumPy time
   - Torch native `conv2d` time
   - QPU host prep
   - QPU cached total
   - QPU execute only
   - QPU prep+cached total
5. State clearly whether the comparison is cold-start or steady-state.

## Practical Summary

At the moment:

- `fp32` is reasonable but still behind Torch
- `int32` is promising and can beat Torch and NumPy in the tested steady-state
  benchmark
- public `int16` is correct via widening; packed `int16` assembly remains
  quarantined

The next serious implementation step is not cosmetic tuning. It is:

- fix and revalidate packed `int16`
- then reduce or remove `im2col`
- then specialize direct conv kernels for common shapes

## Packaged QPU-XLA W8A8 Candidates

The packaged runtime now also contains a separate W8A8 candidate family. It
does not change the public `tiledconv2d_int16` contract described above.

- `vc7.tiled_w8a8_gemm` consumes four signed INT8 values packed into every
  `uint32` word and uses signed `v8dot` to accumulate exactly into INT32.
- `vc7.tiled_w8a8_gemm_dequantize` fuses row/column FP32 scaling into the
  native-dot GEMM store path.
- Its logical tile is 16 output rows by 16 output columns by 16 reduction
  values. Host padding and packing are implemented by `PreparedW8A8Linear`.
- `vc7.w8a8_gemv` is the exact single-row decode candidate.
- `vc7.w8a8_dequantize` is an optional tiled INT32-to-FP32 scaling epilogue.
- `vc7.swiglu_fp32` is the fused FP32 SiLU-gate multiplication candidate.
- `Conv2dW8A8Plan` reuses the packed GEMM for dense and grouped convolution.
  Pointwise 1x1 avoids window expansion; general 3x3 still uses vectorized
  im2col and is not a direct spatial QPU kernel.
- Prepared dense and convolution plans support calibrated row and output
  hybrids. The two partition axes are mutually exclusive, and grouped output
  splits require 16 aligned outputs per group. Depthwise output splits are
  explicitly unsupported.

Every W8A8 assembly path has hardware differential tests. Automatic placement
must use an exact-shape `supported-win` record from `CandidateRegistry`.
Correct-but-slower GEMV, dequantization, and convolution candidates remain
available for explicit evaluation but must not be promoted by default.
Full one-layer model regressions also demote standalone stage wins; current
decode shapes and the 2048-hidden/256-token SwiGLU shape are CPU by default.
