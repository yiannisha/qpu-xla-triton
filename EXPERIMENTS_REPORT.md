# Experiments Report

Generated: 2026-04-16T13:57:07-04:00

## Scope

This report is a fresh full-suite rerun of the experiments previously used to characterize the repo.

Coverage in this rerun:

- infrastructure/introspection: `pctr_gpu_clock.py`, `payload.py`
- bandwidth baseline: `scopy.py`
- dense linear algebra: `sgemm.py`, `sgemm_fast.py`, `sgemm_batched_small.py`, `igemm.py`, `igemm_int16.py`
- tensor operators: `minmax.py`, `pool2d.py`, `tiledconv2d.py`, `tiledattention.py`, `tiledmlp.py`
- end-to-end pipeline: `tiledlenet5.py`

Important correction carried through this rerun: for scripts that expose `--num-qpus`, that flag refers to QPU cores, not a higher-level whole-QPU unit. In this suite that affects `minmax.py`, `pool2d.py`, and `tiledlenet5.py`.

The full raw log directory for this rerun is:

- `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun`

## Environment

```text
date=2026-04-16T13:51:03-04:00
pwd=/home/yiannis/side/py-videocore7
python=Python 3.12.13
git_commit=5c13e9714caa9524242affb59bbd8c6d3141082d
uname=Linux yiannis 6.12.75+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.12.75-1+rpt1 (2026-03-11) aarch64 GNU/Linux
```

## Status Summary

Total runs: 20

- Successes: 18
- Failures: 2

| Label | Exit | Duration (s) | Log |
|---|---:|---:|---|
| pctr_gpu_clock | 1 | 0 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/pctr_gpu_clock.log` |
| payload | 0 | 0 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/payload.log` |
| scopy | 0 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/scopy.log` |
| sgemm_default | 0 | 11 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_default.log` |
| sgemm_fast_default | 0 | 6 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_fast_default.log` |
| sgemm_batched_small_default | 0 | 4 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_batched_small_default.log` |
| sgemm_batched_small_batch8 | 0 | 3 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_batched_small_batch8.log` |
| sgemm_batched_small_size128_batch8 | 0 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_batched_small_size128_batch8.log` |
| igemm_default | 0 | 33 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/igemm_default.log` |
| igemm_int16_default | 0 | 32 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/igemm_int16_default.log` |
| minmax_default_core12_invalid | 1 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/minmax_default_core12_invalid.log` |
| minmax_core1 | 0 | 4 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/minmax_core1.log` |
| minmax_core12_validlen | 0 | 4 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/minmax_core12_validlen.log` |
| pool2d_default_core12 | 0 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/pool2d_default_core12.log` |
| pool2d_core1 | 0 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/pool2d_core1.log` |
| tiledattention_default | 0 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledattention_default.log` |
| tiledconv2d_default | 0 | 2 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledconv2d_default.log` |
| tiledlenet5_default_core12 | 0 | 4 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledlenet5_default_core12.log` |
| tiledlenet5_core1 | 0 | 6 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledlenet5_core1.log` |
| tiledmlp_default | 0 | 48 | `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledmlp_default.log` |

## Failure Notes

- `pctr_gpu_clock` fails because `examples/pctr_gpu_clock.py` opens `/dev/mem`, which is not permitted in the current user context.
- `minmax_default_core12_invalid` still fails before execution because the script’s built-in default length is not a valid multiple of the 12-core fp32 chunk requirement.

## Quick Read

- The full suite now reflects one internally consistent rerun rather than a mix of older logs and corrected subset logs.
- The `--num-qpus` wording is corrected throughout the affected runs to mean `1 QPU core` or `12 QPU cores`.
- `scopy.py` still shows the expected built-in scaling from 1 thread to 12 threads, with the 12-thread copy path approaching the CPU copy baseline.
- `sgemm_fast.py` improves throughput over the naive SGEMM kernel, but the fast variants still report `NaN`/`Inf` outputs in this live rerun.
- Several higher-level ML paths in the current snapshot still show large max errors or `NaN` outputs in live benchmarking, notably `tiledconv2d.py`, `tiledattention.py`, `tiledmlp.py`, and `tiledlenet5.py`. The performance story and the correctness story should therefore be presented separately.

## Raw Outputs

### pctr_gpu_clock

- Notes: V3D clock counter probe; currently fails without `/dev/mem` access.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/pctr_gpu_clock.py`
- Exit status: 1
- Duration: 0s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/pctr_gpu_clock.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/pctr_gpu_clock.py
Traceback (most recent call last):
  File "/home/yiannis/side/py-videocore7/examples/pctr_gpu_clock.py", line 26, in <module>
    with RegisterMapping() as reg:
         ^^^^^^^^^^^^^^^^^
  File "/home/yiannis/side/py-videocore7/src/_videocore7/v3d.py", line 510, in __init__
    fd = os.open("/dev/mem", os.O_RDWR)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: '/dev/mem'
```

### payload

- Notes: Payload/workgroup introspection dump.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/payload.py`
- Exit status: 0
- Duration: 0s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/payload.log`
- Output in this markdown is truncated to the first 120 lines to avoid bloating the appendix; the full log remains at the path above.

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/payload.py
[[[[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 2  2  2  2  2  2  2  2  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  3  3  3  3  3  3  3  3]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 4  4  4  4  4  4  4  4  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  5  5  5  5  5  5  5  5]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 6  6  6  6  6  6  6  6  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  7  7  7  7  7  7  7  7]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 8  8  8  8  8  8  8  8  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  9  9  9  9  9  9  9  9]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[10 10 10 10 10 10 10 10  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0 11 11 11 11 11 11 11 11]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[12 12 12 12 12 12 12 12  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0 13 13 13 13 13 13 13 13]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[14 14 14 14 14 14 14 14  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]]


  [[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0]
    [ 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  2  2  2  2  2  2  2  2]
    [ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 3  3  3  3  3  3  3  3  0  0  0  0  0  0  0  0]
    [ 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  4  4  4  4  4  4  4  4]
    [ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 5  5  5  5  5  5  5  5  0  0  0  0  0  0  0  0]
    [ 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  6  6  6  6  6  6  6  6]
    [ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  1  2  3  4  5  6  7]]

   [[ 7  7  7  7  7  7  7  7  0  0  0  0  0  0  0  0]
    [ 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 0  1  2  3  4  5  6  7  0  0  0  0  0  0  0  0]]

   [[ 0  0  0  0  0  0  0  0  8  8  8  8  8  8  8  8]
    [ 0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
...
[truncated after 120 lines]
```

### scopy

- Notes: Built-in CPU vs 1-thread vs 12-thread copy benchmark.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/scopy.py`
- Exit status: 0
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/scopy.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/scopy.py
==== CPU scopy example (24.0 Mi elements) ====
0.021210130071267486 sec, 4746.000880794434 MB/s
==== QPU 1 thread scopy example (24.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.057327570975758135 sec, 1755.931644872361 MB/s
==== QPU 12 threads scopy example (24.0 Mi elements) ====
Preparing for buffers...
Executing on QPU...
0.024132244056090713 sec, 4171.319325547501 MB/s
```

### sgemm_default

- Notes: Naive FP32 SGEMM sweeps, including batched single-dispatch cases in the current snapshot.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm.py`
- Exit status: 0
- Duration: 11s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm.py
/home/yiannis/side/py-videocore7/examples/sgemm.py:661: RuntimeWarning: overflow encountered in divide
  print(f"Maximum relative error: {np.max(np.abs((c - expected) / expected))}")
/home/yiannis/side/py-videocore7/examples/sgemm.py:673: RuntimeWarning: overflow encountered in divide
  "max_rel_error": float(np.max(np.abs((c - expected) / expected))),

==== sgemm example (256x256 times 256x256) ====
numpy: 0.0012 sec, 28.7423 Gflop/s
torch: 0.0201 sec, 1.6796 Gflop/s
QPU:   0.0053 sec, 6.3976 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 143.4606170654297
Minimum relative error: 0.0
Maximum relative error: 4197.4638671875

==== sgemm example (512x512 times 512x512) ====
numpy: 0.0052 sec, 51.7915 Gflop/s
torch: 0.0149 sec, 18.1111 Gflop/s
QPU:   0.0123 sec, 21.9670 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 229.41651916503906
Minimum relative error: 0.0
Maximum relative error: 10315.4462890625

==== sgemm example (768x768 times 768x768) ====
numpy: 0.0157 sec, 57.8879 Gflop/s
torch: 0.0433 sec, 20.9660 Gflop/s
QPU:   0.0429 sec, 21.1441 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 315.7813415527344
Minimum relative error: 0.0
Maximum relative error: 114232.3828125

==== sgemm example (1024x1024 times 1024x1024) ====
numpy: 0.0386 sec, 55.6800 Gflop/s
torch: 0.0446 sec, 48.2101 Gflop/s
QPU:   0.0984 sec, 21.8613 Gflop/s
Minimum absolute error: 0.0
Maximum absolute error: 330.2965393066406
Minimum relative error: 0.0
Maximum relative error: 11605.98046875

==== sgemm example (1536x1536 times 1536x1536) ====
numpy: 0.1119 sec, 64.8354 Gflop/s
torch: 0.1095 sec, 66.2521 Gflop/s
QPU:   0.3214 sec, 22.5735 Gflop/s
Minimum absolute error: nan
Maximum absolute error: nan
Minimum relative error: nan
Maximum relative error: nan

==== sgemm example (2048x2048 times 2048x2048) ====
numpy: 0.2414 sec, 71.2270 Gflop/s
torch: 0.2032 sec, 84.6010 Gflop/s
QPU:   0.7577 sec, 22.6912 Gflop/s
Minimum absolute error: nan
Maximum absolute error: nan
Minimum relative error: nan
Maximum relative error: nan

==== sgemm size sweep summary ====
  size   numpy GF/s   torch GF/s     QPU GF/s    QPU sec    max abs err
   256      28.7423       1.6796       6.3976     0.0053        143.461
   512      51.7915      18.1111      21.9670     0.0123        229.417
   768      57.8879      20.9660      21.1441     0.0429        315.781
  1024      55.6800      48.2101      21.8613     0.0984        330.297
  1536      64.8354      66.2521      22.5735     0.3214            nan
  2048      71.2270      84.6010      22.6912     0.7577            nan


==== batched sgemm sweep for size 128 ====
==== batched sgemm example (single dispatch, batch=1, 128x128 times 128x128) ====
numpy: 0.0002 sec, 17.5043 Gflop/s
torch: 0.0136 sec, 0.3110 Gflop/s
QPU:   0.0039 sec, 1.0882 Gflop/s
Maximum absolute error: 101.7644271850586
Maximum relative error: 221.2483367919922

==== batched sgemm example (single dispatch, batch=2, 128x128 times 128x128) ====
numpy: 0.0004 sec, 21.1402 Gflop/s
torch: 0.0137 sec, 0.6177 Gflop/s
QPU:   0.0006 sec, 14.9327 Gflop/s
Maximum absolute error: 87.40705108642578
Maximum relative error: 11870.302734375

==== batched sgemm example (single dispatch, batch=4, 128x128 times 128x128) ====
numpy: 0.0005 sec, 33.8377 Gflop/s
torch: 0.0335 sec, 0.5063 Gflop/s
QPU:   0.0012 sec, 14.7562 Gflop/s
Maximum absolute error: 104.28404998779297
Maximum relative error: 4398.39013671875

==== batched sgemm example (single dispatch, batch=8, 128x128 times 128x128) ====
numpy: 0.0008 sec, 42.0284 Gflop/s
torch: 0.0684 sec, 0.4965 Gflop/s
QPU:   0.0103 sec, 3.2891 Gflop/s
Maximum absolute error: 103.19575500488281
Maximum relative error: 10959.978515625

==== batched sgemm example (single dispatch, batch=16, 128x128 times 128x128) ====
numpy: 0.0015 sec, 46.3285 Gflop/s
torch: 0.0833 sec, 0.8152 Gflop/s
QPU:   0.0072 sec, 9.3662 Gflop/s
Maximum absolute error: 112.175537109375
Maximum relative error: 32516.771484375

==== batched sgemm example (single dispatch, batch=32, 128x128 times 128x128) ====
numpy: 0.0030 sec, 45.9510 Gflop/s
torch: 0.0889 sec, 1.5269 Gflop/s
QPU:   0.0102 sec, 13.2917 Gflop/s
Maximum absolute error: 113.43132019042969
Maximum relative error: 11617.388671875


==== batched sgemm sweep for size 256 ====
==== batched sgemm example (single dispatch, batch=1, 256x256 times 256x256) ====
numpy: 0.0012 sec, 28.1727 Gflop/s
torch: 0.0211 sec, 1.5963 Gflop/s
QPU:   0.0036 sec, 9.3026 Gflop/s
Maximum absolute error: 140.39231872558594
Maximum relative error: 4197.4638671875

==== batched sgemm example (single dispatch, batch=2, 256x256 times 256x256) ====
numpy: 0.0152 sec, 4.4348 Gflop/s
torch: 0.0301 sec, 2.2422 Gflop/s
QPU:   0.0120 sec, 5.6346 Gflop/s
Maximum absolute error: 143.51516723632812
Maximum relative error: 691.0121459960938

==== batched sgemm example (single dispatch, batch=4, 256x256 times 256x256) ====
numpy: 0.0026 sec, 51.4068 Gflop/s
torch: 0.0320 sec, 4.2197 Gflop/s
QPU:   0.0063 sec, 21.2797 Gflop/s
Maximum absolute error: 164.609375
Maximum relative error: 20902.5703125

==== batched sgemm example (single dispatch, batch=8, 256x256 times 256x256) ====
numpy: 0.0049 sec, 54.6729 Gflop/s
torch: 0.0776 sec, 3.4810 Gflop/s
QPU:   0.0160 sec, 16.9093 Gflop/s
Maximum absolute error: 178.93585205078125
Maximum relative error: 6498.12353515625

==== batched sgemm example (single dispatch, batch=16, 256x256 times 256x256) ====
numpy: 0.0104 sec, 51.9291 Gflop/s
torch: 0.0928 sec, 5.8173 Gflop/s
QPU:   0.0283 sec, 19.0757 Gflop/s
Maximum absolute error: inf
Maximum relative error: inf

==== batched sgemm example (single dispatch, batch=32, 256x256 times 256x256) ====
numpy: 0.0267 sec, 40.5255 Gflop/s
torch: 0.1032 sec, 10.4647 Gflop/s
QPU:   0.0529 sec, 20.4216 Gflop/s
Maximum absolute error: 186.0936279296875
Maximum relative error: 56909.6015625


==== batched sgemm sweep for size 512 ====
==== batched sgemm example (single dispatch, batch=1, 512x512 times 512x512) ====
numpy: 0.0051 sec, 52.4517 Gflop/s
torch: 0.0288 sec, 9.3334 Gflop/s
QPU:   0.0123 sec, 21.9666 Gflop/s
Maximum absolute error: 234.5069580078125
Maximum relative error: 10315.4462890625

==== batched sgemm example (single dispatch, batch=2, 512x512 times 512x512) ====
numpy: 0.0098 sec, 55.1763 Gflop/s
torch: 0.0583 sec, 9.2365 Gflop/s
QPU:   0.0277 sec, 19.4419 Gflop/s
Maximum absolute error: 222.89633178710938
Maximum relative error: 5525.31103515625

==== batched sgemm example (single dispatch, batch=4, 512x512 times 512x512) ====
numpy: 0.0213 sec, 50.4419 Gflop/s
torch: 0.0849 sec, 12.6817 Gflop/s
QPU:   0.0517 sec, 20.8213 Gflop/s
Maximum absolute error: 243.04116821289062
Maximum relative error: 66504.9375

==== batched sgemm example (single dispatch, batch=8, 512x512 times 512x512) ====
numpy: 0.0430 sec, 50.0881 Gflop/s
torch: 0.1043 sec, 20.6558 Gflop/s
QPU:   0.0996 sec, 21.6302 Gflop/s
Maximum absolute error: 246.42210388183594
Maximum relative error: 114611.6953125

==== batched sgemm example (single dispatch, batch=16, 512x512 times 512x512) ====
numpy: 0.0786 sec, 54.8226 Gflop/s
torch: 0.1319 sec, 32.6504 Gflop/s
QPU:   0.1954 sec, 22.0425 Gflop/s
Maximum absolute error: 253.0756072998047
Maximum relative error: 55175.33203125

==== batched sgemm example (single dispatch, batch=32, 512x512 times 512x512) ====
numpy: 0.1592 sec, 54.1155 Gflop/s
torch: 0.1915 sec, 44.9975 Gflop/s
QPU:   0.3869 sec, 22.2677 Gflop/s
Maximum absolute error: inf
Maximum relative error: inf

==== batched sgemm summary ====
  size  batch   numpy GF/s   torch GF/s     QPU GF/s    QPU sec
   128      1      17.5043       0.3110       1.0882     0.0039
   128      2      21.1402       0.6177      14.9327     0.0006
   128      4      33.8377       0.5063      14.7562     0.0012
   128      8      42.0284       0.4965       3.2891     0.0103
   128     16      46.3285       0.8152       9.3662     0.0072
   128     32      45.9510       1.5269      13.2917     0.0102
   256      1      28.1727       1.5963       9.3026     0.0036
   256      2       4.4348       2.2422       5.6346     0.0120
   256      4      51.4068       4.2197      21.2797     0.0063
   256      8      54.6729       3.4810      16.9093     0.0160
   256     16      51.9291       5.8173      19.0757     0.0283
   256     32      40.5255      10.4647      20.4216     0.0529
   512      1      52.4517       9.3334      21.9666     0.0123
   512      2      55.1763       9.2365      19.4419     0.0277
   512      4      50.4419      12.6817      20.8213     0.0517
   512      8      50.0881      20.6558      21.6302     0.0996
   512     16      54.8226      32.6504      22.0425     0.1954
   512     32      54.1155      44.9975      22.2677     0.3869
```

### sgemm_fast_default

- Notes: Optimized FP32 SGEMM variants vs naive baseline.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_fast.py`
- Exit status: 0
- Duration: 6s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_fast_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_fast.py
==== fast sgemm example (1024x1024 times 1024x1024) ====
numpy: 0.0377 sec, 57.0699 Gflop/s
torch: 0.0730 sec, 29.4604 Gflop/s
QPU naive: 0.1146 sec, 18.7701 Gflop/s
QPU fast payload:    0.0948 sec, 22.6760 Gflop/s
QPU fast qpu-aware:  0.0975 sec, 22.0471 Gflop/s
Speedup over naive: 1.208x
Naive NaN count: 0
Naive Inf count: 0
Naive minimum absolute error: 0.0
Naive maximum absolute error: 333.2373046875
Naive minimum relative error: 0.0
Naive maximum relative error: 11605.98046875
Fast payload NaN count: 416
Fast payload Inf count: 188
Fast payload minimum absolute error: 0.00018310546875
Fast payload maximum absolute error: inf
Fast payload minimum relative error: 2.2590065782424062e-06
Fast payload maximum relative error: inf
Fast qpu-aware NaN count: 256
Fast qpu-aware Inf count: 306
Fast qpu-aware minimum absolute error: 0.00018310546875
Fast qpu-aware maximum absolute error: inf
Fast qpu-aware minimum relative error: 2.2590065782424062e-06
Fast qpu-aware maximum relative error: inf
```

### sgemm_batched_small_default

- Notes: Specialized small batched SGEMM sweep.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_batched_small.py`
- Exit status: 0
- Duration: 4s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_batched_small_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_batched_small.py

==== small batched sgemm (128x128, batch=1) ====
numpy:        0.0003 sec, 15.9242 Gflop/s
torch:        0.0113 sec, 0.3760 Gflop/s
QPU generic:  0.0003 sec, 14.2763 Gflop/s (median of 5)
QPU small:    0.0003 sec, 14.1458 Gflop/s (median of 5)
Generic max abs error: 101.39855194091797
Small max abs error:   101.39855194091797

==== small batched sgemm (128x128, batch=2) ====
numpy:        0.0003 sec, 28.1389 Gflop/s
torch:        0.0189 sec, 0.4488 Gflop/s
QPU generic:  0.0005 sec, 17.3795 Gflop/s (median of 5)
QPU small:    0.0005 sec, 17.4477 Gflop/s (median of 5)
Generic max abs error: 85.41255187988281
Small max abs error:   87.40705108642578

==== small batched sgemm (128x128, batch=4) ====
numpy:        0.0005 sec, 37.5809 Gflop/s
torch:        0.0310 sec, 0.5473 Gflop/s
QPU generic:  0.0009 sec, 18.8488 Gflop/s (median of 5)
QPU small:    0.0009 sec, 18.8364 Gflop/s (median of 5)
Generic max abs error: 104.6340103149414
Small max abs error:   104.6340103149414

==== small batched sgemm (128x128, batch=8) ====
numpy:        0.0009 sec, 38.6310 Gflop/s
torch:        0.0670 sec, 0.5069 Gflop/s
QPU generic:  0.0017 sec, 19.8977 Gflop/s (median of 5)
QPU small:    0.0017 sec, 19.8761 Gflop/s (median of 5)
Generic max abs error: 102.7874755859375
Small max abs error:   103.19575500488281

==== small batched sgemm (128x128, batch=16) ====
numpy:        0.0129 sec, 5.2795 Gflop/s
torch:        0.0838 sec, 0.8101 Gflop/s
QPU generic:  0.0033 sec, 20.2749 Gflop/s (median of 5)
QPU small:    0.0033 sec, 20.2848 Gflop/s (median of 5)
Generic max abs error: 108.17369079589844
Small max abs error:   112.175537109375

==== small batched sgemm (256x256, batch=1) ====
numpy:        0.0012 sec, 28.9229 Gflop/s
torch:        0.0163 sec, 2.0654 Gflop/s
QPU generic:  0.0016 sec, 20.4779 Gflop/s (median of 5)
QPU small:    0.0016 sec, 20.5187 Gflop/s (median of 5)
Generic max abs error: 146.27500915527344
Small max abs error:   146.27500915527344

==== small batched sgemm (256x256, batch=2) ====
numpy:        0.0016 sec, 42.9296 Gflop/s
torch:        0.0203 sec, 3.3300 Gflop/s
QPU generic:  0.0032 sec, 21.3414 Gflop/s (median of 5)
QPU small:    0.0032 sec, 21.3219 Gflop/s (median of 5)
Generic max abs error: 143.51516723632812
Small max abs error:   143.51516723632812

==== small batched sgemm (256x256, batch=4) ====
numpy:        0.0027 sec, 50.1504 Gflop/s
torch:        0.0403 sec, 3.3535 Gflop/s
QPU generic:  0.0063 sec, 21.5302 Gflop/s (median of 5)
QPU small:    0.0063 sec, 21.5441 Gflop/s (median of 5)
Generic max abs error: 164.609375
Small max abs error:   164.609375

==== small batched sgemm (256x256, batch=8) ====
numpy:        0.0054 sec, 49.8810 Gflop/s
torch:        0.0596 sec, 4.5302 Gflop/s
QPU generic:  0.0124 sec, 21.7773 Gflop/s (median of 5)
QPU small:    0.0124 sec, 21.7746 Gflop/s (median of 5)
Generic max abs error: 178.93585205078125
Small max abs error:   178.93585205078125

==== small batched sgemm (256x256, batch=16) ====
numpy:        0.0098 sec, 54.8924 Gflop/s
torch:        0.0871 sec, 6.2013 Gflop/s
QPU generic:  0.0247 sec, 21.8635 Gflop/s (median of 5)
QPU small:    0.0247 sec, 21.8632 Gflop/s (median of 5)
Generic max abs error: 186.0501708984375
Small max abs error:   186.0501708984375

==== small batched sgemm summary ====
  size  batch   numpy GF/s   torch GF/s   generic GF/s   small GF/s   speedup
   128      1      15.9242       0.3760        14.2763      14.1458     0.991x
   128      2      28.1389       0.4488        17.3795      17.4477     1.004x
   128      4      37.5809       0.5473        18.8488      18.8364     0.999x
   128      8      38.6310       0.5069        19.8977      19.8761     0.999x
   128     16       5.2795       0.8101        20.2749      20.2848     1.000x
   256      1      28.9229       2.0654        20.4779      20.5187     1.002x
   256      2      42.9296       3.3300        21.3414      21.3219     0.999x
   256      4      50.1504       3.3535        21.5302      21.5441     1.001x
   256      8      49.8810       4.5302        21.7773      21.7746     1.000x
   256     16      54.8924       6.2013        21.8635      21.8632     1.000x
```

### sgemm_batched_small_batch8

- Notes: Specialized small batched SGEMM restricted to batch 8.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_batched_small.py --batch 8`
- Exit status: 0
- Duration: 3s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_batched_small_batch8.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_batched_small.py --batch 8

==== small batched sgemm (128x128, batch=8) ====
numpy:        0.0010 sec, 32.8535 Gflop/s
torch:        0.0529 sec, 0.6417 Gflop/s
QPU generic:  0.0017 sec, 19.7978 Gflop/s (median of 5)
QPU small:    0.0017 sec, 19.7995 Gflop/s (median of 5)
Generic max abs error: 102.7874755859375
Small max abs error:   103.19575500488281

==== small batched sgemm (256x256, batch=8) ====
numpy:        0.0166 sec, 16.3088 Gflop/s
torch:        0.0616 sec, 4.3830 Gflop/s
QPU generic:  0.0124 sec, 21.7934 Gflop/s (median of 5)
QPU small:    0.0124 sec, 21.7923 Gflop/s (median of 5)
Generic max abs error: 178.93585205078125
Small max abs error:   178.93585205078125

==== small batched sgemm summary ====
  size  batch   numpy GF/s   torch GF/s   generic GF/s   small GF/s   speedup
   128      8      32.8535       0.6417        19.7978      19.7995     1.000x
   256      8      16.3088       4.3830        21.7934      21.7923     1.000x
```

### sgemm_batched_small_size128_batch8

- Notes: Specialized small batched SGEMM at size 128, batch 8.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_batched_small.py --size 128 --batch 8`
- Exit status: 0
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/sgemm_batched_small_size128_batch8.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/sgemm_batched_small.py --size 128 --batch 8
==== small batched sgemm (128x128, batch=8) ====
numpy:        0.0009 sec, 37.9041 Gflop/s
torch:        0.0365 sec, 0.9304 Gflop/s
QPU generic:  0.0017 sec, 19.8610 Gflop/s (median of 5)
QPU small:    0.0017 sec, 19.8727 Gflop/s (median of 5)
Generic max abs error: 102.7874755859375
Small max abs error:   102.7874755859375
```

### igemm_default

- Notes: INT32 GEMM sweep.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/igemm.py`
- Exit status: 0
- Duration: 33s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/igemm_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/igemm.py

==== igemm example (256x256 times 256x256) ====
Kernel contract: inputs must fit in signed 24-bit integers because it uses smul24.
numpy: 0.0284 sec, 1.1796 Gop/s
QPU:   0.0053 sec, 6.3012 Gop/s
Maximum absolute error: 1620066157

==== igemm example (512x512 times 512x512) ====
Kernel contract: inputs must fit in signed 24-bit integers because it uses smul24.
numpy: 0.4742 sec, 0.5661 Gop/s
QPU:   0.0161 sec, 16.6905 Gop/s
Maximum absolute error: 1032671774

==== igemm example (768x768 times 768x768) ====
Kernel contract: inputs must fit in signed 24-bit integers because it uses smul24.
numpy: 1.6027 sec, 0.5653 Gop/s
QPU:   0.0442 sec, 20.4908 Gop/s
Maximum absolute error: 958161664

==== igemm example (1024x1024 times 1024x1024) ====
Kernel contract: inputs must fit in signed 24-bit integers because it uses smul24.
numpy: 10.0292 sec, 0.2141 Gop/s
QPU:   0.2001 sec, 10.7326 Gop/s
Maximum absolute error: 882835775

==== igemm size sweep summary ====
  size   numpy GO/s     QPU GO/s    QPU sec    max abs err
   256       1.1796       6.3012     0.0053     1620066157
   512       0.5661      16.6905     0.0161     1032671774
   768       0.5653      20.4908     0.0442      958161664
  1024       0.2141      10.7326     0.2001      882835775
```

### igemm_int16_default

- Notes: Packed INT16-input GEMM with INT32 accumulation sweep.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/igemm_int16.py`
- Exit status: 0
- Duration: 32s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/igemm_int16_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/igemm_int16.py

==== packed int16 igemm example (256x256 times 256x256) ====
Kernel contract: A/B are packed int16 pairs in int32 words; accumulation is signed int32 via smul24.
This reduces input bandwidth, but it is not a packed int8/int16 dot-product kernel.
numpy: 0.0288 sec, 1.1666 Gop/s
QPU:   0.0053 sec, 6.2981 Gop/s
Maximum absolute error: 62050486

==== packed int16 igemm example (512x512 times 512x512) ====
Kernel contract: A/B are packed int16 pairs in int32 words; accumulation is signed int32 via smul24.
This reduces input bandwidth, but it is not a packed int8/int16 dot-product kernel.
numpy: 0.4739 sec, 0.5664 Gop/s
QPU:   0.0160 sec, 16.8201 Gop/s
Maximum absolute error: 79513364

==== packed int16 igemm example (768x768 times 768x768) ====
Kernel contract: A/B are packed int16 pairs in int32 words; accumulation is signed int32 via smul24.
This reduces input bandwidth, but it is not a packed int8/int16 dot-product kernel.
numpy: 1.4990 sec, 0.6044 Gop/s
QPU:   0.0442 sec, 20.4931 Gop/s
Maximum absolute error: 86806904

==== packed int16 igemm example (1024x1024 times 1024x1024) ====
Kernel contract: A/B are packed int16 pairs in int32 words; accumulation is signed int32 via smul24.
This reduces input bandwidth, but it is not a packed int8/int16 dot-product kernel.
numpy: 9.3535 sec, 0.2296 Gop/s
QPU:   0.0991 sec, 21.6689 Gop/s
Maximum absolute error: 100577955

==== packed int16 igemm size sweep summary ====
  size   numpy GO/s     QPU GO/s    QPU sec    max abs err
   256       1.1666       6.2981     0.0053       62050486
   512       0.5664      16.8201     0.0160       79513364
   768       0.6044      20.4931     0.0442       86806904
  1024       0.2296      21.6689     0.0991      100577955
```

### minmax_default_core12_invalid

- Notes: Default `minmax.py` path, where `--num-qpus` implicitly means 12 QPU cores.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/minmax.py`
- Exit status: 1
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/minmax_default_core12_invalid.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/minmax.py
Traceback (most recent call last):
  File "/home/yiannis/side/py-videocore7/examples/minmax.py", line 604, in <module>
    main()
  File "/home/yiannis/side/py-videocore7/examples/minmax.py", line 583, in main
    iterations_by_dtype = validate_length(args.length, args.num_qpus, configs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yiannis/side/py-videocore7/examples/minmax.py", line 352, in validate_length
    raise ValueError(f"length must be a multiple of {chunk} for dtype={config.name} and num_qpus={num_qpus}")
ValueError: length must be a multiple of 192 for dtype=fp32 and num_qpus=12
```

### minmax_core1

- Notes: Explicit 1-core `minmax.py` run.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/minmax.py --num-qpus 1`
- Exit status: 0
- Duration: 4s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/minmax_core1.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/minmax.py --num-qpus 1
length: 4194304 scalar values
repeat: 5
dtypes: fp32, int32, int16

==== fp32 elementwise min/max (4.0 Mi elements, 1 QPU) ====
numpy:     0.004212 sec, 11.128 GiB/s
torch:     0.006710 sec, 6.986 GiB/s
QPU:       0.044090 sec, 1.063 GiB/s

numpy:     0.004219 sec, 11.110 GiB/s
torch:     0.006745 sec, 6.950 GiB/s
QPU:       0.044078 sec, 1.063 GiB/s

==== int32 elementwise min/max (4.0 Mi elements, 1 QPU) ====
numpy:     0.007117 sec, 6.586 GiB/s
torch:     0.007731 sec, 6.063 GiB/s
QPU:       0.044094 sec, 1.063 GiB/s

numpy:     0.006938 sec, 6.757 GiB/s
torch:     0.007647 sec, 6.130 GiB/s
QPU:       0.044096 sec, 1.063 GiB/s

==== int16 elementwise min/max (4.0 Mi elements, 1 QPU) ====
numpy:     0.003376 sec, 6.942 GiB/s
torch:     0.003716 sec, 6.307 GiB/s
QPU:       0.025360 sec, 0.924 GiB/s

numpy:     0.003369 sec, 6.958 GiB/s
torch:     0.003678 sec, 6.372 GiB/s
QPU:       0.025348 sec, 0.925 GiB/s
```

### minmax_core12_validlen

- Notes: Explicit 12-core `minmax.py` rerun with a valid length.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/minmax.py --num-qpus 12 --length 4194432`
- Exit status: 0
- Duration: 4s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/minmax_core12_validlen.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/minmax.py --num-qpus 12 --length 4194432
length: 4194432 scalar values
repeat: 5
dtypes: fp32, int32, int16

==== fp32 elementwise min/max (4.0 Mi elements, 12 QPUs) ====
numpy:     0.004584 sec, 10.227 GiB/s
torch:     0.006766 sec, 6.928 GiB/s
QPU:       0.008162 sec, 5.743 GiB/s

numpy:     0.004323 sec, 10.843 GiB/s
torch:     0.006683 sec, 7.014 GiB/s
QPU:       0.007107 sec, 6.595 GiB/s

==== int32 elementwise min/max (4.0 Mi elements, 12 QPUs) ====
numpy:     0.007479 sec, 6.268 GiB/s
torch:     0.007600 sec, 6.168 GiB/s
QPU:       0.007261 sec, 6.456 GiB/s

numpy:     0.007477 sec, 6.269 GiB/s
torch:     0.007654 sec, 6.125 GiB/s
QPU:       0.007216 sec, 6.496 GiB/s

==== int16 elementwise min/max (4.0 Mi elements, 12 QPUs) ====
numpy:     0.003491 sec, 6.714 GiB/s
torch:     0.003746 sec, 6.257 GiB/s
QPU:       0.003907 sec, 5.998 GiB/s

numpy:     0.003504 sec, 6.690 GiB/s
torch:     0.003762 sec, 6.230 GiB/s
QPU:       0.003891 sec, 6.024 GiB/s
```

### pool2d_default_core12

- Notes: Default `pool2d.py` path, meaning 12 QPU cores.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/pool2d.py`
- Exit status: 0
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/pool2d_default_core12.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/pool2d.py
Pool contract: NCHW, kernel=2x2, stride=2, padding=0.
Integer avgpool uses truncation toward zero to match the Torch baseline.
dtypes: fp32, int32, int16

==== fp32 2x2/2 pool (1x32x144x144 -> 1x32x72x72, 12 QPUs) ====
maxpool:
numpy:     0.001162 sec, 2.659 GiB/s
torch:     0.001020 sec, 3.030 GiB/s
QPU:       0.001165 sec, 2.651 GiB/s

avgpool:
numpy:     0.001296 sec, 2.384 GiB/s
torch:     0.001242 sec, 2.488 GiB/s
QPU:       0.001164 sec, 2.654 GiB/s


==== int32 2x2/2 pool (1x32x144x144 -> 1x32x72x72, 12 QPUs) ====
maxpool:
numpy:     0.001173 sec, 2.634 GiB/s
torch:     0.001011 sec, 3.055 GiB/s
QPU:       0.001222 sec, 2.528 GiB/s

avgpool:
numpy:     0.004288 sec, 0.721 GiB/s
torch:     0.004460 sec, 0.693 GiB/s
QPU:       0.001181 sec, 2.617 GiB/s


==== int16 2x2/2 pool (1x32x144x144 -> 1x32x72x72, 12 QPUs) ====
maxpool:
numpy:     0.000411 sec, 3.758 GiB/s
torch:     0.000379 sec, 4.077 GiB/s
QPU:       0.001095 sec, 1.411 GiB/s

avgpool:
numpy:     0.003747 sec, 0.412 GiB/s
torch:     0.003786 sec, 0.408 GiB/s
QPU:       0.001112 sec, 1.389 GiB/s

```

### pool2d_core1

- Notes: Explicit 1-core `pool2d.py` run.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/pool2d.py --num-qpus 1`
- Exit status: 0
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/pool2d_core1.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/pool2d.py --num-qpus 1
Pool contract: NCHW, kernel=2x2, stride=2, padding=0.
Integer avgpool uses truncation toward zero to match the Torch baseline.
dtypes: fp32, int32, int16

==== fp32 2x2/2 pool (1x32x144x144 -> 1x32x72x72, 1 QPU) ====
maxpool:
numpy:     0.001195 sec, 2.586 GiB/s
torch:     0.001029 sec, 3.004 GiB/s
QPU:       0.005156 sec, 0.599 GiB/s

avgpool:
numpy:     0.001334 sec, 2.317 GiB/s
torch:     0.001252 sec, 2.468 GiB/s
QPU:       0.005189 sec, 0.595 GiB/s


==== int32 2x2/2 pool (1x32x144x144 -> 1x32x72x72, 1 QPU) ====
maxpool:
numpy:     0.001208 sec, 2.559 GiB/s
torch:     0.001041 sec, 2.967 GiB/s
QPU:       0.005143 sec, 0.601 GiB/s

avgpool:
numpy:     0.004226 sec, 0.731 GiB/s
torch:     0.004435 sec, 0.697 GiB/s
QPU:       0.005321 sec, 0.581 GiB/s


==== int16 2x2/2 pool (1x32x144x144 -> 1x32x72x72, 1 QPU) ====
maxpool:
numpy:     0.000413 sec, 3.743 GiB/s
torch:     0.000359 sec, 4.300 GiB/s
QPU:       0.003340 sec, 0.462 GiB/s

avgpool:
numpy:     0.003220 sec, 0.480 GiB/s
torch:     0.003763 sec, 0.411 GiB/s
QPU:       0.003601 sec, 0.429 GiB/s

```

### tiledattention_default

- Notes: Tiled attention core benchmark.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledattention.py`
- Exit status: 0
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledattention_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledattention.py
==== tiledattention fp32 example ====
Operator: single-head dot-product attention core O = (Q @ K^T) @ V
Dimensions: q=128x128, k=128x128, v=128x128
Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.
QPU setup (excluded): 0.0199 sec
-- Score stage (Q @ K^T) --
numpy: 0.0001 sec, 78.6703 Gop/s
torch matmul: 0.0012 sec, 3.5259 Gop/s
QPU host prep: 0.0000 sec
QPU cached total: 0.0004 sec, 11.3486 Gop/s
QPU execute only: 0.0003 sec, 14.2899 Gop/s
QPU prep+cached total: 0.0004 sec, 10.1832 Gop/s
Maximum absolute error: 19.59319281578064

-- Value stage (Scores @ V, reference scores) --
numpy: 0.0001 sec, 80.4600 Gop/s
torch matmul: 0.0038 sec, 1.1171 Gop/s
QPU host prep: 0.0000 sec
QPU cached total: 0.0004 sec, 11.7580 Gop/s
QPU execute only: 0.0003 sec, 14.2926 Gop/s
QPU prep+cached total: 0.0004 sec, 11.3208 Gop/s
Maximum absolute error: 514.3143844604492

-- Attention total --
numpy: 0.0001 sec, 82.3462 Gop/s
torch matmul attention core: 0.0016 sec, 5.2768 Gop/s
torch native sdpa: 0.0014 sec, 5.9664 Gop/s
torch native sdpa note: includes softmax with scale=1.0; speed baseline only, not a correctness reference.
QPU host prep: 0.0000 sec
QPU cached total: 0.0007 sec, 12.7380 Gop/s
QPU execute only: 0.0006 sec, 14.3138 Gop/s
QPU prep+cached total: 0.0007 sec, 12.0361 Gop/s
Maximum absolute error: 547.2508546113968


==== tiledattention int32 example ====
Operator: single-head dot-product attention core O = (Q @ K^T) @ V
Dimensions: q=128x128, k=128x128, v=128x128
Kernel contract: q, k, and v must fit the signed 24-bit range, and the intermediate score matrix must also stay within it.
Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.
QPU setup (excluded): 0.0171 sec
-- Score stage (Q @ K^T) --
numpy: 0.0054 sec, 0.7782 Gop/s
torch int32 matmul: 0.0041 sec, 1.0283 Gop/s
QPU host prep: 0.0001 sec
QPU cached total: 0.0004 sec, 11.4115 Gop/s
QPU execute only: 0.0003 sec, 14.3080 Gop/s
QPU prep+cached total: 0.0005 sec, 8.3593 Gop/s
Maximum absolute error: 1563.0

-- Value stage (Scores @ V, reference scores) --
numpy: 0.0020 sec, 2.1040 Gop/s
torch int32 matmul: 0.0006 sec, 6.9953 Gop/s
QPU host prep: 0.0001 sec
QPU cached total: 0.0004 sec, 11.7930 Gop/s
QPU execute only: 0.0003 sec, 14.3315 Gop/s
QPU prep+cached total: 0.0005 sec, 9.2500 Gop/s
Maximum absolute error: 1824924.0

-- Attention total --
numpy: 0.0075 sec, 1.1255 Gop/s
torch int32 attention core: 0.0047 sec, 1.7893 Gop/s
QPU host prep: 0.0001 sec
QPU cached total: 0.0007 sec, 12.8083 Gop/s
QPU execute only: 0.0006 sec, 14.4122 Gop/s
QPU prep+cached total: 0.0008 sec, 10.7032 Gop/s
Maximum absolute error: 78106785.0

```

### tiledconv2d_default

- Notes: Current GEMM-backed tiled conv2d benchmark.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledconv2d.py`
- Exit status: 0
- Duration: 2s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledconv2d_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledconv2d.py
==== tiledconv2d fp32 example ====
numpy: 0.0020 sec, 21.6917 Gop/s
torch native conv2d: 0.0033 sec, 12.7378 Gop/s
QPU host prep: 0.0009 sec
QPU cached total: 0.0026 sec, 16.2549 Gop/s
QPU execute only: 0.0021 sec, 19.8662 Gop/s
QPU prep+cached total: 0.0035 sec, 12.0349 Gop/s
Maximum absolute error: 37.749603271484375

==== tiledconv2d int32 example ====
numpy: 0.0083 sec, 1.4354 Gop/s
torch native conv2d: 0.0014 sec, 8.2980 Gop/s
QPU host prep: 0.0003 sec
QPU cached total: 0.0008 sec, 15.0788 Gop/s
QPU execute only: 0.0006 sec, 18.6018 Gop/s
QPU prep+cached total: 0.0011 sec, 10.8520 Gop/s
Maximum absolute error: 670088

==== tiledconv2d int16 example ====
Kernel contract: int16 inputs are packed as int32 pairs and accumulated into int32.
numpy: 0.0083 sec, 1.4458 Gop/s
torch native conv2d: 0.0008 sec, 15.4468 Gop/s
torch note: native int16 conv2d returns int16 on this build, so it is a speed baseline only.
QPU host prep: 0.0004 sec
QPU cached total: 0.0007 sec, 15.9736 Gop/s
QPU execute only: 0.0006 sec, 18.4269 Gop/s
QPU prep+cached total: 0.0012 sec, 10.2894 Gop/s
Maximum absolute error: 3993623
```

### tiledlenet5_default_core12

- Notes: Default LeNet-5 pipeline run, meaning 12 QPU cores.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledlenet5.py`
- Exit status: 0
- Duration: 4s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledlenet5_default_core12.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledlenet5.py
==== LeNet-5 fp32 example ====
Architecture: Conv5x5(1->6) -> ReLU -> AvgPool -> Conv5x5(6->16) -> ReLU -> AvgPool -> Linear(400->120) -> ReLU -> Linear(120->84) -> ReLU -> Linear(84->10)
Batch: 48, num_qpus: 12
QPU steady-state layout: on-device gather lowering -> position-major matrices -> zero-copy flatten -> FC.
Steady-state CPU compute in the QPU path: none.
Excluded one-time CPU setup: weight reshape/padding, metadata generation, kernel assembly, buffer allocation.
QPU setup (excluded): 0.5569 sec
numpy: 0.0181 sec, 2.2557 Gop/s
torch native lenet5: 0.0037 sec, 11.0738 Gop/s
QPU host prep: 0.0000 sec
QPU cached total: 0.0101 sec, 4.0679 Gop/s
QPU execute only: 0.0100 sec, 4.0829 Gop/s
QPU prep+cached total: 0.0101 sec, 4.0679 Gop/s
QPU upload+readback overhead inside cached total: 0.0000 sec
Maximum absolute error: nan
QPU execute breakdown:
  conv1 gather             0.001881 sec
  conv1 gemm+bias+relu     0.003619 sec
  pool1 avgpool            0.000542 sec
  conv2 gather             0.001723 sec
  conv2 gemm+bias+relu     0.001415 sec
  pool2 avgpool            0.000118 sec
  fc1 gemm+bias+relu       0.000345 sec
  fc2 gemm+bias+relu       0.000190 sec
  fc3 gemm+bias            0.000137 sec


==== LeNet-5 int32 example ====
Architecture: Conv5x5(1->6) -> ReLU -> AvgPool -> Conv5x5(6->16) -> ReLU -> AvgPool -> Linear(400->120) -> ReLU -> Linear(120->84) -> ReLU -> Linear(84->10)
Batch: 48, num_qpus: 12
QPU steady-state layout: on-device gather lowering -> position-major matrices -> zero-copy flatten -> FC.
Steady-state CPU compute in the QPU path: none.
Excluded one-time CPU setup: weight reshape/padding, metadata generation, kernel assembly, buffer allocation.
Actual smul24 stage maxima: input=2, pool1=24, pool2_flat=342, fc1=4579, fc2=22929
QPU setup (excluded): 0.5594 sec
numpy: 0.0494 sec, 0.8277 Gop/s
torch native lenet5: 0.0054 sec, 7.6148 Gop/s
QPU host prep: 0.0000 sec
QPU cached total: 0.0101 sec, 4.0427 Gop/s
QPU execute only: 0.0100 sec, 4.0754 Gop/s
QPU prep+cached total: 0.0101 sec, 4.0427 Gop/s
QPU upload+readback overhead inside cached total: 0.0001 sec
Maximum absolute error: 87016300.0
QPU execute breakdown:
  conv1 gather             0.001907 sec
  conv1 gemm+bias+relu     0.003617 sec
  pool1 avgpool            0.000552 sec
  conv2 gather             0.001739 sec
  conv2 gemm+bias+relu     0.001413 sec
  pool2 avgpool            0.000118 sec
  fc1 gemm+bias+relu       0.000345 sec
  fc2 gemm+bias+relu       0.000190 sec
  fc3 gemm+bias            0.000136 sec

```

### tiledlenet5_core1

- Notes: Explicit 1-core LeNet-5 pipeline run.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledlenet5.py --num-qpus 1`
- Exit status: 0
- Duration: 6s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledlenet5_core1.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledlenet5.py --num-qpus 1
==== LeNet-5 fp32 example ====
Architecture: Conv5x5(1->6) -> ReLU -> AvgPool -> Conv5x5(6->16) -> ReLU -> AvgPool -> Linear(400->120) -> ReLU -> Linear(120->84) -> ReLU -> Linear(84->10)
Batch: 48, num_qpus: 1
QPU steady-state layout: on-device gather lowering -> position-major matrices -> zero-copy flatten -> FC.
Steady-state CPU compute in the QPU path: none.
Excluded one-time CPU setup: weight reshape/padding, metadata generation, kernel assembly, buffer allocation.
QPU setup (excluded): 0.5594 sec
numpy: 0.0182 sec, 2.2438 Gop/s
torch native lenet5: 0.0037 sec, 11.0430 Gop/s
QPU host prep: 0.0000 sec
QPU cached total: 0.0315 sec, 1.2993 Gop/s
QPU execute only: 0.0314 sec, 1.3012 Gop/s
QPU prep+cached total: 0.0315 sec, 1.2993 Gop/s
QPU upload+readback overhead inside cached total: 0.0000 sec
Maximum absolute error: nan
QPU execute breakdown:
  conv1 gather             0.012180 sec
  conv1 gemm+bias+relu     0.003617 sec
  pool1 avgpool            0.003574 sec
  conv2 gather             0.009476 sec
  conv2 gemm+bias+relu     0.001413 sec
  pool2 avgpool            0.000484 sec
  fc1 gemm+bias+relu       0.000345 sec
  fc2 gemm+bias+relu       0.000188 sec
  fc3 gemm+bias            0.000138 sec


==== LeNet-5 int32 example ====
Architecture: Conv5x5(1->6) -> ReLU -> AvgPool -> Conv5x5(6->16) -> ReLU -> AvgPool -> Linear(400->120) -> ReLU -> Linear(120->84) -> ReLU -> Linear(84->10)
Batch: 48, num_qpus: 1
QPU steady-state layout: on-device gather lowering -> position-major matrices -> zero-copy flatten -> FC.
Steady-state CPU compute in the QPU path: none.
Excluded one-time CPU setup: weight reshape/padding, metadata generation, kernel assembly, buffer allocation.
Actual smul24 stage maxima: input=2, pool1=24, pool2_flat=342, fc1=4579, fc2=22929
QPU setup (excluded): 0.5692 sec
numpy: 0.0498 sec, 0.8211 Gop/s
torch native lenet5: 0.0055 sec, 7.3982 Gop/s
QPU host prep: 0.0000 sec
QPU cached total: 0.0317 sec, 1.2913 Gop/s
QPU execute only: 0.0315 sec, 1.2970 Gop/s
QPU prep+cached total: 0.0317 sec, 1.2913 Gop/s
QPU upload+readback overhead inside cached total: 0.0001 sec
Maximum absolute error: 81424898.0
QPU execute breakdown:
  conv1 gather             0.012172 sec
  conv1 gemm+bias+relu     0.003615 sec
  pool1 avgpool            0.003746 sec
  conv2 gather             0.009469 sec
  conv2 gemm+bias+relu     0.001412 sec
  pool2 avgpool            0.000496 sec
  fc1 gemm+bias+relu       0.000343 sec
  fc2 gemm+bias+relu       0.000187 sec
  fc3 gemm+bias            0.000137 sec

```

### tiledmlp_default

- Notes: Tiled two-layer MLP benchmark.
- Command: `/home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledmlp.py`
- Exit status: 0
- Duration: 48s
- Log path: `/home/yiannis/side/py-videocore7/experiment_logs/20260416-135103-full-rerun/tiledmlp_default.log`

```text
$ /home/yiannis/side/py-videocore7/.venv/bin/python examples/tiledmlp.py
==== tiledmlp fp32 example ====
Operator: Linear -> ReLU -> Linear
Dimensions: x=256x1024, w1=1024x1024, w2=1024x512
Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.
QPU setup (excluded): 0.0447 sec

-- MLP total --
numpy: 0.0120 sec, 67.4173 Gop/s
torch native mlp: 0.0102 sec, 79.0828 Gop/s
QPU host prep: 0.0001 sec
QPU cached total: 0.0371 sec, 21.7194 Gop/s
QPU execute only: 0.0366 sec, 22.0089 Gop/s
QPU prep+cached total: 0.0372 sec, 21.6452 Gop/s
Maximum absolute error: nan

-- Layer1 only --
numpy: 0.0079 sec, 68.1484 Gop/s
torch native linear: 0.0064 sec, 84.4416 Gop/s
QPU host prep: 0.0001 sec
QPU cached total: 0.0252 sec, 21.3293 Gop/s
QPU execute only: 0.0244 sec, 22.0344 Gop/s
QPU prep+cached total: 0.0253 sec, 21.2242 Gop/s
Maximum absolute error: nan

-- ReLU only --
numpy: 0.0002 sec, 1.5481 Gop/s
torch relu: 0.0001 sec, 2.1306 Gop/s
QPU host prep: 0.0003 sec
QPU cached total: 0.0011 sec, 0.2364 Gop/s
QPU execute only: 0.0003 sec, 0.9181 Gop/s
QPU prep+cached total: 0.0014 sec, 0.1883 Gop/s
Maximum absolute error: 45.80464172363281

-- Layer2 only --
numpy: 0.0039 sec, 68.7269 Gop/s
torch native linear: 0.0032 sec, 83.6757 Gop/s
QPU host prep: 0.0003 sec
QPU cached total: 0.0127 sec, 21.0745 Gop/s
QPU execute only: 0.0122 sec, 21.9357 Gop/s
QPU prep+cached total: 0.0130 sec, 20.6166 Gop/s
Maximum absolute error: 281079029.14691734


==== tiledmlp int32 example ====
Operator: Linear -> ReLU -> Linear
Dimensions: x=256x1024, w1=1024x1024, w2=1024x512
Kernel contract: x, w1, and w2 must fit the signed 24-bit range, and layer1 output must also stay within it.
Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.
QPU setup (excluded): 0.0501 sec

-- MLP total --
numpy: 2.9992 sec, 0.2687 Gop/s
torch int32 mlp: 0.0921 sec, 8.7475 Gop/s
QPU host prep: 0.0002 sec
QPU cached total: 0.0372 sec, 21.6573 Gop/s
QPU execute only: 0.0366 sec, 22.0083 Gop/s
QPU prep+cached total: 0.0374 sec, 21.5491 Gop/s
Maximum absolute error: 2147657713.0

-- Layer1 only --
numpy: 2.3862 sec, 0.2251 Gop/s
torch int32 linear: 0.0606 sec, 8.8682 Gop/s
QPU host prep: 0.0001 sec
QPU cached total: 0.0253 sec, 21.2128 Gop/s
QPU execute only: 0.0244 sec, 22.0316 Gop/s
QPU prep+cached total: 0.0254 sec, 21.1199 Gop/s
Maximum absolute error: 272796693.0

-- ReLU only --
numpy: 0.0002 sec, 1.3809 Gop/s
torch int32 relu: 0.0001 sec, 2.6386 Gop/s
QPU host prep: 0.0003 sec
QPU cached total: 0.0012 sec, 0.2167 Gop/s
QPU execute only: 0.0003 sec, 0.9151 Gop/s
QPU prep+cached total: 0.0015 sec, 0.1752 Gop/s
Maximum absolute error: 3530.0

-- Layer2 only --
numpy: 0.6028 sec, 0.4455 Gop/s
torch int32 linear: 0.0316 sec, 8.5115 Gop/s
QPU host prep: 0.0003 sec
QPU cached total: 0.0128 sec, 20.9306 Gop/s
QPU execute only: 0.0122 sec, 21.9341 Gop/s
QPU prep+cached total: 0.0131 sec, 20.5054 Gop/s
Maximum absolute error: 2147747576.0

```
