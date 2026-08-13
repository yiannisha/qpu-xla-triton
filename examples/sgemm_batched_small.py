# Copyright (c) 2025- Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from collections import deque
import argparse
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from sgemm import qpu_sgemm_rnn_naive_batched
except ImportError:
    from examples.sgemm import qpu_sgemm_rnn_naive_batched

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def gflops(p: int, q: int, r: int, sec: float) -> float:
    return (2 * p * q * r + 3 * p * r) / sec * 1e-9


def batch_gflops(batch: int, p: int, q: int, r: int, sec: float) -> float:
    return batch * gflops(p, q, r, sec)


def median_sec(times: list[float]) -> float:
    return float(np.median(np.asarray(times, dtype=np.float64)))


@qpu
def qpu_sgemm_rnn_small_batched(asm: Assembly, *, size: int) -> None:
    assert size in (128, 256)

    row_shift = 9 if size == 128 else 10
    batch_shift = 16 if size == 128 else 18
    k_iter_shift = 5 if size == 128 else 6
    batch_shift_0 = batch_shift // 2
    batch_shift_1 = batch_shift - batch_shift_0

    reg_batch = rf0
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]
    reg_batch_offset = rf32

    mov(reg_batch, rf2.unpack("ul"))
    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    mov(reg_a_stride, 1)
    shl(reg_a_stride, reg_a_stride, row_shift)
    mov(reg_b_stride, 1)
    shl(reg_b_stride, reg_b_stride, row_shift)
    mov(reg_c_stride, 1)
    shl(reg_c_stride, reg_c_stride, row_shift)

    mov(reg_batch_offset, reg_batch)
    shl(reg_batch_offset, reg_batch_offset, batch_shift_0)
    shl(reg_batch_offset, reg_batch_offset, batch_shift_1)

    # a_base = a[batch, i, :]
    nop(sig=ldunifrf(reg_a_base))
    add(reg_a_base, reg_a_base, reg_batch_offset)
    umul24(rf3, reg_tile_i, reg_a_stride)
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3)

    # b_base = b[batch, :, j]
    nop(sig=ldunifrf(reg_b_base))
    add(reg_b_base, reg_b_base, reg_batch_offset)
    shl(rf3, reg_tile_j, 6)
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)

    umul24(rf4, rf3, reg_a_stride)
    del reg_a_stride
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)
    del reg_b_stride
    add(reg_b_base, reg_b_base, rf3)

    bnot(tmuc, 3)
    mov(tmua, reg_a_base)

    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    # c_base = c[batch, i, j]
    nop(sig=ldunifrf(reg_c_base))
    add(reg_c_base, reg_c_base, reg_batch_offset)
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j
    del reg_batch
    eidx(rf0).add(rf3, rf3, rf0)
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride
    add(reg_c_base, reg_c_base, rf0)
    add(reg_c_base, reg_c_base, rf3)

    mov(reg_i, 1)
    shl(reg_i, reg_i, k_iter_shift)

    for i in range(8):
        r1 = reg_accum[i]
        r2 = reg_accum[i + 8]
        bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu((reg_a + reg_b)[i]))
    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b_order = deque(reg_b * 16)

        def rotate_broadcast_reg_b() -> None:
            r = rotate_broadcast_reg_b_order.popleft()
            rotate(r, r, 1).mov(rep, r)

        rotate_broadcast_reg_b()
        sub(reg_i, reg_i, 1, cond="pushz").fmul(rf1, rf0, reg_a[0])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[0])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[1])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[1])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[2])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[2])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[3])
        for i in range(8):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[8], reg_accum[8], rf1, sig=ldtmu(reg_a[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[9], reg_accum[9], rf1, sig=ldtmu(reg_a[1])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[10], reg_accum[10], rf1, sig=ldtmu(reg_a[2])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).fmul(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        rotate_broadcast_reg_b()
        fadd(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).fmul(rf1, rf0, reg_a[3])
        fadd(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)

    del reg_a
    del reg_b
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b_base
    del reg_batch_offset

    reg_tmuc_vec_4_cfg = rf12
    reg_alpha = rf13
    reg_beta = rf14
    bnot(reg_tmuc_vec_4_cfg, 3)
    nop(sig=ldunifrf(reg_alpha))
    mov(tmuc, reg_tmuc_vec_4_cfg, sig=ldunifrf(reg_beta)).fmul(reg_accum[0], reg_accum[0], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[1], reg_accum[1], reg_alpha)
    fmul(reg_accum[2], reg_accum[2], reg_alpha)
    fmul(reg_accum[3], reg_accum[3], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[0], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[1], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[2], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[3], rf2)
    mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[4], reg_accum[4], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[5], reg_accum[5], reg_alpha)
    fmul(reg_accum[6], reg_accum[6], reg_alpha)
    fmul(reg_accum[7], reg_accum[7], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[4], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[5], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[6], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[7], rf2)
    mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[8], reg_accum[8], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[9], reg_accum[9], reg_alpha)
    fmul(reg_accum[10], reg_accum[10], reg_alpha)
    fmul(reg_accum[11], reg_accum[11], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[8], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[9], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[10], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[11], rf2)
    mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[12], reg_accum[12], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[13], reg_accum[13], reg_alpha)
    fmul(reg_accum[14], reg_accum[14], reg_alpha)
    fmul(reg_accum[15], reg_accum[15], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[12], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[13], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[14], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[15], rf2)
    mov(tmua, reg_c_base)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def run_generic_batched(
    drv: Driver,
    code: Array[np.uint64],
    a: Array[np.float32],
    b: Array[np.float32],
    c: Array[np.float32],
    *,
    batch: int,
    size: int,
    alpha: float,
    beta: float,
) -> float:
    tile_p = size // 16
    tile_r = size // 16
    unif: Array[np.uint32] = drv.alloc(12, dtype=np.uint32)

    assert a.strides[0] % 256 == 0
    assert b.strides[0] % 256 == 0
    assert c.strides[0] % 256 == 0

    unif[0] = a.strides[0] >> 8
    unif[1] = a.strides[1]
    unif[2] = a.addresses()[0, 0, 0]
    unif[3] = b.strides[0] >> 8
    unif[4] = b.strides[1]
    unif[5] = b.addresses()[0, 0, 0]
    unif[6] = c.strides[0] >> 8
    unif[7] = c.strides[1]
    unif[8] = c.addresses()[0, 0, 0]
    unif[9] = size
    unif[10] = np.float32(alpha).view(np.uint32).item()
    unif[11] = np.float32(beta).view(np.uint32).item()

    start = getsec()
    drv.execute(
        code,
        local_invocation=(16, 1, 1),
        uniforms=unif.addresses().item(0),
        workgroup=(tile_r, tile_p, batch),
        wgs_per_sg=24,
        thread=batch * tile_p * tile_r,
    )
    return getsec() - start


def run_small_batched(
    drv: Driver,
    code: Array[np.uint64],
    a: Array[np.float32],
    b: Array[np.float32],
    c: Array[np.float32],
    *,
    batch: int,
    size: int,
    alpha: float,
    beta: float,
) -> float:
    tile_p = size // 16
    tile_r = size // 16
    unif: Array[np.uint32] = drv.alloc(5, dtype=np.uint32)

    unif[0] = a.addresses()[0, 0, 0]
    unif[1] = b.addresses()[0, 0, 0]
    unif[2] = c.addresses()[0, 0, 0]
    unif[3] = np.float32(alpha).view(np.uint32).item()
    unif[4] = np.float32(beta).view(np.uint32).item()

    start = getsec()
    drv.execute(
        code,
        local_invocation=(16, 1, 1),
        uniforms=unif.addresses().item(0),
        workgroup=(tile_r, tile_p, batch),
        wgs_per_sg=24,
        thread=batch * tile_p * tile_r,
    )
    return getsec() - start


def benchmark_small_batched(size: int, batch: int) -> dict[str, float]:
    assert size in (128, 256)
    assert batch > 0

    data_area_size = batch * 4 * size * size * np.dtype(np.float32).itemsize + 16384

    with Driver(data_area_size=data_area_size) as drv:
        generic_code = drv.program(qpu_sgemm_rnn_naive_batched)
        small_code = drv.program(qpu_sgemm_rnn_small_batched, size=size)

        a: Array[np.float32] = drv.alloc((batch, size, size), dtype=np.float32)
        b: Array[np.float32] = drv.alloc((batch, size, size), dtype=np.float32)
        c_generic: Array[np.float32] = drv.alloc((batch, size, size), dtype=np.float32)
        c_small: Array[np.float32] = drv.alloc((batch, size, size), dtype=np.float32)

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        a_ref = np.random.randn(*a.shape).astype(a.dtype)
        b_ref = np.random.randn(*b.shape).astype(b.dtype)
        c_ref = np.random.randn(*c_generic.shape).astype(c_generic.dtype)
        expected = np.empty(c_generic.shape, dtype=c_generic.dtype)

        a[:] = a_ref
        b[:] = b_ref
        c_generic[:] = c_ref
        c_small[:] = c_ref

        start = getsec()
        for i in range(batch):
            expected[i] = alpha * a_ref[i].dot(b_ref[i]) + beta * c_ref[i]
        time_numpy = getsec() - start

        time_torch: float | None = None
        if torch is not None:
            a_torch = torch.from_numpy(a_ref)
            b_torch = torch.from_numpy(b_ref)
            c_torch = torch.from_numpy(c_ref)
            start = getsec()
            with torch.no_grad():
                _ = alpha * torch.matmul(a_torch, b_torch) + beta * c_torch
            time_torch = getsec() - start

        # Warm up both kernels once before timing to reduce first-dispatch noise.
        c_generic[:] = c_ref
        run_generic_batched(
            drv,
            generic_code,
            a,
            b,
            c_generic,
            batch=batch,
            size=size,
            alpha=alpha,
            beta=beta,
        )
        c_small[:] = c_ref
        run_small_batched(
            drv,
            small_code,
            a,
            b,
            c_small,
            batch=batch,
            size=size,
            alpha=alpha,
            beta=beta,
        )

        generic_times: list[float] = []
        small_times: list[float] = []
        n_trials = BENCH_TRIALS
        for _ in range(n_trials):
            c_generic[:] = c_ref
            generic_times.append(
                run_generic_batched(
                    drv,
                    generic_code,
                    a,
                    b,
                    c_generic,
                    batch=batch,
                    size=size,
                    alpha=alpha,
                    beta=beta,
                )
            )
            c_small[:] = c_ref
            small_times.append(
                run_small_batched(
                    drv,
                    small_code,
                    a,
                    b,
                    c_small,
                    batch=batch,
                    size=size,
                    alpha=alpha,
                    beta=beta,
                )
            )

        time_generic = median_sec(generic_times)
        time_small = median_sec(small_times)

        print(f"==== small batched sgemm ({size}x{size}, batch={batch}) ====")
        print(f"numpy:        {time_numpy:.4f} sec, {batch_gflops(batch, size, size, size, time_numpy):.4f} Gflop/s")
        if time_torch is None:
            print("torch:        n/a (torch is not installed)")
        else:
            print(f"torch:        {time_torch:.4f} sec, {batch_gflops(batch, size, size, size, time_torch):.4f} Gflop/s")
        print(f"QPU generic:  {time_generic:.4f} sec, {batch_gflops(batch, size, size, size, time_generic):.4f} Gflop/s (median of {n_trials})")
        print(f"QPU small:    {time_small:.4f} sec, {batch_gflops(batch, size, size, size, time_small):.4f} Gflop/s (median of {n_trials})")
        print(f"Generic max abs error: {np.max(np.abs(c_generic - expected))}")
        print(f"Small max abs error:   {np.max(np.abs(c_small - expected))}")

        return {
            "size": float(size),
            "batch": float(batch),
            "numpy_gflops": batch_gflops(batch, size, size, size, time_numpy),
            "torch_gflops": float("nan") if time_torch is None else batch_gflops(batch, size, size, size, time_torch),
            "generic_gflops": batch_gflops(batch, size, size, size, time_generic),
            "small_gflops": batch_gflops(batch, size, size, size, time_small),
            "generic_sec": time_generic,
            "small_sec": time_small,
            "generic_max_abs_error": float(np.max(np.abs(c_generic - expected))),
            "small_max_abs_error": float(np.max(np.abs(c_small - expected))),
        }


def sweep_small_batched(
    sizes: tuple[int, ...] = (128, 256),
    batches: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> None:
    results: list[dict[str, float]] = []

    for size in sizes:
        for batch in batches:
            print()
            results.append(benchmark_small_batched(size, batch))

    print()
    print("==== small batched sgemm summary ====")
    print(
        f"{'size':>6} {'batch':>6} {'numpy GF/s':>12} {'torch GF/s':>12} "
        f"{'generic GF/s':>14} {'small GF/s':>12} {'speedup':>9}"
    )
    for result in results:
        torch_gflops = result["torch_gflops"]
        torch_text = "n/a" if np.isnan(torch_gflops) else f"{torch_gflops:.4f}"
        speedup = result["small_gflops"] / result["generic_gflops"]
        print(
            f"{int(result['size']):>6} "
            f"{int(result['batch']):>6} "
            f"{result['numpy_gflops']:>12.4f} "
            f"{torch_text:>12} "
            f"{result['generic_gflops']:>14.4f} "
            f"{result['small_gflops']:>12.4f} "
            f"{speedup:>9.3f}x"
        )


BENCH_TRIALS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(128, 256))
    parser.add_argument("--batch", type=int)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    global BENCH_TRIALS

    args = parse_args()
    BENCH_TRIALS = max(args.trials, 1)

    if args.size is not None:
        batch = 1 if args.batch is None else args.batch
        benchmark_small_batched(args.size, batch)
        return

    if args.batch is not None:
        sweep_small_batched(batches=(args.batch,))
        return

    sweep_small_batched()


if __name__ == "__main__":
    main()
