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
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np
try:
    import torch
except ImportError:
    torch = None

# from videocore7 import pack_unpack
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def gflops(p: int, q: int, r: int, sec: float) -> float:
    return (2 * p * q * r + 3 * p * r) / sec * 1e-9


def batch_gflops(batch: int, p: int, q: int, r: int, sec: float) -> float:
    return batch * gflops(p, q, r, sec)


@qpu
def qpu_sgemm_rnn_naive(asm: Assembly) -> None:
    # rf0 - rf15: regs
    # rf16 - rf31: 16x16 accumulators
    # rf32 - rf63: (unused)

    # params
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11  # use after reg_c_stride is released
    reg_a_stride = rf9  # use for init, unused in loop
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10  # use for init, unused in loop
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]  # accumulators

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    # a_base = a[i,:] = a_item0_addr + 16 * i * a_stride
    # b_base = b[:,j] = b_item0_addr + 16 * j * 4
    nop(sig=ldunifrf(reg_a_stride))  # load a_stride
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))  # load a_item0 addr
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))  # load b_stride
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))  # load b_item0 addr
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)  # eidx for next block

    # calc base addr of A and B
    # eidx(rf3)  # merged pred instruction
    umul24(rf4, rf3, reg_a_stride)
    del reg_a_stride  # reg_a_stride is released
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)  # keep 4*b_stride
    del reg_b_stride  # reg_b_stride is released
    add(reg_b_base, reg_b_base, rf3)

    # start loading A
    bnot(tmuc, 3)
    mov(tmua, reg_a_base)

    # start loading B
    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    # calc base addr of C
    nop(sig=ldunifrf(reg_c_stride))  # load c_stride
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i  # reg_tile_i is released
    del reg_tile_j  # reg_tile_j is released
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))  # c_item0_addr
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride  # reg_c_stride is released
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)  # load q
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)  # c_base[0] = c_item0_addr + 16 * i * c_stride + 16 * j * 4

    for i in range(8):
        r1 = reg_accum[i]
        r2 = reg_accum[i + 8]
        bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu((reg_a + reg_b)[i]))  # load reg_a and reg_b
    with loop as lk:
        # start loading next A
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        # start loading next B
        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b_order = deque(reg_b * 16)

        def rotate_broadcast_reg_b() -> None:
            r = rotate_broadcast_reg_b_order.popleft()
            rotate(r, r, 1).mov(rep, r)

        # 16x4x16 FMA
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
        fadd(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).fmul(rf1, rf0, reg_a[3])  # reg_a[3] is using
        rotate_broadcast_reg_b()
        fadd(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).fmul(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        rotate_broadcast_reg_b()  # delay slot
        fadd(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).fmul(rf1, rf0, reg_a[3])  # delay slot
        fadd(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)  # delay slot

    # rf3-14 is released
    del reg_a
    del reg_b
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b_base

    # load C and store αAB+βC
    reg_tmuc_vec_4_cfg = rf12
    reg_alpha = rf13
    reg_beta = rf14
    bnot(reg_tmuc_vec_4_cfg, 3)  # pixel, regular, vec4
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


@qpu
def qpu_sgemm_rnn_naive_batched(asm: Assembly) -> None:
    # rf0 - rf15: regs
    # rf16 - rf31: 16x16 accumulators
    # rf32 - rf35: batch-stride helpers

    reg_batch = rf0
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11  # use after reg_c_stride is released
    reg_a_stride = rf9  # use for init, unused in loop
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10  # use for init, unused in loop
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]  # accumulators
    reg_a_batch_stride = rf32
    reg_b_batch_stride = rf33
    reg_c_batch_stride = rf34
    reg_batch_offset = rf35

    mov(reg_batch, rf2.unpack("ul"))
    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    # a_base = a[batch, i, :] = a_item0_addr + batch * a_batch_stride + 16 * i * a_stride
    nop(sig=ldunifrf(reg_a_batch_stride))
    umul24(reg_batch_offset, reg_batch, reg_a_batch_stride, sig=ldunifrf(reg_a_stride))
    shl(reg_batch_offset, reg_batch_offset, 8)
    nop(sig=ldunifrf(reg_a_base))
    add(reg_a_base, reg_a_base, reg_batch_offset)
    umul24(rf3, reg_tile_i, reg_a_stride)
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3)

    # b_base = b[batch, :, j] = b_item0_addr + batch * b_batch_stride + 16 * j * 4
    nop(sig=ldunifrf(reg_b_batch_stride))
    umul24(reg_batch_offset, reg_batch, reg_b_batch_stride, sig=ldunifrf(reg_b_stride))
    shl(reg_batch_offset, reg_batch_offset, 8)
    nop(sig=ldunifrf(reg_b_base))
    add(reg_b_base, reg_b_base, reg_batch_offset)
    shl(rf3, reg_tile_j, 6)
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)  # eidx for next block

    # calc base addr of A and B
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

    # c_base = c[batch, i, j] = c_item0_addr + batch * c_batch_stride + 16 * i * c_stride + 16 * j * 4
    nop(sig=ldunifrf(reg_c_batch_stride))
    umul24(reg_batch_offset, reg_batch, reg_c_batch_stride, sig=ldunifrf(reg_c_stride))
    shl(reg_batch_offset, reg_batch_offset, 8)
    nop(sig=ldunifrf(reg_c_base))
    add(reg_c_base, reg_c_base, reg_batch_offset)
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j
    del reg_batch
    eidx(rf0).add(rf3, rf3, rf0)
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)

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
    del reg_a_batch_stride
    del reg_b_batch_stride
    del reg_c_batch_stride
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


def sgemm_rnn_naive(p: int = 1024, q: int = 1024, r: int = 1024) -> dict[str, float]:
    assert p > 0
    assert q > 0
    assert r > 0

    assert p % 16 == 0
    assert q % 4 == 0
    assert r % 16 == 0

    tile_p = p // 16
    tile_r = r // 16
    data_area_size = (p * q + q * r + p * r) * np.dtype(np.float32).itemsize + 4096

    with Driver(data_area_size=data_area_size) as drv:
        code = drv.program(qpu_sgemm_rnn_naive)

        a: Array[np.float32] = drv.alloc((p, q), dtype=np.float32)
        b: Array[np.float32] = drv.alloc((q, r), dtype=np.float32)
        c: Array[np.float32] = drv.alloc((p, r), dtype=np.float32)

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        a_ref = np.random.randn(*a.shape).astype(a.dtype)
        b_ref = np.random.randn(*b.shape).astype(b.dtype)
        c_ref = np.random.randn(*c.shape).astype(c.dtype)
        expected = np.empty(c.shape, dtype=c.dtype)

        a[:] = a_ref
        b[:] = b_ref
        c[:] = c_ref

        start = getsec()
        expected[:] = alpha * a_ref.dot(b_ref) + beta * c_ref
        time_ref = getsec() - start
        time_torch: float | None = None
        if torch is not None:
            a_torch = torch.from_numpy(a_ref)
            b_torch = torch.from_numpy(b_ref)
            c_torch = torch.from_numpy(c_ref)
            start = getsec()
            with torch.no_grad():
                _ = alpha * torch.matmul(a_torch, b_torch) + beta * c_torch
            time_torch = getsec() - start

        unif: Array[np.uint32] = drv.alloc(9, dtype=np.uint32)
        unif[0] = a.strides[0]
        unif[1] = a.addresses().item(0)
        unif[2] = b.strides[0]
        unif[3] = b.addresses().item(0)
        unif[4] = c.strides[0]
        unif[5] = c.addresses().item(0)
        unif[6] = q
        unif[7] = np.float32(alpha).view(np.uint32).item()
        unif[8] = np.float32(beta).view(np.uint32).item()

        start = getsec()
        drv.execute(
            code,
            local_invocation=(16, 1, 1),
            uniforms=unif.addresses().item(0),
            workgroup=(tile_r, tile_p, 1),
            wgs_per_sg=24,
            thread=tile_p * tile_r,
        )
        time_gpu = getsec() - start

        print(f"==== sgemm example ({p}x{q} times {q}x{r}) ====")
        print(f"numpy: {time_ref:.4f} sec, {gflops(p, q, r, time_ref):.4f} Gflop/s")
        if time_torch is None:
            print("torch: n/a (torch is not installed)")
        else:
            print(f"torch: {time_torch:.4f} sec, {gflops(p, q, r, time_torch):.4f} Gflop/s")
        print(f"QPU:   {time_gpu:.4f} sec, {gflops(p, q, r, time_gpu):.4f} Gflop/s")
        print(f"Minimum absolute error: {np.min(np.abs(c - expected))}")
        print(f"Maximum absolute error: {np.max(np.abs(c - expected))}")
        print(f"Minimum relative error: {np.min(np.abs((c - expected) / expected))}")
        print(f"Maximum relative error: {np.max(np.abs((c - expected) / expected))}")

        return {
            "p": float(p),
            "q": float(q),
            "r": float(r),
            "numpy_sec": time_ref,
            "numpy_gflops": gflops(p, q, r, time_ref),
            "torch_sec": float("nan") if time_torch is None else time_torch,
            "torch_gflops": float("nan") if time_torch is None else gflops(p, q, r, time_torch),
            "qpu_sec": time_gpu,
            "qpu_gflops": gflops(p, q, r, time_gpu),
            "max_abs_error": float(np.max(np.abs(c - expected))),
            "max_rel_error": float(np.max(np.abs((c - expected) / expected))),
        }


def sweep_sgemm_rnn_naive(sizes: list[int] | tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048)) -> None:
    results: list[dict[str, float]] = []

    for size in sizes:
        print()
        results.append(sgemm_rnn_naive(p=size, q=size, r=size))

    print()
    print("==== sgemm size sweep summary ====")
    print(f"{'size':>6} {'numpy GF/s':>12} {'torch GF/s':>12} {'QPU GF/s':>12} {'QPU sec':>10} {'max abs err':>14}")
    for result in results:
        size = int(result["p"])
        torch_gflops = result["torch_gflops"]
        torch_text = "n/a" if np.isnan(torch_gflops) else f"{torch_gflops:.4f}"
        print(
            f"{size:>6} "
            f"{result['numpy_gflops']:>12.4f} "
            f"{torch_text:>12} "
            f"{result['qpu_gflops']:>12.4f} "
            f"{result['qpu_sec']:>10.4f} "
            f"{result['max_abs_error']:>14.6g}"
        )


def batched_sgemm_rnn_naive(
    *,
    size: int = 256,
    batch: int = 8,
) -> dict[str, float]:
    assert size > 0
    assert batch > 0
    assert size % 16 == 0
    assert size % 4 == 0

    p = q = r = size
    tile_p = p // 16
    tile_r = r // 16
    data_area_size = (
        batch * (p * q + q * r + p * r) * np.dtype(np.float32).itemsize
        + 12 * np.dtype(np.uint32).itemsize
        + 4096
    )

    with Driver(data_area_size=data_area_size) as drv:
        code = drv.program(qpu_sgemm_rnn_naive_batched)

        a: Array[np.float32] = drv.alloc((batch, p, q), dtype=np.float32)
        b: Array[np.float32] = drv.alloc((batch, q, r), dtype=np.float32)
        c: Array[np.float32] = drv.alloc((batch, p, r), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc(12, dtype=np.uint32)

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        a_ref = np.random.randn(*a.shape).astype(a.dtype)
        b_ref = np.random.randn(*b.shape).astype(b.dtype)
        c_ref = np.random.randn(*c.shape).astype(c.dtype)
        expected = np.empty(c.shape, dtype=c.dtype)

        a[:] = a_ref
        b[:] = b_ref
        c[:] = c_ref

        start = getsec()
        for i in range(batch):
            expected[i] = alpha * a_ref[i].dot(b_ref[i]) + beta * c_ref[i]
        time_ref = getsec() - start

        time_torch: float | None = None
        if torch is not None:
            a_torch = torch.from_numpy(a_ref)
            b_torch = torch.from_numpy(b_ref)
            c_torch = torch.from_numpy(c_ref)
            start = getsec()
            with torch.no_grad():
                _ = alpha * torch.matmul(a_torch, b_torch) + beta * c_torch
            time_torch = getsec() - start

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
        unif[9] = q
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
        time_gpu = getsec() - start

        print(f"==== batched sgemm example (single dispatch, batch={batch}, {p}x{q} times {q}x{r}) ====")
        print(f"numpy: {time_ref:.4f} sec, {batch_gflops(batch, p, q, r, time_ref):.4f} Gflop/s")
        if time_torch is None:
            print("torch: n/a (torch is not installed)")
        else:
            print(f"torch: {time_torch:.4f} sec, {batch_gflops(batch, p, q, r, time_torch):.4f} Gflop/s")
        print(f"QPU:   {time_gpu:.4f} sec, {batch_gflops(batch, p, q, r, time_gpu):.4f} Gflop/s")
        print(f"Maximum absolute error: {np.max(np.abs(c - expected))}")
        print(f"Maximum relative error: {np.max(np.abs((c - expected) / expected))}")

        return {
            "size": float(size),
            "batch": float(batch),
            "numpy_sec": time_ref,
            "numpy_gflops": batch_gflops(batch, p, q, r, time_ref),
            "torch_sec": float("nan") if time_torch is None else time_torch,
            "torch_gflops": float("nan") if time_torch is None else batch_gflops(batch, p, q, r, time_torch),
            "qpu_sec": time_gpu,
            "qpu_gflops": batch_gflops(batch, p, q, r, time_gpu),
            "max_abs_error": float(np.max(np.abs(c - expected))),
            "max_rel_error": float(np.max(np.abs((c - expected) / expected))),
        }


def sweep_batched_sgemm_rnn_naive(
    sizes: list[int] | tuple[int, ...] = (128, 256, 512),
    batch_sizes: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> None:
    results: list[dict[str, float]] = []

    for size in sizes:
        print()
        print(f"==== batched sgemm sweep for size {size} ====")
        for batch in batch_sizes:
            results.append(batched_sgemm_rnn_naive(size=size, batch=batch))
            print()

    print("==== batched sgemm summary ====")
    print(f"{'size':>6} {'batch':>6} {'numpy GF/s':>12} {'torch GF/s':>12} {'QPU GF/s':>12} {'QPU sec':>10}")
    for result in results:
        torch_gflops = result["torch_gflops"]
        torch_text = "n/a" if np.isnan(torch_gflops) else f"{torch_gflops:.4f}"
        print(
            f"{int(result['size']):>6} "
            f"{int(result['batch']):>6} "
            f"{result['numpy_gflops']:>12.4f} "
            f"{torch_text:>12} "
            f"{result['qpu_gflops']:>12.4f} "
            f"{result['qpu_sec']:>10.4f}"
        )


def main() -> None:
    sweep_sgemm_rnn_naive()
    print()
    sweep_batched_sgemm_rnn_naive()


if __name__ == "__main__":
    main()
