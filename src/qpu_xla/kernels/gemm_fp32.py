"""Cached tiled FP32 GEMM kernel for VideoCore VII."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu


@qpu
def qpu_tiled_sgemm(asm: Assembly) -> None:
    """Compute one 16x16 FP32 output tile using four K values per iteration."""
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
    reg_accum = [rf[index] for index in range(16, 32)]

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_a_stride))
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))
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
    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)
    for index in range(8):
        bxor(reg_accum[index], reg_accum[index], reg_accum[index]).sub(
            reg_accum[index + 8], reg_accum[index + 8], reg_accum[index + 8], sig=ldtmu((reg_a + reg_b)[index])
        )

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)
        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)
        sub(reg_a_base, reg_a_base, -16)
        broadcast_order = deque(reg_b * 16)

        def broadcast_next() -> None:
            register = broadcast_order.popleft()
            rotate(register, register, 1).mov(rep, register)

        broadcast_next()
        sub(reg_i, reg_i, 1, cond="pushz").fmul(rf1, rf0, reg_a[0])
        for index in range(15):
            broadcast_next()
            fadd(reg_accum[index], reg_accum[index], rf1).fmul(rf1, rf0, reg_a[0])
        broadcast_next()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[1])
        for index in range(15):
            broadcast_next()
            fadd(reg_accum[index], reg_accum[index], rf1).fmul(rf1, rf0, reg_a[1])
        broadcast_next()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[2])
        for index in range(15):
            broadcast_next()
            fadd(reg_accum[index], reg_accum[index], rf1).fmul(rf1, rf0, reg_a[2])
        broadcast_next()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[3])
        for index in range(8):
            broadcast_next()
            fadd(reg_accum[index], reg_accum[index], rf1).fmul(rf1, rf0, reg_a[3])
        broadcast_next()
        fadd(reg_accum[8], reg_accum[8], rf1, sig=ldtmu(reg_a[0])).fmul(rf1, rf0, reg_a[3])
        broadcast_next()
        fadd(reg_accum[9], reg_accum[9], rf1, sig=ldtmu(reg_a[1])).fmul(rf1, rf0, reg_a[3])
        broadcast_next()
        fadd(reg_accum[10], reg_accum[10], rf1, sig=ldtmu(reg_a[2])).fmul(rf1, rf0, reg_a[3])
        broadcast_next()
        fadd(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).fmul(rf1, rf0, reg_a[3])
        broadcast_next()
        fadd(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).fmul(rf1, rf0, reg_a[3])
        broadcast_next()
        fadd(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).fmul(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        broadcast_next()
        fadd(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).fmul(rf1, rf0, reg_a[3])
        fadd(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)

    mov(tmuc, -1)
    for index in range(0, 16, 4):
        mov(tmud, reg_accum[index])
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
        mov(tmud, reg_accum[index + 1])
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
        mov(tmud, reg_accum[index + 2])
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
        mov(tmud, reg_accum[index + 3])
        mov(tmua, reg_c_base)
        tmuwt()
        if index < 12:
            sub(reg_c_base, reg_c_base, -4)
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@dataclass(slots=True)
class _ProgramState:
    """Driver-local assembled code and reusable uniform storage."""

    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, _ProgramState] = WeakKeyDictionary()


def supports_tiled_fp32_gemm(left: Tensor, right: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the exact tiled FP32 GEMM contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (left, right, destination)):
        return False
    if len(left.shape) != 2 or len(right.shape) != 2 or len(destination.shape) != 2:
        return False
    p, q = left.shape
    q_right, r = right.shape
    if q != q_right or destination.shape != (p, r) or p % 16 or q % 4 or r % 16:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (left, right, destination))


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    """Assemble the FP32 kernel once for each hardware backend."""
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(code=driver.program(qpu_tiled_sgemm), uniforms=driver.alloc(7, dtype=np.uint32))
            _PROGRAMS[backend] = state
        return state


def _execute_tiled_fp32_gemm(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    """Dispatch cached FP32 tiled GEMM over all 16x16 output tiles."""
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("tiled FP32 GEMM expects (left_tensor, right_tensor, destination_tensor)")
    left, right, destination = args
    if not supports_tiled_fp32_gemm(left, right, destination, backend):
        raise KernelError(
            "tiled FP32 GEMM requires contiguous float32 (P,Q)@(Q,R) tensors with P/R multiples of 16 and Q of 4"
        )
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("tiled FP32 GEMM requires PyVideoCore7Backend")
    expected_grid = (right.shape[1] // 16, left.shape[0] // 16, 1)
    if grid != expected_grid:
        raise KernelError(f"tiled FP32 GEMM grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            left.numpy().strides[0],
            left.address,
            right.numpy().strides[0],
            right.address,
            destination.numpy().strides[0],
            destination.address,
            left.shape[1],
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=grid[0] * grid[1],
        )


TILED_FP32_GEMM_KERNEL = Kernel("vc7.tiled_fp32_gemm", _execute_tiled_fp32_gemm)
