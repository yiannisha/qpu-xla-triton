"""Packed W8A8 GEMM candidate with exact INT32 accumulation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import numpy.typing as npt

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu


def pack_int8_quads(values: npt.NDArray[np.int8]) -> npt.NDArray[np.uint32]:
    """Pack groups of four adjacent signed bytes along the final axis."""
    if values.ndim != 2 or values.dtype != np.dtype(np.int8) or values.shape[1] % 4:
        raise ValueError("packed INT8 matrices must be rank-2 int8 with a final dimension divisible by four")
    unsigned = values.view(np.uint8).astype(np.uint32)
    packed = unsigned[:, 0::4] | (unsigned[:, 1::4] << 8) | (unsigned[:, 2::4] << 16) | (unsigned[:, 3::4] << 24)
    return np.ascontiguousarray(packed)


def pack_int8_gemm_operands(
    left: npt.NDArray[np.int8], right: npt.NDArray[np.int8]
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Pack logical ``(P,Q)`` and ``(Q,R)`` operands in the kernel's row layouts."""
    if left.ndim != 2 or right.ndim != 2 or left.dtype != np.dtype(np.int8) or right.dtype != np.dtype(np.int8):
        raise ValueError("W8A8 GEMM operands must be rank-2 int8 matrices")
    if left.shape[1] != right.shape[0] or left.shape[1] % 4:
        raise ValueError("W8A8 GEMM reduction dimensions must align and be divisible by four")
    return pack_int8_quads(left), np.ascontiguousarray(pack_int8_quads(right.T).T)


@qpu
def qpu_tiled_w8a8_gemm(asm: Assembly, *, dequantize: bool = False) -> None:
    """Compute one 16x16 tile, optionally applying FP32 row/column scales."""
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
    reg_row_scale_base = rf32
    reg_column_scale_base = rf33
    reg_column_scale = rf34
    reg_row_scale = rf35
    reg_output = rf36

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
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)
    if dequantize:
        nop(sig=ldunifrf(reg_row_scale_base))
        shl(rf0, reg_tile_i, 6)
        add(reg_row_scale_base, reg_row_scale_base, rf0, sig=ldunifrf(reg_column_scale_base))
        shl(rf0, reg_tile_j, 6)
        eidx(rf3).add(reg_column_scale_base, reg_column_scale_base, rf0)
        shl(rf3, rf3, 2)
        add(reg_row_scale_base, reg_row_scale_base, rf3)
    del reg_tile_i
    del reg_tile_j

    for index in range(8):
        bxor(reg_accum[index], reg_accum[index], reg_accum[index]).sub(
            reg_accum[index + 8],
            reg_accum[index + 8],
            reg_accum[index + 8],
            sig=ldtmu((reg_a + reg_b)[index]),
        )

    setnnmode_ss()
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
        sub(reg_i, reg_i, 1, cond="pushz").v8dot(rf1, rf0, reg_a[0])
        for index in range(15):
            broadcast_next()
            add(reg_accum[index], reg_accum[index], rf1).v8dot(rf1, rf0, reg_a[0])
        broadcast_next()
        add(reg_accum[15], reg_accum[15], rf1).v8dot(rf1, rf0, reg_a[1])
        for index in range(15):
            broadcast_next()
            add(reg_accum[index], reg_accum[index], rf1).v8dot(rf1, rf0, reg_a[1])
        broadcast_next()
        add(reg_accum[15], reg_accum[15], rf1).v8dot(rf1, rf0, reg_a[2])
        for index in range(15):
            broadcast_next()
            add(reg_accum[index], reg_accum[index], rf1).v8dot(rf1, rf0, reg_a[2])
        broadcast_next()
        add(reg_accum[15], reg_accum[15], rf1).v8dot(rf1, rf0, reg_a[3])
        for index in range(8):
            broadcast_next()
            add(reg_accum[index], reg_accum[index], rf1).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_accum[8], reg_accum[8], rf1, sig=ldtmu(reg_a[0])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_accum[9], reg_accum[9], rf1, sig=ldtmu(reg_a[1])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_accum[10], reg_accum[10], rf1, sig=ldtmu(reg_a[2])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).v8dot(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        broadcast_next()
        add(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).v8dot(rf1, rf0, reg_a[3])
        add(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)

    mov(tmuc, -1)
    if dequantize:
        mov(tmua, reg_row_scale_base, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_row_scale))
        for index in range(16):
            mov(tmua, reg_column_scale_base, sig=thrsw)
            nop()
            nop()
            nop(sig=ldtmu(reg_column_scale))
            itof(reg_output, reg_accum[index])
            fmul(reg_output, reg_output, reg_row_scale)
            fmul(reg_output, reg_output, reg_column_scale)
            mov(tmud, reg_output)
            mov(tmua, reg_c_base)
            tmuwt()
            if index < 15:
                add(reg_c_base, reg_c_base, 4)
                add(reg_column_scale_base, reg_column_scale_base, 4)
    else:
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
    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[bool, _ProgramState]] = WeakKeyDictionary()


def _has_contiguous_columns(tensor: Tensor) -> bool:
    """Return whether each logical row can be traversed as adjacent words."""
    array = tensor.numpy()
    return array.strides[-1] == array.dtype.itemsize


def supports_tiled_w8a8_gemm(left: Tensor, right: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether packed tensors meet the exact W8A8 tile contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if left.dtype != np.dtype(np.uint32) or right.dtype != np.dtype(np.uint32):
        return False
    if destination.dtype != np.dtype(np.int32) or any(len(tensor.shape) != 2 for tensor in (left, right, destination)):
        return False
    p, q_words = left.shape
    right_q_words, r = right.shape
    if q_words != right_q_words or destination.shape != (p, r) or p % 16 or q_words % 4 or r % 16:
        return False
    return all(_has_contiguous_columns(tensor) for tensor in (left, right, destination))


def supports_tiled_w8a8_gemm_dequantize(
    left: Tensor,
    right: Tensor,
    row_scales: Tensor,
    column_scales: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether packed GEMM can directly store scaled FP32 output."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if left.dtype != np.dtype(np.uint32) or right.dtype != np.dtype(np.uint32):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (row_scales, column_scales, destination)):
        return False
    if any(len(tensor.shape) != 2 for tensor in (left, right, destination)):
        return False
    rows, q_words = left.shape
    right_q_words, columns = right.shape
    if (
        q_words != right_q_words
        or destination.shape != (rows, columns)
        or rows % 16
        or q_words % 4
        or columns % 16
        or row_scales.shape != (rows,)
        or column_scales.shape != (columns,)
    ):
        return False
    return all(_has_contiguous_columns(tensor) for tensor in (left, right, row_scales, column_scales, destination))


def _program_state(backend: PyVideoCore7Backend, *, dequantize: bool = False) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(dequantize)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_tiled_w8a8_gemm, dequantize=dequantize),
                    uniforms=driver.alloc(9 if dequantize else 7, dtype=np.uint32),
                )
            states[dequantize] = state
        return state


def _execute_tiled_w8a8_gemm(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("packed W8A8 GEMM expects (left, right, destination)")
    left, right, destination = args
    if not supports_tiled_w8a8_gemm(left, right, destination, backend):
        raise KernelError("packed W8A8 GEMM requires uint32 packed inputs, int32 output, and 16x16x16 logical tiles")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("packed W8A8 GEMM requires PyVideoCore7Backend")
    expected_grid = (right.shape[1] // 16, left.shape[0] // 16, 1)
    if grid != expected_grid:
        raise KernelError(f"packed W8A8 GEMM grid must be {expected_grid}, got {grid}")
    logical_reduction = left.shape[1] * 4
    if logical_reduction * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError("packed W8A8 GEMM accumulation can overflow int32")
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


TILED_W8A8_GEMM_KERNEL = Kernel("vc7.tiled_w8a8_gemm", _execute_tiled_w8a8_gemm)


def _execute_tiled_w8a8_gemm_dequantize(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 5 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("fused W8A8 GEMM expects (left, right, row_scales, column_scales, destination)")
    left, right, row_scales, column_scales, destination = args
    if not supports_tiled_w8a8_gemm_dequantize(
        left,
        right,
        row_scales,
        column_scales,
        destination,
        backend,
    ):
        raise KernelError("fused W8A8 GEMM requires packed inputs, aligned FP32 output, and matching scales")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("fused W8A8 GEMM requires PyVideoCore7Backend")
    expected_grid = (right.shape[1] // 16, left.shape[0] // 16, 1)
    if grid != expected_grid:
        raise KernelError(f"fused W8A8 GEMM grid must be {expected_grid}, got {grid}")
    if left.shape[1] * 4 * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError("packed W8A8 GEMM accumulation can overflow int32")
    state = _program_state(backend, dequantize=True)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            left.numpy().strides[0],
            left.address,
            right.numpy().strides[0],
            right.address,
            destination.numpy().strides[0],
            destination.address,
            left.shape[1],
            row_scales.address,
            column_scales.address,
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=grid[0] * grid[1],
        )


TILED_W8A8_GEMM_DEQUANTIZE_KERNEL = Kernel(
    "vc7.tiled_w8a8_gemm_dequantize",
    _execute_tiled_w8a8_gemm_dequantize,
)

__all__: Sequence[str] = (
    "TILED_W8A8_GEMM_KERNEL",
    "TILED_W8A8_GEMM_DEQUANTIZE_KERNEL",
    "pack_int8_gemm_operands",
    "pack_int8_quads",
    "supports_tiled_w8a8_gemm",
    "supports_tiled_w8a8_gemm_dequantize",
)
