"""Cached 2x2/stride-2 INT32 pooling kernels for VideoCore VII."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from weakref import WeakKeyDictionary

import numpy as np

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu

PoolMode = Literal["max", "avg"]


def _emit_trunc_div4_int(dst: Register, src: Register, sign: Register, bias: Register, shift31: Register) -> None:
    asr(sign, src, shift31)
    band(bias, sign, 3)
    add(dst, src, bias)
    asr(dst, dst, 2)


@qpu
def qpu_pool2d_int32(asm: Assembly, *, mode: PoolMode) -> None:
    """Pool 16 independent INT32 output pixels per QPU work item."""
    if mode not in ("max", "avg"):
        raise ValueError("pool mode must be 'max' or 'avg'")
    reg_iters, reg_meta, reg_dst, reg_row_stride = rf0, rf1, rf2, rf3
    reg_word_stride, reg_shift31 = rf6, rf7
    reg_base, reg_v0, reg_v1, reg_v2, reg_v3 = rf10, rf11, rf12, rf13, rf14
    reg_tmp, reg_out, reg_sign, reg_bias = rf15, rf16, rf17, rf18

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_meta))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_row_stride))
    mov(reg_shift31, -1)
    mov(reg_word_stride, 1)
    shl(reg_word_stride, reg_word_stride, 6)
    eidx(rf31)
    shl(rf31, rf31, 2)
    add(reg_meta, reg_meta, rf31)
    add(reg_dst, reg_dst, rf31)

    with loop as lk:
        mov(tmua, reg_meta, sig=thrsw).add(reg_meta, reg_meta, reg_word_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_base))
        mov(tmua, reg_base, sig=thrsw)
        nop()
        add(rf31, reg_base, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v0))
        add(rf31, reg_base, reg_row_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v1))
        add(rf31, rf31, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v2))
        nop()
        nop(sig=ldtmu(reg_v3))
        if mode == "max":
            imax(reg_out, reg_v0, reg_v1)
            imax(reg_out, reg_out, reg_v2)
            imax(reg_out, reg_out, reg_v3)
        else:
            add(reg_out, reg_v0, reg_v1)
            add(reg_tmp, reg_v2, reg_v3)
            add(reg_out, reg_out, reg_tmp)
            _emit_trunc_div4_int(reg_out, reg_out, reg_sign, reg_bias, reg_shift31)
        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_word_stride)
        tmuwt()
        lk.b(cond="na0")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_pool2d_fp32(asm: Assembly, *, mode: PoolMode, num_qpus: int) -> None:
    """Pool 16 independent FP32 output pixels per QPU work item."""
    if mode not in ("max", "avg"):
        raise ValueError("pool mode must be 'max' or 'avg'")
    if not 1 <= num_qpus <= 12:
        raise ValueError("FP32 pool QPU count must be between 1 and 12")
    reg_iters, reg_meta, reg_dst, reg_row_stride = rf0, rf1, rf2, rf3
    reg_qpu_num, reg_offset, reg_word_stride = rf4, rf5, rf6
    reg_base, reg_v0, reg_v1, reg_v2, reg_v3 = rf10, rf11, rf12, rf13, rf14
    reg_tmp, reg_out = rf15, rf16
    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_meta))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_row_stride))
    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_word_stride, 1)
        shl(reg_word_stride, reg_word_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_word_stride, num_qpus)
        shl(reg_word_stride, reg_word_stride, 6)
    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_meta, reg_meta, reg_offset)
    add(reg_dst, reg_dst, reg_offset)
    with loop as lk:
        mov(tmua, reg_meta, sig=thrsw).add(reg_meta, reg_meta, reg_word_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_base))
        mov(tmua, reg_base, sig=thrsw)
        nop()
        add(rf31, reg_base, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v0))
        add(rf31, reg_base, reg_row_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v1))
        add(rf31, rf31, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v2))
        nop()
        nop(sig=ldtmu(reg_v3))
        if mode == "max":
            fmax(reg_out, reg_v0, reg_v1)
            fmax(reg_out, reg_out, reg_v2)
            fmax(reg_out, reg_out, reg_v3)
        else:
            fadd(reg_out, reg_v0, reg_v1)
            fadd(reg_tmp, reg_v2, reg_v3)
            fadd(reg_out, reg_out, reg_tmp)
            fmul(reg_out, reg_out, 0.25)
        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_word_stride)
        tmuwt()
        lk.b(cond="na0")
        nop()
        nop()
        nop()
    barrierid(syncb, sig=thrsw)
    nop()
    nop()
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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[PoolMode, _ProgramState]] = WeakKeyDictionary()
_FP32_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[PoolMode, int], _ProgramState]] = (
    WeakKeyDictionary()
)


def supports_pool2d_int32(source: Tensor, destination: Tensor, metadata: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the scalar 16-output pooling kernel contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.int32) or destination.dtype != np.dtype(np.int32):
        return False
    if metadata.dtype != np.dtype(np.uint32) or len(source.shape) != 4 or len(destination.shape) != 4:
        return False
    if metadata.shape != (int(np.prod(destination.shape)),) or metadata.shape[0] % 16:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, destination, metadata))


def supports_pool2d_fp32(source: Tensor, destination: Tensor, metadata: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the scalar FP32 pooling kernel contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        return False
    if metadata.dtype != np.dtype(np.uint32) or len(source.shape) != 4 or len(destination.shape) != 4:
        return False
    if metadata.shape != (int(np.prod(destination.shape)),) or metadata.shape[0] % 16:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, destination, metadata))


def _program_state(backend: PyVideoCore7Backend, mode: PoolMode) -> _ProgramState:
    with _STATE_LOCK:
        programs = _PROGRAMS.setdefault(backend, {})
        state = programs.get(mode)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_pool2d_int32, mode=mode),
                    uniforms=driver.alloc(4, dtype=np.uint32),
                )
            programs[mode] = state
        return state


def _fp32_program_state(backend: PyVideoCore7Backend, mode: PoolMode, num_qpus: int) -> _ProgramState:
    """Assemble each FP32 pooling mode once for a hardware backend."""
    with _STATE_LOCK:
        programs = _FP32_PROGRAMS.setdefault(backend, {})
        key = (mode, num_qpus)
        state = programs.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_pool2d_fp32, mode=mode, num_qpus=num_qpus),
                    uniforms=driver.alloc(4, dtype=np.uint32),
                )
            programs[key] = state
        return state


def _execute_pool2d(mode: PoolMode, backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("INT32 pool2d expects (source_tensor, destination_tensor, metadata_tensor)")
    source, destination, metadata = args
    if not supports_pool2d_int32(source, destination, metadata, backend):
        raise KernelError(
            "INT32 pool2d requires contiguous NCHW int32 tensors and a 16-aligned uint32 metadata stream"
        )
    if grid != (1, 1, 1):
        raise KernelError("INT32 pool2d uses one workgroup")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("INT32 pool2d requires PyVideoCore7Backend")
    state = _program_state(backend, mode)
    with backend.driver_session() as driver:
        state.uniforms[:] = (metadata.shape[0] // 16, metadata.address, destination.address, source.numpy().strides[2])
        driver.execute(state.code, local_invocation=(16, 1, 1), uniforms=state.uniforms.addresses()[0], thread=1)


MAXPOOL2D_INT32_KERNEL = Kernel(
    "vc7.maxpool2d_int32", lambda backend, args, grid: _execute_pool2d("max", backend, args, grid)
)
AVGPOOL2D_INT32_KERNEL = Kernel(
    "vc7.avgpool2d_int32", lambda backend, args, grid: _execute_pool2d("avg", backend, args, grid)
)


def _execute_pool2d_fp32(mode: PoolMode, backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    """Dispatch one cached FP32 pooling specialization over metadata addresses."""
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("FP32 pool2d expects (source_tensor, destination_tensor, metadata_tensor)")
    source, destination, metadata = args
    if not supports_pool2d_fp32(source, destination, metadata, backend):
        raise KernelError(
            "FP32 pool2d requires contiguous NCHW float32 tensors and a 16-aligned uint32 metadata stream"
        )
    if grid != (1, 1, 1):
        raise KernelError("FP32 pool2d uses one workgroup")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("FP32 pool2d requires PyVideoCore7Backend")
    vectors = metadata.shape[0] // 16
    num_qpus = next(candidate for candidate in range(min(vectors, 12), 0, -1) if vectors % candidate == 0)
    state = _fp32_program_state(backend, mode, num_qpus)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            metadata.shape[0] // (16 * num_qpus),
            metadata.address,
            destination.address,
            source.numpy().strides[2],
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            thread=num_qpus,
        )


MAXPOOL2D_FP32_KERNEL = Kernel(
    "vc7.maxpool2d_fp32", lambda backend, args, grid: _execute_pool2d_fp32("max", backend, args, grid)
)
AVGPOOL2D_FP32_KERNEL = Kernel(
    "vc7.avgpool2d_fp32", lambda backend, args, grid: _execute_pool2d_fp32("avg", backend, args, grid)
)
