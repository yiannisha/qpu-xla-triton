"""Row-parallel numerically stable FP32 softmax kernel."""

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


@qpu
def qpu_softmax_fp32(asm: Assembly, *, vector_width: Literal[1, 4]) -> None:
    """Compute stable softmax with scalar or vec4 TMU accesses."""
    if vector_width not in {1, 4}:
        raise ValueError("softmax vector width must be 1 or 4")

    reg_row = rf0
    reg_iterations = rf1
    reg_source = rf2
    reg_destination = rf4
    reg_row_stride = rf5
    reg_log2_e = rf6
    reg_negative_inf = rf7
    reg_lane_offset = rf8
    reg_vector_stride = rf9
    reg_pointer = rf10
    reg_maximum = rf15
    reg_sum = rf16
    reg_temporary = rf17
    reg_inverse_sum = rf18
    reg_count = rf19
    reg_workgroup_offset = rf20

    values = [rf11, rf12, rf13, rf14]

    mov(reg_row, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_row_stride))
    nop(sig=ldunifrf(reg_log2_e))
    nop(sig=ldunifrf(reg_negative_inf))

    umul24(reg_temporary, reg_row, reg_row_stride)
    add(reg_source, reg_source, reg_temporary)
    add(reg_destination, reg_destination, reg_temporary)
    eidx(reg_lane_offset)
    shl(reg_lane_offset, reg_lane_offset, 4 if vector_width == 4 else 2)
    add(reg_source, reg_source, reg_lane_offset)
    add(reg_destination, reg_destination, reg_lane_offset)
    mov(reg_vector_stride, 1)
    shl(reg_vector_stride, reg_vector_stride, 8 if vector_width == 4 else 6)
    if vector_width == 4:
        bnot(reg_workgroup_offset, 3)

    mov(reg_maximum, reg_negative_inf)
    mov(reg_pointer, reg_source)
    mov(reg_count, reg_iterations)
    with loop as maximum_loop:
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_pointer, sig=thrsw).add(reg_pointer, reg_pointer, reg_vector_stride)
        nop()
        nop()
        for value in values[:vector_width]:
            nop(sig=ldtmu(value))
        for value in values[:vector_width]:
            fmax(reg_maximum, reg_maximum, value)
        sub(reg_count, reg_count, 1, cond="pushz")
        maximum_loop.b(cond="na0")
        nop()
        nop()
        nop()
    for distance in (8, 4, 2, 1):
        rotate(reg_temporary, reg_maximum, distance)
        fmax(reg_maximum, reg_maximum, reg_temporary)

    bxor(reg_sum, reg_sum, reg_sum)
    mov(reg_pointer, reg_source)
    mov(reg_count, reg_iterations)
    with loop as exponential_loop:
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_pointer, sig=thrsw).add(reg_pointer, reg_pointer, reg_vector_stride)
        nop()
        nop()
        for value in values[:vector_width]:
            nop(sig=ldtmu(value))
        for value in values[:vector_width]:
            fsub(value, value, reg_maximum)
            fmul(value, value, reg_log2_e)
            exp(value, value)
            fadd(reg_sum, reg_sum, value)
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        for value in values[:vector_width]:
            mov(tmud, value)
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_vector_stride)
        tmuwt()
        sub(reg_count, reg_count, 1, cond="pushz")
        exponential_loop.b(cond="na0")
        nop()
        nop()
        nop()
    for distance in (8, 4, 2, 1):
        rotate(reg_temporary, reg_sum, distance)
        fadd(reg_sum, reg_sum, reg_temporary)
    recip(reg_inverse_sum, reg_sum)

    sub(reg_destination, reg_destination, reg_row_stride)
    mov(reg_count, reg_iterations)
    with loop as normalize_loop:
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_destination, sig=thrsw)
        nop()
        nop()
        for value in values[:vector_width]:
            nop(sig=ldtmu(value))
        for value in values[:vector_width]:
            fmul(value, value, reg_inverse_sum)
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        for value in values[:vector_width]:
            mov(tmud, value)
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_vector_stride)
        tmuwt()
        sub(reg_count, reg_count, 1, cond="pushz")
        normalize_loop.b(cond="na0")
        nop()
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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[int, _ProgramState]] = WeakKeyDictionary()
_WGS_PER_SG = 48


def supports_softmax_fp32(source: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the contiguous row-softmax contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        return False
    if len(source.shape) != 2 or destination.shape != source.shape or source.shape[1] % 16:
        return False
    if source.shape[1] * np.dtype(np.float32).itemsize >= 1 << 24:
        return False
    return source.numpy().flags.c_contiguous and destination.numpy().flags.c_contiguous


def _program_state(backend: PyVideoCore7Backend, vector_width: Literal[1, 4]) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(vector_width)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_softmax_fp32, vector_width=vector_width),
                    uniforms=driver.alloc(64, dtype=np.uint32),
                )
            states[vector_width] = state
        return state


def _execute_softmax_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 2 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("softmax expects (source, destination)")
    source, destination = args
    if not supports_softmax_fp32(source, destination, backend):
        raise KernelError("softmax requires equal contiguous FP32 rows with 16-aligned width")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("softmax requires PyVideoCore7Backend")
    expected_grid = (source.shape[0], 1, 1)
    if grid != expected_grid:
        raise KernelError(f"softmax grid must be {expected_grid}, got {grid}")
    vector_width: Literal[1, 4] = 4 if source.shape[1] % 64 == 0 else 1
    state = _program_state(backend, vector_width)
    state.uniforms[:6] = (
        source.shape[1] // (16 * vector_width),
        source.address,
        destination.address,
        source.numpy().strides[0],
        np.asarray(np.log2(np.e), dtype=np.float32).view(np.uint32),
        np.asarray(-np.inf, dtype=np.float32).view(np.uint32),
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=expected_grid,
            wgs_per_sg=_WGS_PER_SG,
            thread=source.shape[0],
        )


SOFTMAX_FP32_KERNEL = Kernel("vc7.softmax_fp32", _execute_softmax_fp32)

__all__ = ["SOFTMAX_FP32_KERNEL", "supports_softmax_fp32"]
