"""Row-parallel FP32 scalar-scale plus column-bias kernel."""

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
def qpu_affine_fp32(asm: Assembly, *, vector_width: Literal[1, 4]) -> None:
    """Apply ``destination = source * scale + bias`` to one row/workgroup."""
    if vector_width not in {1, 4}:
        raise ValueError("affine vector width must be 1 or 4")
    reg_row = rf0
    reg_iterations = rf1
    reg_source = rf2
    reg_bias = rf4
    reg_destination = rf5
    reg_row_stride = rf6
    reg_scale = rf7
    reg_offset = rf8
    reg_stride = rf9
    reg_count = rf10
    reg_workgroup_offset = rf11
    source_values = [rf12, rf13, rf14, rf15]
    bias_values = [rf16, rf17, rf18, rf19]

    mov(reg_row, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_bias))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_row_stride))
    nop(sig=ldunifrf(reg_scale))
    umul24(reg_offset, reg_row, reg_row_stride)
    add(reg_source, reg_source, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    eidx(reg_offset)
    shl(reg_offset, reg_offset, 4 if vector_width == 4 else 2)
    add(reg_source, reg_source, reg_offset)
    add(reg_bias, reg_bias, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 8 if vector_width == 4 else 6)
    if vector_width == 4:
        bnot(reg_workgroup_offset, 3)
    mov(reg_count, reg_iterations)

    with loop as affine:
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_stride)
        nop()
        nop()
        for value in source_values[:vector_width]:
            nop(sig=ldtmu(value))
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_bias, sig=thrsw).add(reg_bias, reg_bias, reg_stride)
        nop()
        nop()
        for value in bias_values[:vector_width]:
            nop(sig=ldtmu(value))
        for value, bias_value in zip(source_values[:vector_width], bias_values[:vector_width], strict=True):
            fmul(value, value, reg_scale)
            fadd(value, value, bias_value)
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        for value in source_values[:vector_width]:
            mov(tmud, value)
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
        tmuwt()
        sub(reg_count, reg_count, 1, cond="pushz")
        affine.b(cond="na0")
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


def supports_affine_fp32(
    source: Tensor,
    bias: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors satisfy the row-parallel affine contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (source, bias, destination)):
        return False
    if (
        len(source.shape) != 2
        or destination.shape != source.shape
        or bias.shape != (source.shape[1],)
        or source.shape[1] % 16
    ):
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, bias, destination))


def _program_state(backend: PyVideoCore7Backend, vector_width: Literal[1, 4]) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(vector_width)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_affine_fp32, vector_width=vector_width),
                    uniforms=driver.alloc(7, dtype=np.uint32),
                )
            states[vector_width] = state
        return state


def _execute_affine_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 4 or not all(isinstance(argument, Tensor) for argument in args[:3]):
        raise KernelError("affine expects (source, bias, destination, scale)")
    source, bias, destination, scale = args
    if not isinstance(scale, float) or not np.isfinite(scale):
        raise KernelError("affine scale must be finite")
    if not supports_affine_fp32(source, bias, destination, backend):
        raise KernelError("affine requires contiguous FP32 rows with a 16-aligned width")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("affine requires PyVideoCore7Backend")
    expected = (source.shape[0], 1, 1)
    if grid != expected:
        raise KernelError(f"affine grid must be {expected}, got {grid}")
    vector_width: Literal[1, 4] = 4 if source.shape[1] % 64 == 0 else 1
    state = _program_state(backend, vector_width)
    state.uniforms[:] = (
        source.shape[1] // (16 * vector_width),
        source.address,
        bias.address,
        destination.address,
        source.numpy().strides[0],
        np.asarray(scale, dtype=np.float32).view(np.uint32),
        0,
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=source.shape[0],
        )


AFFINE_FP32_KERNEL = Kernel("vc7.affine_fp32", _execute_affine_fp32)

__all__ = ["AFFINE_FP32_KERNEL", "supports_affine_fp32"]
