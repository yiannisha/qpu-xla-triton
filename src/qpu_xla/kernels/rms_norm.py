"""Row-parallel FP32 RMSNorm kernel for VideoCore VII."""

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
def qpu_rms_norm_fp32(asm: Assembly, *, vector_width: Literal[1, 4]) -> None:
    """Normalize one row per workgroup with scalar or vec4 TMU accesses."""
    if vector_width not in {1, 4}:
        raise ValueError("RMSNorm vector width must be 1 or 4")

    reg_row = rf0
    reg_iterations = rf1
    reg_source = rf2
    reg_weight = rf4
    reg_destination = rf5
    reg_row_stride = rf6
    reg_epsilon = rf7
    reg_inverse_width = rf8
    reg_lane_offset = rf9
    reg_vector_stride = rf10
    reg_sum = rf19
    reg_temporary = rf20
    reg_scale = rf21
    reg_output = rf22
    reg_count = rf26
    reg_workgroup_offset = rf27

    source_values = [rf11, rf12, rf13, rf14]
    weight_values = [rf15, rf16, rf17, rf18]

    mov(reg_row, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_weight))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_row_stride))
    nop(sig=ldunifrf(reg_epsilon))
    nop(sig=ldunifrf(reg_inverse_width))

    umul24(reg_temporary, reg_row, reg_row_stride)
    add(reg_source, reg_source, reg_temporary)
    add(reg_destination, reg_destination, reg_temporary)
    eidx(reg_lane_offset)
    shl(reg_lane_offset, reg_lane_offset, 4 if vector_width == 4 else 2)
    add(reg_source, reg_source, reg_lane_offset)
    add(reg_weight, reg_weight, reg_lane_offset)
    add(reg_destination, reg_destination, reg_lane_offset)
    mov(reg_vector_stride, 1)
    shl(reg_vector_stride, reg_vector_stride, 8 if vector_width == 4 else 6)
    if vector_width == 4:
        bnot(reg_workgroup_offset, 3)
    bxor(reg_sum, reg_sum, reg_sum)
    mov(reg_count, reg_iterations)
    mov(reg_temporary, reg_source)

    with loop as reduction:
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_temporary, sig=thrsw).add(reg_temporary, reg_temporary, reg_vector_stride)
        nop()
        nop()
        for source_value in source_values[:vector_width]:
            nop(sig=ldtmu(source_value))
        for source_value in source_values[:vector_width]:
            fmul(reg_output, source_value, source_value)
            fadd(reg_sum, reg_sum, reg_output)
        sub(reg_count, reg_count, 1, cond="pushz")
        reduction.b(cond="na0")
        nop()
        nop()
        nop()

    for distance in (8, 4, 2, 1):
        rotate(reg_temporary, reg_sum, distance)
        fadd(reg_sum, reg_sum, reg_temporary)
    fmul(reg_sum, reg_sum, reg_inverse_width)
    fadd(reg_sum, reg_sum, reg_epsilon)
    rsqrt(reg_scale, reg_sum)

    mov(reg_count, reg_iterations)
    with loop as output_loop:
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_vector_stride)
        nop()
        nop()
        for source_value in source_values[:vector_width]:
            nop(sig=ldtmu(source_value))
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        mov(tmua, reg_weight, sig=thrsw).add(reg_weight, reg_weight, reg_vector_stride)
        nop()
        nop()
        for weight_value in weight_values[:vector_width]:
            nop(sig=ldtmu(weight_value))
        for source_value, weight_value in zip(
            source_values[:vector_width], weight_values[:vector_width], strict=True
        ):
            fmul(source_value, source_value, reg_scale)
            fmul(source_value, source_value, weight_value)
        if vector_width == 4:
            mov(tmuc, reg_workgroup_offset)
        for source_value in source_values[:vector_width]:
            mov(tmud, source_value)
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_vector_stride)
        tmuwt()
        sub(reg_count, reg_count, 1, cond="pushz")
        output_loop.b(cond="na0")
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
_WGS_PER_SG = 24


def supports_rms_norm_fp32(source: Tensor, weight: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the row-parallel RMSNorm contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (source, weight, destination)):
        return False
    if len(source.shape) != 2 or destination.shape != source.shape or weight.shape != (source.shape[1],):
        return False
    if source.shape[1] % 16 or source.shape[1] * np.dtype(np.float32).itemsize >= 1 << 24:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, weight, destination))


def _program_state(backend: PyVideoCore7Backend, vector_width: Literal[1, 4]) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(vector_width)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_rms_norm_fp32, vector_width=vector_width),
                    uniforms=driver.alloc(64, dtype=np.uint32),
                )
            states[vector_width] = state
        return state


def _execute_rms_norm_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 4 or not all(isinstance(argument, Tensor) for argument in args[:3]):
        raise KernelError("RMSNorm expects (source, weight, destination, epsilon)")
    source, weight, destination, epsilon = args
    if not isinstance(epsilon, float) or not np.isfinite(epsilon) or epsilon <= 0:
        raise KernelError("RMSNorm epsilon must be a finite positive float")
    if not supports_rms_norm_fp32(source, weight, destination, backend):
        raise KernelError("RMSNorm requires contiguous FP32 rows with a 16-aligned width")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("RMSNorm requires PyVideoCore7Backend")
    expected_grid = (source.shape[0], 1, 1)
    if grid != expected_grid:
        raise KernelError(f"RMSNorm grid must be {expected_grid}, got {grid}")
    vector_width: Literal[1, 4] = 4 if source.shape[1] % 64 == 0 else 1
    state = _program_state(backend, vector_width)
    state.uniforms[:7] = (
        source.shape[1] // (16 * vector_width),
        source.address,
        weight.address,
        destination.address,
        source.numpy().strides[0],
        np.asarray(epsilon, dtype=np.float32).view(np.uint32),
        np.asarray(1.0 / source.shape[1], dtype=np.float32).view(np.uint32),
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


RMS_NORM_FP32_KERNEL = Kernel("vc7.rms_norm_fp32", _execute_rms_norm_fp32)

__all__ = ["RMS_NORM_FP32_KERNEL", "supports_rms_norm_fp32"]
