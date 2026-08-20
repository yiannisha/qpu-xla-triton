"""Cached VideoCore VII FP32 residual-add kernel."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from weakref import WeakKeyDictionary

import numpy as np

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.kernels.swiglu import _workgroup_count
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu


@qpu
def qpu_residual_add_fp32(asm: Assembly, *, vector_width: Literal[1, 4]) -> None:
    if vector_width not in {1, 4}:
        raise ValueError("residual-add vector width must be 1 or 4")

    reg_iterations = rf0
    reg_left = rf1
    reg_right = rf2
    reg_destination = rf3
    reg_offset = rf4
    reg_stride = rf5
    reg_workgroup = rf6
    reg_workgroup_offset = rf7
    reg_left_value = rf10
    reg_right_value = rf11
    reg_output = rf12
    reg_tmu_config = rf13

    mov(reg_workgroup, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_left))
    nop(sig=ldunifrf(reg_right))
    nop(sig=ldunifrf(reg_destination))
    shl(reg_workgroup_offset, reg_iterations, 8 if vector_width == 4 else 6)
    umul24(reg_workgroup_offset, reg_workgroup_offset, reg_workgroup)
    add(reg_left, reg_left, reg_workgroup_offset)
    add(reg_right, reg_right, reg_workgroup_offset)
    add(reg_destination, reg_destination, reg_workgroup_offset)
    eidx(reg_offset)
    shl(reg_offset, reg_offset, 4 if vector_width == 4 else 2)
    add(reg_left, reg_left, reg_offset)
    add(reg_right, reg_right, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 8 if vector_width == 4 else 6)

    if vector_width == 4:
        left_values = [rf10, rf11, rf12, rf14]
        right_values = [rf15, rf16, rf17, rf18]
        outputs = [rf19, rf20, rf21, rf22]
        bnot(reg_tmu_config, 3)

        with loop as vector_loop:
            mov(tmuc, reg_tmu_config)
            mov(tmua, reg_left, sig=thrsw)
            add(reg_left, reg_left, reg_stride)
            nop()
            nop()
            for value in left_values:
                nop(sig=ldtmu(value))

            mov(tmuc, reg_tmu_config)
            mov(tmua, reg_right, sig=thrsw)
            add(reg_right, reg_right, reg_stride)
            nop()
            nop()
            for value in right_values:
                nop(sig=ldtmu(value))

            for output, left_value, right_value in zip(outputs, left_values, right_values, strict=True):
                fadd(output, left_value, right_value)

            mov(tmuc, reg_tmu_config)
            for output in outputs:
                mov(tmud, output)
            mov(tmua, reg_destination)
            add(reg_destination, reg_destination, reg_stride)
            tmuwt()
            sub(reg_iterations, reg_iterations, 1, cond="pushz")

            vector_loop.b(cond="na0")
            nop()
            nop()
            nop()
    else:
        with loop as scalar_loop:
            mov(tmua, reg_left, sig=thrsw).add(reg_left, reg_left, reg_stride)
            nop()
            mov(tmua, reg_right, sig=thrsw).add(reg_right, reg_right, reg_stride)
            nop(sig=ldtmu(reg_left_value))
            nop()
            nop(sig=ldtmu(reg_right_value))
            fadd(reg_output, reg_left_value, reg_right_value)
            mov(tmud, reg_output)
            sub(reg_iterations, reg_iterations, 1, cond="pushz")
            mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
            tmuwt()
            scalar_loop.b(cond="na0")
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


def supports_residual_add_fp32(left: Tensor, right: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the contiguous FP32 vector contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    tensors = (left, right, destination)
    if any(t.dtype != np.dtype(np.float32) or t.shape != destination.shape for t in tensors):
        return False
    return destination.nbytes % 64 == 0 and all(t.numpy().flags.c_contiguous for t in tensors)


def _program_state(backend: PyVideoCore7Backend, vector_width: Literal[1, 4]) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(vector_width)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    driver.program(qpu_residual_add_fp32, vector_width=vector_width),
                    driver.alloc(4, dtype=np.uint32),
                )
            states[vector_width] = state
        return state


def _execute(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(arg, Tensor) for arg in args):
        raise KernelError("residual add expects (left, right, destination)")
    left, right, destination = args
    if not supports_residual_add_fp32(left, right, destination, backend) or not isinstance(
        backend, PyVideoCore7Backend
    ):
        raise KernelError("residual add requires equal contiguous FP32 tensors aligned to 16 values")
    vectors = destination.nbytes // 64
    workgroups = _workgroup_count(vectors)
    expected = (workgroups, 1, 1)
    if grid != expected:
        raise KernelError(f"residual add grid must be {expected}, got {grid}")
    vector_width: Literal[1, 4] = 1
    work_items = vectors
    if vectors % 4 == 0 and workgroups >= 6 and (vectors // 4) % workgroups == 0:
        vector_width = 4
        work_items = vectors // 4
    state = _program_state(backend, vector_width)
    with backend.driver_session() as driver:
        state.uniforms[:] = (work_items // workgroups, left.address, right.address, destination.address)
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=workgroups,
        )


RESIDUAL_ADD_FP32_KERNEL = Kernel("vc7.residual_add_fp32", _execute)

__all__ = ["RESIDUAL_ADD_FP32_KERNEL", "supports_residual_add_fp32"]
