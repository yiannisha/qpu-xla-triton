"""Cached VideoCore VII fused FP32 SiLU-times-gate kernel."""

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
def qpu_swiglu_fp32(
    asm: Assembly,
    *,
    vector_width: Literal[1, 4],
    num_qpus: int,
) -> None:
    """Apply ``silu(gate) * up`` to one or four words per SIMD lane."""
    if vector_width not in {1, 4}:
        raise ValueError("SwiGLU vector width must be 1 or 4")
    if not 1 <= num_qpus <= 12:
        raise ValueError("SwiGLU QPU count must be between 1 and 12")

    reg_iterations = rf0
    reg_gate = rf1
    reg_up = rf2
    reg_destination = rf3
    reg_log2_e = rf4
    reg_offset = rf5
    reg_stride = rf6
    reg_workgroup = rf7
    reg_workgroup_offset = rf8
    reg_remainder = rf9
    reg_gate_value = rf10
    reg_up_value = rf11
    reg_exp = rf12
    reg_output = rf13

    if vector_width == 1:
        mov(reg_workgroup, rf3.unpack("ul"))
        nop(sig=ldunifrf(reg_iterations))
        nop(sig=ldunifrf(reg_gate))
        nop(sig=ldunifrf(reg_up))
        nop(sig=ldunifrf(reg_destination))
        nop(sig=ldunifrf(reg_log2_e))
    else:
        nop(sig=ldunifrf(reg_iterations))
        nop(sig=ldunifrf(reg_remainder))
        nop(sig=ldunifrf(reg_gate))
        nop(sig=ldunifrf(reg_up))
        nop(sig=ldunifrf(reg_destination))
        nop(sig=ldunifrf(reg_log2_e))

    if vector_width == 4:
        reg_base_iterations = rf26
        reg_extra_item = rf27
        reg_first_item = rf28
        mov(reg_base_iterations, reg_iterations)
        tidx(reg_workgroup)
        shr(reg_workgroup, reg_workgroup, 2)
        band(reg_workgroup, reg_workgroup, 0b1111)
        mov(reg_extra_item, reg_workgroup)
        sub(null, reg_workgroup, reg_remainder, cond="pushn")
        mov(reg_extra_item, reg_remainder, cond="ifna")
        add(reg_iterations, reg_iterations, 1, cond="ifa")
        umul24(reg_first_item, reg_workgroup, reg_base_iterations)
        add(reg_first_item, reg_first_item, reg_extra_item)
        shl(reg_offset, reg_first_item, 8)
        eidx(rf31)
        shl(rf31, rf31, 4)
        add(reg_offset, reg_offset, rf31)
        add(reg_gate, reg_gate, reg_offset)
        add(reg_up, reg_up, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 8)
        bnot(reg_workgroup_offset, 3)

        gate_values = [rf10, rf11, rf12, rf13]
        up_values = [rf14, rf15, rf16, rf17]
        exponentials = [rf18, rf19, rf20, rf21]
        outputs = [rf22, rf23, rf24, rf25]

        nop(sig=thrsw)
        nop()
        nop()
        with loop as vector_loop:
            mov(tmuc, reg_workgroup_offset)
            mov(tmua, reg_gate)
            add(reg_gate, reg_gate, reg_stride)
            nop()
            nop()
            for value in gate_values:
                nop(sig=ldtmu(value))

            mov(tmuc, reg_workgroup_offset)
            mov(tmua, reg_up)
            add(reg_up, reg_up, reg_stride)
            nop()
            nop()
            for value in up_values:
                nop(sig=ldtmu(value))

            for exponential, gate_value in zip(exponentials, gate_values, strict=True):
                fmul(exponential, gate_value, reg_log2_e)
            for exponential in exponentials:
                exp(exponential, exponential)
            for exponential in exponentials:
                fadd(exponential, exponential, 1.0)
            for exponential in exponentials:
                recip(exponential, exponential)
            for output, gate_value, exponential, up_value in zip(
                outputs,
                gate_values,
                exponentials,
                up_values,
                strict=True,
            ):
                fmul(output, gate_value, exponential)
                fmul(output, output, up_value)

            mov(tmuc, reg_workgroup_offset)
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
        barrierid(syncb, sig=thrsw)
        nop()
        nop()
    else:
        shl(reg_workgroup_offset, reg_iterations, 6)
        umul24(reg_workgroup_offset, reg_workgroup_offset, reg_workgroup)
        add(reg_gate, reg_gate, reg_workgroup_offset)
        add(reg_up, reg_up, reg_workgroup_offset)
        add(reg_destination, reg_destination, reg_workgroup_offset)
        eidx(reg_offset)
        shl(reg_offset, reg_offset, 2)
        add(reg_gate, reg_gate, reg_offset)
        add(reg_up, reg_up, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)

        with loop as scalar_loop:
            mov(tmua, reg_gate, sig=thrsw).add(reg_gate, reg_gate, reg_stride)
            nop()
            mov(tmua, reg_up, sig=thrsw).add(reg_up, reg_up, reg_stride)
            nop(sig=ldtmu(reg_gate_value))
            nop()
            nop(sig=ldtmu(reg_up_value))

            fmul(reg_exp, reg_gate_value, reg_log2_e)
            exp(reg_exp, reg_exp)
            fadd(reg_exp, reg_exp, 1.0)
            recip(reg_exp, reg_exp)
            fmul(reg_output, reg_gate_value, reg_exp)
            fmul(reg_output, reg_output, reg_up_value)
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
    """Driver-local program and uniform storage."""

    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[int, int], _ProgramState]] = (
    WeakKeyDictionary()
)


def supports_swiglu_fp32(gate: Tensor, up: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether the fused contiguous FP32 specialization applies."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    tensors = (gate, up, destination)
    if any(tensor.dtype != np.dtype(np.float32) or tensor.shape != destination.shape for tensor in tensors):
        return False
    if destination.nbytes == 0 or destination.nbytes % (16 * np.dtype(np.float32).itemsize):
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in tensors)


def _workgroup_count(vector_count: int) -> int:
    """Use the widest exact partition that does not require a tail branch."""
    for count in range(min(12, vector_count), 0, -1):
        if vector_count % count == 0:
            return count
    return 1


def _program_state(
    backend: PyVideoCore7Backend,
    vector_width: Literal[1, 4],
    num_qpus: int,
) -> _ProgramState:
    """Assemble the fused kernel once per live backend."""
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        key = (vector_width, num_qpus)
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(
                        qpu_swiglu_fp32,
                        vector_width=vector_width,
                        num_qpus=num_qpus,
                    ),
                    uniforms=driver.alloc(64, dtype=np.uint32),
                )
            states[key] = state
        return state


def _execute_swiglu_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    """Validate and dispatch one fused SwiGLU operation."""
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("SwiGLU expects (gate, up, destination)")
    gate, up, destination = args
    if not supports_swiglu_fp32(gate, up, destination, backend):
        raise KernelError("SwiGLU requires equal contiguous FP32 tensors with a multiple of 16 elements")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("SwiGLU requires PyVideoCore7Backend")
    vector_count = destination.nbytes // (16 * np.dtype(np.float32).itemsize)
    workgroups = _workgroup_count(vector_count)
    expected_grid = (workgroups, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"SwiGLU grid must be {expected_grid}, got {grid}")
    vector_width: Literal[1, 4] = 1
    work_items = vector_count
    num_qpus = workgroups
    if vector_count % 4 == 0:
        vector_work_items = vector_count // 4
        vector_qpus = min(12, vector_work_items)
        if vector_qpus >= 6:
            vector_width = 4
            work_items = vector_work_items
            num_qpus = vector_qpus
    state = _program_state(backend, vector_width, num_qpus)
    with backend.driver_session() as driver:
        if vector_width == 4:
            base_iterations, remainder = divmod(work_items, num_qpus)
            state.uniforms[:6] = (
                base_iterations,
                remainder,
                gate.address,
                up.address,
                destination.address,
                np.asarray(-np.log2(np.e), dtype=np.float32).view(np.uint32),
            )
            driver.execute(
                state.code,
                local_invocation=(16, 1, 1),
                uniforms=state.uniforms.addresses()[0],
                thread=num_qpus,
            )
        else:
            state.uniforms[:6] = (
                work_items // num_qpus,
                gate.address,
                up.address,
                destination.address,
                np.asarray(-np.log2(np.e), dtype=np.float32).view(np.uint32),
                0,
            )
            driver.execute(
                state.code,
                local_invocation=(16, 1, 1),
                uniforms=state.uniforms.addresses()[0],
                workgroup=expected_grid,
                wgs_per_sg=24,
                thread=workgroups,
            )


SWIGLU_FP32_KERNEL = Kernel("vc7.swiglu_fp32", _execute_swiglu_fp32)


__all__ = ["SWIGLU_FP32_KERNEL", "supports_swiglu_fp32"]
