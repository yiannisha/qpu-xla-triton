"""Cached-table FP32 rotary embedding kernel."""

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
def qpu_rope_fp32(
    asm: Assembly,
    *,
    vector_width: Literal[1, 4],
) -> None:
    """Rotate cached pairs with scalar or vec4 TMU accesses."""
    if vector_width not in {1, 4}:
        raise ValueError("RoPE vector width must be 1 or 4")

    reg_iterations = rf0
    reg_source = rf1
    reg_cosine = rf2
    reg_signed_sine = rf4
    reg_destination = rf5
    reg_workgroup = rf6
    reg_offset = rf7
    reg_stride = rf8
    if vector_width == 4:
        reg_remainder = rf9
        reg_base_iterations = rf10
        reg_extra_item = rf11
        reg_first_item = rf12
        reg_workgroup_offset = rf13
        source_values = [rf14, rf15, rf16, rf17]
        cosine_values = [rf18, rf19, rf20, rf21]
        sine_values = [rf22, rf23, rf24, rf25]
        outputs = [rf26, rf27, rf28, rf29]
        reg_temporary = rf30

        nop(sig=ldunifrf(reg_iterations))
        nop(sig=ldunifrf(reg_remainder))
        nop(sig=ldunifrf(reg_source))
        nop(sig=ldunifrf(reg_cosine))
        nop(sig=ldunifrf(reg_signed_sine))
        nop(sig=ldunifrf(reg_destination))

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
        add(reg_source, reg_source, reg_offset)
        add(reg_cosine, reg_cosine, reg_offset)
        add(reg_signed_sine, reg_signed_sine, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 8)
        bnot(reg_workgroup_offset, 3)

        nop(sig=thrsw)
        nop()
        nop()
        with loop as vector_rotation:
            mov(tmuc, reg_workgroup_offset)
            mov(tmua, reg_source).add(reg_source, reg_source, reg_stride)
            nop()
            nop()
            for value in source_values:
                nop(sig=ldtmu(value))

            mov(tmuc, reg_workgroup_offset)
            mov(tmua, reg_cosine).add(reg_cosine, reg_cosine, reg_stride)
            nop()
            nop()
            for value in cosine_values:
                nop(sig=ldtmu(value))

            mov(tmuc, reg_workgroup_offset)
            mov(tmua, reg_signed_sine).add(reg_signed_sine, reg_signed_sine, reg_stride)
            nop()
            nop()
            for value in sine_values:
                nop(sig=ldtmu(value))

            for output, source_value, adjacent, cosine_value, sine_value in (
                (outputs[0], source_values[0], source_values[1], cosine_values[0], sine_values[0]),
                (outputs[1], source_values[1], source_values[0], cosine_values[1], sine_values[1]),
                (outputs[2], source_values[2], source_values[3], cosine_values[2], sine_values[2]),
                (outputs[3], source_values[3], source_values[2], cosine_values[3], sine_values[3]),
            ):
                fmul(output, source_value, cosine_value)
                fmul(reg_temporary, adjacent, sine_value)
                fadd(output, output, reg_temporary)

            mov(tmuc, reg_workgroup_offset)
            for output in outputs:
                mov(tmud, output)
            sub(reg_iterations, reg_iterations, 1, cond="pushz")
            mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
            tmuwt()
            vector_rotation.b(cond="na0")
            nop()
            nop()
            nop()
        barrierid(syncb, sig=thrsw)
        nop()
        nop()
    else:
        reg_source_value = rf9
        reg_cosine_value = rf10
        reg_sine_value = rf11
        reg_adjacent = rf12
        reg_indices = rf13
        reg_output = rf14

        mov(reg_workgroup, rf3.unpack("ul"))
        nop(sig=ldunifrf(reg_iterations))
        nop(sig=ldunifrf(reg_source))
        nop(sig=ldunifrf(reg_cosine))
        nop(sig=ldunifrf(reg_signed_sine))
        nop(sig=ldunifrf(reg_destination))

        shl(reg_offset, reg_iterations, 6)
        umul24(reg_offset, reg_offset, reg_workgroup)
        add(reg_source, reg_source, reg_offset)
        add(reg_cosine, reg_cosine, reg_offset)
        add(reg_signed_sine, reg_signed_sine, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        eidx(reg_indices)
        shl(reg_offset, reg_indices, 2)
        add(reg_source, reg_source, reg_offset)
        add(reg_cosine, reg_cosine, reg_offset)
        add(reg_signed_sine, reg_signed_sine, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        bxor(reg_indices, reg_indices, 1)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)

        with loop as scalar_rotation:
            mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_stride)
            nop()
            mov(tmua, reg_cosine, sig=thrsw).add(reg_cosine, reg_cosine, reg_stride)
            nop(sig=ldtmu(reg_source_value))
            mov(tmua, reg_signed_sine, sig=thrsw).add(reg_signed_sine, reg_signed_sine, reg_stride)
            nop(sig=ldtmu(reg_cosine_value))
            nop()
            nop(sig=ldtmu(reg_sine_value))
            shuffle(reg_adjacent, reg_source_value, reg_indices)
            fmul(reg_output, reg_source_value, reg_cosine_value)
            fmul(reg_adjacent, reg_adjacent, reg_sine_value)
            fadd(reg_output, reg_output, reg_adjacent)
            mov(tmud, reg_output)
            sub(reg_iterations, reg_iterations, 1, cond="pushz")
            mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
            tmuwt()
            scalar_rotation.b(cond="na0")
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


def supports_rope_fp32(
    source: Tensor,
    cosine: Tensor,
    signed_sine: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether cached tables and values meet the vector kernel contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    tensors = (source, cosine, signed_sine, destination)
    if any(tensor.dtype != np.dtype(np.float32) or tensor.shape != source.shape for tensor in tensors):
        return False
    if len(source.shape) != 2 or source.shape[1] % 2 or source.nbytes % 64:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in tensors)


def _workgroup_count(vector_count: int) -> int:
    if vector_count >= 4 and vector_count % 4 == 0:
        return min(12, vector_count // 4)
    for count in range(min(12, vector_count), 0, -1):
        if vector_count % count == 0:
            return count
    return 1


def _program_state(backend: PyVideoCore7Backend, vector_width: Literal[1, 4]) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(vector_width)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_rope_fp32, vector_width=vector_width),
                    uniforms=driver.alloc(64, dtype=np.uint32),
                )
            states[vector_width] = state
        return state


def _execute_rope_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 4 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("RoPE expects (source, cosine, signed_sine, destination)")
    source, cosine, signed_sine, destination = args
    if not supports_rope_fp32(source, cosine, signed_sine, destination, backend):
        raise KernelError("RoPE requires equal contiguous FP32 tensors and complete 16-value vectors")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("RoPE requires PyVideoCore7Backend")
    vector_count = source.nbytes // 64
    workgroups = _workgroup_count(vector_count)
    expected_grid = (workgroups, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"RoPE grid must be {expected_grid}, got {grid}")
    vector_width: Literal[1, 4] = 4 if vector_count >= 4 and vector_count % 4 == 0 else 1
    state = _program_state(backend, vector_width)
    if vector_width == 4:
        work_items = vector_count // 4
        base_iterations, remainder = divmod(work_items, workgroups)
        state.uniforms[:6] = (
            base_iterations,
            remainder,
            source.address,
            cosine.address,
            signed_sine.address,
            destination.address,
        )
    else:
        state.uniforms[:5] = (
            vector_count // workgroups,
            source.address,
            cosine.address,
            signed_sine.address,
            destination.address,
        )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=expected_grid,
            wgs_per_sg=_WGS_PER_SG,
            thread=workgroups,
        )


ROPE_FP32_KERNEL = Kernel("vc7.rope_fp32", _execute_rope_fp32)

__all__ = ["ROPE_FP32_KERNEL", "supports_rope_fp32"]
