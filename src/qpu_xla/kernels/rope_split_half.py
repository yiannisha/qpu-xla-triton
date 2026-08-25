"""Split-half FP32 rotary embedding kernel used by SmolVLA/Gemma."""

from __future__ import annotations

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
def qpu_rope_split_half_fp32(asm: Assembly) -> None:
    """Rotate one split-half head row per workgroup."""
    reg_row = rf0
    reg_iterations = rf1
    reg_source = rf2
    reg_cosine = rf4
    reg_sine = rf5
    reg_destination = rf6
    reg_row_stride = rf7
    reg_half_offset = rf8
    reg_lane_offset = rf9
    reg_stride = rf10
    reg_second_source = rf11
    reg_second_destination = rf12
    reg_first = rf13
    reg_second = rf14
    reg_cosine_value = rf15
    reg_sine_value = rf16
    reg_temporary = rf17
    reg_first_output = rf18
    reg_second_output = rf19
    reg_count = rf20
    reg_row_offset = rf21

    mov(reg_row, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_cosine))
    nop(sig=ldunifrf(reg_sine))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_row_stride))
    nop(sig=ldunifrf(reg_half_offset))

    umul24(reg_row_offset, reg_row, reg_row_stride)
    add(reg_source, reg_source, reg_row_offset)
    add(reg_destination, reg_destination, reg_row_offset)
    shr(reg_row_offset, reg_row_offset, 1)
    add(reg_cosine, reg_cosine, reg_row_offset)
    add(reg_sine, reg_sine, reg_row_offset)
    eidx(reg_lane_offset)
    shl(reg_lane_offset, reg_lane_offset, 2)
    add(reg_source, reg_source, reg_lane_offset)
    add(reg_cosine, reg_cosine, reg_lane_offset)
    add(reg_sine, reg_sine, reg_lane_offset)
    add(reg_destination, reg_destination, reg_lane_offset)
    add(reg_second_source, reg_source, reg_half_offset)
    add(reg_second_destination, reg_destination, reg_half_offset)
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6)
    mov(reg_count, reg_iterations)

    with loop as rotation:
        mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_stride)
        nop()
        mov(tmua, reg_second_source, sig=thrsw).add(reg_second_source, reg_second_source, reg_stride)
        nop(sig=ldtmu(reg_first))
        mov(tmua, reg_cosine, sig=thrsw).add(reg_cosine, reg_cosine, reg_stride)
        nop(sig=ldtmu(reg_second))
        mov(tmua, reg_sine, sig=thrsw).add(reg_sine, reg_sine, reg_stride)
        nop(sig=ldtmu(reg_cosine_value))
        nop()
        nop(sig=ldtmu(reg_sine_value))

        fmul(reg_first_output, reg_first, reg_cosine_value)
        fmul(reg_temporary, reg_second, reg_sine_value)
        fsub(reg_first_output, reg_first_output, reg_temporary)
        fmul(reg_second_output, reg_second, reg_cosine_value)
        fmul(reg_temporary, reg_first, reg_sine_value)
        fadd(reg_second_output, reg_second_output, reg_temporary)

        mov(tmud, reg_first_output)
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
        tmuwt()
        mov(tmud, reg_second_output)
        mov(tmua, reg_second_destination).add(reg_second_destination, reg_second_destination, reg_stride)
        tmuwt()
        sub(reg_count, reg_count, 1, cond="pushz")
        rotation.b(cond="na0")
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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, _ProgramState] = WeakKeyDictionary()
_WGS_PER_SG = 24


def supports_rope_split_half_fp32(
    source: Tensor,
    cosine: Tensor,
    sine: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors meet the one-head-per-row split-half contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (source, cosine, sine, destination)):
        return False
    if (
        len(source.shape) != 2
        or destination.shape != source.shape
        or source.shape[1] % 32
        or cosine.shape != (source.shape[0], source.shape[1] // 2)
        or sine.shape != cosine.shape
    ):
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, cosine, sine, destination))


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_rope_split_half_fp32),
                    uniforms=driver.alloc(64, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute_rope_split_half_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 4 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("split-half RoPE expects (source, cosine, sine, destination)")
    source, cosine, sine, destination = args
    if not supports_rope_split_half_fp32(source, cosine, sine, destination, backend):
        raise KernelError("split-half RoPE requires contiguous FP32 head rows with a 32-aligned width")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("split-half RoPE requires PyVideoCore7Backend")
    expected_grid = (source.shape[0], 1, 1)
    if grid != expected_grid:
        raise KernelError(f"split-half RoPE grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    state.uniforms[:7] = (
        source.shape[1] // 2 // 16,
        source.address,
        cosine.address,
        sine.address,
        destination.address,
        source.numpy().strides[0],
        source.shape[1] // 2 * np.dtype(np.float32).itemsize,
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


ROPE_SPLIT_HALF_FP32_KERNEL = Kernel("vc7.rope_split_half_fp32", _execute_rope_split_half_fp32)

__all__ = ["ROPE_SPLIT_HALF_FP32_KERNEL", "supports_rope_split_half_fp32"]
