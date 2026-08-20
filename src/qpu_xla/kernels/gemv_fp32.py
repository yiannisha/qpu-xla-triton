"""Single-row FP32 GEMV candidate for autoregressive decode."""

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
def qpu_fp32_gemv(asm: Assembly) -> None:
    """Compute 16 output columns for one FP32 activation row."""
    reg_tile = rf1
    reg_i = rf2
    reg_source_base = rf3
    reg_weight_stride = rf4
    reg_weight_base = rf5
    reg_destination_base = rf6
    reg_source_value = rf7
    reg_weight_value = rf8
    reg_accum = rf11
    reg_product = rf12
    reg_tmp = rf13

    mov(reg_tile, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_i))
    nop(sig=ldunifrf(reg_source_base))
    nop(sig=ldunifrf(reg_weight_stride))
    nop(sig=ldunifrf(reg_weight_base))
    nop(sig=ldunifrf(reg_destination_base))

    shl(reg_tmp, reg_tile, 6)
    add(reg_weight_base, reg_weight_base, reg_tmp)
    add(reg_destination_base, reg_destination_base, reg_tmp)
    eidx(reg_tmp)
    shl(reg_tmp, reg_tmp, 2)
    add(reg_weight_base, reg_weight_base, reg_tmp)
    add(reg_destination_base, reg_destination_base, reg_tmp)
    bxor(reg_accum, reg_accum, reg_accum)
    mov(tmuc, -1)

    with loop as reduction:
        mov(tmua, reg_source_base, sig=thrsw)
        add(reg_source_base, reg_source_base, 4)
        nop()
        mov(tmua, reg_weight_base, sig=thrsw)
        add(reg_weight_base, reg_weight_base, reg_weight_stride)
        nop()
        nop(sig=ldtmu(reg_source_value))
        nop(sig=ldtmu(reg_weight_value))
        fmul(reg_product, reg_source_value, reg_weight_value)
        fadd(reg_accum, reg_accum, reg_product)
        sub(reg_i, reg_i, 1, cond="pushz")
        reduction.b(cond="na0")
        nop()
        nop()
        nop()

    mov(tmud, reg_accum)
    mov(tmua, reg_destination_base)
    tmuwt()
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


def supports_fp32_gemv(source: Tensor, weight: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the one-row FP32 GEMV contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (source, weight, destination)):
        return False
    if len(source.shape) != 2 or len(weight.shape) != 2 or len(destination.shape) != 2:
        return False
    if source.shape[0] != 1 or destination.shape[0] != 1:
        return False
    if source.shape[1] <= 0 or weight.shape[0] != source.shape[1]:
        return False
    if destination.shape[1] != weight.shape[1] or weight.shape[1] % 16:
        return False
    return all(tensor.numpy().strides[-1] == tensor.dtype.itemsize for tensor in (source, weight, destination))


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(code=driver.program(qpu_fp32_gemv), uniforms=driver.alloc(5, dtype=np.uint32))
            _PROGRAMS[backend] = state
        return state


def _execute_fp32_gemv(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("FP32 GEMV expects (source, weight, destination)")
    source, weight, destination = args
    if not supports_fp32_gemv(source, weight, destination, backend):
        raise KernelError("FP32 GEMV requires one source row and 16-aligned output columns")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("FP32 GEMV requires PyVideoCore7Backend")
    expected_grid = (weight.shape[1] // 16, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"FP32 GEMV grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            source.shape[1],
            source.address,
            weight.numpy().strides[0],
            weight.address,
            destination.address,
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=grid[0],
        )


FP32_GEMV_KERNEL = Kernel("vc7.fp32_gemv", _execute_fp32_gemv)

__all__ = ["FP32_GEMV_KERNEL", "supports_fp32_gemv"]
