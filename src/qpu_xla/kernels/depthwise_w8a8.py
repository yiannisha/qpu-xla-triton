"""Direct packed W8A8 depthwise 3x3 kernel for VideoCore VII."""

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
def qpu_depthwise_w8a8_3x3(asm: Assembly) -> None:
    """Compute 16 independently quantized depthwise outputs per workgroup."""
    reg_workgroup = rf0
    reg_lane = rf1
    reg_flat = rf2
    reg_group = rf3
    reg_source = rf4
    reg_weight = rf5
    reg_row_scale = rf6
    reg_column_scale = rf7
    reg_destination = rf8
    reg_group_mask = rf9
    reg_offset = rf10
    reg_source_word = [rf11, rf12, rf13, rf14]
    reg_weight_word = [rf15, rf16, rf17, rf18]
    reg_source_value = rf19
    reg_weight_value = rf20
    reg_row_scale_value = rf21
    reg_column_scale_value = rf22
    reg_accumulator = rf23
    reg_product = rf24
    reg_output = rf25

    mov(reg_workgroup, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_weight))
    nop(sig=ldunifrf(reg_row_scale))
    nop(sig=ldunifrf(reg_column_scale))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_group_mask))

    eidx(reg_lane)
    shl(reg_flat, reg_workgroup, 4)
    add(reg_flat, reg_flat, reg_lane)
    band(reg_group, reg_flat, reg_group_mask)

    shl(reg_offset, reg_flat, 4)
    add(reg_source, reg_source, reg_offset)
    shl(reg_offset, reg_group, 4)
    add(reg_weight, reg_weight, reg_offset)
    shl(reg_offset, reg_flat, 2)
    add(reg_row_scale, reg_row_scale, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    shl(reg_offset, reg_group, 2)
    add(reg_column_scale, reg_column_scale, reg_offset)

    mov(reg_offset, 4)
    for index in range(4):
        mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_offset)
        nop()
        mov(tmua, reg_weight, sig=thrsw).add(reg_weight, reg_weight, reg_offset)
        nop(sig=ldtmu(reg_source_value))
        nop()
        nop(sig=ldtmu(reg_weight_value))
        mov(reg_source_word[index], reg_source_value)
        mov(reg_weight_word[index], reg_weight_value)

    mov(tmua, reg_row_scale, sig=thrsw)
    nop()
    mov(tmua, reg_column_scale, sig=thrsw)
    nop(sig=ldtmu(reg_row_scale_value))
    nop()
    nop(sig=ldtmu(reg_column_scale_value))

    setnnmode_ss()
    bxor(reg_accumulator, reg_accumulator, reg_accumulator)
    for index in range(4):
        v8dot(reg_product, reg_source_word[index], reg_weight_word[index])
        add(reg_accumulator, reg_accumulator, reg_product)
    itof(reg_output, reg_accumulator)
    fmul(reg_output, reg_output, reg_row_scale_value)
    fmul(reg_output, reg_output, reg_column_scale_value)
    mov(tmud, reg_output)
    mov(tmua, reg_destination)
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


def supports_depthwise_w8a8_3x3(
    source: Tensor,
    weight: Tensor,
    row_scales: Tensor,
    column_scales: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors meet the direct depthwise packed-K16 contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.uint32) or weight.dtype != np.dtype(np.uint32):
        return False
    if any(tensor.dtype != np.dtype(np.float32) for tensor in (row_scales, column_scales, destination)):
        return False
    if len(source.shape) != 2 or source.shape[1] != 4 or weight.shape[1:] != (4,):
        return False
    elements = source.shape[0]
    groups = weight.shape[0]
    if elements % 16 or groups < 16 or groups & (groups - 1):
        return False
    if row_scales.shape != (elements,) or column_scales.shape != (groups,) or destination.shape != (elements,):
        return False
    return all(
        tensor.numpy().flags.c_contiguous for tensor in (source, weight, row_scales, column_scales, destination)
    )


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_depthwise_w8a8_3x3),
                    uniforms=driver.alloc(6, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 5 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("depthwise W8A8 3x3 expects packed source, weight, scales, and destination")
    source, weight, row_scales, column_scales, destination = args
    if not supports_depthwise_w8a8_3x3(
        source,
        weight,
        row_scales,
        column_scales,
        destination,
        backend,
    ):
        raise KernelError("depthwise W8A8 3x3 requires power-of-two groups and packed K16 operands")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("depthwise W8A8 3x3 requires PyVideoCore7Backend")
    expected_grid = (source.shape[0] // 16, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"depthwise W8A8 3x3 grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            source.address,
            weight.address,
            row_scales.address,
            column_scales.address,
            destination.address,
            weight.shape[0] - 1,
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=expected_grid,
            wgs_per_sg=24,
            thread=expected_grid[0],
        )


DEPTHWISE_W8A8_3X3_KERNEL = Kernel("vc7.depthwise_w8a8_3x3", _execute)

__all__ = ["DEPTHWISE_W8A8_3X3_KERNEL", "supports_depthwise_w8a8_3x3"]
