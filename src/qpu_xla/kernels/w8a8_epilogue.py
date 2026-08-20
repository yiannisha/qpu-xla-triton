"""QPU dequantization epilogue for tiled W8A8 INT32 accumulators."""

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
def qpu_w8a8_dequantize(asm: Assembly) -> None:
    """Scale one 16x16 INT32 tile by its FP32 row and column scales."""
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_source_stride = rf3
    reg_source_base = rf4
    reg_row_scale_base = rf5
    reg_column_scale_base = rf6
    reg_destination_stride = rf7
    reg_destination_base = rf8
    reg_source_row = rf9
    reg_destination_row = rf10
    reg_column_scale = rf11
    reg_row_scale = rf12
    reg_value = rf13
    reg_output = rf14
    reg_tmp = rf15

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_source_stride))
    nop(sig=ldunifrf(reg_source_base))
    nop(sig=ldunifrf(reg_row_scale_base))
    nop(sig=ldunifrf(reg_column_scale_base))
    nop(sig=ldunifrf(reg_destination_stride))
    nop(sig=ldunifrf(reg_destination_base))

    shl(reg_tmp, reg_tile_i, 4)
    umul24(reg_source_row, reg_tmp, reg_source_stride)
    umul24(reg_destination_row, reg_tmp, reg_destination_stride)
    shl(reg_tmp, reg_tmp, 2)
    add(reg_row_scale_base, reg_row_scale_base, reg_tmp)
    shl(reg_tmp, reg_tile_j, 6)
    add(reg_source_row, reg_source_base, reg_source_row)
    add(reg_source_row, reg_source_row, reg_tmp)
    add(reg_destination_row, reg_destination_base, reg_destination_row)
    add(reg_destination_row, reg_destination_row, reg_tmp)
    add(reg_column_scale_base, reg_column_scale_base, reg_tmp)
    eidx(reg_tmp)
    shl(reg_tmp, reg_tmp, 2)
    add(reg_source_row, reg_source_row, reg_tmp)
    add(reg_destination_row, reg_destination_row, reg_tmp)
    add(reg_column_scale_base, reg_column_scale_base, reg_tmp)

    mov(tmua, reg_column_scale_base, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(reg_column_scale))
    mov(tmuc, -1)
    for index in range(16):
        mov(tmua, reg_source_row, sig=thrsw)
        nop()
        mov(tmua, reg_row_scale_base, sig=thrsw)
        nop(sig=ldtmu(reg_value))
        nop()
        nop(sig=ldtmu(reg_row_scale))
        itof(reg_output, reg_value)
        fmul(reg_output, reg_output, reg_row_scale)
        fmul(reg_output, reg_output, reg_column_scale)
        mov(tmud, reg_output)
        mov(tmua, reg_destination_row)
        tmuwt()
        if index < 15:
            add(reg_source_row, reg_source_row, reg_source_stride)
            add(reg_destination_row, reg_destination_row, reg_destination_stride)
            add(reg_row_scale_base, reg_row_scale_base, 4)

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


def supports_w8a8_dequantize(
    source: Tensor,
    row_scales: Tensor,
    column_scales: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors meet the tiled dequantization contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.int32) or any(
        tensor.dtype != np.dtype(np.float32) for tensor in (row_scales, column_scales, destination)
    ):
        return False
    if len(source.shape) != 2 or destination.shape != source.shape:
        return False
    rows, columns = source.shape
    if rows % 16 or columns % 16 or row_scales.shape != (rows,) or column_scales.shape != (columns,):
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, row_scales, column_scales, destination))


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_w8a8_dequantize),
                    uniforms=driver.alloc(6, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute_w8a8_dequantize(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 4 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("W8A8 dequantization expects (source, row_scales, column_scales, destination)")
    source, row_scales, column_scales, destination = args
    if not supports_w8a8_dequantize(source, row_scales, column_scales, destination, backend):
        raise KernelError("W8A8 dequantization requires contiguous 16x16-aligned INT32-to-FP32 tensors")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("W8A8 dequantization requires PyVideoCore7Backend")
    expected_grid = (source.shape[1] // 16, source.shape[0] // 16, 1)
    if grid != expected_grid:
        raise KernelError(f"W8A8 dequantization grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            source.numpy().strides[0],
            source.address,
            row_scales.address,
            column_scales.address,
            destination.numpy().strides[0],
            destination.address,
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=grid[0] * grid[1],
        )


W8A8_DEQUANTIZE_KERNEL = Kernel("vc7.w8a8_dequantize", _execute_w8a8_dequantize)

__all__ = ["W8A8_DEQUANTIZE_KERNEL", "supports_w8a8_dequantize"]
