"""Row-parallel FP32 argmax reduction for greedy sampling."""

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
def qpu_argmax_fp32(asm: Assembly) -> None:
    reg_row = rf0
    reg_iterations = rf1
    reg_source = rf2
    reg_destination = rf4
    reg_source_stride = rf5
    reg_destination_stride = rf6
    reg_negative_inf = rf7
    reg_lane = rf8
    reg_vector_stride = rf9
    reg_pointer = rf10
    reg_value = rf11
    reg_maximum = rf12
    reg_maximum_index = rf13
    reg_current_index = rf14
    reg_index_stride = rf15
    reg_count = rf16
    reg_other_maximum = rf17
    reg_other_index = rf18
    reg_offset = rf19

    mov(reg_row, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_source_stride))
    nop(sig=ldunifrf(reg_destination_stride))
    nop(sig=ldunifrf(reg_negative_inf))
    umul24(reg_offset, reg_row, reg_source_stride)
    add(reg_source, reg_source, reg_offset)
    umul24(reg_offset, reg_row, reg_destination_stride)
    add(reg_destination, reg_destination, reg_offset)
    eidx(reg_lane)
    shl(reg_offset, reg_lane, 2)
    add(reg_source, reg_source, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_vector_stride, 1)
    shl(reg_vector_stride, reg_vector_stride, 6)
    mov(reg_index_stride, 1)
    shl(reg_index_stride, reg_index_stride, 4)
    mov(reg_pointer, reg_source)
    mov(reg_current_index, reg_lane)
    mov(reg_maximum, reg_negative_inf)
    mov(reg_maximum_index, reg_lane)
    mov(reg_count, reg_iterations)

    with loop as scan_loop:
        mov(tmua, reg_pointer, sig=thrsw).add(reg_pointer, reg_pointer, reg_vector_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_value))
        fcmp(null, reg_maximum, reg_value, cond="pushn")
        mov(reg_maximum, reg_value, cond="ifa")
        mov(reg_maximum_index, reg_current_index, cond="ifa")
        add(reg_current_index, reg_current_index, reg_index_stride)
        sub(reg_count, reg_count, 1, cond="pushz")
        scan_loop.b(cond="na0")
        nop()
        nop()
        nop()

    for distance in (8, 4, 2, 1):
        rotate(reg_other_maximum, reg_maximum, distance)
        rotate(reg_other_index, reg_maximum_index, distance)
        # Resolve equal values to the smallest index, matching NumPy/Torch.
        fcmp(null, reg_maximum, reg_other_maximum, cond="pushz")
        sub(null, reg_other_index, reg_maximum_index, cond="andn")
        mov(reg_maximum_index, reg_other_index, cond="ifa")
        fcmp(null, reg_maximum, reg_other_maximum, cond="pushn")
        mov(reg_maximum, reg_other_maximum, cond="ifa")
        mov(reg_maximum_index, reg_other_index, cond="ifa")

    mov(tmud, reg_maximum_index)
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


_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, _ProgramState] = WeakKeyDictionary()


def supports_argmax_fp32(source: Tensor, scratch: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the row-reduction scratch contract."""
    return (
        isinstance(backend, PyVideoCore7Backend)
        and source.dtype == np.dtype(np.float32)
        and scratch.dtype == np.dtype(np.int32)
        and len(source.shape) == 2
        and source.shape[1] % 16 == 0
        and scratch.shape == (source.shape[0], 16)
        and source.numpy().flags.c_contiguous
        and scratch.numpy().flags.c_contiguous
    )


def _state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(driver.program(qpu_argmax_fp32), driver.alloc(6, dtype=np.uint32))
            _PROGRAMS[backend] = state
        return state


def _execute(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 2 or not all(isinstance(arg, Tensor) for arg in args):
        raise KernelError("argmax expects (source, scratch)")
    source, scratch = args
    if not supports_argmax_fp32(source, scratch, backend) or not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("argmax requires contiguous 16-aligned FP32 rows and rows-by-16 INT32 scratch")
    expected = (source.shape[0], 1, 1)
    if grid != expected:
        raise KernelError(f"argmax grid must be {expected}, got {grid}")
    state = _state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            source.shape[1] // 16,
            source.address,
            scratch.address,
            source.numpy().strides[0],
            scratch.numpy().strides[0],
            np.asarray(-np.inf, np.float32).view(np.uint32),
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=source.shape[0],
        )


ARGMAX_FP32_KERNEL = Kernel("vc7.argmax_fp32", _execute)

__all__ = ["ARGMAX_FP32_KERNEL", "supports_argmax_fp32"]
