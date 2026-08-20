"""Row-parallel FP32 embedding gather kernel."""

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
def qpu_embedding_lookup_fp32(asm: Assembly) -> None:
    reg_row = rf0
    reg_iterations = rf1
    reg_ids = rf2
    reg_table = rf4
    reg_table_stride = rf5
    reg_destination = rf6
    reg_destination_stride = rf7
    reg_id = rf8
    reg_offset = rf9
    reg_vector_stride = rf10
    reg_value = rf11

    mov(reg_row, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_ids))
    nop(sig=ldunifrf(reg_table))
    nop(sig=ldunifrf(reg_table_stride))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_destination_stride))
    shl(reg_offset, reg_row, 2)
    add(reg_ids, reg_ids, reg_offset)
    mov(tmua, reg_ids, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(reg_id))
    umul24(reg_offset, reg_id, reg_table_stride)
    add(reg_table, reg_table, reg_offset)
    umul24(reg_offset, reg_row, reg_destination_stride)
    add(reg_destination, reg_destination, reg_offset)
    eidx(reg_offset)
    shl(reg_offset, reg_offset, 2)
    add(reg_table, reg_table, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_vector_stride, 1)
    shl(reg_vector_stride, reg_vector_stride, 6)

    with loop as copy_loop:
        mov(tmua, reg_table, sig=thrsw).add(reg_table, reg_table, reg_vector_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_value))
        mov(tmud, reg_value)
        sub(reg_iterations, reg_iterations, 1, cond="pushz")
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_vector_stride)
        tmuwt()
        copy_loop.b(cond="na0")
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


def supports_embedding_lookup_fp32(token_ids: Tensor, table: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the row-parallel gather contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if (
        token_ids.dtype != np.dtype(np.int32)
        or table.dtype != np.dtype(np.float32)
        or destination.dtype != np.dtype(np.float32)
    ):
        return False
    if len(token_ids.shape) != 1 or len(table.shape) != 2 or destination.shape != (token_ids.shape[0], table.shape[1]):
        return False
    if table.shape[1] % 16:
        return False
    return all(t.numpy().flags.c_contiguous for t in (token_ids, table, destination))


def _state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(driver.program(qpu_embedding_lookup_fp32), driver.alloc(6, dtype=np.uint32))
            _PROGRAMS[backend] = state
        return state


def _execute(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(arg, Tensor) for arg in args):
        raise KernelError("embedding lookup expects (token_ids, table, destination)")
    token_ids, table, destination = args
    if not supports_embedding_lookup_fp32(token_ids, table, destination, backend) or not isinstance(
        backend, PyVideoCore7Backend
    ):
        raise KernelError("embedding lookup requires contiguous tensors and a 16-aligned FP32 width")
    expected = (token_ids.shape[0], 1, 1)
    if grid != expected:
        raise KernelError(f"embedding lookup grid must be {expected}, got {grid}")
    state = _state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (
            table.shape[1] // 16,
            token_ids.address,
            table.address,
            table.numpy().strides[0],
            destination.address,
            destination.numpy().strides[0],
        )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=token_ids.shape[0],
        )


EMBEDDING_LOOKUP_FP32_KERNEL = Kernel("vc7.embedding_lookup_fp32", _execute)

__all__ = ["EMBEDDING_LOOKUP_FP32_KERNEL", "supports_embedding_lookup_fp32"]
