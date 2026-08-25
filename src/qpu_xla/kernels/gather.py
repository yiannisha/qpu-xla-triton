"""Address-table FP32 gather kernel for fixed tensor permutations."""

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
def qpu_indexed_gather_fp32(asm: Assembly, *, num_qpus: int) -> None:
    """Gather 16 independently addressed FP32 values per SIMD iteration."""
    if not 1 <= num_qpus <= 12:
        raise ValueError("gather QPU count must be between 1 and 12")
    reg_iterations = rf0
    reg_metadata = rf1
    reg_destination = rf2
    reg_qpu = rf4
    reg_offset = rf5
    reg_stride = rf6
    reg_address = rf7
    reg_value = rf8

    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_metadata))
    nop(sig=ldunifrf(reg_destination))
    if num_qpus == 1:
        mov(reg_qpu, 0)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu, reg_offset, 0b1111)
        mov(reg_stride, num_qpus)
        shl(reg_stride, reg_stride, 6)
    shl(reg_offset, reg_qpu, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_metadata, reg_metadata, reg_offset)
    add(reg_destination, reg_destination, reg_offset)

    with loop as gather:
        mov(tmua, reg_metadata, sig=thrsw).add(reg_metadata, reg_metadata, reg_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_address))
        mov(tmua, reg_address, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_value))
        mov(tmud, reg_value)
        sub(reg_iterations, reg_iterations, 1, cond="pushz")
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
        tmuwt()
        gather.b(cond="na0")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
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


def supports_indexed_gather_fp32(
    source: Tensor,
    metadata: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors satisfy the flat address-table gather contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        return False
    if metadata.dtype != np.dtype(np.uint32) or metadata.shape != (int(np.prod(destination.shape)),):
        return False
    if destination.nbytes == 0 or destination.nbytes % 64:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, metadata, destination))


def _workgroup_count(vector_count: int) -> int:
    for count in range(min(12, vector_count), 0, -1):
        if vector_count % count == 0:
            return count
    return 1


def _program_state(backend: PyVideoCore7Backend, num_qpus: int) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(num_qpus)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_indexed_gather_fp32, num_qpus=num_qpus),
                    uniforms=driver.alloc(3, dtype=np.uint32),
                )
            states[num_qpus] = state
        return state


def _execute_indexed_gather_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("indexed gather expects (source, metadata, destination)")
    source, metadata, destination = args
    if not supports_indexed_gather_fp32(source, metadata, destination, backend):
        raise KernelError("indexed gather requires FP32 tensors and one aligned uint32 address per output")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("indexed gather requires PyVideoCore7Backend")
    vectors = destination.nbytes // 64
    expected = (_workgroup_count(vectors), 1, 1)
    if grid != expected:
        raise KernelError(f"indexed gather grid must be {expected}, got {grid}")
    state = _program_state(backend, grid[0])
    state.uniforms[:] = (vectors // grid[0], metadata.address, destination.address)
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=grid[0],
            thread=grid[0],
        )


INDEXED_GATHER_FP32_KERNEL = Kernel("vc7.indexed_gather_fp32", _execute_indexed_gather_fp32)

__all__ = ["INDEXED_GATHER_FP32_KERNEL", "supports_indexed_gather_fp32"]
