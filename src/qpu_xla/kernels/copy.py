"""Cached VideoCore VII kernel for contiguous four-byte tensor copies."""

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
def qpu_copy_words(asm: Assembly) -> None:
    """Copy one 16-lane word vector per loop iteration on a single QPU core."""
    reg_iterations = rf0
    reg_source = rf1
    reg_destination = rf2
    reg_value = rf3
    reg_offset = rf4
    reg_stride = rf5

    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_destination))

    eidx(reg_offset)
    shl(reg_offset, reg_offset, 2)
    add(reg_source, reg_source, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6)

    with loop as l:  # noqa: E741
        mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_value))
        mov(tmud, reg_value)
        sub(reg_iterations, reg_iterations, 1, cond="pushz")
        mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
        tmuwt()

        l.b(cond="na0")
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
class _CopyProgramState:
    """Driver-local code and uniform storage reused by every copy dispatch."""

    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, _CopyProgramState] = WeakKeyDictionary()


def supports_word_copy(source: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether the contiguous four-byte QPU specialization is applicable."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != destination.dtype or source.dtype.itemsize != 4 or source.shape != destination.shape:
        return False
    if source.nbytes == 0 or source.nbytes % (16 * 4):
        return False
    return source.numpy().flags.c_contiguous and destination.numpy().flags.c_contiguous


def _program_state(backend: PyVideoCore7Backend) -> _CopyProgramState:
    """Build the assembly and uniform block exactly once per live backend."""
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is not None:
            return state
        with backend.driver_session() as driver:
            state = _CopyProgramState(
                code=driver.program(qpu_copy_words),
                uniforms=driver.alloc(3, dtype=np.uint32),
            )
        _PROGRAMS[backend] = state
        return state


def _execute_word_copy(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    """Run one cached word-copy dispatch through the existing synchronous driver."""
    if grid != (1, 1, 1):
        raise KernelError("word-copy uses a fixed single-workgroup launch")
    if len(args) != 2 or not isinstance(args[0], Tensor) or not isinstance(args[1], Tensor):
        raise KernelError("word-copy expects (source_tensor, destination_tensor)")
    source, destination = args
    if not supports_word_copy(source, destination, backend):
        raise KernelError("word-copy requires equal contiguous four-byte tensors with a multiple of 16 elements")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("word-copy requires PyVideoCore7Backend")

    state = _program_state(backend)
    with backend.driver_session() as driver:
        state.uniforms[:] = (source.nbytes // (16 * 4), source.address, destination.address)
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            thread=1,
        )


WORD_COPY_KERNEL = Kernel("vc7.copy_words", _execute_word_copy)
