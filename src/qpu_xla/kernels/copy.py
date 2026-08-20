"""Cached VideoCore VII kernel for contiguous four-byte tensor copies."""

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
def qpu_copy_words(asm: Assembly, *, num_qpus: int, vector_width: Literal[1, 4]) -> None:
    """Copy one or four words per lane using exact QPU stripes."""
    if not 1 <= num_qpus <= 12:
        raise ValueError("word-copy QPU count must be between 1 and 12")
    if vector_width not in {1, 4}:
        raise ValueError("word-copy vector width must be 1 or 4")

    reg_iterations = rf0
    reg_source = rf1
    reg_destination = rf2
    reg_value = rf3
    reg_offset = rf4
    reg_stride = rf5
    reg_qpu = rf6
    reg_tmu_config = rf7

    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_destination))

    if num_qpus == 1:
        mov(reg_qpu, 0)
    else:
        tidx(reg_qpu)
        shr(reg_qpu, reg_qpu, 2)
        band(reg_qpu, reg_qpu, 0b1111)
    shl(reg_offset, reg_qpu, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 4 if vector_width == 4 else 2)
    add(reg_source, reg_source, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_stride, num_qpus)
    shl(reg_stride, reg_stride, 8 if vector_width == 4 else 6)

    if vector_width == 4:
        values = [rf10, rf11, rf12, rf13]
        bnot(reg_tmu_config, 3)
        with loop as vector_loop:
            mov(tmuc, reg_tmu_config)
            mov(tmua, reg_source, sig=thrsw)
            add(reg_source, reg_source, reg_stride)
            nop()
            nop()
            for value in values:
                nop(sig=ldtmu(value))

            mov(tmuc, reg_tmu_config)
            for value in values:
                mov(tmud, value)
            mov(tmua, reg_destination)
            add(reg_destination, reg_destination, reg_stride)
            tmuwt()
            sub(reg_iterations, reg_iterations, 1, cond="pushz")

            vector_loop.b(cond="na0")
            nop()
            nop()
            nop()
    else:
        with loop as scalar_loop:
            mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_stride)
            nop()
            nop()
            nop(sig=ldtmu(reg_value))
            mov(tmud, reg_value)
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
class _CopyProgramState:
    """Driver-local code and uniform storage reused by every copy dispatch."""

    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[int, int], _CopyProgramState]] = (
    WeakKeyDictionary()
)


def supports_word_copy(source: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether the contiguous four-byte QPU specialization is applicable."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != destination.dtype or source.dtype.itemsize != 4 or source.shape != destination.shape:
        return False
    if source.nbytes == 0 or source.nbytes % (16 * 4):
        return False
    return source.numpy().flags.c_contiguous and destination.numpy().flags.c_contiguous


def _program_state(
    backend: PyVideoCore7Backend,
    num_qpus: int,
    vector_width: Literal[1, 4],
) -> _CopyProgramState:
    """Build the assembly and uniform block exactly once per live backend."""
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        key = (num_qpus, vector_width)
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _CopyProgramState(
                    code=driver.program(qpu_copy_words, num_qpus=num_qpus, vector_width=vector_width),
                    uniforms=driver.alloc(3, dtype=np.uint32),
                )
            states[key] = state
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

    vectors = source.nbytes // (16 * 4)
    vector_width: Literal[1, 4] = 1
    work_items = vectors
    num_qpus = next(candidate for candidate in range(min(work_items, 12), 0, -1) if work_items % candidate == 0)
    if vectors % 4 == 0:
        vector_work_items = vectors // 4
        vector_qpus = next(
            candidate
            for candidate in range(min(vector_work_items, 12), 0, -1)
            if vector_work_items % candidate == 0
        )
        if vector_qpus >= 6:
            vector_width = 4
            work_items = vector_work_items
            num_qpus = vector_qpus
    state = _program_state(backend, num_qpus, vector_width)
    with backend.driver_session() as driver:
        state.uniforms[:] = (work_items // num_qpus, source.address, destination.address)
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            thread=num_qpus,
        )


WORD_COPY_KERNEL = Kernel("vc7.copy_words", _execute_word_copy)
