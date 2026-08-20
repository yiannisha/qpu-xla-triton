"""Cached VideoCore VII FP32 and INT32 elementwise min/max kernels."""

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

MinMaxOperation = Literal["minimum", "maximum"]


@qpu
def qpu_minmax_words(
    asm: Assembly,
    *,
    dtype: Literal["float32", "int32"],
    op: MinMaxOperation,
    num_qpus: int,
    vector_width: Literal[1, 4],
) -> None:
    """Apply one compile-time-selected min/max operation to 16 or 64 words."""
    if not 1 <= num_qpus <= 12:
        raise ValueError("word min/max QPU count must be between 1 and 12")
    if vector_width not in {1, 4}:
        raise ValueError("word min/max vector width must be 1 or 4")
    operation: Any
    if dtype == "float32":
        operation = fmin if op == "minimum" else fmax
    elif dtype == "int32":
        operation = imin if op == "minimum" else imax
    else:
        raise ValueError(f"unsupported min/max dtype {dtype!r}")

    reg_iterations = rf0
    reg_left = rf1
    reg_right = rf2
    reg_destination = rf3
    reg_offset = rf4
    reg_stride = rf5
    reg_left_value = rf10
    reg_right_value = rf11
    reg_output = rf12
    reg_qpu = rf13
    reg_tmu_config = rf14

    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_left))
    nop(sig=ldunifrf(reg_right))
    nop(sig=ldunifrf(reg_destination))

    if num_qpus == 1:
        mov(reg_qpu, 0)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)
    else:
        tidx(reg_qpu)
        shr(reg_qpu, reg_qpu, 2)
        band(reg_qpu, reg_qpu, 0b1111)
        mov(reg_stride, num_qpus)
        shl(reg_stride, reg_stride, 6)
    shl(reg_offset, reg_qpu, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 4 if vector_width == 4 else 2)
    add(reg_left, reg_left, reg_offset)
    add(reg_right, reg_right, reg_offset)
    add(reg_destination, reg_destination, reg_offset)

    if vector_width == 4:
        mov(reg_stride, num_qpus)
        shl(reg_stride, reg_stride, 8)
        bnot(reg_tmu_config, 3)

        left_values = [rf10, rf11, rf12, rf15]
        right_values = [rf16, rf17, rf18, rf19]
        outputs = [rf20, rf21, rf22, rf23]

        with loop as vector_loop:
            mov(tmuc, reg_tmu_config)
            mov(tmua, reg_left, sig=thrsw)
            add(reg_left, reg_left, reg_stride)
            nop()
            nop()
            for value in left_values:
                nop(sig=ldtmu(value))

            mov(tmuc, reg_tmu_config)
            mov(tmua, reg_right, sig=thrsw)
            add(reg_right, reg_right, reg_stride)
            nop()
            nop()
            for value in right_values:
                nop(sig=ldtmu(value))

            for output, left_value, right_value in zip(outputs, left_values, right_values, strict=True):
                operation(output, left_value, right_value)

            mov(tmuc, reg_tmu_config)
            for output in outputs:
                mov(tmud, output)
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
            mov(tmua, reg_left, sig=thrsw).add(reg_left, reg_left, reg_stride)
            nop()
            mov(tmua, reg_right, sig=thrsw).add(reg_right, reg_right, reg_stride)
            nop(sig=ldtmu(reg_left_value))
            nop()
            nop(sig=ldtmu(reg_right_value))

            operation(reg_output, reg_left_value, reg_right_value)
            mov(tmud, reg_output)
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
class _ProgramState:
    """Driver-local code and uniform storage for one min/max specialization."""

    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[
    PyVideoCore7Backend,
    dict[tuple[str, MinMaxOperation, int, int], _ProgramState],
] = (
    WeakKeyDictionary()
)


def supports_word_minmax(left: Tensor, right: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether this exact four-byte QPU min/max specialization applies."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    tensors = (left, right, destination)
    if any(tensor.dtype != destination.dtype or tensor.shape != destination.shape for tensor in tensors):
        return False
    if destination.dtype not in {np.dtype(np.float32), np.dtype(np.int32)}:
        return False
    if destination.nbytes % (16 * 4):
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in tensors)


def _dtype_name(dtype: np.dtype[np.generic]) -> Literal["float32", "int32"]:
    """Map supported NumPy dtypes to stable assembly specialization keys."""
    if dtype == np.dtype(np.float32):
        return "float32"
    if dtype == np.dtype(np.int32):
        return "int32"
    raise KernelError(f"unsupported min/max QPU dtype {dtype}")


def _program_state(
    backend: PyVideoCore7Backend,
    dtype: str,
    op: MinMaxOperation,
    num_qpus: int,
    vector_width: Literal[1, 4],
) -> _ProgramState:
    """Build one program/uniform pair per backend, dtype, and operation."""
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        key = (dtype, op, num_qpus, vector_width)
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(
                        qpu_minmax_words,
                        dtype=dtype,
                        op=op,
                        num_qpus=num_qpus,
                        vector_width=vector_width,
                    ),
                    uniforms=driver.alloc(4, dtype=np.uint32),
                )
            states[key] = state
        return state


def _execute(op: MinMaxOperation, backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    """Dispatch one cached min/max kernel with the supplied tensors."""
    if grid != (1, 1, 1):
        raise KernelError("word min/max uses a fixed single-workgroup launch")
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("word min/max expects (left_tensor, right_tensor, destination_tensor)")
    left, right, destination = args
    if not supports_word_minmax(left, right, destination, backend):
        raise KernelError(
            "word min/max requires equal contiguous float32 or int32 tensors with a multiple of 16 elements"
        )
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("word min/max requires PyVideoCore7Backend")

    vectors = destination.nbytes // (16 * 4)
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
    state = _program_state(backend, _dtype_name(destination.dtype), op, num_qpus, vector_width)
    with backend.driver_session() as driver:
        state.uniforms[:] = (work_items // num_qpus, left.address, right.address, destination.address)
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            thread=num_qpus,
        )


MINIMUM_WORD_KERNEL = Kernel("vc7.minimum_words", lambda backend, args, grid: _execute("minimum", backend, args, grid))
MAXIMUM_WORD_KERNEL = Kernel("vc7.maximum_words", lambda backend, args, grid: _execute("maximum", backend, args, grid))
