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
def qpu_minmax_words(asm: Assembly, *, dtype: Literal["float32", "int32"], op: MinMaxOperation) -> None:
    """Apply one compile-time-selected min/max operation to 16 words per iteration."""
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

    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_left))
    nop(sig=ldunifrf(reg_right))
    nop(sig=ldunifrf(reg_destination))

    eidx(reg_offset)
    shl(reg_offset, reg_offset, 2)
    add(reg_left, reg_left, reg_offset)
    add(reg_right, reg_right, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6)

    with loop as l:  # noqa: E741
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
class _ProgramState:
    """Driver-local code and uniform storage for one min/max specialization."""

    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[str, MinMaxOperation], _ProgramState]] = (
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


def _program_state(backend: PyVideoCore7Backend, dtype: str, op: MinMaxOperation) -> _ProgramState:
    """Build one program/uniform pair per backend, dtype, and operation."""
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        key = (dtype, op)
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_minmax_words, dtype=dtype, op=op),
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

    state = _program_state(backend, _dtype_name(destination.dtype), op)
    with backend.driver_session() as driver:
        state.uniforms[:] = (destination.nbytes // (16 * 4), left.address, right.address, destination.address)
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            thread=1,
        )


MINIMUM_WORD_KERNEL = Kernel("vc7.minimum_words", lambda backend, args, grid: _execute("minimum", backend, args, grid))
MAXIMUM_WORD_KERNEL = Kernel("vc7.maximum_words", lambda backend, args, grid: _execute("maximum", backend, args, grid))
