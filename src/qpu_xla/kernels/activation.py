"""Standalone FP32 SiLU and tanh-approximate GELU kernels."""

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

Activation = Literal["silu", "gelu_tanh"]


@qpu
def qpu_activation_fp32(
    asm: Assembly,
    *,
    activation: Activation,
    vector_width: Literal[1, 4],
    num_qpus: int,
) -> None:
    """Apply one activation to one or four FP32 words per SIMD lane."""
    if activation not in {"silu", "gelu_tanh"}:
        raise ValueError("unsupported activation")
    if vector_width not in {1, 4} or not 1 <= num_qpus <= 12:
        raise ValueError("invalid activation kernel specialization")

    reg_iterations = rf0
    reg_source = rf1
    reg_destination = rf2
    reg_log2_e = rf4
    reg_offset = rf5
    reg_stride = rf6
    reg_workgroup = rf7
    reg_workgroup_offset = rf8
    reg_remainder = rf9
    reg_gelu_cubic = rf29
    reg_gelu_scale = rf30

    if vector_width == 1:
        mov(reg_workgroup, rf3.unpack("ul"))
        nop(sig=ldunifrf(reg_iterations))
        nop(sig=ldunifrf(reg_source))
        nop(sig=ldunifrf(reg_destination))
        nop(sig=ldunifrf(reg_log2_e))
    else:
        nop(sig=ldunifrf(reg_iterations))
        nop(sig=ldunifrf(reg_remainder))
        nop(sig=ldunifrf(reg_source))
        nop(sig=ldunifrf(reg_destination))
        nop(sig=ldunifrf(reg_log2_e))
    if activation == "gelu_tanh":
        nop(sig=ldunifrf(reg_gelu_cubic))
        nop(sig=ldunifrf(reg_gelu_scale))

    if vector_width == 4:
        reg_base_iterations = rf26
        reg_extra_item = rf27
        reg_first_item = rf28
        mov(reg_base_iterations, reg_iterations)
        tidx(reg_workgroup)
        shr(reg_workgroup, reg_workgroup, 2)
        band(reg_workgroup, reg_workgroup, 0b1111)
        mov(reg_extra_item, reg_workgroup)
        sub(null, reg_workgroup, reg_remainder, cond="pushn")
        mov(reg_extra_item, reg_remainder, cond="ifna")
        add(reg_iterations, reg_iterations, 1, cond="ifa")
        umul24(reg_first_item, reg_workgroup, reg_base_iterations)
        add(reg_first_item, reg_first_item, reg_extra_item)
        shl(reg_offset, reg_first_item, 8)
        eidx(rf31)
        shl(rf31, rf31, 4)
        add(reg_offset, reg_offset, rf31)
        add(reg_source, reg_source, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 8)
        bnot(reg_workgroup_offset, 3)

        values = [rf10, rf11, rf12, rf13]
        temporaries = [rf14, rf15, rf16, rf17]
        exponentials = [rf18, rf19, rf20, rf21]
        outputs = [rf22, rf23, rf24, rf25]

        nop(sig=thrsw)
        nop()
        nop()
        with loop as vector_loop:
            mov(tmuc, reg_workgroup_offset)
            mov(tmua, reg_source).add(reg_source, reg_source, reg_stride)
            nop()
            nop()
            for value in values:
                nop(sig=ldtmu(value))
            if activation == "silu":
                for exponential, value in zip(exponentials, values, strict=True):
                    fmul(exponential, value, reg_log2_e)
            else:
                for temporary, exponential, value in zip(temporaries, exponentials, values, strict=True):
                    fmul(temporary, value, value)
                    fmul(temporary, temporary, value)
                    fmul(temporary, temporary, reg_gelu_cubic)
                    fadd(temporary, temporary, value)
                    fmul(temporary, temporary, reg_gelu_scale)
                    fmul(exponential, temporary, reg_log2_e)
                    fmul(exponential, exponential, 2.0)
            for exponential in exponentials:
                exp(exponential, exponential)
                fadd(exponential, exponential, 1.0)
                recip(exponential, exponential)
            for output, value, exponential in zip(outputs, values, exponentials, strict=True):
                fmul(output, value, exponential)
            mov(tmuc, reg_workgroup_offset)
            for output in outputs:
                mov(tmud, output)
            mov(tmua, reg_destination).add(reg_destination, reg_destination, reg_stride)
            tmuwt()
            sub(reg_iterations, reg_iterations, 1, cond="pushz")
            vector_loop.b(cond="na0")
            nop()
            nop()
            nop()
        barrierid(syncb, sig=thrsw)
        nop()
        nop()
    else:
        reg_value = rf10
        reg_temporary = rf11
        reg_exponential = rf12
        reg_output = rf13
        shl(reg_workgroup_offset, reg_iterations, 6)
        umul24(reg_workgroup_offset, reg_workgroup_offset, reg_workgroup)
        add(reg_source, reg_source, reg_workgroup_offset)
        add(reg_destination, reg_destination, reg_workgroup_offset)
        eidx(reg_offset)
        shl(reg_offset, reg_offset, 2)
        add(reg_source, reg_source, reg_offset)
        add(reg_destination, reg_destination, reg_offset)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)

        with loop as scalar_loop:
            mov(tmua, reg_source, sig=thrsw).add(reg_source, reg_source, reg_stride)
            nop()
            nop()
            nop(sig=ldtmu(reg_value))
            if activation == "silu":
                fmul(reg_exponential, reg_value, reg_log2_e)
            else:
                fmul(reg_temporary, reg_value, reg_value)
                fmul(reg_temporary, reg_temporary, reg_value)
                fmul(reg_temporary, reg_temporary, reg_gelu_cubic)
                fadd(reg_temporary, reg_temporary, reg_value)
                fmul(reg_temporary, reg_temporary, reg_gelu_scale)
                fmul(reg_exponential, reg_temporary, reg_log2_e)
                fmul(reg_exponential, reg_exponential, 2.0)
            exp(reg_exponential, reg_exponential)
            fadd(reg_exponential, reg_exponential, 1.0)
            recip(reg_exponential, reg_exponential)
            fmul(reg_output, reg_value, reg_exponential)
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
    code: Any
    uniforms: Any


_STATE_LOCK = Lock()
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[str, int, int], _ProgramState]] = WeakKeyDictionary()


def supports_activation_fp32(source: Tensor, destination: Tensor, backend: Backend) -> bool:
    """Return whether tensors meet the contiguous vector activation contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.float32) or destination.dtype != source.dtype or destination.shape != source.shape:
        return False
    if source.nbytes == 0 or source.nbytes % 64:
        return False
    return source.numpy().flags.c_contiguous and destination.numpy().flags.c_contiguous


def _workgroup_count(vector_count: int) -> int:
    for count in range(min(12, vector_count), 0, -1):
        if vector_count % count == 0:
            return count
    return 1


def _program_state(
    backend: PyVideoCore7Backend,
    activation: Activation,
    vector_width: Literal[1, 4],
    num_qpus: int,
) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        key = (activation, vector_width, num_qpus)
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(
                        qpu_activation_fp32,
                        activation=activation,
                        vector_width=vector_width,
                        num_qpus=num_qpus,
                    ),
                    uniforms=driver.alloc(64, dtype=np.uint32),
                )
            states[key] = state
        return state


def _execute_activation(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
    *,
    activation: Activation,
) -> None:
    if len(args) != 2 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("activation expects (source, destination)")
    source, destination = args
    if not supports_activation_fp32(source, destination, backend):
        raise KernelError("activation requires equal contiguous FP32 tensors aligned to 16 values")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("activation requires PyVideoCore7Backend")
    vectors = source.nbytes // 64
    expected_grid = (_workgroup_count(vectors), 1, 1)
    if grid != expected_grid:
        raise KernelError(f"activation grid must be {expected_grid}, got {grid}")
    vector_width: Literal[1, 4] = 4 if vectors // grid[0] >= 4 else 1
    items_per_qpu = vectors // grid[0]
    if vector_width == 4:
        iterations, remainder = divmod(items_per_qpu, 4)
        uniforms = (
            iterations,
            remainder,
            source.address,
            destination.address,
            np.asarray(-np.log2(np.e), dtype=np.float32).view(np.uint32),
        )
    else:
        uniforms = (
            items_per_qpu,
            source.address,
            destination.address,
            np.asarray(-np.log2(np.e), dtype=np.float32).view(np.uint32),
        )
    state = _program_state(backend, activation, vector_width, grid[0])
    if activation == "gelu_tanh":
        uniforms = (
            *uniforms,
            np.asarray(0.044715, dtype=np.float32).view(np.uint32),
            np.asarray(np.sqrt(2.0 / np.pi), dtype=np.float32).view(np.uint32),
        )
    state.uniforms[: len(uniforms)] = uniforms
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=grid[0],
            thread=grid[0],
        )


def _execute_silu_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    _execute_activation(backend, args, grid, activation="silu")


def _execute_gelu_tanh_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    _execute_activation(backend, args, grid, activation="gelu_tanh")


SILU_FP32_KERNEL = Kernel("vc7.silu_fp32", _execute_silu_fp32)
GELU_TANH_FP32_KERNEL = Kernel("vc7.gelu_tanh_fp32", _execute_gelu_tanh_fp32)

__all__ = [
    "GELU_TANH_FP32_KERNEL",
    "SILU_FP32_KERNEL",
    "supports_activation_fp32",
]
