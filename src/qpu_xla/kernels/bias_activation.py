"""Cached tiled bias and ReLU epilogue kernels."""

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

BiasActivationDType = Literal["float32", "int32"]


@qpu
def qpu_relu_words(
    asm: Assembly,
    *,
    dtype: BiasActivationDType,
    num_qpus: int,
    vector_width: Literal[1, 4],
) -> None:
    """Apply ReLU to one or four contiguous words per lane."""
    if dtype not in {"float32", "int32"}:
        raise ValueError(f"unsupported ReLU dtype {dtype!r}")
    if not 1 <= num_qpus <= 12:
        raise ValueError("ReLU QPU count must be between 1 and 12")
    if vector_width not in {1, 4}:
        raise ValueError("ReLU vector width must be 1 or 4")

    iterations = rf0
    source = rf1
    destination = rf2
    qpu_id = rf3
    offset = rf4
    stride = rf5
    tmu_config = rf6
    zero = rf7

    nop(sig=ldunifrf(iterations))
    nop(sig=ldunifrf(source))
    nop(sig=ldunifrf(destination))
    if num_qpus == 1:
        mov(qpu_id, 0)
    else:
        tidx(qpu_id)
        shr(qpu_id, qpu_id, 2)
        band(qpu_id, qpu_id, 0b1111)
    shl(offset, qpu_id, 4)
    eidx(rf31)
    add(offset, offset, rf31)
    shl(offset, offset, 4 if vector_width == 4 else 2)
    add(source, source, offset)
    add(destination, destination, offset)
    mov(stride, num_qpus)
    shl(stride, stride, 8 if vector_width == 4 else 6)
    mov(zero, 0.0 if dtype == "float32" else 0)

    operation = fmax if dtype == "float32" else imax
    if vector_width == 4:
        values = [rf10, rf11, rf12, rf13]
        bnot(tmu_config, 3)
        with loop as vector_loop:
            mov(tmuc, tmu_config)
            mov(tmua, source, sig=thrsw)
            add(source, source, stride)
            nop()
            nop()
            for value in values:
                nop(sig=ldtmu(value))
            for value in values:
                operation(value, value, zero)

            mov(tmuc, tmu_config)
            for value in values:
                mov(tmud, value)
            mov(tmua, destination)
            add(destination, destination, stride)
            tmuwt()
            sub(iterations, iterations, 1, cond="pushz")

            vector_loop.b(cond="na0")
            nop()
            nop()
            nop()
    else:
        value = rf10
        with loop as scalar_loop:
            mov(tmua, source, sig=thrsw).add(source, source, stride)
            nop()
            nop()
            nop(sig=ldtmu(value))
            operation(value, value, zero)
            mov(tmud, value)
            mov(tmua, destination).add(destination, destination, stride)
            tmuwt()
            sub(iterations, iterations, 1, cond="pushz")

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


@qpu
def qpu_tiled_bias_activation(
    asm: Assembly,
    *,
    dtype: BiasActivationDType,
    use_bias: bool,
    apply_relu: bool,
) -> None:
    """Apply optional per-column bias and ReLU to one 16x16 output tile."""
    if dtype not in {"float32", "int32"}:
        raise ValueError(f"unsupported bias activation dtype {dtype!r}")

    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_stride = rf4
    reg_base = rf5
    reg_bias_base = rf6
    reg_row_ptr = rf7
    reg_bias = rf8
    reg_value = rf9
    reg_output = rf10
    reg_zero = rf11
    reg_tmp = rf12
    reg_destination_base = rf13
    reg_destination_row = rf14

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_stride))
    umul24(reg_tmp, reg_tile_i, reg_stride, sig=ldunifrf(reg_base))
    shl(reg_tmp, reg_tmp, 4)
    if use_bias:
        add(reg_base, reg_base, reg_tmp, sig=ldunifrf(reg_bias_base))
    else:
        add(reg_base, reg_base, reg_tmp)

    shl(reg_tmp, reg_tile_j, 6)
    eidx(rf0)
    shl(rf0, rf0, 2)
    add(reg_row_ptr, reg_base, reg_tmp)
    add(reg_row_ptr, reg_row_ptr, rf0)

    mov(reg_zero, 0.0 if dtype == "float32" else 0)
    if use_bias:
        add(reg_bias_base, reg_bias_base, reg_tmp)
        add(reg_bias_base, reg_bias_base, rf0)
        mov(tmua, reg_bias_base, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_bias))
    else:
        mov(reg_bias, 0.0 if dtype == "float32" else 0)

    nop(sig=ldunifrf(reg_destination_base))
    umul24(reg_destination_row, reg_tile_i, reg_stride)
    shl(reg_destination_row, reg_destination_row, 4)
    add(reg_destination_row, reg_destination_base, reg_destination_row)
    shl(reg_tmp, reg_tile_j, 6)
    add(reg_destination_row, reg_destination_row, reg_tmp)
    add(reg_destination_row, reg_destination_row, rf0)

    mov(tmuc, -1)
    for index in range(16):
        mov(tmua, reg_row_ptr, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_value))
        if use_bias:
            if dtype == "float32":
                fadd(reg_output, reg_value, reg_bias)
            else:
                add(reg_output, reg_value, reg_bias)
        else:
            mov(reg_output, reg_value)
        if apply_relu:
            if dtype == "float32":
                fmax(reg_output, reg_output, reg_zero)
            else:
                imax(reg_output, reg_output, reg_zero)
        mov(tmud, reg_output)
        mov(tmua, reg_destination_row)
        tmuwt()
        if index < 15:
            add(reg_row_ptr, reg_row_ptr, reg_stride)
            add(reg_destination_row, reg_destination_row, reg_stride)

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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[str, bool, bool], _ProgramState]] = (
    WeakKeyDictionary()
)
_RELU_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[tuple[str, int, int], _ProgramState]] = (
    WeakKeyDictionary()
)


def supports_bias_activation(
    source: Tensor,
    bias: Tensor | None,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether the tiled 16x16 epilogue can process these tensors."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype not in {np.dtype(np.float32), np.dtype(np.int32)}:
        return False
    if source.dtype != destination.dtype or source.shape != destination.shape:
        return False
    if len(source.shape) != 2 or source.shape[0] % 16 or source.shape[1] % 16:
        return False
    if not source.numpy().flags.c_contiguous or not destination.numpy().flags.c_contiguous:
        return False
    if bias is not None and (
        bias.dtype != source.dtype
        or bias.shape != (source.shape[1],)
        or not bias.numpy().flags.c_contiguous
    ):
        return False
    return True


def _dtype_name(dtype: np.dtype[np.generic]) -> BiasActivationDType:
    if dtype == np.dtype(np.float32):
        return "float32"
    if dtype == np.dtype(np.int32):
        return "int32"
    raise KernelError(f"unsupported bias activation dtype {dtype}")


def _program_state(
    backend: PyVideoCore7Backend,
    dtype: BiasActivationDType,
    use_bias: bool,
    apply_relu: bool,
) -> _ProgramState:
    key = (dtype, use_bias, apply_relu)
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(
                        qpu_tiled_bias_activation,
                        dtype=dtype,
                        use_bias=use_bias,
                        apply_relu=apply_relu,
                    ),
                    uniforms=driver.alloc(4, dtype=np.uint32),
                )
            states[key] = state
        return state


def _relu_program_state(
    backend: PyVideoCore7Backend,
    dtype: BiasActivationDType,
    num_qpus: int,
    vector_width: Literal[1, 4],
) -> _ProgramState:
    key = (dtype, num_qpus, vector_width)
    with _STATE_LOCK:
        states = _RELU_PROGRAMS.setdefault(backend, {})
        state = states.get(key)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(
                        qpu_relu_words,
                        dtype=dtype,
                        num_qpus=num_qpus,
                        vector_width=vector_width,
                    ),
                    uniforms=driver.alloc(3, dtype=np.uint32),
                )
            states[key] = state
        return state


def _execute_relu_streaming(
    backend: PyVideoCore7Backend,
    source: Tensor,
    destination: Tensor,
) -> None:
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

    state = _relu_program_state(backend, _dtype_name(source.dtype), num_qpus, vector_width)
    with backend.driver_session() as driver:
        state.uniforms[:] = (work_items // num_qpus, source.address, destination.address)
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            thread=num_qpus,
        )


def _execute(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
    *,
    use_bias: bool,
    apply_relu: bool,
) -> None:
    if len(args) != 3 or not isinstance(args[0], Tensor) or not isinstance(args[2], Tensor):
        raise KernelError("bias activation expects (source, bias, destination)")
    source, bias, destination = args
    if use_bias and not isinstance(bias, Tensor):
        raise KernelError("bias activation requires a bias tensor")
    if not use_bias and bias is not None:
        raise KernelError("bias activation without bias expects None")
    if not supports_bias_activation(source, bias if isinstance(bias, Tensor) else None, destination, backend):
        raise KernelError("bias activation requires contiguous tiled FP32 or INT32 tensors")
    expected_grid = (source.shape[1] // 16, source.shape[0] // 16, 1)
    if grid != expected_grid:
        raise KernelError(f"bias activation grid must be {expected_grid}, got {grid}")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("bias activation requires PyVideoCore7Backend")

    if not use_bias and apply_relu:
        _execute_relu_streaming(backend, source, destination)
        return

    dtype = _dtype_name(source.dtype)
    state = _program_state(backend, dtype, use_bias, apply_relu)
    workgroups = grid[0] * grid[1]
    wgs_per_sg = 48 if workgroups >= 64 else 24
    with backend.driver_session() as driver:
        if use_bias:
            state.uniforms[:] = (
                source.numpy().strides[0],
                source.address,
                bias.address,
                destination.address,
            )
        else:
            # This assembly specialization consumes only stride, source, and
            # destination. The fourth allocated uniform word is unused.
            state.uniforms[:] = (
                source.numpy().strides[0],
                source.address,
                destination.address,
                0,
            )
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=wgs_per_sg,
            thread=workgroups,
        )


def _kernel(name: str, *, use_bias: bool, apply_relu: bool) -> Kernel:
    return Kernel(
        name,
        lambda backend, args, grid: _execute(
            backend,
            args,
            grid,
            use_bias=use_bias,
            apply_relu=apply_relu,
        ),
    )


BIAS_RELU_FP32_KERNEL = _kernel("vc7.bias_relu_fp32", use_bias=True, apply_relu=True)
BIAS_ONLY_FP32_KERNEL = _kernel("vc7.bias_fp32", use_bias=True, apply_relu=False)
RELU_ONLY_FP32_KERNEL = _kernel("vc7.relu_fp32", use_bias=False, apply_relu=True)
BIAS_RELU_INT32_KERNEL = _kernel("vc7.bias_relu_int32", use_bias=True, apply_relu=True)
BIAS_ONLY_INT32_KERNEL = _kernel("vc7.bias_int32", use_bias=True, apply_relu=False)
RELU_ONLY_INT32_KERNEL = _kernel("vc7.relu_int32", use_bias=False, apply_relu=True)


def bias_activation_kernel(
    dtype: np.dtype[np.generic],
    *,
    use_bias: bool,
    apply_relu: bool,
) -> Kernel:
    """Return the cached kernel handle for one dtype/epilogue specialization."""
    if dtype == np.dtype(np.float32):
        kernels = {
            (True, True): BIAS_RELU_FP32_KERNEL,
            (True, False): BIAS_ONLY_FP32_KERNEL,
            (False, True): RELU_ONLY_FP32_KERNEL,
        }
    elif dtype == np.dtype(np.int32):
        kernels = {
            (True, True): BIAS_RELU_INT32_KERNEL,
            (True, False): BIAS_ONLY_INT32_KERNEL,
            (False, True): RELU_ONLY_INT32_KERNEL,
        }
    else:
        raise ValueError(f"unsupported bias activation dtype {dtype}")
    if (use_bias, apply_relu) not in kernels:
        raise ValueError("bias activation must apply bias, ReLU, or both")
    return kernels[(use_bias, apply_relu)]


__all__ = [
    "BIAS_ONLY_FP32_KERNEL",
    "BIAS_ONLY_INT32_KERNEL",
    "BIAS_RELU_FP32_KERNEL",
    "BIAS_RELU_INT32_KERNEL",
    "RELU_ONLY_FP32_KERNEL",
    "RELU_ONLY_INT32_KERNEL",
    "bias_activation_kernel",
    "supports_bias_activation",
]
