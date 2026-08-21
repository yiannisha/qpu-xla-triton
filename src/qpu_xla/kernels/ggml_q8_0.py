"""Native GGML Q8_0 by Q8_0 four-row linear candidate."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import numpy.typing as npt

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.kernels.ggml_q4_0 import unpack_ggml_q8_0_blocks
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu

QK8_0 = 32
Q8_0_BLOCK_BYTES = 34
OUTPUT_TILE = 16


def ggml_q8_0_q8_0_reference(
    activation_blocks: npt.NDArray[np.uint8],
    weight_blocks: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Accumulate exact per-block Q8_0 integer dots and apply FP16 scales."""
    if activation_blocks.ndim != 3 or weight_blocks.ndim != 3:
        raise ValueError("activation and weight blocks must both be rank three")
    rows, blocks, activation_bytes = activation_blocks.shape
    outputs, weight_block_count, weight_bytes = weight_blocks.shape
    if activation_bytes != Q8_0_BLOCK_BYTES or weight_bytes != Q8_0_BLOCK_BYTES:
        raise ValueError("native Q8_0 block extents do not match")
    if blocks != weight_block_count:
        raise ValueError("activation and weight reduction block counts do not match")
    activation_scales, activations = unpack_ggml_q8_0_blocks(activation_blocks)
    weight_scales, weights = unpack_ggml_q8_0_blocks(weight_blocks)
    result = np.zeros((rows, outputs), dtype=np.float32)
    for block in range(blocks):
        dots = (
            activations[:, block].astype(np.int32)
            @ weights[:, block].astype(np.int32).T
        )
        contribution = dots.astype(np.float32)
        contribution *= activation_scales[:, block, None]
        contribution *= weight_scales[None, :, block]
        result += contribution
    return result


def _load_tmu_word(address: Register, destination: Register) -> None:
    mov(tmua, address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(destination))


@qpu
def qpu_ggml_q8_0_q8_0_linear_m4(asm: Assembly) -> None:
    """Compute a native Q8_0 by Q8_0 four-row output tile."""
    reg_tile = rf0
    reg_blocks = rf1
    reg_activation_stride = rf2
    reg_activation_base = [rf3, rf4, rf5, rf6]
    reg_weight_stride = rf7
    reg_weight_base = rf8
    reg_output_stride = rf9
    reg_output_base = rf10
    reg_weight_column_offset = rf11
    reg_output_column_offset = rf12
    reg_block_stride = rf13
    reg_lane = rf14
    reg_column = rf15
    reg_temporary = rf16
    reg_weight_pointer = rf17
    reg_weight_data = rf18
    reg_weight_scale = rf19
    reg_activation_data = [rf20, rf21, rf22, rf23]
    reg_activation_scale = [rf24, rf25, rf26, rf27]
    reg_dot = [rf28, rf29, rf30, rf31]
    reg_accumulator = [rf32, rf33, rf34, rf35]
    reg_weight_word = rf36
    reg_activation_word = rf37
    reg_product = rf38
    reg_word_count = rf39
    reg_output_pointer = rf40
    reg_block_count = rf41
    reg_float_temporary = rf42

    mov(reg_tile, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_blocks))
    nop(sig=ldunifrf(reg_activation_stride))
    nop(sig=ldunifrf(reg_activation_base[0]))
    nop(sig=ldunifrf(reg_weight_stride))
    nop(sig=ldunifrf(reg_weight_base))
    nop(sig=ldunifrf(reg_output_stride))
    nop(sig=ldunifrf(reg_output_base))
    nop(sig=ldunifrf(reg_weight_column_offset))
    nop(sig=ldunifrf(reg_output_column_offset))
    nop(sig=ldunifrf(reg_block_stride))

    for row in range(1, 4):
        add(reg_activation_base[row], reg_activation_base[row - 1], reg_activation_stride)
    eidx(reg_lane)
    shl(reg_column, reg_tile, 4)
    add(reg_column, reg_column, reg_lane)
    add(reg_temporary, reg_column, reg_weight_column_offset)
    umul24(reg_temporary, reg_temporary, reg_weight_stride)
    add(reg_weight_pointer, reg_weight_base, reg_temporary)
    add(reg_temporary, reg_column, reg_output_column_offset)
    shl(reg_temporary, reg_temporary, 2)
    add(reg_output_pointer, reg_output_base, reg_temporary)
    mov(reg_block_count, reg_blocks)
    for row in range(4):
        bxor(reg_accumulator[row], reg_accumulator[row], reg_accumulator[row])
    setnnmode_ss()

    with loop as block_loop:
        _load_tmu_word(reg_weight_pointer, reg_weight_scale)
        fmov(reg_weight_scale, reg_weight_scale.unpack("l"))
        add(reg_weight_data, reg_weight_pointer, 2)
        add(reg_weight_pointer, reg_weight_pointer, reg_block_stride)
        for row in range(4):
            _load_tmu_word(reg_activation_base[row], reg_activation_scale[row])
            fmov(reg_activation_scale[row], reg_activation_scale[row].unpack("l"))
            add(reg_activation_data[row], reg_activation_base[row], 2)
            add(reg_activation_base[row], reg_activation_base[row], reg_block_stride)
            bxor(reg_dot[row], reg_dot[row], reg_dot[row])
        mov(reg_word_count, 8)
        with loop as word_loop:
            _load_tmu_word(reg_weight_data, reg_weight_word)
            add(reg_weight_data, reg_weight_data, 4)
            for row in range(4):
                _load_tmu_word(reg_activation_data[row], reg_activation_word)
                add(reg_activation_data[row], reg_activation_data[row], 4)
                v8dot(reg_product, reg_weight_word, reg_activation_word)
                add(reg_dot[row], reg_dot[row], reg_product)
            sub(reg_word_count, reg_word_count, 1, cond="pushz")
            word_loop.b(cond="na0")
            nop()
            nop()
            nop()
        for row in range(4):
            itof(reg_float_temporary, reg_dot[row])
            fmul(reg_float_temporary, reg_float_temporary, reg_weight_scale)
            fmul(reg_float_temporary, reg_float_temporary, reg_activation_scale[row])
            fadd(reg_accumulator[row], reg_accumulator[row], reg_float_temporary)
        sub(reg_block_count, reg_block_count, 1, cond="pushz")
        block_loop.b(cond="na0")
        nop()
        nop()
        nop()

    for row in range(4):
        mov(tmud, reg_accumulator[row])
        mov(tmua, reg_output_pointer)
        tmuwt()
        if row + 1 < 4:
            add(reg_output_pointer, reg_output_pointer, reg_output_stride)
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


def supports_ggml_q8_0_q8_0_linear_m4(
    activation: Tensor,
    weight: Tensor,
    destination: Tensor,
    weight_column_start: int,
    output_column_start: int,
    column_count: int,
    backend: Backend,
) -> bool:
    """Return whether native Q8_0 block tensors meet the exact M=4 contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if activation.dtype != np.dtype(np.uint8) or weight.dtype != np.dtype(np.uint8):
        return False
    if destination.dtype != np.dtype(np.float32):
        return False
    if len(activation.shape) != 3 or len(weight.shape) != 3 or len(destination.shape) != 2:
        return False
    rows, blocks, activation_bytes = activation.shape
    weight_outputs, weight_blocks, weight_bytes = weight.shape
    if rows != 4 or activation_bytes != Q8_0_BLOCK_BYTES or weight_bytes != Q8_0_BLOCK_BYTES:
        return False
    if blocks != weight_blocks or destination.shape[0] != rows:
        return False
    if weight_column_start < 0 or output_column_start < 0:
        return False
    if weight_column_start % OUTPUT_TILE or output_column_start % OUTPUT_TILE:
        return False
    if column_count <= 0 or column_count % OUTPUT_TILE:
        return False
    if weight_column_start + column_count > weight_outputs:
        return False
    if output_column_start + column_count > destination.shape[1]:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (activation, weight, destination))


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_ggml_q8_0_q8_0_linear_m4),
                    uniforms=driver.alloc(10, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute_ggml_q8_0_q8_0_linear_m4(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 6 or not all(isinstance(argument, Tensor) for argument in args[:3]):
        raise KernelError(
            "GGML Q8_0 linear expects "
            "(activation, weight, destination, weight_column_start, output_column_start, column_count)"
        )
    activation, weight, destination, weight_column_start, output_column_start, column_count = args
    if not all(
        isinstance(value, int)
        for value in (weight_column_start, output_column_start, column_count)
    ):
        raise KernelError("GGML Q8_0 column bounds must be integers")
    if not supports_ggml_q8_0_q8_0_linear_m4(
        activation,
        weight,
        destination,
        weight_column_start,
        output_column_start,
        column_count,
        backend,
    ):
        raise KernelError("GGML Q8_0 linear requires contiguous native M=4 blocks and aligned columns")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("GGML Q8_0 linear requires PyVideoCore7Backend")
    expected_grid = (column_count // OUTPUT_TILE, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"GGML Q8_0 linear grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    state.uniforms[:] = (
        activation.shape[1],
        activation.numpy().strides[0],
        activation.address,
        weight.numpy().strides[0],
        weight.address,
        destination.numpy().strides[0],
        destination.address,
        weight_column_start,
        output_column_start,
        Q8_0_BLOCK_BYTES,
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=expected_grid,
            wgs_per_sg=24,
            thread=expected_grid[0],
        )


GGML_Q8_0_Q8_0_LINEAR_M4_KERNEL = Kernel(
    "vc7.ggml_q8_0_q8_0_linear_m4",
    _execute_ggml_q8_0_q8_0_linear_m4,
)

__all__ = [
    "GGML_Q8_0_Q8_0_LINEAR_M4_KERNEL",
    "ggml_q8_0_q8_0_reference",
    "supports_ggml_q8_0_q8_0_linear_m4",
]
