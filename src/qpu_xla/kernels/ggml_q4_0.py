"""Native GGML Q4_0 by Q8_0 small-row linear candidate."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from weakref import WeakKeyDictionary

import numpy as np
import numpy.typing as npt

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu

Q4_0_BLOCK_ELEMENTS = 32
Q4_0_BLOCK_BYTES = 18
Q8_0_BLOCK_BYTES = 34
OUTPUT_TILE = 16


def pack_ggml_q4_0_blocks(
    scales: npt.NDArray[np.floating[Any]],
    codes: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint8]:
    """Pack explicit 0..15 codes into native ``block_q4_0`` bytes."""
    scales = np.asarray(scales, dtype=np.float16)
    codes = np.asarray(codes)
    if codes.shape != (*scales.shape, Q4_0_BLOCK_ELEMENTS) or np.any((codes < 0) | (codes > 15)):
        raise ValueError("Q4_0 codes must have scales.shape + (32,) and values in 0..15")
    packed = np.empty((*scales.shape, Q4_0_BLOCK_BYTES), dtype=np.uint8)
    packed[..., :2] = scales.astype("<f2", copy=False).view(np.uint8).reshape(*scales.shape, 2)
    packed[..., 2:] = (
        codes[..., :16].astype(np.uint8) | (codes[..., 16:].astype(np.uint8) << 4)
    )
    return np.ascontiguousarray(packed)


def pack_ggml_q8_0_blocks(
    scales: npt.NDArray[np.floating[Any]],
    values: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint8]:
    """Pack explicit signed bytes into native ``block_q8_0`` bytes."""
    scales = np.asarray(scales, dtype=np.float16)
    values = np.asarray(values)
    if values.shape != (*scales.shape, Q4_0_BLOCK_ELEMENTS) or np.any((values < -128) | (values > 127)):
        raise ValueError("Q8_0 values must have scales.shape + (32,) and values in -128..127")
    packed = np.empty((*scales.shape, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    packed[..., :2] = scales.astype("<f2", copy=False).view(np.uint8).reshape(*scales.shape, 2)
    packed[..., 2:] = values.astype(np.int8).view(np.uint8)
    return np.ascontiguousarray(packed)


def unpack_ggml_q4_0_blocks(blocks: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    """Decode native Q4_0 blocks independently of the QPU implementation."""
    blocks = np.asarray(blocks)
    if blocks.dtype != np.dtype(np.uint8) or blocks.ndim < 1 or blocks.shape[-1] != Q4_0_BLOCK_BYTES:
        raise ValueError("Q4_0 blocks must be uint8 with a final extent of 18 bytes")
    scales = blocks[..., :2].copy().reshape(-1, 2).view("<f2").reshape(blocks.shape[:-1]).astype(np.float32)
    packed = blocks[..., 2:]
    values = np.empty((*blocks.shape[:-1], Q4_0_BLOCK_ELEMENTS), dtype=np.int8)
    values[..., :16] = (packed & 0x0F).astype(np.int8) - 8
    values[..., 16:] = (packed >> 4).astype(np.int8) - 8
    return scales, values


def unpack_ggml_q8_0_blocks(blocks: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    """Decode native Q8_0 activation blocks."""
    blocks = np.asarray(blocks)
    if blocks.dtype != np.dtype(np.uint8) or blocks.ndim < 1 or blocks.shape[-1] != Q8_0_BLOCK_BYTES:
        raise ValueError("Q8_0 blocks must be uint8 with a final extent of 34 bytes")
    scales = blocks[..., :2].copy().reshape(-1, 2).view("<f2").reshape(blocks.shape[:-1])
    values = blocks[..., 2:].view(np.int8)
    return scales.astype(np.float32), values


def ggml_q4_0_q8_0_reference(
    activation_blocks: npt.NDArray[np.uint8],
    weight_blocks: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Accumulate exact integer block dots and apply GGML block scales."""
    if activation_blocks.ndim != 3 or weight_blocks.ndim != 3:
        raise ValueError("activation and weight blocks must both be rank three")
    rows, blocks, activation_bytes = activation_blocks.shape
    outputs, weight_block_count, weight_bytes = weight_blocks.shape
    if activation_bytes != Q8_0_BLOCK_BYTES or weight_bytes != Q4_0_BLOCK_BYTES or blocks != weight_block_count:
        raise ValueError("native Q8_0/Q4_0 block extents or reduction block counts do not match")
    activation_scales, activations = unpack_ggml_q8_0_blocks(activation_blocks)
    weight_scales, weights = unpack_ggml_q4_0_blocks(weight_blocks)
    result = np.zeros((rows, outputs), dtype=np.float32)
    for block in range(blocks):
        dots = activations[:, block].astype(np.int32) @ weights[:, block].astype(np.int32).T
        contribution = dots.astype(np.float32) * weight_scales[None, :, block]
        contribution *= activation_scales[:, block, None]
        result += contribution
    return result


def _load_tmu_word(address: Register, destination: Register) -> None:
    mov(tmua, address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(destination))


def _sign_extend_nibbles(
    source: Register,
    destination: Register,
    sign: Register,
    temporary: Register,
    mask: Register,
    sign_bit: Register,
    *,
    high: bool,
) -> None:
    if high:
        shr(destination, source, 4)
        band(destination, destination, mask)
    else:
        band(destination, source, mask)
    bxor(destination, destination, sign_bit)
    band(sign, destination, sign_bit)
    for shift in (1, 2, 3, 4):
        shl(temporary, sign, shift)
        bor(destination, destination, temporary)


@qpu
def qpu_ggml_q4_0_q8_0_linear(asm: Assembly, *, rows: Literal[1, 4]) -> None:
    """Compute four physical rows; the native adapter expands logical M=1."""
    if rows not in {1, 4}:
        raise ValueError("native Q4_0 linear supports exactly one or four rows")

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
    reg_mask = rf13
    reg_sign_bit = rf14
    reg_lane = rf15
    reg_column = rf16
    reg_temporary = rf17
    reg_weight_pointer = rf18
    reg_weight_data = rf19
    reg_weight_header = rf20
    reg_weight_scale = rf21
    reg_activation_data = [rf22, rf23, rf24, rf25]
    reg_activation_scale = [rf26, rf27, rf28, rf29]
    reg_integer_accumulator = [rf30, rf31, rf32, rf33]
    reg_float_accumulator = [rf34, rf35, rf36, rf37]
    reg_q4_word = rf38
    reg_q4_low = rf39
    reg_q4_high = rf40
    reg_sign = rf41
    reg_shifted_sign = rf42
    reg_q8_word = rf43
    reg_product = rf44
    reg_block_count = rf45
    reg_output_pointer = rf46
    reg_q4_block_stride = rf47
    reg_q8_block_stride = rf48
    reg_selected_column = rf49

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
    nop(sig=ldunifrf(reg_mask))
    nop(sig=ldunifrf(reg_sign_bit))
    nop(sig=ldunifrf(reg_q4_block_stride))
    nop(sig=ldunifrf(reg_q8_block_stride))

    pipeline_rows = 4
    for row in range(1, pipeline_rows):
        add(reg_activation_base[row], reg_activation_base[row - 1], reg_activation_stride)
    eidx(reg_lane)
    shl(reg_column, reg_tile, 4)
    add(reg_column, reg_column, reg_lane)
    add(reg_selected_column, reg_column, reg_weight_column_offset)
    umul24(reg_temporary, reg_selected_column, reg_weight_stride)
    add(reg_weight_pointer, reg_weight_base, reg_temporary)
    add(reg_selected_column, reg_column, reg_output_column_offset)
    shl(reg_temporary, reg_selected_column, 2)
    add(reg_output_pointer, reg_output_base, reg_temporary)
    mov(reg_block_count, reg_blocks)
    for row in range(pipeline_rows):
        bxor(reg_float_accumulator[row], reg_float_accumulator[row], reg_float_accumulator[row])
    setnnmode_ss()

    with loop as block_loop:
        mov(reg_weight_data, reg_weight_pointer)
        add(reg_weight_data, reg_weight_data, 2)
        _load_tmu_word(reg_weight_pointer, reg_weight_header)
        add(reg_weight_pointer, reg_weight_pointer, reg_q4_block_stride)
        fmov(reg_weight_scale, reg_weight_header.unpack("l"))
        for row in range(pipeline_rows):
            mov(reg_activation_data[row], reg_activation_base[row])
            add(reg_activation_data[row], reg_activation_data[row], 2)
            _load_tmu_word(reg_activation_base[row], reg_activation_scale[row])
            add(reg_activation_base[row], reg_activation_base[row], reg_q8_block_stride)
            nop()
            fmov(reg_activation_scale[row], reg_activation_scale[row].unpack("l"))
            bxor(
                reg_integer_accumulator[row],
                reg_integer_accumulator[row],
                reg_integer_accumulator[row],
            )

        for _ in range(4):
            _load_tmu_word(reg_weight_data, reg_q4_word)
            add(reg_weight_data, reg_weight_data, 4)
            _sign_extend_nibbles(
                reg_q4_word,
                reg_q4_low,
                reg_sign,
                reg_shifted_sign,
                reg_mask,
                reg_sign_bit,
                high=False,
            )
            _sign_extend_nibbles(
                reg_q4_word,
                reg_q4_high,
                reg_sign,
                reg_shifted_sign,
                reg_mask,
                reg_sign_bit,
                high=True,
            )
            for row in range(pipeline_rows):
                _load_tmu_word(reg_activation_data[row], reg_q8_word)
                add(reg_temporary, reg_activation_data[row], 8)
                add(reg_temporary, reg_temporary, 8)
                v8dot(reg_product, reg_q4_low, reg_q8_word)
                add(reg_integer_accumulator[row], reg_integer_accumulator[row], reg_product)
                _load_tmu_word(reg_temporary, reg_q8_word)
                v8dot(reg_product, reg_q4_high, reg_q8_word)
                add(reg_integer_accumulator[row], reg_integer_accumulator[row], reg_product)
                add(reg_activation_data[row], reg_activation_data[row], 4)

        for row in range(pipeline_rows):
            itof(reg_temporary, reg_integer_accumulator[row])
            fmul(reg_temporary, reg_temporary, reg_weight_scale)
            fmul(reg_temporary, reg_temporary, reg_activation_scale[row])
            fadd(reg_float_accumulator[row], reg_float_accumulator[row], reg_temporary)
        sub(reg_block_count, reg_block_count, 1, cond="pushz")
        block_loop.b(cond="na0")
        nop()
        nop()
        nop()

    for row in range(pipeline_rows):
        mov(tmud, reg_float_accumulator[row])
        mov(tmua, reg_output_pointer)
        tmuwt()
        if row + 1 < pipeline_rows:
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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[int, _ProgramState]] = WeakKeyDictionary()


def supports_ggml_q4_0_q8_0_linear(
    activation: Tensor,
    weight: Tensor,
    destination: Tensor,
    weight_column_start: int,
    output_column_start: int,
    column_count: int,
    backend: Backend,
) -> bool:
    """Return whether native block tensors meet the exact small-M contract."""
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
    if rows != 4 or activation_bytes != Q8_0_BLOCK_BYTES or weight_bytes != Q4_0_BLOCK_BYTES:
        return False
    if blocks != weight_blocks or destination.shape[0] != rows:
        return False
    starts = (weight_column_start, output_column_start)
    if any(start < 0 or start % OUTPUT_TILE for start in starts) or column_count <= 0 or column_count % OUTPUT_TILE:
        return False
    if weight_column_start + column_count > weight_outputs:
        return False
    if output_column_start + column_count > destination.shape[1]:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (activation, weight, destination))


def _program_state(backend: PyVideoCore7Backend, rows: Literal[1, 4]) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(rows)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_ggml_q4_0_q8_0_linear, rows=rows),
                    uniforms=driver.alloc(13, dtype=np.uint32),
                )
            states[rows] = state
        return state


def _execute_ggml_q4_0_q8_0_linear(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 6 or not all(isinstance(argument, Tensor) for argument in args[:3]):
        raise KernelError(
            "GGML Q4_0 linear expects "
            "(activation, weight, destination, weight_column_start, output_column_start, column_count)"
        )
    activation, weight, destination, weight_column_start, output_column_start, column_count = args
    if not all(isinstance(value, int) for value in (weight_column_start, output_column_start, column_count)):
        raise KernelError("GGML Q4_0 column bounds must be integers")
    if not supports_ggml_q4_0_q8_0_linear(
        activation,
        weight,
        destination,
        weight_column_start,
        output_column_start,
        column_count,
        backend,
    ):
        raise KernelError("GGML Q4_0 linear requires native contiguous Q8_0/Q4_0 blocks and aligned columns")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("GGML Q4_0 linear requires PyVideoCore7Backend")
    expected_grid = (column_count // OUTPUT_TILE, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"GGML Q4_0 linear grid must be {expected_grid}, got {grid}")
    rows = activation.shape[0]
    assert rows in {1, 4}
    state = _program_state(backend, rows)
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
        0x0F0F0F0F,
        0x08080808,
        Q4_0_BLOCK_BYTES,
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


GGML_Q4_0_Q8_0_LINEAR_KERNEL = Kernel("vc7.ggml_q4_0_q8_0_linear", _execute_ggml_q4_0_q8_0_linear)

__all__ = [
    "GGML_Q4_0_Q8_0_LINEAR_KERNEL",
    "ggml_q4_0_q8_0_reference",
    "pack_ggml_q4_0_blocks",
    "pack_ggml_q8_0_blocks",
    "supports_ggml_q4_0_q8_0_linear",
    "unpack_ggml_q4_0_blocks",
    "unpack_ggml_q8_0_blocks",
]
