"""Native GGML Q4_0 by Q8_0 small-row linear candidate."""

from __future__ import annotations

from collections import deque
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
MX_ROW_TILE = 16


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


def _emit_tiled_mx_q4_0_q8_0_linear() -> None:
    """Emit a 16x16 output tile with pipelined native-block TMU loads."""
    reg_tile = rf0
    reg_blocks = rf1
    reg_activation_stride = rf2
    reg_activation_block = rf3
    reg_weight_stride = rf4
    reg_weight_base = rf5
    reg_output_stride = rf6
    reg_output_base = rf7
    reg_weight_column_offset = rf8
    reg_output_column_offset = rf9
    reg_mask = rf10
    reg_sign_bit = rf11
    reg_q4_block_stride = rf12
    reg_q8_block_stride = rf13
    reg_row_group = rf14
    reg_lane = rf15
    reg_column = rf16
    reg_temporary = rf17
    reg_weight_pointer = rf18
    reg_weight_data = rf19
    reg_header_or_product = rf20
    reg_weight_scale = rf21
    reg_activation_row = rf22
    reg_activation_scale = rf23
    reg_integer_accumulator = rf24
    reg_block_count = rf25
    reg_output_pointer = rf26
    reg_selected_column = rf27
    reg_sign = rf28
    reg_shifted_sign = rf29
    reg_q4_packed = [rf30, rf31, rf32, rf33]
    reg_q4_values = [rf[index] for index in range(34, 42)]
    reg_float_accumulator = [rf[index] for index in range(42, 42 + MX_ROW_TILE)]
    reg_q8_word = rf58

    mov(reg_tile, rf3.unpack("ul"))
    mov(reg_row_group, rf3.unpack("uh"))
    nop(sig=ldunifrf(reg_blocks))
    nop(sig=ldunifrf(reg_activation_stride))
    nop(sig=ldunifrf(reg_activation_block))
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

    umul24(reg_temporary, reg_row_group, reg_activation_stride)
    shl(reg_temporary, reg_temporary, 4)
    add(reg_activation_block, reg_activation_block, reg_temporary)
    umul24(reg_temporary, reg_row_group, reg_output_stride)
    shl(reg_temporary, reg_temporary, 4)
    add(reg_output_base, reg_output_base, reg_temporary)
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
    for accumulator in reg_float_accumulator:
        bxor(accumulator, accumulator, accumulator)
    setnnmode_ss()

    with loop as block_loop:
        mov(tmuc, -1)
        mov(tmua, reg_weight_pointer, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_header_or_product))
        sub(reg_weight_data, reg_weight_pointer, -16)
        bnot(tmuc, 3)
        mov(tmua, reg_weight_data, sig=thrsw)
        nop()
        nop()
        for word in range(4):
            nop(sig=ldtmu(reg_q4_packed[word]))
        fmov(reg_weight_scale, reg_header_or_product.unpack("l"))
        add(reg_weight_pointer, reg_weight_pointer, reg_q4_block_stride)

        for word in range(4):
            _sign_extend_nibbles(
                reg_q4_packed[word],
                reg_q4_values[word],
                reg_sign,
                reg_shifted_sign,
                reg_mask,
                reg_sign_bit,
                high=False,
            )
            _sign_extend_nibbles(
                reg_q4_packed[word],
                reg_q4_values[4 + word],
                reg_sign,
                reg_shifted_sign,
                reg_mask,
                reg_sign_bit,
                high=True,
            )

        mov(reg_activation_row, reg_activation_block)
        for row in range(MX_ROW_TILE):
            mov(tmuc, -1)
            mov(tmua, reg_activation_row, sig=thrsw)
            nop()
            nop()
            nop(sig=ldtmu(reg_header_or_product))
            fmov(reg_activation_scale, reg_header_or_product.unpack("l"))
            sub(reg_temporary, reg_activation_row, -16)
            bxor(
                reg_integer_accumulator,
                reg_integer_accumulator,
                reg_integer_accumulator,
            )
            bnot(tmuc, 3)
            mov(tmua, reg_temporary)
            sub(reg_temporary, reg_temporary, -16)
            bnot(tmuc, 3)
            mov(tmua, reg_temporary, sig=thrsw)
            nop()
            nop()
            for word in range(8):
                nop(sig=ldtmu(reg_q8_word))
                v8dot(
                    reg_header_or_product,
                    reg_q4_values[word],
                    reg_q8_word,
                )
                add(
                    reg_integer_accumulator,
                    reg_integer_accumulator,
                    reg_header_or_product,
                )
            itof(reg_temporary, reg_integer_accumulator)
            fmul(reg_temporary, reg_temporary, reg_weight_scale)
            fmul(reg_temporary, reg_temporary, reg_activation_scale)
            fadd(
                reg_float_accumulator[row],
                reg_float_accumulator[row],
                reg_temporary,
            )
            if row + 1 < MX_ROW_TILE:
                add(reg_activation_row, reg_activation_row, reg_activation_stride)

        add(reg_activation_block, reg_activation_block, reg_q8_block_stride)
        sub(reg_block_count, reg_block_count, 1, cond="pushz")
        block_loop.b(cond="na0")
        nop()
        nop()
        nop()

    mov(tmuc, -1)
    for row in range(MX_ROW_TILE):
        mov(tmud, reg_float_accumulator[row])
        mov(tmua, reg_output_pointer)
        tmuwt()
        if row + 1 < MX_ROW_TILE:
            add(reg_output_pointer, reg_output_pointer, reg_output_stride)
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_ggml_q4_0_q8_0_tiled_gemm(
    asm: Assembly, *, column_weight_scale: bool = False
) -> None:
    """Compute 16x16 tiles with exact or approximate column weight scales."""
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10
    reg_c_base = rf15
    reg_integer_accumulator = [rf[index] for index in range(16, 32)]
    reg_block_count = rf32
    reg_activation_scale_stride = rf33
    reg_activation_scale_pointer = rf34
    reg_weight_scale_block_stride = rf35
    reg_weight_scale_pointer = rf36
    reg_activation_scale = rf37
    reg_weight_scale = rf38
    reg_float_accumulator = [rf[index] for index in range(39, 55)]
    reg_float_temporary = rf55

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_a_stride))
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)
    umul24(rf4, rf3, reg_a_stride)
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)
    add(reg_b_base, reg_b_base, rf3)

    # Prime the same K16 double-buffered TMU stream used by tiled W8A8.
    bnot(tmuc, 3)
    mov(tmua, reg_a_base)
    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)
    sub(reg_a_base, reg_a_base, -16)

    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    add(reg_c_base, reg_c_base, rf0)
    add(reg_c_base, reg_c_base, rf3, sig=ldunifrf(reg_block_count))

    nop(sig=ldunifrf(reg_activation_scale_stride))
    umul24(rf3, reg_tile_i, reg_activation_scale_stride,
        sig=ldunifrf(reg_activation_scale_pointer))
    shl(rf3, rf3, 4)
    add(reg_activation_scale_pointer, reg_activation_scale_pointer, rf3)
    eidx(rf3)
    umul24(rf3, rf3, reg_activation_scale_stride)
    add(reg_activation_scale_pointer, reg_activation_scale_pointer, rf3)

    if not column_weight_scale:
        nop(sig=ldunifrf(reg_weight_scale_block_stride))
    nop(sig=ldunifrf(reg_weight_scale_pointer))
    shl(rf3, reg_tile_j, 6 if column_weight_scale else 5)
    add(reg_weight_scale_pointer, reg_weight_scale_pointer, rf3)
    eidx(rf3)
    shl(rf3, rf3, 2 if column_weight_scale else 1)
    add(reg_weight_scale_pointer, reg_weight_scale_pointer, rf3)

    for index in range(8):
        bxor(reg_float_accumulator[index], reg_float_accumulator[index],
            reg_float_accumulator[index]).sub(
                reg_float_accumulator[index + 8],
                reg_float_accumulator[index + 8],
                reg_float_accumulator[index + 8],
                sig=ldtmu((reg_a + reg_b)[index]),
            )
    setnnmode_ss()

    def emit_k16() -> None:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)
        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(
            reg_b_base, reg_b_base, reg_b_stride_x4)
        sub(reg_a_base, reg_a_base, -16)

        broadcast_order = deque(reg_b * 16)

        def broadcast_next() -> None:
            register = broadcast_order.popleft()
            rotate(register, register, 1).mov(rep, register)

        broadcast_next()
        v8dot(rf1, rf0, reg_a[0])
        for index in range(15):
            broadcast_next()
            add(reg_integer_accumulator[index], reg_integer_accumulator[index], rf1).v8dot(
                rf1, rf0, reg_a[0])
        broadcast_next()
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1).v8dot(
            rf1, rf0, reg_a[1])
        for index in range(15):
            broadcast_next()
            add(reg_integer_accumulator[index], reg_integer_accumulator[index], rf1).v8dot(
                rf1, rf0, reg_a[1])
        broadcast_next()
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1).v8dot(
            rf1, rf0, reg_a[2])
        for index in range(15):
            broadcast_next()
            add(reg_integer_accumulator[index], reg_integer_accumulator[index], rf1).v8dot(
                rf1, rf0, reg_a[2])
        broadcast_next()
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1).v8dot(
            rf1, rf0, reg_a[3])
        for index in range(8):
            broadcast_next()
            add(reg_integer_accumulator[index], reg_integer_accumulator[index], rf1).v8dot(
                rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[8], reg_integer_accumulator[8], rf1,
            sig=ldtmu(reg_a[0])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[9], reg_integer_accumulator[9], rf1,
            sig=ldtmu(reg_a[1])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[10], reg_integer_accumulator[10], rf1,
            sig=ldtmu(reg_a[2])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[11], reg_integer_accumulator[11], rf1,
            sig=ldtmu(rf2)).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[12], reg_integer_accumulator[12], rf1,
            sig=ldtmu(reg_b[0])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[13], reg_integer_accumulator[13], rf1,
            sig=ldtmu(reg_b[1])).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(reg_integer_accumulator[14], reg_integer_accumulator[14], rf1,
            sig=ldtmu(reg_b[2])).v8dot(rf1, rf0, reg_a[3])
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1,
            sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)

    with loop as block_loop:
        for index in range(8):
            bxor(reg_integer_accumulator[index], reg_integer_accumulator[index],
                reg_integer_accumulator[index]).sub(
                    reg_integer_accumulator[index + 8],
                    reg_integer_accumulator[index + 8],
                    reg_integer_accumulator[index + 8],
                )
        emit_k16()
        emit_k16()

        mov(tmuc, -1)
        if column_weight_scale:
            _load_tmu_word(reg_activation_scale_pointer, reg_activation_scale)
        else:
            mov(tmua, reg_activation_scale_pointer)
            mov(tmua, reg_weight_scale_pointer, sig=thrsw)
            nop()
            nop()
            nop(sig=ldtmu(reg_activation_scale))
            nop(sig=ldtmu(reg_weight_scale))
        fmov(reg_activation_scale, reg_activation_scale.unpack("l"))
        add(reg_activation_scale_pointer, reg_activation_scale_pointer, 2)
        if not column_weight_scale:
            fmov(reg_weight_scale, reg_weight_scale.unpack("l"))
            add(reg_weight_scale_pointer, reg_weight_scale_pointer,
                reg_weight_scale_block_stride)

        for index in range(16):
            if not column_weight_scale:
                rotate(reg_weight_scale, reg_weight_scale, 1).mov(rep, reg_weight_scale)
            itof(reg_float_temporary, reg_integer_accumulator[index])
            fmul(reg_float_temporary, reg_float_temporary, reg_activation_scale)
            if not column_weight_scale:
                fmul(reg_float_temporary, reg_float_temporary, rf0)
            fadd(reg_float_accumulator[index], reg_float_accumulator[index],
                reg_float_temporary)

        sub(reg_block_count, reg_block_count, 1, cond="pushz")
        block_loop.b(cond="na0")
        nop()
        nop()
        nop()

    if column_weight_scale:
        mov(tmuc, -1)
        _load_tmu_word(reg_weight_scale_pointer, reg_weight_scale)
        for index in range(16):
            rotate(reg_weight_scale, reg_weight_scale, 1).mov(rep, reg_weight_scale)
            fmul(reg_float_accumulator[index], reg_float_accumulator[index], rf0)

    mov(tmuc, -1)
    for index in range(0, 16, 4):
        mov(tmud, reg_float_accumulator[index])
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
        mov(tmud, reg_float_accumulator[index + 1])
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
        mov(tmud, reg_float_accumulator[index + 2])
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
        mov(tmud, reg_float_accumulator[index + 3])
        mov(tmua, reg_c_base)
        tmuwt()
        if index < 12:
            sub(reg_c_base, reg_c_base, -4)
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_ggml_q4_0_q8_0_linear(asm: Assembly, *, rows: Literal[0, 1, 4]) -> None:
    """Compute native Q4_0 by Q8_0; ``rows=0`` selects 16x16 tiled M."""
    if rows not in {0, 1, 4}:
        raise ValueError("native Q4_0 linear supports one, four, or batched rows")
    if rows == 0:
        _emit_tiled_mx_q4_0_q8_0_linear()
        return

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
    if (
        rows <= 0
        or (rows != 4 and rows % MX_ROW_TILE)
        or activation_bytes != Q8_0_BLOCK_BYTES
        or weight_bytes != Q4_0_BLOCK_BYTES
    ):
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


def _program_state(backend: PyVideoCore7Backend, rows: Literal[0, 1, 4]) -> _ProgramState:
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
    rows = activation.shape[0]
    row_tile = 4 if rows == 4 else MX_ROW_TILE
    expected_grid = (column_count // OUTPUT_TILE, rows // row_tile, 1)
    if grid != expected_grid:
        raise KernelError(f"GGML Q4_0 linear grid must be {expected_grid}, got {grid}")
    program_rows: Literal[0, 1, 4] = 4 if rows == 4 else 0
    state = _program_state(backend, program_rows)
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
            thread=expected_grid[0] * expected_grid[1],
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
