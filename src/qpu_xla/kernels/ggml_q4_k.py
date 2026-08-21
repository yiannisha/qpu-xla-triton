"""Native GGML Q4_K by Q8_K four-row linear candidate."""

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
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu

QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q8_K_BLOCK_BYTES = 292
OUTPUT_TILE = 16


def pack_ggml_q4_k_blocks(
    super_scales: npt.NDArray[np.floating[Any]],
    super_mins: npt.NDArray[np.floating[Any]],
    sub_scales: npt.NDArray[np.integer[Any]],
    sub_mins: npt.NDArray[np.integer[Any]],
    codes: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint8]:
    """Pack explicit Q4_K fields into native 144-byte super-blocks."""
    super_scales = np.asarray(super_scales, dtype=np.float16)
    super_mins = np.asarray(super_mins, dtype=np.float16)
    sub_scales = np.asarray(sub_scales)
    sub_mins = np.asarray(sub_mins)
    codes = np.asarray(codes)
    base_shape = super_scales.shape
    if super_mins.shape != base_shape:
        raise ValueError("Q4_K super scale and min shapes must match")
    if sub_scales.shape != (*base_shape, 8) or sub_mins.shape != (*base_shape, 8):
        raise ValueError("Q4_K sub-scales and sub-mins must have a final extent of 8")
    if np.any((sub_scales < 0) | (sub_scales > 63)) or np.any((sub_mins < 0) | (sub_mins > 63)):
        raise ValueError("Q4_K sub-scales and sub-mins must fit unsigned 6-bit fields")
    if codes.shape != (*base_shape, QK_K) or np.any((codes < 0) | (codes > 15)):
        raise ValueError("Q4_K codes must have a final extent of 256 and values in 0..15")
    result = np.empty((*base_shape, Q4_K_BLOCK_BYTES), dtype=np.uint8)
    result[..., 0:2] = super_scales.astype("<f2", copy=False).view(np.uint8).reshape(*base_shape, 2)
    result[..., 2:4] = super_mins.astype("<f2", copy=False).view(np.uint8).reshape(*base_shape, 2)
    encoded = result[..., 4:16]
    scales = sub_scales.astype(np.uint8)
    mins = sub_mins.astype(np.uint8)
    encoded[..., 0:4] = scales[..., 0:4] | ((scales[..., 4:8] >> 4) << 6)
    encoded[..., 4:8] = mins[..., 0:4] | ((mins[..., 4:8] >> 4) << 6)
    encoded[..., 8:12] = (scales[..., 4:8] & 15) | ((mins[..., 4:8] & 15) << 4)
    packed_codes = result[..., 16:]
    for group in range(4):
        start = group * 64
        packed_codes[..., group * 32 : (group + 1) * 32] = (
            codes[..., start : start + 32].astype(np.uint8)
            | (codes[..., start + 32 : start + 64].astype(np.uint8) << 4)
        )
    return np.ascontiguousarray(result)


def unpack_ggml_q4_k_blocks(
    blocks: npt.NDArray[np.uint8],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    """Decode native Q4_K blocks independently of the QPU implementation."""
    blocks = np.asarray(blocks)
    if blocks.dtype != np.dtype(np.uint8) or blocks.ndim < 1 or blocks.shape[-1] != Q4_K_BLOCK_BYTES:
        raise ValueError("Q4_K blocks must be uint8 with a final extent of 144 bytes")
    base_shape = blocks.shape[:-1]
    super_scales = blocks[..., 0:2].copy().reshape(-1, 2).view("<f2").reshape(base_shape)
    super_mins = blocks[..., 2:4].copy().reshape(-1, 2).view("<f2").reshape(base_shape)
    encoded = blocks[..., 4:16]
    sub_scales = np.empty((*base_shape, 8), dtype=np.uint8)
    sub_mins = np.empty((*base_shape, 8), dtype=np.uint8)
    sub_scales[..., :4] = encoded[..., :4] & 63
    sub_mins[..., :4] = encoded[..., 4:8] & 63
    sub_scales[..., 4:] = (encoded[..., 8:12] & 15) | ((encoded[..., :4] >> 6) << 4)
    sub_mins[..., 4:] = (encoded[..., 8:12] >> 4) | ((encoded[..., 4:8] >> 6) << 4)
    codes = np.empty((*base_shape, QK_K), dtype=np.uint8)
    packed = blocks[..., 16:]
    for group in range(4):
        start = group * 64
        source = packed[..., group * 32 : (group + 1) * 32]
        codes[..., start : start + 32] = source & 15
        codes[..., start + 32 : start + 64] = source >> 4
    return (
        super_scales.astype(np.float32),
        super_mins.astype(np.float32),
        sub_scales,
        sub_mins,
        codes,
    )


def pack_ggml_q8_k_blocks(
    scales: npt.NDArray[np.floating[Any]],
    values: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint8]:
    """Pack Q8_K scale, signed values, and derived 16-value sums."""
    scales = np.asarray(scales, dtype=np.float32)
    values = np.asarray(values)
    if values.shape != (*scales.shape, QK_K) or np.any((values < -128) | (values > 127)):
        raise ValueError("Q8_K values must have scales.shape + (256,) and fit int8")
    result = np.empty((*scales.shape, Q8_K_BLOCK_BYTES), dtype=np.uint8)
    result[..., :4] = scales.astype("<f4", copy=False).view(np.uint8).reshape(*scales.shape, 4)
    signed = values.astype(np.int8)
    result[..., 4:260] = signed.view(np.uint8)
    sums = signed.astype(np.int16).reshape(*scales.shape, 16, 16).sum(axis=-1, dtype=np.int16)
    result[..., 260:] = sums.astype("<i2", copy=False).view(np.uint8).reshape(*scales.shape, 32)
    return np.ascontiguousarray(result)


def unpack_ggml_q8_k_blocks(
    blocks: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8], npt.NDArray[np.int16]]:
    """Decode native Q8_K activation blocks."""
    blocks = np.asarray(blocks)
    if blocks.dtype != np.dtype(np.uint8) or blocks.ndim < 1 or blocks.shape[-1] != Q8_K_BLOCK_BYTES:
        raise ValueError("Q8_K blocks must be uint8 with a final extent of 292 bytes")
    base_shape = blocks.shape[:-1]
    scales = blocks[..., :4].copy().reshape(-1, 4).view("<f4").reshape(base_shape)
    values = blocks[..., 4:260].view(np.int8)
    sums = blocks[..., 260:].copy().reshape(-1, 32).view("<i2").reshape(*base_shape, 16)
    return scales.astype(np.float32), values, sums


def ggml_q4_k_q8_k_reference(
    activation_blocks: npt.NDArray[np.uint8],
    weight_blocks: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Accumulate exact Q4_K/Q8_K integer terms and apply native scales."""
    if activation_blocks.ndim != 3 or weight_blocks.ndim != 3:
        raise ValueError("activation and weight blocks must both be rank three")
    rows, blocks, activation_bytes = activation_blocks.shape
    outputs, weight_block_count, weight_bytes = weight_blocks.shape
    if activation_bytes != Q8_K_BLOCK_BYTES or weight_bytes != Q4_K_BLOCK_BYTES:
        raise ValueError("native Q8_K/Q4_K block extents do not match")
    if blocks != weight_block_count:
        raise ValueError("activation and weight reduction block counts do not match")
    activation_scales, activations, _ = unpack_ggml_q8_k_blocks(activation_blocks)
    weight_scales, weight_mins, sub_scales, sub_mins, codes = unpack_ggml_q4_k_blocks(
        weight_blocks
    )
    result = np.zeros((rows, outputs), dtype=np.float32)
    for block in range(blocks):
        for subblock in range(8):
            start = subblock * 32
            stop = start + 32
            q8 = activations[:, block, start:stop].astype(np.int32)
            q4 = codes[:, block, start:stop].astype(np.int32)
            dots = q8 @ q4.T
            sums = q8.sum(axis=1, dtype=np.int32)
            contribution = dots.astype(np.float32) * sub_scales[None, :, block, subblock]
            contribution *= weight_scales[None, :, block]
            contribution -= (
                sums[:, None].astype(np.float32)
                * sub_mins[None, :, block, subblock]
                * weight_mins[None, :, block]
            )
            contribution *= activation_scales[:, block, None]
            result += contribution
    return result


def _load_tmu_word(address: Register, destination: Register) -> None:
    mov(tmua, address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(destination))


@qpu
def qpu_ggml_q4_k_q8_k_linear_m4(asm: Assembly) -> None:
    """Compute a native Q4_K by Q8_K four-row output tile."""
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
    reg_nibble_mask = rf13
    reg_ones = rf14
    reg_scale_mask = rf15
    reg_lane = rf16
    reg_column = rf17
    reg_temporary = rf18
    reg_weight_pointer = rf19
    reg_weight_header = rf20
    reg_weight_scale = rf21
    reg_weight_min = rf22
    reg_activation_data = [rf23, rf24, rf25, rf26]
    reg_activation_scale = [rf27, rf28, rf29, rf30]
    reg_dot = [rf31, rf32, rf33, rf34]
    reg_sum = [rf35, rf36, rf37, rf38]
    reg_accumulator = [rf39, rf40, rf41, rf42]
    reg_q4_word = rf43
    reg_q4_values = rf44
    reg_q8_word = rf45
    reg_product = rf46
    reg_block_count = rf47
    reg_output_pointer = rf48
    reg_q4_stride = rf49
    reg_q8_stride = rf50
    reg_selected_column = rf51
    reg_scale_word0 = rf52
    reg_scale_word1 = rf53
    reg_scale_word2 = rf54
    reg_sub_scale = rf55
    reg_sub_min = rf56
    reg_scale_float = rf57
    reg_min_float = rf58
    reg_byte = rf59
    reg_q4_pointer = rf60
    reg_float_temporary = rf61
    reg_float_minimum = rf62

    def accumulate_subblock(*, high_nibble: bool) -> None:
        mov(reg_temporary, reg_q4_pointer)
        for active_row in range(4):
            bxor(reg_dot[active_row], reg_dot[active_row], reg_dot[active_row])
            bxor(reg_sum[active_row], reg_sum[active_row], reg_sum[active_row])
        mov(reg_byte, 8)
        with loop as word_loop:
            _load_tmu_word(reg_temporary, reg_q4_word)
            add(reg_temporary, reg_temporary, 4)
            if high_nibble:
                shr(reg_q4_values, reg_q4_word, 4)
                band(reg_q4_values, reg_q4_values, reg_nibble_mask)
            else:
                band(reg_q4_values, reg_q4_word, reg_nibble_mask)
            for active_row in range(4):
                _load_tmu_word(reg_activation_data[active_row], reg_q8_word)
                add(reg_activation_data[active_row], reg_activation_data[active_row], 4)
                v8dot(reg_product, reg_q4_values, reg_q8_word)
                add(reg_dot[active_row], reg_dot[active_row], reg_product)
                v8dot(reg_product, reg_ones, reg_q8_word)
                add(reg_sum[active_row], reg_sum[active_row], reg_product)
            sub(reg_byte, reg_byte, 1, cond="pushz")
            word_loop.b(cond="na0")
            nop()
            nop()
            nop()
        for active_row in range(4):
            itof(reg_float_temporary, reg_dot[active_row])
            fmul(reg_float_temporary, reg_float_temporary, reg_scale_float)
            fmul(reg_float_temporary, reg_float_temporary, reg_activation_scale[active_row])
            itof(reg_float_minimum, reg_sum[active_row])
            fmul(reg_float_minimum, reg_float_minimum, reg_min_float)
            fmul(reg_float_minimum, reg_float_minimum, reg_activation_scale[active_row])
            fsub(reg_float_temporary, reg_float_temporary, reg_float_minimum)
            fadd(reg_accumulator[active_row], reg_accumulator[active_row], reg_float_temporary)

    def prepare_scale_min(*, high_group: bool) -> None:
        if high_group:
            band(reg_sub_scale, reg_scale_word2, 15)
            shr(reg_byte, reg_scale_word0, 6)
            band(reg_byte, reg_byte, 3)
            shl(reg_byte, reg_byte, 4)
            bor(reg_sub_scale, reg_sub_scale, reg_byte)
            shr(reg_sub_min, reg_scale_word2, 4)
            band(reg_sub_min, reg_sub_min, 15)
            shr(reg_byte, reg_scale_word1, 6)
            band(reg_byte, reg_byte, 3)
            shl(reg_byte, reg_byte, 4)
            bor(reg_sub_min, reg_sub_min, reg_byte)
        else:
            band(reg_sub_scale, reg_scale_word0, reg_scale_mask)
            band(reg_sub_min, reg_scale_word1, reg_scale_mask)
        itof(reg_scale_float, reg_sub_scale)
        fmul(reg_scale_float, reg_scale_float, reg_weight_scale)
        itof(reg_min_float, reg_sub_min)
        fmul(reg_min_float, reg_min_float, reg_weight_min)

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
    nop(sig=ldunifrf(reg_nibble_mask))
    nop(sig=ldunifrf(reg_ones))
    nop(sig=ldunifrf(reg_scale_mask))
    nop(sig=ldunifrf(reg_q4_stride))
    nop(sig=ldunifrf(reg_q8_stride))

    for row in range(1, 4):
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
    for row in range(4):
        bxor(reg_accumulator[row], reg_accumulator[row], reg_accumulator[row])
    setnnmode_ss()

    with loop as block_loop:
        _load_tmu_word(reg_weight_pointer, reg_weight_header)
        add(reg_temporary, reg_weight_pointer, 4)
        _load_tmu_word(reg_temporary, reg_scale_word0)
        add(reg_temporary, reg_weight_pointer, 8)
        _load_tmu_word(reg_temporary, reg_scale_word1)
        add(reg_temporary, reg_weight_pointer, 12)
        _load_tmu_word(reg_temporary, reg_scale_word2)
        add(reg_temporary, reg_weight_pointer, 8)
        add(reg_q4_pointer, reg_temporary, 8)
        add(reg_weight_pointer, reg_weight_pointer, reg_q4_stride)
        fmov(reg_weight_scale, reg_weight_header.unpack("l"))
        fmov(reg_weight_min, reg_weight_header.unpack("h"))
        mov(reg_column, reg_scale_word0)
        mov(reg_selected_column, reg_scale_word1)
        for row in range(4):
            _load_tmu_word(reg_activation_base[row], reg_activation_scale[row])
            add(reg_activation_data[row], reg_activation_base[row], 4)

        # The first four Q4_K subblocks consume the low six-bit fields and
        # the first half of Q8_K data.  Two packed nibble subblocks share each
        # 32-byte Q4 payload, so only that pair is emitted statically while
        # the eight packed words and two payload groups loop in QPU code.
        mov(reg_output_column_offset, 2)
        with loop as low_group_loop:
            prepare_scale_min(high_group=False)
            accumulate_subblock(high_nibble=False)
            shr(reg_scale_word0, reg_scale_word0, 8)
            shr(reg_scale_word1, reg_scale_word1, 8)
            prepare_scale_min(high_group=False)
            accumulate_subblock(high_nibble=True)
            shr(reg_scale_word0, reg_scale_word0, 8)
            shr(reg_scale_word1, reg_scale_word1, 8)
            add(reg_q4_pointer, reg_q4_pointer, 15)
            add(reg_q4_pointer, reg_q4_pointer, 15)
            add(reg_q4_pointer, reg_q4_pointer, 2)
            sub(reg_output_column_offset, reg_output_column_offset, 1, cond="pushz")
            low_group_loop.b(cond="na0")
            nop()
            nop()
            nop()

        # Restore the packed high-bit carriers, then consume subblocks 4..7.
        mov(reg_scale_word0, reg_column)
        mov(reg_scale_word1, reg_selected_column)
        mov(reg_output_column_offset, 2)
        with loop as high_group_loop:
            prepare_scale_min(high_group=True)
            accumulate_subblock(high_nibble=False)
            shr(reg_scale_word0, reg_scale_word0, 8)
            shr(reg_scale_word1, reg_scale_word1, 8)
            shr(reg_scale_word2, reg_scale_word2, 8)
            prepare_scale_min(high_group=True)
            accumulate_subblock(high_nibble=True)
            shr(reg_scale_word0, reg_scale_word0, 8)
            shr(reg_scale_word1, reg_scale_word1, 8)
            shr(reg_scale_word2, reg_scale_word2, 8)
            add(reg_q4_pointer, reg_q4_pointer, 15)
            add(reg_q4_pointer, reg_q4_pointer, 15)
            add(reg_q4_pointer, reg_q4_pointer, 2)
            sub(reg_output_column_offset, reg_output_column_offset, 1, cond="pushz")
            high_group_loop.b(cond="na0")
            nop()
            nop()
            nop()
        for row in range(4):
            add(reg_activation_base[row], reg_activation_base[row], reg_q8_stride)
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


def supports_ggml_q4_k_q8_k_linear_m4(
    activation: Tensor,
    weight: Tensor,
    destination: Tensor,
    weight_column_start: int,
    output_column_start: int,
    column_count: int,
    backend: Backend,
) -> bool:
    """Return whether native block tensors meet the exact M=4 contract."""
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
    if rows != 4 or activation_bytes != Q8_K_BLOCK_BYTES or weight_bytes != Q4_K_BLOCK_BYTES:
        return False
    if blocks != weight_blocks or destination.shape[0] != rows:
        return False
    starts = (weight_column_start, output_column_start)
    if any(start < 0 or start % OUTPUT_TILE for start in starts):
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
                    code=driver.program(qpu_ggml_q4_k_q8_k_linear_m4),
                    uniforms=driver.alloc(14, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute_ggml_q4_k_q8_k_linear_m4(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 6 or not all(isinstance(argument, Tensor) for argument in args[:3]):
        raise KernelError(
            "GGML Q4_K linear expects "
            "(activation, weight, destination, weight_column_start, output_column_start, column_count)"
        )
    activation, weight, destination, weight_column_start, output_column_start, column_count = args
    if not all(isinstance(value, int) for value in (weight_column_start, output_column_start, column_count)):
        raise KernelError("GGML Q4_K column bounds must be integers")
    if not supports_ggml_q4_k_q8_k_linear_m4(
        activation,
        weight,
        destination,
        weight_column_start,
        output_column_start,
        column_count,
        backend,
    ):
        raise KernelError("GGML Q4_K linear requires contiguous Q8_K/Q4_K M=4 blocks and aligned columns")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("GGML Q4_K linear requires PyVideoCore7Backend")
    expected_grid = (column_count // OUTPUT_TILE, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"GGML Q4_K linear grid must be {expected_grid}, got {grid}")
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
        0x0F0F0F0F,
        0x01010101,
        0x0000003F,
        Q4_K_BLOCK_BYTES,
        Q8_K_BLOCK_BYTES,
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


GGML_Q4_K_Q8_K_LINEAR_M4_KERNEL = Kernel(
    "vc7.ggml_q4_k_q8_k_linear_m4",
    _execute_ggml_q4_k_q8_k_linear_m4,
)

__all__ = [
    "GGML_Q4_K_Q8_K_LINEAR_M4_KERNEL",
    "ggml_q4_k_q8_k_reference",
    "pack_ggml_q4_k_blocks",
    "pack_ggml_q8_k_blocks",
    "supports_ggml_q4_k_q8_k_linear_m4",
    "unpack_ggml_q4_k_blocks",
    "unpack_ggml_q8_k_blocks",
]
