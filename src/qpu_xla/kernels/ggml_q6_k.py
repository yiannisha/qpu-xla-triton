"""Native GGML Q6_K by Q8_K four-row linear candidate."""

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
from qpu_xla.kernels.ggml_q4_k import (
    Q8_K_BLOCK_BYTES,
    pack_ggml_q8_k_blocks,
    unpack_ggml_q8_k_blocks,
)
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu

QK_K = 256
Q6_K_BLOCK_BYTES = 210
OUTPUT_TILE = 16


def pack_ggml_q6_k_blocks(
    super_scales: npt.NDArray[np.floating[Any]],
    sub_scales: npt.NDArray[np.integer[Any]],
    codes: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint8]:
    """Pack explicit signed Q6_K values into native 210-byte blocks."""
    super_scales = np.asarray(super_scales, dtype=np.float16)
    sub_scales = np.asarray(sub_scales)
    codes = np.asarray(codes)
    base_shape = super_scales.shape
    if sub_scales.shape != (*base_shape, 16) or np.any(
        (sub_scales < -128) | (sub_scales > 127)
    ):
        raise ValueError("Q6_K sub-scales must have a final extent of 16 and fit int8")
    if codes.shape != (*base_shape, QK_K) or np.any((codes < -32) | (codes > 31)):
        raise ValueError("Q6_K codes must have a final extent of 256 and values in -32..31")
    result = np.zeros((*base_shape, Q6_K_BLOCK_BYTES), dtype=np.uint8)
    unsigned = (codes.astype(np.int16) + 32).astype(np.uint8)
    ql = result[..., :128]
    qh = result[..., 128:192]
    for half in range(2):
        code_base = half * 128
        ql_base = half * 64
        qh_base = half * 32
        for group in range(4):
            source = unsigned[..., code_base + group * 32 : code_base + (group + 1) * 32]
            destination = ql[..., ql_base + (group % 2) * 32 : ql_base + (group % 2 + 1) * 32]
            if group < 2:
                destination |= source & 15
            else:
                destination |= (source & 15) << 4
            qh[..., qh_base : qh_base + 32] |= ((source >> 4) & 3) << (2 * group)
    result[..., 192:208] = sub_scales.astype(np.int8).view(np.uint8)
    result[..., 208:210] = super_scales.astype("<f2", copy=False).view(np.uint8).reshape(
        *base_shape, 2
    )
    return np.ascontiguousarray(result)


def unpack_ggml_q6_k_blocks(
    blocks: npt.NDArray[np.uint8],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
]:
    """Decode native Q6_K super-scales, signed sub-scales, and signed values."""
    blocks = np.asarray(blocks)
    if blocks.dtype != np.dtype(np.uint8) or blocks.ndim < 1 or blocks.shape[-1] != Q6_K_BLOCK_BYTES:
        raise ValueError("Q6_K blocks must be uint8 with a final extent of 210 bytes")
    base_shape = blocks.shape[:-1]
    ql = blocks[..., :128]
    qh = blocks[..., 128:192]
    codes = np.empty((*base_shape, QK_K), dtype=np.int8)
    for half in range(2):
        code_base = half * 128
        ql_base = half * 64
        qh_base = half * 32
        for group in range(4):
            low_source = ql[
                ..., ql_base + (group % 2) * 32 : ql_base + (group % 2 + 1) * 32
            ]
            low = low_source & 15 if group < 2 else low_source >> 4
            high = (qh[..., qh_base : qh_base + 32] >> (2 * group)) & 3
            codes[..., code_base + group * 32 : code_base + (group + 1) * 32] = (
                low.astype(np.int16) | (high.astype(np.int16) << 4)
            ).astype(np.int8) - np.int8(32)
    sub_scales = blocks[..., 192:208].view(np.int8)
    super_scales = (
        blocks[..., 208:210].copy().reshape(-1, 2).view("<f2").reshape(base_shape)
    )
    return super_scales.astype(np.float32), sub_scales, codes


def ggml_q6_k_q8_k_reference(
    activation_blocks: npt.NDArray[np.uint8],
    weight_blocks: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Accumulate exact Q6_K/Q8_K integer terms before applying FP32 scales."""
    if activation_blocks.ndim != 3 or weight_blocks.ndim != 3:
        raise ValueError("activation and weight blocks must both be rank three")
    rows, blocks, activation_bytes = activation_blocks.shape
    outputs, weight_block_count, weight_bytes = weight_blocks.shape
    if activation_bytes != Q8_K_BLOCK_BYTES or weight_bytes != Q6_K_BLOCK_BYTES:
        raise ValueError("native Q8_K/Q6_K block extents do not match")
    if blocks != weight_block_count:
        raise ValueError("activation and weight reduction block counts do not match")
    activation_scales, activations, _ = unpack_ggml_q8_k_blocks(activation_blocks)
    weight_scales, sub_scales, weights = unpack_ggml_q6_k_blocks(weight_blocks)
    result = np.zeros((rows, outputs), dtype=np.float32)
    for block in range(blocks):
        integer = np.zeros((rows, outputs), dtype=np.int32)
        for subblock in range(16):
            span = slice(subblock * 16, (subblock + 1) * 16)
            dots = (
                activations[:, block, span].astype(np.int32)
                @ weights[:, block, span].astype(np.int32).T
            )
            integer += dots * sub_scales[None, :, block, subblock].astype(np.int32)
        contribution = integer.astype(np.float32)
        contribution *= weight_scales[None, :, block]
        contribution *= activation_scales[:, block, None]
        result += contribution
    return result


def _load_tmu_word(address: Register, destination: Register) -> None:
    mov(tmua, address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(destination))


@qpu
def qpu_ggml_q6_k_q8_k_linear_m4(asm: Assembly) -> None:
    """Compute a native Q6_K by Q8_K four-row output tile."""
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
    reg_two_bit_mask = rf14
    reg_ones = rf15
    reg_q6_stride = rf16
    reg_q8_stride = rf17
    reg_ql_half_bytes = rf18
    reg_lane = rf19
    reg_column = rf20
    reg_temporary = rf21
    reg_weight_pointer = rf22
    reg_ql_pointer = rf23
    reg_qh_base = rf24
    reg_qh_pointer = rf25
    reg_scale_pointer = rf26
    reg_ql_word = rf27
    reg_qh_word = rf28
    reg_q6_values = rf29
    reg_qh_values = rf30
    reg_scale_word = rf31
    reg_scale = rf32
    reg_qh_shift = rf33
    reg_word_count = rf34
    reg_group_count = rf35
    reg_q8_word = rf36
    reg_product = rf37
    reg_activation_data = [rf38, rf39, rf40, rf41]
    reg_activation_scale = [rf42, rf43, rf44, rf45]
    reg_dot = [rf46, rf47, rf48, rf49]
    reg_sum = [rf50, rf51, rf52, rf53]
    reg_block_accumulator = [rf54, rf55, rf56, rf57]
    reg_accumulator = [rf58, rf59, rf60, rf61]
    reg_weight_scale = rf62
    reg_output_pointer = rf63

    def extract_signed_scale() -> None:
        mov(reg_scale, reg_scale_word)
        for _ in range(3):
            shl(reg_scale, reg_scale, 8)
        for _ in range(3):
            asr(reg_scale, reg_scale, 8)
        shr(reg_scale_word, reg_scale_word, 8)

    def accumulate_subblock(*, high_nibble: bool) -> None:
        extract_signed_scale()
        for active_row in range(4):
            bxor(reg_dot[active_row], reg_dot[active_row], reg_dot[active_row])
            bxor(reg_sum[active_row], reg_sum[active_row], reg_sum[active_row])
        mov(reg_word_count, 4)
        with loop as word_loop:
            _load_tmu_word(reg_ql_pointer, reg_ql_word)
            add(reg_ql_pointer, reg_ql_pointer, 4)
            _load_tmu_word(reg_qh_pointer, reg_qh_word)
            add(reg_qh_pointer, reg_qh_pointer, 4)
            if high_nibble:
                shr(reg_q6_values, reg_ql_word, 4)
                band(reg_q6_values, reg_q6_values, reg_nibble_mask)
            else:
                band(reg_q6_values, reg_ql_word, reg_nibble_mask)
            shr(reg_qh_values, reg_qh_word, reg_qh_shift)
            band(reg_qh_values, reg_qh_values, reg_two_bit_mask)
            shl(reg_qh_values, reg_qh_values, 4)
            bor(reg_q6_values, reg_q6_values, reg_qh_values)
            for active_row in range(4):
                _load_tmu_word(reg_activation_data[active_row], reg_q8_word)
                add(reg_activation_data[active_row], reg_activation_data[active_row], 4)
                v8dot(reg_product, reg_q6_values, reg_q8_word)
                add(reg_dot[active_row], reg_dot[active_row], reg_product)
                v8dot(reg_product, reg_ones, reg_q8_word)
                add(reg_sum[active_row], reg_sum[active_row], reg_product)
            sub(reg_word_count, reg_word_count, 1, cond="pushz")
            word_loop.b(cond="na0")
            nop()
            nop()
            nop()
        for active_row in range(4):
            shl(reg_sum[active_row], reg_sum[active_row], 5)
            sub(reg_dot[active_row], reg_dot[active_row], reg_sum[active_row])
            smul24(reg_product, reg_dot[active_row], reg_scale)
            add(
                reg_block_accumulator[active_row],
                reg_block_accumulator[active_row],
                reg_product,
            )

    def process_phase(*, high_nibble: bool) -> None:
        _load_tmu_word(reg_scale_pointer, reg_scale_word)
        add(reg_scale_pointer, reg_scale_pointer, 4)
        mov(reg_group_count, 2)
        with loop as group_loop:
            mov(reg_qh_pointer, reg_qh_base)
            accumulate_subblock(high_nibble=high_nibble)
            accumulate_subblock(high_nibble=high_nibble)
            add(reg_qh_shift, reg_qh_shift, 2)
            sub(reg_group_count, reg_group_count, 1, cond="pushz")
            group_loop.b(cond="na0")
            nop()
            nop()
            nop()

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
    nop(sig=ldunifrf(reg_two_bit_mask))
    nop(sig=ldunifrf(reg_ones))
    nop(sig=ldunifrf(reg_q6_stride))
    nop(sig=ldunifrf(reg_q8_stride))
    nop(sig=ldunifrf(reg_ql_half_bytes))

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
    for row in range(4):
        bxor(reg_accumulator[row], reg_accumulator[row], reg_accumulator[row])
    setnnmode_ss()

    with loop as block_loop:
        mov(reg_ql_pointer, reg_weight_pointer)
        add(reg_qh_base, reg_weight_pointer, reg_ql_half_bytes)
        add(reg_qh_base, reg_qh_base, reg_ql_half_bytes)
        add(reg_scale_pointer, reg_qh_base, reg_ql_half_bytes)
        add(reg_weight_pointer, reg_weight_pointer, reg_q6_stride)
        for row in range(4):
            _load_tmu_word(reg_activation_base[row], reg_activation_scale[row])
            add(reg_activation_data[row], reg_activation_base[row], 4)
            bxor(
                reg_block_accumulator[row],
                reg_block_accumulator[row],
                reg_block_accumulator[row],
            )
        # Two 128-value halves.  Each half uses low-nibble groups 0/1,
        # rewinds QL by 64 bytes, then uses high-nibble groups 2/3.
        mov(reg_column, 2)
        with loop as half_loop:
            bxor(reg_qh_shift, reg_qh_shift, reg_qh_shift)
            process_phase(high_nibble=False)
            sub(reg_ql_pointer, reg_ql_pointer, reg_ql_half_bytes)
            process_phase(high_nibble=True)
            shr(reg_temporary, reg_ql_half_bytes, 1)
            add(reg_qh_base, reg_qh_base, reg_temporary)
            sub(reg_column, reg_column, 1, cond="pushz")
            half_loop.b(cond="na0")
            nop()
            nop()
            nop()

        _load_tmu_word(reg_scale_pointer, reg_weight_scale)
        fmov(reg_weight_scale, reg_weight_scale.unpack("l"))
        for row in range(4):
            itof(reg_temporary, reg_block_accumulator[row])
            fmul(reg_temporary, reg_temporary, reg_weight_scale)
            fmul(reg_temporary, reg_temporary, reg_activation_scale[row])
            fadd(reg_accumulator[row], reg_accumulator[row], reg_temporary)
            add(reg_activation_base[row], reg_activation_base[row], reg_q8_stride)
        sub(reg_blocks, reg_blocks, 1, cond="pushz")
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


def supports_ggml_q6_k_q8_k_linear_m4(
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
    if rows != 4 or activation_bytes != Q8_K_BLOCK_BYTES or weight_bytes != Q6_K_BLOCK_BYTES:
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
                    code=driver.program(qpu_ggml_q6_k_q8_k_linear_m4),
                    uniforms=driver.alloc(15, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute_ggml_q6_k_q8_k_linear_m4(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 6 or not all(isinstance(argument, Tensor) for argument in args[:3]):
        raise KernelError(
            "GGML Q6_K linear expects "
            "(activation, weight, destination, weight_column_start, output_column_start, column_count)"
        )
    activation, weight, destination, weight_column_start, output_column_start, column_count = args
    if not all(
        isinstance(value, int)
        for value in (weight_column_start, output_column_start, column_count)
    ):
        raise KernelError("GGML Q6_K column bounds must be integers")
    if not supports_ggml_q6_k_q8_k_linear_m4(
        activation,
        weight,
        destination,
        weight_column_start,
        output_column_start,
        column_count,
        backend,
    ):
        raise KernelError("GGML Q6_K linear requires contiguous Q8_K/Q6_K M=4 blocks and aligned columns")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("GGML Q6_K linear requires PyVideoCore7Backend")
    expected_grid = (column_count // OUTPUT_TILE, 1, 1)
    if grid != expected_grid:
        raise KernelError(f"GGML Q6_K linear grid must be {expected_grid}, got {grid}")
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
        0x03030303,
        0x01010101,
        Q6_K_BLOCK_BYTES,
        Q8_K_BLOCK_BYTES,
        64,
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


GGML_Q6_K_Q8_K_LINEAR_M4_KERNEL = Kernel(
    "vc7.ggml_q6_k_q8_k_linear_m4",
    _execute_ggml_q6_k_q8_k_linear_m4,
)

__all__ = [
    "GGML_Q6_K_Q8_K_LINEAR_M4_KERNEL",
    "ggml_q6_k_q8_k_reference",
    "pack_ggml_q6_k_blocks",
    "pack_ggml_q8_k_blocks",
    "supports_ggml_q6_k_q8_k_linear_m4",
    "unpack_ggml_q6_k_blocks",
    "unpack_ggml_q8_k_blocks",
]
