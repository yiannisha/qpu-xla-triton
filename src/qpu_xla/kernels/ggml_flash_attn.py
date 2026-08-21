"""Independent reference semantics for GGML ``FLASH_ATTN_EXT`` fixtures.

The logical axis order in this module is the native GGML ``ne`` order rather
than NumPy's usual row-major convention.  Arrays therefore use
``[embedding, row, head, batch]`` for Q/K/V and
``[kv_row, query_row, head, batch]`` for the optional mask.
"""

from __future__ import annotations

import math
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

GEMMA_HEAD_DIM = 256


def _require_rank_four(name: str, value: npt.NDArray[Any]) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must have four GGML dimensions")


def ggml_flash_attn_ext_reference(
    query: npt.NDArray[np.floating[Any]],
    key: npt.NDArray[np.floating[Any]],
    value: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]] | None,
    *,
    scale: float,
    max_bias: float,
    logit_softcap: float,
    sinks: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.float32]:
    """Evaluate the public GGML flash-attention contract in FP32.

    This deliberately does not call llama.cpp or reuse a staged project
    attention implementation.  It is a fixture oracle for validating a future
    fused streaming QPU kernel.  Small accumulation-order differences from the
    optimized CPU implementation are expected and must be measured explicitly.
    """
    query = np.asarray(query)
    key = np.asarray(key)
    value = np.asarray(value)
    _require_rank_four("query", query)
    _require_rank_four("key", key)
    _require_rank_four("value", value)
    dk, query_rows, query_heads, batches = map(int, query.shape)
    key_dk, kv_rows, key_heads, key_batches = map(int, key.shape)
    dv, value_rows, value_heads, value_batches = map(int, value.shape)
    if dk != key_dk or kv_rows != value_rows:
        raise ValueError("Q/K embedding or K/V row dimensions differ")
    if query_heads % key_heads or query_heads % value_heads:
        raise ValueError("query heads are not divisible by K/V heads")
    if batches % key_batches or batches % value_batches:
        raise ValueError("query batches are not divisible by K/V batches")
    if not all(math.isfinite(number) for number in (scale, max_bias, logit_softcap)):
        raise ValueError("attention parameters must be finite")
    if max_bias < 0.0:
        raise ValueError("max_bias must be nonnegative")

    mask_array: npt.NDArray[Any] | None = None
    if mask is not None:
        mask_array = np.asarray(mask)
        _require_rank_four("mask", mask_array)
        if int(mask_array.shape[0]) != kv_rows or int(mask_array.shape[1]) != query_rows:
            raise ValueError("mask row dimensions do not match Q/K")
        if query_heads % int(mask_array.shape[2]) or batches % int(mask_array.shape[3]):
            raise ValueError("mask head/batch dimensions do not broadcast to Q")
    elif max_bias > 0.0:
        raise ValueError("positive max_bias requires a mask")

    sink_array: npt.NDArray[np.float32] | None = None
    if sinks is not None:
        sink_array = np.asarray(sinks, dtype=np.float32).reshape(-1)
        if sink_array.size != query_heads:
            raise ValueError("attention sinks must contain one value per query head")

    result = np.empty((dv, query_heads, query_rows, batches), dtype=np.float32)
    key_ratio = query_heads // key_heads
    value_ratio = query_heads // value_heads
    key_batch_ratio = batches // key_batches
    value_batch_ratio = batches // value_batches
    head_power = 1 << int(math.floor(math.log2(query_heads)))
    m0 = math.pow(2.0, -max_bias / head_power)
    m1 = math.pow(2.0, -(max_bias / 2.0) / head_power)
    effective_scale = scale / logit_softcap if logit_softcap != 0.0 else scale

    for batch in range(batches):
        key_batch = batch // key_batch_ratio
        value_batch = batch // value_batch_ratio
        for head in range(query_heads):
            key_head = head // key_ratio
            value_head = head // value_ratio
            if max_bias > 0.0:
                slope = (
                    math.pow(m0, head + 1)
                    if head < head_power
                    else math.pow(m1, 2 * (head - head_power) + 1)
                )
            else:
                slope = 1.0
            keys = np.asarray(key[:, :, key_head, key_batch], dtype=np.float32)
            values = np.asarray(value[:, :, value_head, value_batch], dtype=np.float32)
            for row in range(query_rows):
                q = np.asarray(query[:, row, head, batch], dtype=np.float32)
                scores = np.asarray(keys.T @ q, dtype=np.float32)
                scores *= np.float32(effective_scale)
                if logit_softcap != 0.0:
                    scores = np.tanh(scores).astype(np.float32, copy=False)
                    scores *= np.float32(logit_softcap)
                if mask_array is not None:
                    mask_head = head % int(mask_array.shape[2])
                    mask_batch = batch % int(mask_array.shape[3])
                    scores += (
                        np.asarray(mask_array[:, row, mask_head, mask_batch], dtype=np.float32)
                        * np.float32(slope)
                    )

                finite = np.isfinite(scores)
                if np.any(finite):
                    maximum = np.max(scores[finite])
                    weights = np.zeros(kv_rows, dtype=np.float32)
                    weights[finite] = np.exp(scores[finite] - maximum).astype(
                        np.float32, copy=False
                    )
                    denominator = np.sum(weights, dtype=np.float32)
                    if sink_array is not None:
                        sink = sink_array[head]
                        if sink > maximum:
                            weights *= np.exp(maximum - sink, dtype=np.float32)
                            denominator = denominator * np.exp(
                                maximum - sink, dtype=np.float32
                            ) + np.float32(1.0)
                        else:
                            denominator += np.exp(sink - maximum, dtype=np.float32)
                    result[:, head, row, batch] = (
                        values @ weights / denominator
                        if denominator != 0.0
                        else np.float32(0.0)
                    )
                elif sink_array is not None:
                    result[:, head, row, batch] = np.float32(0.0)
                else:
                    result[:, head, row, batch] = np.float32(0.0)
    return result


def _load_tmu_word(address: Register, destination: Register) -> None:
    mov(tmua, address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(destination))


@qpu
def qpu_ggml_gemma_flash_attn_f16_m1(asm: Assembly) -> None:
    """Fused Gemma M=1 GQA for one F16 KV head and 256-wide heads.

    One workgroup owns one query head.  Its 16 SIMD lanes jointly reduce QK
    while retaining all 256 online-softmax value accumulators in registers, so
    scores are never materialized.  The first prototype intentionally accepts
    only the exact no-ALiBi, no-softcap, no-sinks graph record.
    """
    query_values = [rf[index] for index in range(16)]
    accumulators = [rf[index] for index in range(16, 32)]
    reg_pair_count = rf32
    reg_key_pointer = rf33
    reg_key_stride = rf34
    reg_value_pointer = rf35
    reg_value_stride = rf36
    reg_mask_pointer = rf37
    reg_output_base = rf38
    reg_scale = rf39
    reg_log2_e = rf40
    reg_negative_inf = rf41
    reg_even_lane_offset = rf42
    reg_temporary = rf43
    reg_mask_word = rf44
    reg_key_word = rf45
    reg_value_word = rf46
    reg_dot = rf47
    reg_product = rf48
    reg_rotated = rf49
    reg_score = rf50
    reg_maximum = rf51
    reg_new_maximum = rf52
    reg_old_scale = rf53
    reg_value_scale = rf54
    reg_sum = rf55
    reg_inverse_sum = rf56
    reg_count = rf57
    reg_even = rf58
    reg_odd = rf59
    reg_value_even = rf60
    reg_value_odd = rf61
    reg_query_pointer = rf62
    reg_output_pointer = rf63

    mov(reg_temporary, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_pair_count))
    nop(sig=ldunifrf(reg_query_pointer))
    nop(sig=ldunifrf(reg_count))  # query-head stride
    nop(sig=ldunifrf(reg_key_pointer))
    nop(sig=ldunifrf(reg_key_stride))
    nop(sig=ldunifrf(reg_value_pointer))
    nop(sig=ldunifrf(reg_value_stride))
    nop(sig=ldunifrf(reg_mask_pointer))
    nop(sig=ldunifrf(reg_output_base))
    nop(sig=ldunifrf(reg_output_pointer))  # output-head stride
    nop(sig=ldunifrf(reg_scale))
    nop(sig=ldunifrf(reg_log2_e))
    nop(sig=ldunifrf(reg_negative_inf))

    umul24(reg_count, reg_temporary, reg_count)
    add(reg_query_pointer, reg_query_pointer, reg_count)
    umul24(reg_output_pointer, reg_temporary, reg_output_pointer)
    add(reg_output_base, reg_output_base, reg_output_pointer)
    eidx(reg_even_lane_offset)
    shl(reg_even_lane_offset, reg_even_lane_offset, 3)
    add(reg_query_pointer, reg_query_pointer, reg_even_lane_offset)
    add(reg_output_base, reg_output_base, reg_even_lane_offset)
    mov(reg_count, 1)
    shl(reg_count, reg_count, 7)

    for pair in range(8):
        _load_tmu_word(reg_query_pointer, query_values[2 * pair])
        add(reg_temporary, reg_query_pointer, 4)
        _load_tmu_word(reg_temporary, query_values[2 * pair + 1])
        add(reg_query_pointer, reg_query_pointer, reg_count)

    for accumulator in accumulators:
        bxor(accumulator, accumulator, accumulator)
    bxor(reg_sum, reg_sum, reg_sum)
    mov(reg_maximum, reg_negative_inf)
    shr(reg_even_lane_offset, reg_even_lane_offset, 1)
    mov(reg_count, reg_pair_count)

    def process_key(unpack: str, label: str) -> None:
        fmov(reg_score, reg_mask_word.unpack(unpack))
        fcmp(null, reg_score, reg_negative_inf, cond="pushz")
        with namespace(label):
            b(R.skip, cond="alla")
            nop()
            nop()
            nop()

            bxor(reg_dot, reg_dot, reg_dot)
            mov(reg_rotated, 1)
            shl(reg_rotated, reg_rotated, 6)
            add(reg_temporary, reg_key_pointer, reg_even_lane_offset)
            for pair in range(8):
                _load_tmu_word(reg_temporary, reg_key_word)
                add(reg_temporary, reg_temporary, reg_rotated)
                fmov(reg_even, reg_key_word.unpack("l"))
                fmov(reg_odd, reg_key_word.unpack("h"))
                fmul(reg_product, reg_even, query_values[2 * pair])
                fadd(reg_dot, reg_dot, reg_product)
                fmul(reg_product, reg_odd, query_values[2 * pair + 1])
                fadd(reg_dot, reg_dot, reg_product)
            for distance in (8, 4, 2, 1):
                rotate(reg_rotated, reg_dot, distance)
                fadd(reg_dot, reg_dot, reg_rotated)
            fmul(reg_dot, reg_dot, reg_scale)
            fadd(reg_score, reg_score, reg_dot)
            fmax(reg_new_maximum, reg_maximum, reg_score)
            fsub(reg_old_scale, reg_maximum, reg_new_maximum)
            fmul(reg_old_scale, reg_old_scale, reg_log2_e)
            exp(reg_old_scale, reg_old_scale)
            fsub(reg_value_scale, reg_score, reg_new_maximum)
            fmul(reg_value_scale, reg_value_scale, reg_log2_e)
            exp(reg_value_scale, reg_value_scale)
            fmul(reg_sum, reg_sum, reg_old_scale)
            fadd(reg_sum, reg_sum, reg_value_scale)
            for accumulator in accumulators:
                fmul(accumulator, accumulator, reg_old_scale)
            mov(reg_rotated, 1)
            shl(reg_rotated, reg_rotated, 6)
            add(reg_temporary, reg_value_pointer, reg_even_lane_offset)
            for pair in range(8):
                _load_tmu_word(reg_temporary, reg_value_word)
                add(reg_temporary, reg_temporary, reg_rotated)
                fmov(reg_value_even, reg_value_word.unpack("l"))
                fmov(reg_value_odd, reg_value_word.unpack("h"))
                fmul(reg_product, reg_value_even, reg_value_scale)
                fadd(accumulators[2 * pair], accumulators[2 * pair], reg_product)
                fmul(reg_product, reg_value_odd, reg_value_scale)
                fadd(accumulators[2 * pair + 1], accumulators[2 * pair + 1], reg_product)
            mov(reg_maximum, reg_new_maximum)
            L.skip
        add(reg_key_pointer, reg_key_pointer, reg_key_stride)
        add(reg_value_pointer, reg_value_pointer, reg_value_stride)

    with loop as kv_loop:
        _load_tmu_word(reg_mask_pointer, reg_mask_word)
        add(reg_mask_pointer, reg_mask_pointer, 4)
        process_key("l", "low")
        process_key("h", "high")
        sub(reg_count, reg_count, 1, cond="pushz")
        kv_loop.b(cond="na0")
        nop()
        nop()
        nop()

    recip(reg_inverse_sum, reg_sum)
    mov(reg_output_pointer, reg_output_base)
    mov(reg_count, 1)
    shl(reg_count, reg_count, 7)
    for pair in range(8):
        fmul(accumulators[2 * pair], accumulators[2 * pair], reg_inverse_sum)
        mov(tmud, accumulators[2 * pair])
        mov(tmua, reg_output_pointer)
        tmuwt()
        add(reg_temporary, reg_output_pointer, 4)
        fmul(accumulators[2 * pair + 1], accumulators[2 * pair + 1], reg_inverse_sum)
        mov(tmud, accumulators[2 * pair + 1])
        mov(tmua, reg_temporary)
        tmuwt()
        add(reg_output_pointer, reg_output_pointer, reg_count)

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


def supports_ggml_gemma_flash_attn_f16_m1(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors match the initial exact Gemma attention record."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if query.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        return False
    if any(tensor.dtype != np.dtype(np.float16) for tensor in (key, value, mask)):
        return False
    if len(query.shape) != 2 or len(key.shape) != 2 or len(value.shape) != 2:
        return False
    if len(mask.shape) != 1 or len(destination.shape) != 2:
        return False
    heads, head_dim = query.shape
    kv_rows, key_dim = key.shape
    if head_dim != GEMMA_HEAD_DIM or key_dim != head_dim:
        return False
    if value.shape != (kv_rows, GEMMA_HEAD_DIM) or mask.shape != (kv_rows,):
        return False
    if destination.shape != query.shape or heads <= 0 or heads > 12 or kv_rows <= 0:
        return False
    if kv_rows % 2:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (query, key, value, mask, destination))


def _program_state(backend: PyVideoCore7Backend) -> _ProgramState:
    with _STATE_LOCK:
        state = _PROGRAMS.get(backend)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_ggml_gemma_flash_attn_f16_m1),
                    uniforms=driver.alloc(13, dtype=np.uint32),
                )
            _PROGRAMS[backend] = state
        return state


def _execute_ggml_gemma_flash_attn_f16_m1(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 6 or not all(isinstance(argument, Tensor) for argument in args[:5]):
        raise KernelError("Gemma flash attention expects (query, key, value, mask, destination, scale)")
    query, key, value, mask, destination, scale = args
    if not isinstance(scale, float | np.floating) or not math.isfinite(float(scale)):
        raise KernelError("Gemma flash attention scale must be a finite float")
    if not supports_ggml_gemma_flash_attn_f16_m1(
        query, key, value, mask, destination, backend
    ):
        raise KernelError("Gemma flash attention requires contiguous M=1 F32/F16 tensors")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("Gemma flash attention requires PyVideoCore7Backend")
    expected_grid = (query.shape[0], 1, 1)
    if grid != expected_grid:
        raise KernelError(f"Gemma flash attention grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    state.uniforms[:] = (
        key.shape[0] // 2,
        query.address,
        query.numpy().strides[0],
        key.address,
        key.numpy().strides[0],
        value.address,
        value.numpy().strides[0],
        mask.address,
        destination.address,
        destination.numpy().strides[0],
        np.asarray(scale, dtype=np.float32).view(np.uint32),
        np.asarray(np.log2(np.e), dtype=np.float32).view(np.uint32),
        np.asarray(-np.inf, dtype=np.float32).view(np.uint32),
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=48,
            thread=query.shape[0],
        )


GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL = Kernel(
    "vc7.ggml_gemma_flash_attn_f16_m1", _execute_ggml_gemma_flash_attn_f16_m1
)


__all__ = [
    "GEMMA_HEAD_DIM",
    "GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL",
    "ggml_flash_attn_ext_reference",
    "supports_ggml_gemma_flash_attn_f16_m1",
]
