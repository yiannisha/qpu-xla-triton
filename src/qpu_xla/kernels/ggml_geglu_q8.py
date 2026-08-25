"""Fused GGML GEGLU and native Q8_0 activation producer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
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
from videocore7.assembler import Assembly, qpu

Q8_0_BLOCK_ELEMENTS = 32
Q8_0_BLOCK_BYTES = 34


@lru_cache(maxsize=1)
def ggml_gelu_fp16_table() -> npt.NDArray[np.uint16]:
    """Build the FP16 lookup table used by GGML's exact F32 GEGLU path."""
    inputs = np.arange(1 << 16, dtype=np.uint16).view(np.float16).astype(np.float32)
    outputs = np.empty_like(inputs)
    for index, value in enumerate(inputs):
        number = float(value)
        outputs[index] = 0.5 * number * (1.0 + math.erf(number / math.sqrt(2.0)))
    return np.ascontiguousarray(outputs.astype(np.float16).view(np.uint16))


def ggml_geglu_q8_0_reference(
    gate: npt.NDArray[np.floating[Any]],
    up: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.uint8]:
    """Apply lookup-table GEGLU and quantize directly to native Q8_0 blocks."""
    gate = np.asarray(gate, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    if gate.shape != up.shape or gate.ndim != 2 or gate.shape[1] % Q8_0_BLOCK_ELEMENTS:
        raise ValueError("gate/up must be equal rank-two F32 arrays with K divisible by 32")
    table = ggml_gelu_fp16_table().view(np.float16)
    indices = gate.astype(np.float16).view(np.uint16)
    activated = table[indices].astype(np.float32) * up
    blocks = activated.reshape(gate.shape[0], -1, Q8_0_BLOCK_ELEMENTS)
    maximum = np.max(np.abs(blocks), axis=2)
    scales = maximum / np.float32(127.0)
    inverse = np.zeros_like(scales)
    np.divide(np.float32(1.0), scales, out=inverse, where=scales != 0.0)
    values = np.rint(blocks * inverse[:, :, None])
    values = np.clip(values, -127.0, 127.0).astype(np.int8)
    packed = np.empty((*scales.shape, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    packed[..., :2] = scales.astype("<f2").view(np.uint8).reshape(*scales.shape, 2)
    packed[..., 2:] = values.view(np.uint8)
    return np.ascontiguousarray(packed)


@qpu
def qpu_ggml_geglu_q8_0(asm: Assembly, *, split_output: bool = False) -> None:
    """Fuse exact table GEGLU with per-32-element Q8_0 quantization."""
    reg_block = rf0
    reg_row = rf1
    reg_blocks = rf2
    reg_gate = rf3
    reg_up = rf4
    reg_destination = rf5
    reg_gelu_table = rf6
    reg_inverse_127 = rf7
    reg_minimum_maximum = rf8
    reg_lane = rf9
    reg_group = rf10
    reg_offset = rf11
    reg_row_stride = rf12
    reg_pointer = rf13
    reg_table_pointer = rf14
    reg_maximum = rf15
    reg_rotated = rf16
    reg_scale = rf17
    reg_inverse_scale = rf18
    reg_pair_low = rf19
    reg_pair_high = rf20
    reg_packed = rf21
    reg_scale_packed = rf22
    gate_values = [rf23, rf24, rf25, rf26]
    up_values = [rf27, rf28, rf29, rf30]
    table_values = [rf31, rf32, rf33, rf34]
    outputs = [rf35, rf36, rf37, rf38]
    codes = [rf39, rf40, rf41, rf42]
    temporaries = [rf43, rf44, rf45, rf46]
    reg_q_destination = rf47

    mov(reg_block, rf3.unpack("ul"))
    mov(reg_row, rf3.unpack("uh"))
    nop(sig=ldunifrf(reg_blocks))
    nop(sig=ldunifrf(reg_gate))
    nop(sig=ldunifrf(reg_up))
    nop(sig=ldunifrf(reg_destination))
    if split_output:
        nop(sig=ldunifrf(reg_q_destination))
    nop(sig=ldunifrf(reg_gelu_table))
    nop(sig=ldunifrf(reg_inverse_127))
    nop(sig=ldunifrf(reg_minimum_maximum))

    eidx(reg_lane)
    band(reg_group, reg_lane, 7)
    shl(reg_group, reg_group, 4)
    shl(reg_offset, reg_block, 7)
    add(reg_offset, reg_offset, reg_group)
    shl(reg_row_stride, reg_blocks, 7)
    umul24(reg_pointer, reg_row, reg_row_stride)
    add(reg_offset, reg_offset, reg_pointer)
    add(reg_gate, reg_gate, reg_offset)
    add(reg_up, reg_up, reg_offset)

    bnot(tmuc, 3)
    mov(tmua, reg_gate, sig=thrsw)
    nop()
    nop()
    for value in gate_values:
        nop(sig=ldtmu(value))
    bnot(tmuc, 3)
    mov(tmua, reg_up, sig=thrsw)
    nop()
    nop()
    for value in up_values:
        nop(sig=ldtmu(value))

    for table_pointer, gate_value in zip(temporaries, gate_values, strict=True):
        fmov(table_pointer.pack("l"), gate_value)
        shl(table_pointer, table_pointer, 8)
        shl(table_pointer, table_pointer, 8)
        shr(table_pointer, table_pointer, 8)
        shr(table_pointer, table_pointer, 8)
        shl(table_pointer, table_pointer, 1)
        add(table_pointer, table_pointer, reg_gelu_table)
    for table_pointer, table_value in zip(temporaries, table_values, strict=True):
        mov(tmua, table_pointer, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(table_value))
        fmov(table_value, table_value.unpack("l"))
    for output, table_value, up_value in zip(
        outputs, table_values, up_values, strict=True
    ):
        fmul(output, table_value, up_value)

    bxor(reg_maximum, reg_maximum, reg_maximum)
    for output in outputs:
        fmax(reg_maximum, reg_maximum, output.unpack("abs"))
    for distance in (8, 4, 2, 1):
        rotate(reg_rotated, reg_maximum, distance)
        fmax(reg_maximum, reg_maximum, reg_rotated)
    fmax(reg_maximum, reg_maximum, reg_minimum_maximum)
    fmul(reg_scale, reg_maximum, reg_inverse_127)
    recip(reg_inverse_scale, reg_scale)
    nop()
    nop()
    for code, output in zip(codes, outputs, strict=True):
        fmul(code, output, reg_inverse_scale)
        ftoin(code, code)

    mov(reg_table_pointer, 1)
    shl(reg_table_pointer, reg_table_pointer, 8)
    sub(reg_table_pointer, reg_table_pointer, 1)
    band(reg_pair_low, codes[0], reg_table_pointer)
    band(reg_offset, codes[1], reg_table_pointer)
    shl(reg_offset, reg_offset, 8)
    shl(reg_offset, reg_offset, 8)
    bor(reg_pair_low, reg_pair_low, reg_offset)
    band(reg_pair_high, codes[2], reg_table_pointer)
    band(reg_offset, codes[3], reg_table_pointer)
    shl(reg_offset, reg_offset, 8)
    shl(reg_offset, reg_offset, 8)
    bor(reg_pair_high, reg_pair_high, reg_offset)
    v8pack(reg_packed, reg_pair_low, reg_pair_high)

    umul24(reg_pointer, reg_row, reg_blocks)
    add(reg_pointer, reg_pointer, reg_block)
    if split_output:
        shl(reg_offset, reg_pointer, 2)
    else:
        shl(reg_offset, reg_pointer, 5)
        shl(reg_row_stride, reg_pointer, 1)
        add(reg_offset, reg_offset, reg_row_stride)
    add(reg_pointer, reg_destination, reg_offset)
    bxor(reg_scale_packed, reg_scale_packed, reg_scale_packed)
    fmov(reg_scale_packed.pack("l"), reg_scale)
    sub(null, reg_lane, 1, cond="pushn")
    mov(tmud, reg_scale_packed, cond="ifa")
    mov(tmua, reg_pointer, cond="ifa")
    tmuwt()

    if split_output:
        shl(reg_row_stride, reg_blocks, 5)
        umul24(reg_pointer, reg_row, reg_row_stride)
        shl(reg_offset, reg_block, 5)
        add(reg_pointer, reg_pointer, reg_offset)
        add(reg_pointer, reg_q_destination, reg_pointer)
    else:
        add(reg_pointer, reg_pointer, 2)
    band(reg_group, reg_lane, 7)
    shl(reg_group, reg_group, 2)
    add(reg_pointer, reg_pointer, reg_group)
    sub(null, reg_lane, 8, cond="pushn")
    mov(tmud, reg_packed, cond="ifa")
    mov(tmua, reg_pointer, cond="ifa")
    tmuwt()

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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[bool, _ProgramState]] = (
    WeakKeyDictionary()
)


def _program_state(backend: PyVideoCore7Backend, split_output: bool = False) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(split_output)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(
                        qpu_ggml_geglu_q8_0,
                        split_output=split_output,
                    ),
                    uniforms=driver.alloc(8 if split_output else 7, dtype=np.uint32),
                )
            states[split_output] = state
        return state


def supports_ggml_geglu_q8_0(
    gate: Tensor,
    up: Tensor,
    destination: Tensor,
    gelu_table: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors match the fused native Q8_0 producer contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if gate.dtype != np.dtype(np.float32) or up.dtype != np.dtype(np.float32):
        return False
    if destination.dtype != np.dtype(np.uint8) or gelu_table.dtype != np.dtype(np.uint16):
        return False
    if len(gate.shape) != 2 or up.shape != gate.shape or gate.shape[1] % Q8_0_BLOCK_ELEMENTS:
        return False
    expected = (gate.shape[0], gate.shape[1] // Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES)
    return bool(
        destination.shape == expected
        and gelu_table.shape == (1 << 16,)
        and all(tensor.numpy().flags.c_contiguous for tensor in (gate, up, destination, gelu_table))
    )


def _execute_ggml_geglu_q8_0(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 4 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("fused GEGLU-Q8_0 expects (gate, up, destination, gelu_table)")
    gate, up, destination, gelu_table = args
    if not supports_ggml_geglu_q8_0(gate, up, destination, gelu_table, backend):
        raise KernelError("fused GEGLU-Q8_0 requires contiguous MxK F32 and Mx(K/32)x34 output")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("fused GEGLU-Q8_0 requires PyVideoCore7Backend")
    expected_grid = (destination.shape[1], destination.shape[0], 1)
    if grid != expected_grid:
        raise KernelError(f"fused GEGLU-Q8_0 grid must be {expected_grid}, got {grid}")
    state = _program_state(backend)
    state.uniforms[:] = (
        destination.shape[1],
        gate.address,
        up.address,
        destination.address,
        gelu_table.address,
        np.asarray(1.0 / 127.0, dtype=np.float32).view(np.uint32),
        np.asarray(2.0**-24, dtype=np.float32).view(np.uint32),
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=grid[0] * grid[1],
        )


GGML_GEGLU_Q8_0_KERNEL = Kernel("vc7.ggml_geglu_q8_0", _execute_ggml_geglu_q8_0)


def supports_ggml_geglu_q8_0_split(
    gate: Tensor,
    up: Tensor,
    scales: Tensor,
    values: Tensor,
    gelu_table: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors match the direct tiled-down producer layout."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if gate.dtype != np.dtype(np.float32) or up.dtype != np.dtype(np.float32):
        return False
    if scales.dtype != np.dtype(np.uint32) or values.dtype != np.dtype(np.uint8):
        return False
    if gelu_table.dtype != np.dtype(np.uint16) or len(gate.shape) != 2 or up.shape != gate.shape:
        return False
    rows, columns = gate.shape
    return bool(
        columns > 0
        and columns % Q8_0_BLOCK_ELEMENTS == 0
        and scales.shape == (rows, columns // Q8_0_BLOCK_ELEMENTS)
        and values.shape == gate.shape
        and gelu_table.shape == (1 << 16,)
        and all(tensor.numpy().flags.c_contiguous for tensor in (gate, up, scales, values, gelu_table))
    )


def _execute_ggml_geglu_q8_0_split(
    backend: Backend,
    args: tuple[Any, ...],
    grid: tuple[int, int, int],
) -> None:
    if len(args) != 5 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("split GEGLU-Q8_0 expects gate, up, scales, values, and table")
    gate, up, scales, values, gelu_table = args
    if not supports_ggml_geglu_q8_0_split(gate, up, scales, values, gelu_table, backend):
        raise KernelError("split GEGLU-Q8_0 requires contiguous MxK tensors")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("split GEGLU-Q8_0 requires PyVideoCore7Backend")
    expected_grid = (scales.shape[1], scales.shape[0], 1)
    if grid != expected_grid:
        raise KernelError(f"split GEGLU-Q8_0 grid must be {expected_grid}, got {grid}")
    state = _program_state(backend, True)
    state.uniforms[:] = (
        scales.shape[1],
        gate.address,
        up.address,
        scales.address,
        values.address,
        gelu_table.address,
        np.asarray(1.0 / 127.0, dtype=np.float32).view(np.uint32),
        np.asarray(2.0**-24, dtype=np.float32).view(np.uint32),
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=24,
            thread=grid[0] * grid[1],
        )


GGML_GEGLU_Q8_0_SPLIT_KERNEL = Kernel(
    "vc7.ggml_geglu_q8_0_split",
    _execute_ggml_geglu_q8_0_split,
)


__all__ = [
    "GGML_GEGLU_Q8_0_KERNEL",
    "GGML_GEGLU_Q8_0_SPLIT_KERNEL",
    "ggml_gelu_fp16_table",
    "ggml_geglu_q8_0_reference",
    "qpu_ggml_geglu_q8_0",
    "supports_ggml_geglu_q8_0",
    "supports_ggml_geglu_q8_0_split",
]
