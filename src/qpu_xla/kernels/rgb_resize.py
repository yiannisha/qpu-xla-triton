"""Prepared bilinear RGB resize, padding, and normalization kernel."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from qpu_xla.backend import Backend, PyVideoCore7Backend
from qpu_xla.errors import KernelError
from qpu_xla.kernel import Kernel
from qpu_xla.memory import Tensor
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu


@qpu
def qpu_rgb_resize_norm_fp32(asm: Assembly, *, num_qpus: int) -> None:
    """Evaluate four-tap bilinear metadata for 16 FP32 outputs per iteration."""
    if not 1 <= num_qpus <= 12:
        raise ValueError("RGB resize QPU count must be between 1 and 12")
    reg_iterations = rf0
    reg_source = rf1
    reg_metadata = rf2
    reg_destination = rf3
    reg_plane_stride = rf4
    reg_qpu = rf5
    reg_offset = rf6
    reg_stride = rf7
    reg_metadata_cursor = rf8
    reg_destination_cursor = rf9
    reg_temporary = rf10
    reg_output = rf11
    offsets = [rf12, rf13, rf14, rf15]
    values = [rf16, rf17, rf18, rf19]
    weights = [rf20, rf21, rf22, rf23]

    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_source))
    nop(sig=ldunifrf(reg_metadata))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_plane_stride))
    if num_qpus == 1:
        mov(reg_qpu, 0)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu, reg_offset, 0b1111)
        mov(reg_stride, num_qpus)
        shl(reg_stride, reg_stride, 6)
    shl(reg_offset, reg_qpu, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_metadata_cursor, reg_metadata, reg_offset)
    add(reg_destination_cursor, reg_destination, reg_offset)

    with loop as resize:
        mov(reg_metadata, reg_metadata_cursor)
        for offset_value in offsets:
            mov(tmua, reg_metadata, sig=thrsw).add(reg_metadata, reg_metadata, reg_plane_stride)
            nop()
            nop()
            nop(sig=ldtmu(offset_value))
        for offset_value, value in zip(offsets, values, strict=True):
            add(reg_temporary, reg_source, offset_value)
            mov(tmua, reg_temporary, sig=thrsw)
            nop()
            nop()
            nop(sig=ldtmu(value))
        for weight in weights:
            mov(tmua, reg_metadata, sig=thrsw).add(reg_metadata, reg_metadata, reg_plane_stride)
            nop()
            nop()
            nop(sig=ldtmu(weight))
        bxor(reg_output, reg_output, reg_output)
        for value, weight in zip(values, weights, strict=True):
            fmul(reg_temporary, value, weight)
            fadd(reg_output, reg_output, reg_temporary)
        fmul(reg_output, reg_output, 2.0)
        fsub(reg_output, reg_output, 1.0)
        mov(tmud, reg_output)
        sub(reg_iterations, reg_iterations, 1, cond="pushz")
        mov(tmua, reg_destination_cursor).add(reg_destination_cursor, reg_destination_cursor, reg_stride)
        add(reg_metadata_cursor, reg_metadata_cursor, reg_stride)
        tmuwt()
        resize.b(cond="na0")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
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
_PROGRAMS: WeakKeyDictionary[PyVideoCore7Backend, dict[int, _ProgramState]] = WeakKeyDictionary()


def supports_rgb_resize_norm_fp32(
    source: Tensor,
    metadata: Tensor,
    destination: Tensor,
    backend: Backend,
) -> bool:
    """Return whether tensors satisfy the prepared bilinear kernel contract."""
    if not isinstance(backend, PyVideoCore7Backend):
        return False
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        return False
    if metadata.dtype != np.dtype(np.uint32) or len(metadata.shape) != 2 or metadata.shape[0] != 8:
        return False
    if metadata.shape[1] < int(np.prod(destination.shape)) or destination.nbytes % 64:
        return False
    return all(tensor.numpy().flags.c_contiguous for tensor in (source, metadata, destination))


def _workgroup_count(vector_count: int) -> int:
    for count in range(min(12, vector_count), 0, -1):
        if vector_count % count == 0:
            return count
    return 1


def _program_state(backend: PyVideoCore7Backend, num_qpus: int) -> _ProgramState:
    with _STATE_LOCK:
        states = _PROGRAMS.setdefault(backend, {})
        state = states.get(num_qpus)
        if state is None:
            with backend.driver_session() as driver:
                state = _ProgramState(
                    code=driver.program(qpu_rgb_resize_norm_fp32, num_qpus=num_qpus),
                    uniforms=driver.alloc(5, dtype=np.uint32),
                )
            states[num_qpus] = state
        return state


def _execute_rgb_resize_norm_fp32(backend: Backend, args: tuple[Any, ...], grid: tuple[int, int, int]) -> None:
    if len(args) != 3 or not all(isinstance(argument, Tensor) for argument in args):
        raise KernelError("RGB resize expects (source, metadata, destination)")
    source, metadata, destination = args
    if not supports_rgb_resize_norm_fp32(source, metadata, destination, backend):
        raise KernelError("RGB resize requires contiguous FP32 tensors and eight metadata planes")
    if not isinstance(backend, PyVideoCore7Backend):
        raise KernelError("RGB resize requires PyVideoCore7Backend")
    vectors = destination.nbytes // 64
    expected = (_workgroup_count(vectors), 1, 1)
    if grid != expected:
        raise KernelError(f"RGB resize grid must be {expected}, got {grid}")
    state = _program_state(backend, grid[0])
    state.uniforms[:] = (
        vectors // grid[0],
        source.address,
        metadata.address,
        destination.address,
        metadata.numpy().strides[0],
    )
    with backend.driver_session() as driver:
        driver.execute(
            state.code,
            local_invocation=(16, 1, 1),
            uniforms=state.uniforms.addresses()[0],
            workgroup=grid,
            wgs_per_sg=grid[0],
            thread=grid[0],
        )


RGB_RESIZE_NORM_FP32_KERNEL = Kernel("vc7.rgb_resize_norm_fp32", _execute_rgb_resize_norm_fp32)

__all__ = ["RGB_RESIZE_NORM_FP32_KERNEL", "supports_rgb_resize_norm_fp32"]
