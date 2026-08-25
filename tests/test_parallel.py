# Copyright (c) 2025- Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time
from typing import Any, Literal

import hypothesis
import hypothesis.stateful
import hypothesis.strategies
import numpy as np

from videocore7.assembler import *
from videocore7.assembler import qpu
from videocore7.driver import Array, Driver


@qpu
def cost(asm: Assembly) -> None:
    mov(rf10, 8)
    shl(rf10, rf10, 8)
    shl(rf10, rf10, 8)
    with loop as l:  # noqa: E741
        sub(rf10, rf10, 1, cond="pushn")
        l.b(cond="anyna")
        nop()
        nop()
        nop()


@qpu
def qpu_serial(asm: Assembly, thread: int) -> None:
    nop(sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    nop(sig=ldunifrf(rf3))

    eidx(rf10)
    shl(rf10, rf10, 2)
    add(rf2, rf2, rf10)
    add(rf3, rf3, rf10)
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    for i in range(thread):
        mov(tmua, rf2, sig=thrsw).add(rf2, rf2, rf13)
        nop()
        nop()
        nop(sig=ldtmu(rf10))
        mov(tmud, rf10)
        mov(tmua, rf3, sig=thrsw).add(rf3, rf3, rf13)
        tmuwt()

    cost(asm)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


# This code requires 12 or 24 thread execution.
# If # of thread == 12, thread id = (tidx & 0b111100) >> 2.
# If # of thread == 24, thread id = (tidx & 0b111110) >> 1.
@qpu
def qpu_parallel(asm: Assembly, thread: Literal[12, 24]) -> None:
    tidx(rf10, sig=ldunifrf(rf0))
    shr(rf10, rf10, 3 - thread // 12)
    mov(rf11, 1)
    shl(rf11, rf11, 3 + thread // 12)
    sub(rf11, rf11, 1)
    band(rf31, rf10, rf11)

    # rf31 = thread_id = (tidx & 0b111100) >> 2

    nop(sig=ldunifrf(rf1))  # rf1 = unif[0,1]
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf31)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)

    # rf10 = thread_id * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))

    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 =  unif[th,2:18]
    bcastf(rf2, rf10)  # rf2 = unif[th,2] (x address)
    rotate(rf10, rf10, 1)
    bcastf(rf3, rf10)  # rf3 = unif[th,3] (y address)

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf2, rf12, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf32))

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf32)
    add(tmua, rf3, rf12)
    tmuwt()

    cost(asm)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_parallel() -> None:
    n_of_threads: list[Literal[12, 24]] = [12, 24]
    for thread in n_of_threads:
        with Driver() as drv:
            serial_code = drv.program(qpu_serial, thread)
            parallel_code = drv.program(qpu_parallel, thread)
            x: Array[np.uint32] = drv.alloc((thread, 16), dtype=np.uint32)
            ys: Array[np.uint32] = drv.alloc((thread, 16), dtype=np.uint32)
            yp: Array[np.uint32] = drv.alloc((thread, 16), dtype=np.uint32)
            unif: Array[np.uint32] = drv.alloc((thread, 4), dtype=np.uint32)

            x[:] = np.arange(thread * 16).reshape(x.shape)

            unif[:, 0] = unif.addresses()[:, 0]
            unif[:, 1] = unif.shape[1]
            unif[:, 2] = x.addresses()[:, 0]

            ys[:] = 0
            unif[:, 3] = ys.addresses()[:, 0]

            start = time.time()
            drv.execute(serial_code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0, 0])
            end = time.time()
            serial_cost = end - start

            yp[:] = 0
            unif[:, 3] = yp.addresses()[:, 0]

            start = time.time()
            drv.execute(
                parallel_code,
                local_invocation=(16, 1, 1),
                uniforms=unif.addresses()[0, 0],
                wgs_per_sg=thread,
                thread=thread,
            )
            end = time.time()
            parallel_cost = end - start

            assert np.all(x == ys)
            assert np.all(x == yp)
            assert parallel_cost < serial_cost * 2, f"{parallel_cost=}, {serial_cost=}"


# If remove `barrierid` in this code, `test_barrier` will fail.
@qpu
def qpu_barrier(asm: Assembly) -> None:
    tidx(rf10, sig=ldunifrf(rf0))  # rf0 = unif[0,0]
    shr(rf12, rf10, 2)
    band(rf11, rf10, 0b11)  # thread_id
    band(rf12, rf12, 0b1111)  # qpu_id
    shr(rf11, rf11, 1)
    shl(rf12, rf12, 1)
    add(rf31, rf11, rf12)  # rf31 = (qpu_id * 2) + (thread_id >> 1)

    nop(sig=ldunifrf(rf1))  # rf1 = unif[0,1]

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf31)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 =  unif[th,2:18]
    bcastf(rf2, rf10)  # rf2 = unif[th,2] (x address)
    rotate(rf10, rf10, 1)
    bcastf(rf3, rf10)  # rf3 = unif[th,3] (y address)

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf2, rf12, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))

    mov(rf11, rf31)
    shl(rf11, rf11, 8)
    L.loop
    sub(rf11, rf11, 1, cond="pushn")
    b(R.loop, cond="anyna")
    nop()
    nop()
    nop()

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf10)
    add(tmua, rf3, rf12)
    tmuwt()

    barrierid(syncb, sig=thrsw)

    add(rf32, rf31, 1)
    mov(rf31, 8)
    add(rf31, rf31, 8)
    add(rf31, rf31, 8)
    sub(null, rf32, rf31, cond="pushz")
    b(R.skip, cond="allna")
    nop()
    nop()
    nop()
    mov(rf32, 0)
    L.skip

    # rf32 = (rf31 + 1) mod 24

    # rf32 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf32)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 = unif[(th+1)%16,2:18]
    bcastf(rf4, rf10)  # rf4 = unif[(th+1)%16,2]
    rotate(rf10, rf10, 1)
    bcastf(rf5, rf10)  # rf5 = unif[(th+1)%16,3]

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf5, rf12, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf10)
    add(tmua, rf3, rf12)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_barrier() -> None:
    with Driver() as drv:
        thread = 24

        code = drv.program(qpu_barrier)
        x: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc((thread, 4), dtype=np.uint32)

        x[:] = np.random.randn(*x.shape)
        y[:] = -1

        unif[:, 0] = unif.addresses()[:, 0]
        unif[:, 1] = unif.shape[1]
        unif[:, 2] = x.addresses()[:, 0]
        unif[:, 3] = y.addresses()[:, 0]

        drv.execute(
            code,
            local_invocation=(16, 1, 1),
            uniforms=unif.addresses()[0, 0],
            wgs_per_sg=thread,
            thread=thread,
        )

        assert np.all(y == np.concatenate([x[1:], x[:1]]))


@qpu
def qpu_parallel_full(asm: Assembly) -> None:
    tidx(rf1, sig=ldunifrf(rf2))
    shl(rf1, rf1, 6)
    add(rf2, rf2, rf1, sig=thrsw)
    eidx(rf1)
    shl(rf1, rf1, 2)
    add(rf2, rf2, rf1)

    tidx(tmud)
    mov(tmua, rf2)
    tmuwt()

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


def test_parallel_full() -> None:
    """48 Threads."""
    with Driver() as drv:
        code = drv.program(qpu_parallel_full)
        dst: Array[np.uint32] = drv.alloc((48, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        dst[:] = 0xDEADBEEF

        unif[0] = dst.addresses()[0, 0]

        drv.execute(
            code,
            local_invocation=(4, 4, 3),
            uniforms=unif.addresses()[0],
            workgroup=(1, 1, 1),
            thread=48,
            threading=True,
        )

        assert np.all(dst == np.arange(48, dtype=np.uint32).reshape(48, 1))


@qpu
def qpu_barrier_memory_handoff_48(asm: Assembly) -> None:
    """Exercise two global TMU handoffs across a 48-thread supergroup."""
    reg_thread = rf0
    reg_base = rf1
    reg_lane_offset = rf2
    reg_row_offset = rf3
    reg_address = rf4
    reg_next_thread = rf5
    reg_value = rf6
    reg_last_thread = rf7
    reg_phase_stride = rf8
    reg_phase_two_offset = rf9
    tidx(reg_thread, sig=ldunifrf(reg_base))
    mov(reg_last_thread, 15)
    shl(reg_last_thread, reg_last_thread, 1)
    add(reg_last_thread, reg_last_thread, 15)
    add(reg_last_thread, reg_last_thread, 2)
    mov(reg_phase_stride, 3)
    shl(reg_phase_stride, reg_phase_stride, 10)
    shl(reg_phase_two_offset, reg_phase_stride, 1)
    eidx(reg_lane_offset)
    shl(reg_lane_offset, reg_lane_offset, 2)
    shl(reg_row_offset, reg_thread, 6)
    add(reg_address, reg_base, reg_row_offset)
    add(reg_address, reg_address, reg_lane_offset)

    mov(tmuc, -1)
    mov(tmud, reg_thread)
    mov(tmua, reg_address)
    tmuwt()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    add(reg_next_thread, reg_thread, 1)
    umin(reg_next_thread, reg_next_thread, reg_last_thread)
    shl(reg_row_offset, reg_next_thread, 6)
    add(reg_address, reg_base, reg_row_offset)
    add(reg_address, reg_address, reg_lane_offset)
    mov(tmua, reg_address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(reg_value))

    add(reg_address, reg_base, reg_phase_stride)
    shl(reg_row_offset, reg_thread, 6)
    add(reg_address, reg_address, reg_row_offset)
    add(reg_address, reg_address, reg_lane_offset)
    mov(tmud, reg_value)
    mov(tmua, reg_address)
    tmuwt()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    add(reg_next_thread, reg_thread, 1)
    umin(reg_next_thread, reg_next_thread, reg_last_thread)
    add(reg_address, reg_base, reg_phase_stride)
    shl(reg_row_offset, reg_next_thread, 6)
    add(reg_address, reg_address, reg_row_offset)
    add(reg_address, reg_address, reg_lane_offset)
    mov(tmua, reg_address, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(reg_value))

    add(reg_address, reg_base, reg_phase_two_offset)
    shl(reg_row_offset, reg_thread, 6)
    add(reg_address, reg_address, reg_row_offset)
    add(reg_address, reg_address, reg_lane_offset)
    mov(tmud, reg_value)
    mov(tmua, reg_address)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_barrier_memory_handoff_48() -> None:
    """All 48 persistent threads observe two completed global-memory phases."""
    with Driver() as drv:
        code = drv.program(qpu_barrier_memory_handoff_48)
        phases: Array[np.uint32] = drv.alloc((3, 48, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        phases[:] = 0xDEADBEEF
        unif[0] = phases.addresses()[0, 0, 0]

        drv.execute(
            code,
            local_invocation=(4, 4, 3),
            uniforms=unif.addresses()[0],
            workgroup=(1, 1, 1),
            thread=48,
            threading=True,
        )

        thread_ids = np.arange(48, dtype=np.uint32)
        expected_phase_0 = np.repeat(thread_ids[:, None], 16, axis=1)
        expected_phase_1 = np.repeat(np.minimum(thread_ids + 1, 47)[:, None], 16, axis=1)
        expected_phase_2 = np.repeat(np.minimum(thread_ids + 2, 47)[:, None], 16, axis=1)
        assert np.array_equal(phases[0], expected_phase_0)
        assert np.array_equal(phases[1], expected_phase_1)
        assert np.array_equal(phases[2], expected_phase_2)


@qpu
def qpu_uniform_stream_24(asm: Assembly) -> None:
    """Load a fixed-stride private uniform stream in 24-thread mode."""
    tidx(rf0, sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    shr(rf6, rf0, 2)
    band(rf6, rf6, 0b1111)
    band(rf0, rf0, 0b11)
    shr(rf0, rf0, 1)
    shl(rf6, rf6, 1)
    add(rf0, rf0, rf6)
    umul24(rf3, rf0, rf2)
    add(rf1, rf1, rf3)
    b(R.stream, cond="always").unif_addr(rf1)
    nop()
    nop()
    nop()

    L.stream
    nop(sig=ldunifrf(rf4))
    nop(sig=ldunifrf(rf5))
    eidx(rf6)
    shl(rf6, rf6, 2)
    add(rf4, rf4, rf6)
    mov(tmud, rf5)
    mov(tmua, rf4)
    tmuwt()

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


def test_uniform_stream_24() -> None:
    with Driver() as drv:
        code = drv.program(qpu_uniform_stream_24)
        output: Array[np.uint32] = drv.alloc((24, 16), dtype=np.uint32)
        streams: Array[np.uint32] = drv.alloc((24, 2), dtype=np.uint32)
        uniforms: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)
        output[:] = 0xDEADBEEF
        for thread in range(24):
            streams[thread] = (output.addresses()[thread, 0], thread + 100)
        uniforms[:] = (streams.addresses()[0, 0], streams.strides[0])

        drv.execute(
            code,
            local_invocation=(16, 1, 1),
            uniforms=uniforms.addresses()[0],
            wgs_per_sg=24,
            thread=24,
        )

        expected = np.repeat((np.arange(24, dtype=np.uint32) + 100)[:, None], 16, axis=1)
        assert np.array_equal(output, expected)
