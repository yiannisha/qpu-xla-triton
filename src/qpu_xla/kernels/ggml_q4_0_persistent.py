"""Experimental persistent-stream tiled GGML Q4_0 by Q8_0 kernel."""

from __future__ import annotations

from collections import deque

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu


@qpu
def qpu_ggml_q4_0_q8_0_tiled_gemm_persistent(
    asm: Assembly,
    *,
    activation_scale_word: bool = False,
) -> None:
    """Run exact 16x16 tiles from 24 persistent per-thread task streams."""
    reg_stream_base = rf61
    reg_task_count = rf62
    reg_stream_stride = rf63
    reg_phase_count = rf59

    # Two hardware threads per QPU form one dense 0..23 persistent index.
    tidx(rf0, sig=ldunifrf(reg_stream_base))
    nop(sig=ldunifrf(reg_stream_stride))
    shr(rf1, rf0, 2)
    band(rf1, rf1, 0b1111)
    band(rf0, rf0, 0b11)
    shr(rf0, rf0, 1)
    shl(rf1, rf1, 1)
    add(rf0, rf0, rf1)
    umul24(rf1, rf0, reg_stream_stride)
    add(reg_stream_base, reg_stream_base, rf1)

    b(R.stream_ready, cond="always").unif_addr(reg_stream_base)
    nop()
    nop()
    nop()

    L.stream_ready
    nop(sig=ldunifrf(reg_phase_count))

    L.phase_loop
    b(R.phase, cond="always", set_link=True)
    nop()
    nop()
    nop()

    # Every persistent thread finishes the projection before the next phase.
    barrierid(syncb, sig=thrsw)
    nop()
    nop()
    sub(reg_phase_count, reg_phase_count, 1, cond="pushz")
    b(R.phase_loop, cond="anyna")
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

    L.phase
    nop(sig=ldunifrf(reg_task_count))

    L.task_loop
    nop(sig=ldunifrf(rf3))

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
    umul24(
        rf3,
        reg_tile_i,
        reg_activation_scale_stride,
        sig=ldunifrf(reg_activation_scale_pointer),
    )
    shl(rf3, rf3, 4)
    add(reg_activation_scale_pointer, reg_activation_scale_pointer, rf3)
    eidx(rf3)
    umul24(rf3, rf3, reg_activation_scale_stride)
    add(reg_activation_scale_pointer, reg_activation_scale_pointer, rf3)

    nop(sig=ldunifrf(reg_weight_scale_block_stride))
    nop(sig=ldunifrf(reg_weight_scale_pointer))
    shl(rf3, reg_tile_j, 5)
    add(reg_weight_scale_pointer, reg_weight_scale_pointer, rf3)
    eidx(rf3)
    shl(rf3, rf3, 1)
    add(reg_weight_scale_pointer, reg_weight_scale_pointer, rf3)

    for index in range(8):
        bxor(
            reg_float_accumulator[index],
            reg_float_accumulator[index],
            reg_float_accumulator[index],
        ).sub(
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
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)
        sub(reg_a_base, reg_a_base, -16)

        broadcast_order = deque(reg_b * 16)

        def broadcast_next() -> None:
            register = broadcast_order.popleft()
            rotate(register, register, 1).mov(rep, register)

        broadcast_next()
        v8dot(rf1, rf0, reg_a[0])
        for index in range(15):
            broadcast_next()
            add(
                reg_integer_accumulator[index],
                reg_integer_accumulator[index],
                rf1,
            ).v8dot(rf1, rf0, reg_a[0])
        broadcast_next()
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1).v8dot(rf1, rf0, reg_a[1])
        for index in range(15):
            broadcast_next()
            add(
                reg_integer_accumulator[index],
                reg_integer_accumulator[index],
                rf1,
            ).v8dot(rf1, rf0, reg_a[1])
        broadcast_next()
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1).v8dot(rf1, rf0, reg_a[2])
        for index in range(15):
            broadcast_next()
            add(
                reg_integer_accumulator[index],
                reg_integer_accumulator[index],
                rf1,
            ).v8dot(rf1, rf0, reg_a[2])
        broadcast_next()
        add(reg_integer_accumulator[15], reg_integer_accumulator[15], rf1).v8dot(rf1, rf0, reg_a[3])
        for index in range(8):
            broadcast_next()
            add(
                reg_integer_accumulator[index],
                reg_integer_accumulator[index],
                rf1,
            ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[8],
            reg_integer_accumulator[8],
            rf1,
            sig=ldtmu(reg_a[0]),
        ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[9],
            reg_integer_accumulator[9],
            rf1,
            sig=ldtmu(reg_a[1]),
        ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[10],
            reg_integer_accumulator[10],
            rf1,
            sig=ldtmu(reg_a[2]),
        ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[11],
            reg_integer_accumulator[11],
            rf1,
            sig=ldtmu(rf2),
        ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[12],
            reg_integer_accumulator[12],
            rf1,
            sig=ldtmu(reg_b[0]),
        ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[13],
            reg_integer_accumulator[13],
            rf1,
            sig=ldtmu(reg_b[1]),
        ).v8dot(rf1, rf0, reg_a[3])
        broadcast_next()
        add(
            reg_integer_accumulator[14],
            reg_integer_accumulator[14],
            rf1,
            sig=ldtmu(reg_b[2]),
        ).v8dot(rf1, rf0, reg_a[3])
        add(
            reg_integer_accumulator[15],
            reg_integer_accumulator[15],
            rf1,
            sig=ldtmu(reg_b[3]),
        ).mov(reg_a[3], rf2)

    with loop as block_loop:
        for index in range(8):
            bxor(
                reg_integer_accumulator[index],
                reg_integer_accumulator[index],
                reg_integer_accumulator[index],
            ).sub(
                reg_integer_accumulator[index + 8],
                reg_integer_accumulator[index + 8],
                reg_integer_accumulator[index + 8],
            )
        emit_k16()
        emit_k16()

        mov(tmuc, -1)
        mov(tmua, reg_activation_scale_pointer)
        mov(tmua, reg_weight_scale_pointer, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_activation_scale))
        nop(sig=ldtmu(reg_weight_scale))
        fmov(reg_activation_scale, reg_activation_scale.unpack("l"))
        add(
            reg_activation_scale_pointer,
            reg_activation_scale_pointer,
            4 if activation_scale_word else 2,
        )
        fmov(reg_weight_scale, reg_weight_scale.unpack("l"))
        add(
            reg_weight_scale_pointer,
            reg_weight_scale_pointer,
            reg_weight_scale_block_stride,
        )

        for index in range(16):
            rotate(reg_weight_scale, reg_weight_scale, 1).mov(rep, reg_weight_scale)
            itof(reg_float_temporary, reg_integer_accumulator[index])
            fmul(reg_float_temporary, reg_float_temporary, reg_activation_scale)
            fmul(reg_float_temporary, reg_float_temporary, rf0)
            fadd(
                reg_float_accumulator[index],
                reg_float_accumulator[index],
                reg_float_temporary,
            )

        sub(reg_block_count, reg_block_count, 1, cond="pushz")
        block_loop.b(cond="na0")
        nop()
        nop()
        nop()

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

    sub(reg_task_count, reg_task_count, 1, cond="pushz")
    b(R.task_loop, cond="anyna")
    nop()
    nop()
    nop()

    b(link, cond="always")
    nop()
    nop()
    nop()


__all__ = ["qpu_ggml_q4_0_q8_0_tiled_gemm_persistent"]
