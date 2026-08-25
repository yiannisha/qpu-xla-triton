"""Native GGML split-GEGLU kernel used by the llama.cpp backend plugin."""

from __future__ import annotations

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu


@qpu
def qpu_ggml_geglu_split_fp32(asm: Assembly) -> None:
    """Apply ``gelu(gate) * up`` to four FP32 words per SIMD lane."""
    reg_iterations = rf0
    reg_gate = rf1
    reg_up = rf2
    reg_workgroup = rf4
    reg_destination = rf5
    reg_gelu_table = rf6
    reg_offset = rf7
    reg_stride = rf30

    mov(reg_workgroup, rf3.unpack("ul"))
    nop(sig=ldunifrf(reg_iterations))
    nop(sig=ldunifrf(reg_gate))
    nop(sig=ldunifrf(reg_up))
    nop(sig=ldunifrf(reg_destination))
    nop(sig=ldunifrf(reg_gelu_table))

    umul24(reg_offset, reg_workgroup, reg_iterations)
    shl(reg_offset, reg_offset, 8)
    eidx(rf31)
    shl(rf31, rf31, 4)
    add(reg_offset, reg_offset, rf31)
    add(reg_gate, reg_gate, reg_offset)
    add(reg_up, reg_up, reg_offset)
    add(reg_destination, reg_destination, reg_offset)
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 8)

    gate_values = [rf10, rf11, rf12, rf13]
    up_values = [rf14, rf15, rf16, rf17]
    table_addresses = [rf18, rf19, rf20, rf21]
    table_values = [rf22, rf23, rf24, rf25]
    outputs = [rf26, rf27, rf28, rf29]

    # Four consecutive words are loaded/stored for every SIMD lane.
    bnot(reg_offset, 3)
    nop(sig=thrsw)
    nop()
    nop()
    with loop as vector_loop:
        mov(tmuc, reg_offset)
        mov(tmua, reg_gate)
        add(reg_gate, reg_gate, reg_stride)
        nop()
        nop()
        for value in gate_values:
            nop(sig=ldtmu(value))

        mov(tmuc, reg_offset)
        mov(tmua, reg_up)
        add(reg_up, reg_up, reg_stride)
        nop()
        nop()
        for value in up_values:
            nop(sig=ldtmu(value))

        # The pinned Cortex-A76 GGML build implements F32 GEGLU through its
        # 65536-entry FP16 GELU table. Convert each gate to the identical FP16
        # index and fetch the precomputed half so the backend matches that
        # production contract, not merely the analytic approximation.
        for address, gate_value in zip(table_addresses, gate_values, strict=True):
            fmov(address.pack("l"), gate_value)
            shl(address, address, 8)
            shl(address, address, 8)
            shr(address, address, 8)
            shr(address, address, 8)
            shl(address, address, 1)
            add(address, address, reg_gelu_table)
        for address, value in zip(table_addresses, table_values, strict=True):
            mov(tmua, address, sig=thrsw)
            nop()
            nop()
            nop(sig=ldtmu(value))
            fmov(value, value.unpack("l"))
        for output, gelu_value, up_value in zip(
            outputs, table_values, up_values, strict=True
        ):
            fmul(output, gelu_value, up_value)

        mov(tmuc, reg_offset)
        for output in outputs:
            mov(tmud, output)
        mov(tmua, reg_destination)
        add(reg_destination, reg_destination, reg_stride)
        tmuwt()
        sub(reg_iterations, reg_iterations, 1, cond="pushz")

        vector_loop.b(cond="na0")
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


__all__ = ["qpu_ggml_geglu_split_fp32"]
