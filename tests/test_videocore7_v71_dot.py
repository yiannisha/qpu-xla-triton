from __future__ import annotations

from videocore7.assembler import *
from videocore7.assembler import Assembly, assemble, qpu


@qpu
def _unsigned_dot(asm: Assembly) -> None:
    setnnmode_uu().v8dot(rf12, rf1, rf2)


def test_v71_signed_v8dot_encoding_matches_mesa_disassembler_fixture() -> None:
    assert assemble(_unsigned_dot) == [0x2C000300BB042030]
