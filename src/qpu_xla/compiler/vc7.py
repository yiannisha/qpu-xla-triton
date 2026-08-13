"""VC7 lowering for the first canonical vector forms of the QPU-XLA DSL."""

from __future__ import annotations

from qpu_xla.compiler.ir import DslProgram
from qpu_xla.errors import DslCompileError
from qpu_xla.kernel import Kernel
from qpu_xla.kernels.copy import WORD_COPY_KERNEL
from qpu_xla.kernels.gemm import TILED_INT32_GEMM_KERNEL
from qpu_xla.kernels.gemm_fp32 import TILED_FP32_GEMM_KERNEL
from qpu_xla.kernels.minmax import MAXIMUM_WORD_KERNEL, MINIMUM_WORD_KERNEL


def _opcodes(program: DslProgram) -> tuple[str, ...]:
    """Return the source-order primitive opcodes used for canonical matching."""
    return tuple(instruction.opcode for instruction in program.instructions)


def _unsupported(program: DslProgram) -> DslCompileError:
    """Produce a source-mapped diagnostic when VC7 has no safe lowering."""
    first_location = program.instructions[0].location
    return DslCompileError(
        f"{first_location.filename}:{first_location.line}:{first_location.column}: "
        f"VC7 lowering does not support DSL kernel {program.name!r}; "
        "supported forms are canonical 16-lane copy, minimum, maximum, and tiled dot"
    )


def lower_vc7(program: DslProgram) -> Kernel:
    """Map a verified canonical vector DSL program to a cached VC7 kernel.

    The returned kernel retains the existing exact launch, contiguity, dtype,
    and lane-alignment checks of the handwritten specialization.  This first
    lowering stage intentionally reuses those verified implementations rather
    than emitting unvalidated assembly for superficially similar ASTs.
    """
    opcodes = _opcodes(program)
    if program.parameters == ("source", "destination") and opcodes in {
        ("arange", "load", "store"),
        ("arange", "program_id", "load", "store"),
    }:
        return WORD_COPY_KERNEL
    if program.parameters == ("left", "right", "destination"):
        if opcodes == ("arange", "dot", "store"):
            constants = dict(program.constants)
            dtype = constants.get("dtype")
            if dtype == "int32":
                return TILED_INT32_GEMM_KERNEL
            if dtype == "float32":
                return TILED_FP32_GEMM_KERNEL
            location = program.instructions[1].location
            raise DslCompileError(
                f"{location.filename}:{location.line}:{location.column}: "
                "DSL dot lowering requires dtype='int32' or dtype='float32'"
            )
        if opcodes in {
            ("arange", "load", "load", "minimum", "store"),
            ("arange", "program_id", "load", "load", "minimum", "store"),
        }:
            return MINIMUM_WORD_KERNEL
        if opcodes in {
            ("arange", "load", "load", "maximum", "store"),
            ("arange", "program_id", "load", "load", "maximum", "store"),
        }:
            return MAXIMUM_WORD_KERNEL
    raise _unsupported(program)
