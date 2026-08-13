"""Restricted kernel DSL capture, typed IR, and validation primitives."""

from qpu_xla.compiler.dsl import (
    JittedKernel,
    arange,
    barrier,
    compile_source,
    dot,
    jit,
    load,
    maximum,
    minimum,
    program_id,
    select,
    store,
)
from qpu_xla.compiler.ir import DslInstruction, DslProgram, SourceLocation
from qpu_xla.compiler.reference import DSL_REFERENCE_VERSION, dsl_reference, write_dsl_reference
from qpu_xla.compiler.vc7 import lower_vc7

__all__ = [
    "DslInstruction",
    "DslProgram",
    "DSL_REFERENCE_VERSION",
    "JittedKernel",
    "SourceLocation",
    "arange",
    "barrier",
    "compile_source",
    "dot",
    "dsl_reference",
    "jit",
    "load",
    "lower_vc7",
    "maximum",
    "minimum",
    "program_id",
    "select",
    "store",
    "write_dsl_reference",
]
