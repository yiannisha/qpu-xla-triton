"""Machine-readable contract for the deliberately narrow v0 DSL surface."""

import json
from os import PathLike
from pathlib import Path

DSL_REFERENCE_VERSION = "0.1"


def dsl_reference() -> dict[str, object]:
    """Return the generator-facing capability contract for supported VC7 DSL code.

    This data deliberately describes only source forms that the compiler can
    prove and lower today. Callers must treat omitted language features as
    unsupported instead of assuming normal Python semantics.
    """
    return {
        "version": DSL_REFERENCE_VERSION,
        "source_contract": {
            "exactly_one_function": True,
            "decorators": False,
            "imports": False,
            "plain_positional_parameters_only": True,
            "python_execution": False,
        },
        "primitives": ["arange", "barrier", "dot", "load", "maximum", "minimum", "program_id", "select", "store"],
        "supported_lowerings": [
            {
                "name": "copy_words",
                "parameters": ["source", "destination"],
                "opcodes": ["arange", "load", "store"],
                "dtype": ["int32", "float32"],
                "element_multiple": 16,
            },
            {
                "name": "minimum_words",
                "parameters": ["left", "right", "destination"],
                "opcodes": ["arange", "load", "load", "minimum", "store"],
                "dtype": ["int32", "float32"],
                "element_multiple": 16,
            },
            {
                "name": "maximum_words",
                "parameters": ["left", "right", "destination"],
                "opcodes": ["arange", "load", "load", "maximum", "store"],
                "dtype": ["int32", "float32"],
                "element_multiple": 16,
            },
            {
                "name": "tiled_dot",
                "parameters": ["left", "right", "destination"],
                "opcodes": ["arange", "dot", "store"],
                "dtype_constant": ["int32", "float32"],
                "left_rows_multiple": 16,
                "reduction_multiple": 4,
                "right_columns_multiple": 16,
            },
        ],
        "candidate_runner": {
            "source_is_never_executed": True,
            "default_max_source_bytes": 65_536,
            "default_max_ast_nodes": 4_096,
            "default_compile_timeout_seconds": 1.0,
            "hardware_differential": "optional; required before promotion",
        },
    }


def write_dsl_reference(path: str | PathLike[str]) -> None:
    """Write the capability contract as stable pretty-printed JSON."""
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(dsl_reference(), output, indent=2, sort_keys=True)
        output.write("\n")
