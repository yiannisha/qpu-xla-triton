"""Small immutable IR emitted by the first QPU-XLA DSL capture pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """One source coordinate retained for actionable compiler diagnostics."""

    filename: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class DslInstruction:
    """One validated DSL primitive invocation in source order."""

    opcode: str
    expression: str
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class DslProgram:
    """Typed, source-mapped program emitted by AST capture and verification."""

    name: str
    parameters: tuple[str, ...]
    constants: tuple[tuple[str, object], ...]
    source_hash: str
    instructions: tuple[DslInstruction, ...]
