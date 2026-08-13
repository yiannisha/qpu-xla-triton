"""A deliberately small, statically validated Python kernel DSL front end."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from textwrap import dedent
from typing import Any, NoReturn, Self

from qpu_xla.compiler.ir import DslInstruction, DslProgram, SourceLocation
from qpu_xla.errors import DslCompileError
from qpu_xla.kernel import Kernel

_PRIMITIVES = frozenset({"arange", "barrier", "dot", "load", "maximum", "minimum", "program_id", "select", "store"})


def _capture_only(name: str) -> NoReturn:
    """Explain that DSL primitives are captured from source, never interpreted by Python."""
    raise RuntimeError(f"{name} is a QPU-XLA DSL primitive and can only be used inside a @jit kernel")


def arange(*args: object, **kwargs: object) -> object:
    """Declare a lane vector in a source-captured ``@jit`` kernel."""
    _capture_only("arange")


def barrier(*args: object, **kwargs: object) -> object:
    """Declare a workgroup barrier in a source-captured ``@jit`` kernel."""
    _capture_only("barrier")


def dot(*args: object, **kwargs: object) -> object:
    """Declare a canonical tiled matrix-product operation in a captured kernel."""
    _capture_only("dot")


def load(*args: object, **kwargs: object) -> object:
    """Declare a masked memory load in a source-captured ``@jit`` kernel."""
    _capture_only("load")


def maximum(*args: object, **kwargs: object) -> object:
    """Declare an elementwise maximum in a source-captured ``@jit`` kernel."""
    _capture_only("maximum")


def minimum(*args: object, **kwargs: object) -> object:
    """Declare an elementwise minimum in a source-captured ``@jit`` kernel."""
    _capture_only("minimum")


def program_id(*args: object, **kwargs: object) -> object:
    """Declare one program identifier query in a source-captured ``@jit`` kernel."""
    _capture_only("program_id")


def select(*args: object, **kwargs: object) -> object:
    """Declare a vector select in a source-captured ``@jit`` kernel."""
    _capture_only("select")


def store(*args: object, **kwargs: object) -> object:
    """Declare a masked memory store in a source-captured ``@jit`` kernel."""
    _capture_only("store")


def _location(node: ast.AST, filename: str) -> SourceLocation:
    """Build a public source location for an AST node."""
    return SourceLocation(filename, getattr(node, "lineno", 1), getattr(node, "col_offset", 0))


class _Verifier:
    """Verify the intentionally narrow expression language and capture calls."""

    def __init__(self: Self, filename: str, symbols: set[str]) -> None:
        self._filename = filename
        self._symbols = symbols
        self.instructions: list[DslInstruction] = []

    def _error(self: Self, node: ast.AST, message: str) -> DslCompileError:
        location = _location(node, self._filename)
        return DslCompileError(f"{location.filename}:{location.line}:{location.column}: {message}")

    def statements(self: Self, statements: list[ast.stmt]) -> None:
        """Validate a function body containing only assignments and primitive calls."""
        for statement in statements:
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise self._error(statement, "DSL assignments require one local name target")
                self.expression(statement.value)
                self._symbols.add(statement.targets[0].id)
            elif isinstance(statement, ast.AnnAssign):
                if not isinstance(statement.target, ast.Name) or statement.value is None:
                    raise self._error(statement, "DSL annotated assignments require one local name and a value")
                self.expression(statement.value)
                self._symbols.add(statement.target.id)
            elif isinstance(statement, ast.Expr):
                if not isinstance(statement.value, ast.Call):
                    raise self._error(statement, "DSL expression statements must be primitive calls")
                self.expression(statement.value)
            elif isinstance(statement, ast.Pass):
                continue
            else:
                raise self._error(statement, "unsupported control flow or statement in DSL kernel")

    def expression(self: Self, expression: ast.expr) -> None:
        """Validate an expression recursively and record primitive invocations."""
        if isinstance(expression, ast.Name):
            if expression.id not in self._symbols:
                raise self._error(expression, f"unknown DSL name {expression.id!r}")
            return
        if isinstance(expression, ast.Constant):
            return
        if isinstance(expression, ast.BinOp):
            self.expression(expression.left)
            self.expression(expression.right)
            return
        if isinstance(expression, ast.UnaryOp):
            self.expression(expression.operand)
            return
        if isinstance(expression, ast.Compare):
            self.expression(expression.left)
            for comparator in expression.comparators:
                self.expression(comparator)
            return
        if isinstance(expression, ast.Call):
            if not isinstance(expression.func, ast.Name) or expression.func.id not in _PRIMITIVES:
                raise self._error(expression, "only QPU-XLA DSL primitives may be called")
            for argument in expression.args:
                self.expression(argument)
            for keyword in expression.keywords:
                if keyword.arg is None:
                    raise self._error(keyword.value, "DSL primitive calls do not accept **kwargs")
                self.expression(keyword.value)
            self.instructions.append(
                DslInstruction(expression.func.id, ast.unparse(expression), _location(expression, self._filename))
            )
            return
        raise self._error(expression, f"unsupported DSL expression {type(expression).__name__}")


def _source(function: Callable[..., Any]) -> tuple[ast.FunctionDef, str, str]:
    """Return the parsed definition and dedented source for a Python function."""
    try:
        lines, start_line = inspect.getsourcelines(function)
    except (OSError, TypeError) as exc:
        raise DslCompileError(f"cannot inspect source for DSL kernel {function.__name__!r}") from exc
    source = dedent("".join(lines))
    module = ast.parse(source)
    definition = next((node for node in module.body if isinstance(node, ast.FunctionDef)), None)
    if definition is None:
        raise DslCompileError(f"cannot find a function definition for DSL kernel {function.__name__!r}")
    ast.increment_lineno(definition, start_line - 1)
    return definition, source, inspect.getsourcefile(function) or "<unknown>"


def _compile_definition(
    definition: ast.FunctionDef,
    source: str,
    filename: str,
    constants: Mapping[str, object],
) -> DslProgram:
    """Verify one parsed function and form its immutable specialization IR."""
    constant_items = tuple(sorted(constants.items()))
    if (
        definition.args.posonlyargs
        or definition.args.vararg is not None
        or definition.args.kwonlyargs
        or definition.args.kwarg is not None
        or definition.args.defaults
        or definition.args.kw_defaults
    ):
        raise DslCompileError(
            f"{filename}:{definition.lineno}:{definition.col_offset}: "
            "DSL kernels require plain positional parameters without defaults"
        )
    parameter_names = tuple(argument.arg for argument in definition.args.args)
    verifier = _Verifier(filename, set(parameter_names) | set(constants))
    verifier.statements(definition.body)
    if not any(instruction.opcode == "store" for instruction in verifier.instructions):
        raise DslCompileError(f"{filename}:{definition.lineno}:0: DSL kernel must contain at least one store")
    digest = sha256()
    digest.update(source.encode())
    digest.update(repr(constant_items).encode())
    return DslProgram(
        definition.name,
        parameter_names,
        constant_items,
        digest.hexdigest(),
        tuple(verifier.instructions),
    )


def compile_source(
    source: str,
    *,
    filename: str = "<qpu-xla-candidate>",
    constants: Mapping[str, object] = {},
) -> DslProgram:
    """Compile exactly one restricted DSL function from inert source text.

    The source is parsed but never passed to :func:`exec` or imported. This is
    the entry point for generated candidate kernels: module-level imports,
    assignments, classes, decorators, and multiple functions are rejected
    before the normal DSL verifier sees the function body.
    """
    if not isinstance(source, str) or not source.strip():
        raise DslCompileError("candidate source must be a non-empty string")
    if not filename:
        raise ValueError("candidate filename must be non-empty")
    try:
        module = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        line = 1 if exc.lineno is None else exc.lineno
        column = 0 if exc.offset is None else exc.offset - 1
        raise DslCompileError(f"{filename}:{line}:{column}: invalid Python syntax") from exc
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise DslCompileError(f"{filename}:1:0: candidate source must define exactly one function")
    definition = module.body[0]
    if definition.decorator_list:
        raise DslCompileError(f"{filename}:{definition.lineno}:{definition.col_offset}: decorators are not allowed")
    return _compile_definition(definition, source, filename, constants)


@dataclass(slots=True)
class JittedKernel:
    """A source-captured kernel that compiles to validated, cacheable DSL IR."""

    function: Callable[..., Any]
    _cache: dict[tuple[tuple[str, object], ...], DslProgram] = field(default_factory=dict)

    @property
    def name(self: Self) -> str:
        """Return the original Python function name."""
        return self.function.__name__

    def compile(self: Self, **constants: object) -> DslProgram:
        """Capture and validate this kernel, specializing it by constant values."""
        constant_items = tuple(sorted(constants.items()))
        try:
            return self._cache[constant_items]
        except KeyError:
            pass
        definition, source, filename = _source(self.function)
        program = _compile_definition(definition, source, filename, dict(constant_items))
        self._cache[constant_items] = program
        return program

    def lower(self: Self, *, target: str = "vc7", **constants: object) -> Kernel:
        """Lower one canonical specialization to the requested executable backend.

        The initial VC7 target deliberately recognizes only the exact vector
        forms backed by hardware-validated assembly kernels.  Programs outside
        that subset fail at their DSL source boundary instead of silently
        falling back to arbitrary Python execution.
        """
        if target != "vc7":
            raise DslCompileError(f"unsupported DSL lowering target {target!r}")
        from qpu_xla.compiler.vc7 import lower_vc7

        return lower_vc7(self.compile(**constants))


def jit(function: Callable[..., Any]) -> JittedKernel:
    """Decorate a restricted DSL function for explicit AST capture and compilation."""
    return JittedKernel(function)
