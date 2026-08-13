"""Kernel handles and launch configuration for the QPU-XLA runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from qpu_xla.backend import Backend, KernelCallable
from qpu_xla.errors import KernelError


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """A three-dimensional launch grid for one kernel invocation."""

    grid: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self: Self) -> None:
        """Reject unsupported empty or malformed grids before submission."""
        if len(self.grid) != 3 or any(dimension <= 0 for dimension in self.grid):
            raise KernelError("kernel grid must contain three positive dimensions")


@dataclass(frozen=True, slots=True)
class Kernel:
    """A backend-neutral kernel handle with an explicit execution adapter."""

    name: str
    executor: KernelCallable

    def execute(self: Self, backend: Backend, args: tuple[Any, ...], launch: LaunchConfig) -> None:
        """Run this kernel through its backend-specific executor."""
        try:
            self.executor(backend, args, launch.grid)
        except KernelError:
            raise
        except BaseException as exc:
            raise KernelError(f"kernel {self.name!r} failed") from exc
