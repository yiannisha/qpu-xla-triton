"""Structured benchmark collection without conflating cold, prep, and QPU timings."""

from __future__ import annotations

import json
import os
import platform
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from os import PathLike
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Self

import numpy as np

from qpu_xla.device import Device


def numpy_blas_metadata() -> dict[str, str]:
    """Return the BLAS implementation linked into the active NumPy build."""
    try:
        configuration = np.show_config(mode="dicts")
    except (AttributeError, TypeError):
        return {"numpy_blas": "unknown"}
    dependencies = configuration.get("Build Dependencies", {})
    blas = dependencies.get("blas", {}) if isinstance(dependencies, dict) else {}
    if not isinstance(blas, dict):
        return {"numpy_blas": "unknown"}
    name = str(blas.get("name", "unknown"))
    version = str(blas.get("version", "unknown"))
    details = str(blas.get("openblas configuration", ""))
    return {
        "numpy_blas": name,
        "numpy_blas_version": version,
        "numpy_blas_configuration": details,
    }


def numpy_backend_label() -> str:
    """Return a concise benchmark label for NumPy's detected BLAS backend."""
    metadata = numpy_blas_metadata()
    name = metadata["numpy_blas"].lower()
    version = metadata.get("numpy_blas_version", "unknown")
    return f"numpy-openblas-{version}" if "openblas" in name else f"numpy-{name}-{version}"


class BenchmarkCategory(Enum):
    """Timing categories that must remain distinct in QPU-XLA reports."""

    NUMPY = "numpy"
    TORCH_NATIVE = "torch_native"
    QPU_HOST_PREP = "qpu_host_prep"
    QPU_CACHED_TOTAL = "qpu_cached_total"
    QPU_EXECUTE_ONLY = "qpu_execute_only"
    QPU_PREP_CACHED_TOTAL = "qpu_prep_cached_total"
    COLD_START = "cold_start"
    CPU_TOTAL = "cpu_total"
    QPU_TOTAL = "qpu_total"
    HYBRID_TOTAL = "hybrid_total"
    KERNEL_ONLY = "kernel_only"
    HOST_PREP = "host_prep"


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Raw timing observations for one named category in seconds."""

    category: BenchmarkCategory
    seconds: tuple[float, ...]

    @property
    def best_seconds(self: Self) -> float:
        """Return the best observed timing, traditionally used for kernel throughput."""
        return min(self.seconds)

    @property
    def median_seconds(self: Self) -> float:
        """Return the robust median timing for regression and scheduler use."""
        return float(median(self.seconds))

    def to_dict(self: Self) -> dict[str, object]:
        """Serialize raw observations and derived summaries without losing samples."""
        return {
            "category": self.category.value,
            "seconds": list(self.seconds),
            "best_seconds": self.best_seconds,
            "median_seconds": self.median_seconds,
        }


def _read_text(path: Path) -> str | None:
    """Read an optional one-line system probe without making it a requirement."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def collect_metadata(device: Device | None = None, *, extra: Mapping[str, str] = {}) -> dict[str, str]:
    """Collect portable host/device metadata plus optional caller provenance.

    Clock governor and thermal data are intentionally best-effort because they
    vary by OS image. Callers can pass additional immutable provenance such as
    git revision, firmware, board model, and benchmark manifest identifiers.
    """
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        **numpy_blas_metadata(),
    }
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(variable)
        if value is not None:
            metadata[variable.lower()] = value
    governor = _read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
    temperature = _read_text(Path("/sys/class/thermal/thermal_zone0/temp"))
    if governor is not None:
        metadata["cpu_governor"] = governor
    if temperature is not None:
        metadata["thermal_zone0_temp_millicelsius"] = temperature
    if device is not None:
        metadata["backend"] = type(device.backend).__name__
    for key, value in extra.items():
        if not key or not isinstance(value, str):
            raise ValueError("benchmark metadata keys and values must be non-empty strings")
        metadata[key] = value
    return metadata


@dataclass(slots=True)
class BenchmarkReport:
    """A JSON-serializable benchmark manifest with separate timing categories."""

    name: str
    metadata: dict[str, str]
    samples: list[BenchmarkSample] = field(default_factory=list)

    def measure(
        self: Self,
        category: BenchmarkCategory,
        fn: Callable[[], object],
        *,
        warmup: int = 1,
        repeat: int = 5,
    ) -> BenchmarkSample:
        """Warm up then measure one callable without interpreting its result."""
        if warmup < 0 or repeat <= 0:
            raise ValueError("warmup must be non-negative and repeat must be positive")
        if any(sample.category is category for sample in self.samples):
            raise ValueError(f"benchmark category {category.value!r} was already recorded")
        for _ in range(warmup):
            fn()
        durations: list[float] = []
        for _ in range(repeat):
            start = perf_counter()
            fn()
            duration = perf_counter() - start
            if duration <= 0:
                raise RuntimeError("benchmark clock returned a non-positive duration")
            durations.append(duration)
        sample = BenchmarkSample(category, tuple(durations))
        self.samples.append(sample)
        return sample

    def to_dict(self: Self) -> dict[str, object]:
        """Return the complete machine-readable benchmark result."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def save(self: Self, path: str | PathLike[str]) -> None:
        """Write this report as stable, human-readable JSON."""
        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, indent=2, sort_keys=True)
