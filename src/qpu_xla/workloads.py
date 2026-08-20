"""Versioned production-shape manifests for accelerator tuning and holdout validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Self


@dataclass(frozen=True, slots=True)
class LlamaWorkload:
    """One dense Llama projection/attention shape used by kernel evaluation."""

    name: str
    phase: Literal["prefill", "decode"]
    tokens: int
    hidden_size: int
    intermediate_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    cache_length: int
    vocabulary_size: int

    def __post_init__(self: Self) -> None:
        """Validate one internally consistent dense decoder shape."""
        values = (
            self.tokens,
            self.hidden_size,
            self.intermediate_size,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.cache_length,
            self.vocabulary_size,
        )
        if not self.name or any(value <= 0 for value in values):
            raise ValueError("Llama workload names and dimensions must be positive")
        if self.phase == "decode" and self.tokens != 1:
            raise ValueError("decode workloads must contain exactly one token")
        if self.hidden_size != self.query_heads * self.head_dim:
            raise ValueError("hidden_size must equal query_heads * head_dim")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")


@dataclass(frozen=True, slots=True)
class YoloConvWorkload:
    """One batch-one YOLO convolution shape used by kernel evaluation."""

    name: str
    height: int
    width: int
    in_channels: int
    out_channels: int
    kernel: int
    stride: int
    groups: int = 1

    def __post_init__(self: Self) -> None:
        """Validate one supported convolution shape class."""
        values = (
            self.height,
            self.width,
            self.in_channels,
            self.out_channels,
            self.kernel,
            self.stride,
            self.groups,
        )
        if not self.name or any(value <= 0 for value in values):
            raise ValueError("YOLO workload names and dimensions must be positive")
        if self.kernel not in (1, 3) or self.stride not in (1, 2):
            raise ValueError("the v1 YOLO manifest covers 1x1/3x3 and stride 1/2")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError("YOLO workload channels must be divisible by groups")


@dataclass(frozen=True, slots=True)
class WorkloadManifest:
    """Immutable tuning and holdout cases for one workload family."""

    name: str
    version: int
    tuning: tuple[LlamaWorkload | YoloConvWorkload, ...]
    holdout: tuple[LlamaWorkload | YoloConvWorkload, ...]

    def __post_init__(self: Self) -> None:
        """Validate manifest identity and tuning/holdout consistency."""
        if not self.name or self.version <= 0 or not self.tuning or not self.holdout:
            raise ValueError("workload manifests require a name, version, tuning cases, and holdout cases")
        names = [case.name for case in (*self.tuning, *self.holdout)]
        if len(names) != len(set(names)):
            raise ValueError("workload case names must be unique within a manifest")
        case_types = {type(case) for case in (*self.tuning, *self.holdout)}
        if len(case_types) != 1:
            raise ValueError("one workload manifest cannot mix model families")

    def to_dict(self: Self) -> dict[str, object]:
        """Return a stable machine-readable representation."""
        return {
            "name": self.name,
            "version": self.version,
            "tuning": [asdict(case) for case in self.tuning],
            "holdout": [asdict(case) for case in self.holdout],
        }


LLAMA_DENSE_V1 = WorkloadManifest(
    "llama-dense",
    1,
    tuning=(
        LlamaWorkload("prefill-h512-t16", "prefill", 16, 512, 1536, 8, 1, 64, 16, 32_000),
        LlamaWorkload("prefill-h1024-t64", "prefill", 64, 1024, 2816, 16, 4, 64, 64, 32_000),
        LlamaWorkload("prefill-h2048-t128", "prefill", 128, 2048, 5632, 32, 4, 64, 128, 32_000),
        LlamaWorkload("decode-h2048-c512", "decode", 1, 2048, 5632, 32, 4, 64, 512, 32_000),
        LlamaWorkload("decode-h2048-c2048", "decode", 1, 2048, 5632, 32, 4, 64, 2048, 32_000),
        LlamaWorkload("prefill-h4096-t16", "prefill", 16, 4096, 11008, 32, 8, 128, 16, 128_000),
    ),
    holdout=(
        LlamaWorkload("prefill-h2048-t256", "prefill", 256, 2048, 5632, 32, 4, 64, 256, 32_000),
        LlamaWorkload("decode-h3072-c4096", "decode", 1, 3072, 8192, 24, 8, 128, 4096, 128_000),
    ),
)


YOLO_DETECTION_V1 = WorkloadManifest(
    "yolo-detection",
    1,
    tuning=(
        YoloConvWorkload("stem-3x3-s2", 640, 640, 3, 16, 3, 2),
        YoloConvWorkload("p3-3x3-s1", 80, 80, 64, 64, 3, 1),
        YoloConvWorkload("p3-1x1", 80, 80, 128, 64, 1, 1),
        YoloConvWorkload("p4-3x3-s2", 80, 80, 128, 256, 3, 2),
        YoloConvWorkload("p5-1x1", 20, 20, 512, 1024, 1, 1),
        YoloConvWorkload("depthwise-p3", 80, 80, 128, 128, 3, 1, 128),
    ),
    holdout=(
        YoloConvWorkload("holdout-p4-1x1", 40, 40, 256, 128, 1, 1),
        YoloConvWorkload("holdout-depthwise-s2", 160, 160, 32, 32, 3, 2, 32),
    ),
)


__all__ = [
    "LLAMA_DENSE_V1",
    "YOLO_DETECTION_V1",
    "LlamaWorkload",
    "WorkloadManifest",
    "YoloConvWorkload",
]
