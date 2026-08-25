"""Evidence-gated placement and memory planning for SmolVLA inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from qpu_xla.benchmark import CandidateRegistry
from qpu_xla.models.smolvla.checkpoint import SmolVLANumerics
from qpu_xla.models.smolvla.config import SmolVLAConfig
from qpu_xla.scheduler import Placement

_ACCELERATED_STAGES = (
    "rgb_preprocess",
    "patchify",
    "linear",
    "layer_norm",
    "rms_norm",
    "rope",
    "attention",
    "activation",
    "swiglu",
    "residual",
    "affine",
    "pixel_shuffle",
    "embedding",
)


@dataclass(frozen=True, slots=True)
class SmolVLAPlacementPolicy:
    """Per-stage placement with exact-shape AUTO evidence gating."""

    default: Placement = Placement.CPU
    overrides: dict[str, Placement] = field(default_factory=dict)
    qpu_fraction: float = 0.5
    candidates: CandidateRegistry | None = None

    def __post_init__(self: Self) -> None:
        """Validate stage overrides and the heterogeneous split fraction."""
        if not 0 < self.qpu_fraction < 1:
            raise ValueError("SmolVLA hybrid QPU fraction must be strictly between zero and one")
        unknown = sorted(set(self.overrides) - set(_ACCELERATED_STAGES))
        if unknown:
            raise ValueError(f"unknown SmolVLA placement stages: {unknown}")

    @classmethod
    def cpu(cls: type[SmolVLAPlacementPolicy]) -> SmolVLAPlacementPolicy:
        """Return an all-native-CPU execution policy."""
        return cls(Placement.CPU)

    @classmethod
    def forced_qpu(cls: type[SmolVLAPlacementPolicy]) -> SmolVLAPlacementPolicy:
        """Return an explicit all-QPU policy for implementation evaluation."""
        return cls(Placement.QPU)

    @classmethod
    def hybrid(cls: type[SmolVLAPlacementPolicy], *, qpu_fraction: float = 0.5) -> SmolVLAPlacementPolicy:
        """Return a true concurrent CPU/QPU policy for every partitionable stage."""
        return cls(Placement.HYBRID, qpu_fraction=qpu_fraction)

    @classmethod
    def auto(cls: type[SmolVLAPlacementPolicy], candidates: CandidateRegistry) -> SmolVLAPlacementPolicy:
        """Return a CPU-safe policy that promotes only exact supported-win records."""
        return cls(Placement.AUTO, candidates=candidates)

    def choose(
        self: Self,
        stage: str,
        *,
        dtype: str,
        shape_class: str,
        model_dtype: str = "fp32",
        model_shape_class: str = "",
    ) -> Placement:
        """Resolve one stage only after exact stage, block, and replay wins."""
        requested = self.overrides.get(stage, self.default)
        if requested is not Placement.AUTO:
            return requested
        if self.candidates is None or not model_shape_class:
            return Placement.CPU
        records = self.candidates.supported_for(
            operation=f"smolvla.{stage}",
            dtype=dtype,
            layout="native-contiguous",
            shape_class=shape_class,
        )
        executable = tuple(record for record in records if record.placement in {"qpu", "hybrid"})
        if not executable:
            return Placement.CPU
        ordered = sorted(
            executable,
            key=lambda record: (
                record.performance.candidate_median_seconds if record.performance is not None else float("inf")
            ),
        )
        block = self.candidates.supported_for(
            operation=f"smolvla.block.{stage}",
            dtype=dtype,
            layout="native-contiguous",
            shape_class=shape_class,
        )
        full = self.candidates.supported_for(
            operation="smolvla.full_replay",
            dtype=model_dtype,
            layout="native-contiguous",
            shape_class=model_shape_class,
        )
        for winner in ordered:
            if any(record.placement == winner.placement for record in block) and any(
                record.placement == winner.placement for record in full
            ):
                return Placement(winner.placement)
        return Placement.CPU

    def row_partition(self: Self, rows: int, *, alignment: int = 1) -> int:
        """Choose an aligned QPU prefix while preserving a non-empty CPU tail."""
        if rows <= 1 or alignment <= 0:
            raise ValueError("hybrid row partition requires multiple rows and positive alignment")
        units = int(rows * self.qpu_fraction) // alignment * alignment
        units = max(alignment, units)
        if units >= rows:
            units = (rows - 1) // alignment * alignment
        if units <= 0 or units >= rows:
            raise ValueError("shape has no non-empty aligned hybrid row partition")
        return units

    def head_partition(self: Self, heads: int) -> int:
        """Choose a non-empty QPU head prefix."""
        return self.row_partition(heads)


@dataclass(frozen=True, slots=True)
class SmolVLAMemoryPlan:
    """Conservative host/QPU memory budget calculated before opening a driver."""

    numerics: SmolVLANumerics
    artifact_bytes: int
    qpu_arena_bytes: int
    activation_bytes: int
    projected_peak_rss_bytes: int
    limit_bytes: int

    @classmethod
    def build(
        cls: type[SmolVLAMemoryPlan],
        config: SmolVLAConfig,
        numerics: SmolVLANumerics,
        *,
        limit_bytes: int = 6 * 1024**3,
    ) -> SmolVLAMemoryPlan:
        """Budget staged weights, persistent workspaces, artifact maps, and headroom."""
        shapes = config.expected_shapes()
        matrix_shapes = [shape for shape in shapes.values() if len(shape) == 2]
        padded = [
            (
                (shape[1] + (15 if numerics is SmolVLANumerics.W8A8 else 3))
                // (16 if numerics is SmolVLANumerics.W8A8 else 4)
                * (16 if numerics is SmolVLANumerics.W8A8 else 4),
                (shape[0] + 15) // 16 * 16,
            )
            for shape in matrix_shapes
        ]
        max_weight_values = max(inputs * outputs for inputs, outputs in padded)
        batches = (
            config.vision_token_count,
            config.image_token_count,
            config.prefix_length,
            config.chunk_size,
            1,
        )
        max_activation_values = max(
            max(
                ((batch + 15) // 16 * 16) * inputs,
                ((batch + 15) // 16 * 16) * outputs,
            )
            for batch in batches
            for inputs, outputs in padded
            if batch * min(inputs, outputs) <= config.vision_token_count * config.vision_intermediate_size
        )
        if numerics is SmolVLANumerics.FP32:
            staged_linear = (max_weight_values + 2 * max_activation_values) * 4
        else:
            staged_linear = max_weight_values + max_activation_values + max_activation_values * 4
        rgb_metadata = 8 * 3 * config.image_size**2 * 4
        image_buffers = (
            config.input_image_height * config.input_image_width * 3
            + config.image_size**2 * 3
            + config.vision_token_count * config.vision_hidden_size * 3
            + config.image_token_count * config.connector_input_size
        ) * 4
        attention = (
            config.vision_token_count**2 * 2
            + config.vision_token_count * config.vision_hidden_size * 5
            + config.prefix_length**2 * 2
        ) * 4
        embedding = config.vocab_size * config.vlm_hidden_size * 4
        transformer = (
            config.prefix_length * config.vlm_intermediate_size * 4
            + config.chunk_size * config.expert_intermediate_size * 4
            + config.vlm_num_layers * config.prefix_length * config.key_value_size * 2 * 4
        )
        activation_bytes = image_buffers + attention + transformer
        raw_arena = staged_linear + rgb_metadata + activation_bytes + embedding + 96 * 1024**2
        qpu_arena = (raw_arena + 16 * 1024**2 - 1) // (16 * 1024**2) * (16 * 1024**2)
        artifact = config.estimate_weight_bytes(numerics=numerics.value)
        transient = max_weight_values * 4 + 512 * 1024**2
        projected = artifact + qpu_arena + transient
        if projected >= limit_bytes:
            raise MemoryError(
                f"SmolVLA {numerics.value} projected peak RSS {projected / 1024**3:.2f} GiB "
                f"exceeds limit {limit_bytes / 1024**3:.2f} GiB"
            )
        return cls(numerics, artifact, qpu_arena, activation_bytes, projected, limit_bytes)


__all__ = ["SmolVLAMemoryPlan", "SmolVLAPlacementPolicy"]
