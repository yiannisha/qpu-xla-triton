from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.replay_llama_cpp_node_fixture import (
    aligned_suffix_partitions,
    build_cpu_repack_command,
    build_native_command,
    calculate_error,
    materialize_native_weight,
)


def test_aligned_suffix_partitions_cover_all_twelfths() -> None:
    partitions = aligned_suffix_partitions(192)
    assert partitions == [(192 - count, count) for count in range(16, 192, 16)]
    assert aligned_suffix_partitions(17) == []


def test_calculate_error_enforces_tolerance_and_greedy_argmax() -> None:
    reference = np.asarray([1.0, 2.0, -3.0], dtype="<f4")
    close = np.asarray([1.0, 2.00001, -3.0], dtype="<f4")
    metrics = calculate_error(reference, close, atol=2e-4, rtol=2e-5)
    assert metrics["passed"] is True
    assert metrics["greedy_argmax_matches"] is True

    changed = np.asarray([3.0, 2.0, -3.0], dtype="<f4")
    metrics = calculate_error(reference, changed, atol=2e-4, rtol=2e-5)
    assert metrics["passed"] is False
    assert metrics["greedy_argmax_matches"] is False


def test_materialize_native_weight_and_build_command(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"skip" + bytes(range(18)))
    destination = tmp_path / "weight.bin"
    fixture = {
        "weight": {
            "bytes": 18,
            "file_offset": 4,
            "model": {"path": str(model)},
        },
        "activation_f32": {"path": "/activation.bin"},
        "dimensions": {"input_columns": 32, "output_columns": 16, "rows": 1},
    }
    metadata = materialize_native_weight(fixture, destination)
    assert destination.read_bytes() == bytes(range(18))
    assert metadata["bytes"] == 18

    command = build_native_command(
        Path("/benchmark"),
        fixture,
        weight_path=destination,
        output_path=Path("/output.bin"),
        mode="hybrid",
        cpu_threads=3,
        qpu_column_start=8,
        qpu_column_count=8,
    )
    assert command[command.index("--output-bin") + 1] == "/output.bin"
    assert command[command.index("--cpu-threads") + 1] == "3"

    cpu_command = build_cpu_repack_command(
        Path("/cpu-repack"),
        fixture,
        weight_path=destination,
        output_path=Path("/cpu-output.bin"),
        cpu_threads=4,
    )
    assert cpu_command[0] == "/cpu-repack"
    assert cpu_command[cpu_command.index("--cpu-threads") + 1] == "4"
