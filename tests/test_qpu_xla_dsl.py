from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device, DslCompileError, arange, dot, jit, load, maximum, minimum, program_id, store
from qpu_xla.compiler import DSL_REFERENCE_VERSION, compile_source, dsl_reference, write_dsl_reference


@jit
def _copy_kernel(source, destination, count):
    offsets = arange(0, 16)
    index = program_id(0) * 16 + offsets
    mask = index < count
    values = load(source + index, mask=mask)
    store(destination + index, values, mask=mask)


@jit
def _invalid_kernel(source, destination):
    if source:
        store(destination, 1)


@jit
def _word_copy(source, destination):
    offsets = arange(0, 16)
    values = load(source + offsets)
    store(destination + offsets, values)


@jit
def _word_minimum(left, right, destination):
    offsets = arange(0, 16)
    left_value = load(left + offsets)
    right_value = load(right + offsets)
    store(destination + offsets, minimum(left_value, right_value))


@jit
def _word_maximum(left, right, destination):
    offsets = arange(0, 16)
    left_value = load(left + offsets)
    right_value = load(right + offsets)
    store(destination + offsets, maximum(left_value, right_value))


@jit
def _word_dot(left, right, destination):
    offsets = arange(0, 16)
    values = dot(left, right)
    store(destination + offsets, values)


def test_jit_captures_source_mapped_ir_and_caches_specializations() -> None:
    first = _copy_kernel.compile(tile=16)
    second = _copy_kernel.compile(tile=16)
    changed = _copy_kernel.compile(tile=32)

    assert first is second
    assert first is not changed
    assert first.parameters == ("source", "destination", "count")
    assert [instruction.opcode for instruction in first.instructions] == ["arange", "program_id", "load", "store"]
    assert first.instructions[-1].location.filename.endswith("test_qpu_xla_dsl.py")


def test_jit_reports_unsupported_control_flow_at_source_location() -> None:
    with pytest.raises(DslCompileError, match=r"test_qpu_xla_dsl.py:\d+:\d+: unsupported control flow"):
        _invalid_kernel.compile()


def test_vc7_lowering_selects_the_verified_vector_specializations() -> None:
    assert _word_copy.lower().name == "vc7.copy_words"
    assert _word_minimum.lower().name == "vc7.minimum_words"
    assert _word_maximum.lower().name == "vc7.maximum_words"
    assert _word_dot.lower(dtype="int32").name == "vc7.tiled_int32_gemm"
    assert _word_dot.lower(dtype="float32").name == "vc7.tiled_fp32_gemm"


def test_vc7_dot_lowering_requires_an_explicit_supported_dtype() -> None:
    with pytest.raises(DslCompileError, match="requires dtype='int32' or dtype='float32'"):
        _word_dot.lower()


def test_vc7_lowering_rejects_a_verified_but_not_yet_canonical_program() -> None:
    with pytest.raises(DslCompileError, match="VC7 lowering does not support"):
        _copy_kernel.lower()


def test_compile_source_verifies_an_inert_single_function_without_executing_it() -> None:
    program = compile_source(
        """
def generated_copy(source, destination):
    offsets = arange(0, 16)
    values = load(source + offsets)
    store(destination + offsets, values)
""",
        filename="generated_candidate.py",
        constants={"tile": 16},
    )

    assert program.name == "generated_copy"
    assert program.parameters == ("source", "destination")
    assert program.constants == (("tile", 16),)
    assert [instruction.opcode for instruction in program.instructions] == ["arange", "load", "store"]


@pytest.mark.parametrize(
    "source, message",
    [
        ("import os\n\ndef generated(source, destination):\n    store(destination, source)", "exactly one function"),
        ("def generated(source, destination):\n    open('unsafe', 'w')", "only QPU-XLA DSL primitives"),
        ("@jit\ndef generated(source, destination):\n    store(destination, source)", "decorators are not allowed"),
        ("def generated(source, destination):\n    store(destination, unknown)", "unknown DSL name 'unknown'"),
        ("def generated(source, destination=0):\n    store(destination, source)", "plain positional parameters"),
    ],
)
def test_compile_source_rejects_non_dsl_or_module_level_candidate_code(source: str, message: str) -> None:
    with pytest.raises(DslCompileError, match=message):
        compile_source(source, filename="untrusted.py")


def test_machine_readable_dsl_reference_is_versioned_and_serializable(tmp_path) -> None:
    reference = dsl_reference()
    output = tmp_path / "dsl-reference.json"
    write_dsl_reference(output)

    assert reference["version"] == DSL_REFERENCE_VERSION
    assert reference["source_contract"] == {
        "exactly_one_function": True,
        "decorators": False,
        "imports": False,
        "plain_positional_parameters_only": True,
        "python_execution": False,
    }
    assert '"copy_words"' in output.read_text(encoding="utf-8")
    assert '"tiled_dot"' in output.read_text(encoding="utf-8")


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_vc7_lowered_dsl_copy_and_minimum_match_numpy() -> None:
    source_value = np.arange(-8, 8, dtype=np.int32)
    right_value = source_value[::-1].copy()
    left_matrix = np.arange(-32, 32, dtype=np.int32).reshape(16, 4)
    right_matrix = np.arange(-8, 8, dtype=np.int32).reshape(4, 4)
    right_matrix = np.pad(right_matrix, ((0, 0), (0, 12)))
    expected_dot = left_matrix @ right_matrix
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.int32)
        right_vector = device.tensor(right_value.shape, np.int32)
        copied = device.tensor(source_value.shape, np.int32)
        minimum_output = device.tensor(source_value.shape, np.int32)
        left = device.tensor(left_matrix.shape, np.int32)
        right = device.tensor(right_matrix.shape, np.int32)
        dot_output = device.tensor(expected_dot.shape, np.int32)
        source.numpy()[:] = source_value
        right_vector.numpy()[:] = right_value

        left.numpy()[:] = left_matrix
        right.numpy()[:] = right_matrix

        copy_event = queue.submit(_word_copy.lower(), (source, copied))
        minimum_event = queue.submit(
            _word_minimum.lower(), (source, right_vector, minimum_output), wait_for=(copy_event,)
        )
        dot_event = queue.submit(_word_dot.lower(dtype="int32"), (left, right, dot_output), wait_for=(minimum_event,))
        dot_event.wait()

        np.testing.assert_array_equal(copied.numpy(), source_value)
        np.testing.assert_array_equal(minimum_output.numpy(), np.minimum(source_value, right_value))
        np.testing.assert_array_equal(dot_output.numpy(), expected_dot)
