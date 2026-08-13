import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "tiledmlp.py"
SPEC = importlib.util.spec_from_file_location("tiledmlp", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PreparedInput = MODULE.PreparedInput
TiledMlpExecutor = MODULE.TiledMlpExecutor
TiledMlpExecutorInt32 = MODULE.TiledMlpExecutorInt32
numpy_mlp_naive = MODULE.numpy_mlp_naive
numpy_mlp_int32 = MODULE.numpy_mlp_int32
reference_mlp_int32 = MODULE.reference_mlp_int32
torch = MODULE.torch
torch_mlp_fp32 = MODULE.torch_mlp_fp32
torch_mlp_int32 = MODULE.torch_mlp_int32
_validate_int32_mlp_contract = MODULE._validate_int32_mlp_contract


def test_numpy_mlp_matches_explicit_reference() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 7), dtype=np.float32)
    w1 = rng.standard_normal((7, 9), dtype=np.float32)
    b1 = rng.standard_normal((9,), dtype=np.float32)
    w2 = rng.standard_normal((9, 4), dtype=np.float32)
    b2 = rng.standard_normal((4,), dtype=np.float32)

    hidden = x.dot(w1) + b1
    hidden = np.maximum(hidden, np.float32(0.0))
    expected = hidden.dot(w2) + b2

    actual = numpy_mlp_naive(x, w1, b1, w2, b2)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_executor_prepare_input_and_padding_preserve_values() -> None:
    rng = np.random.default_rng(1)
    batch = 19
    in_features = 10
    hidden_features = 33
    out_features = 18

    x = rng.standard_normal((batch, in_features), dtype=np.float32)
    w1 = rng.standard_normal((in_features, hidden_features), dtype=np.float32)
    b1 = rng.standard_normal((hidden_features,), dtype=np.float32)
    w2 = rng.standard_normal((hidden_features, out_features), dtype=np.float32)
    b2 = rng.standard_normal((out_features,), dtype=np.float32)

    executor = object.__new__(TiledMlpExecutor)
    executor.batch = batch
    executor.in_features = in_features
    executor.hidden_features = hidden_features
    executor.out_features = out_features
    executor.batch_padded = 32
    executor.in_padded = 12
    executor.hidden_padded = 64
    executor.out_padded = 32
    executor._input_dev = np.empty((32, 12), dtype=np.float32)
    executor._hidden_dev = np.empty((32, 64), dtype=np.float32)

    prepared = TiledMlpExecutor.prepare_input(executor, x)

    assert isinstance(prepared, PreparedInput)
    assert prepared.matrix.shape == (32, 12)
    np.testing.assert_array_equal(prepared.matrix[:batch, :in_features], x)
    assert np.count_nonzero(prepared.matrix[batch:, :]) == 0
    assert np.count_nonzero(prepared.matrix[:, in_features:]) == 0

    padded_w1 = TiledMlpExecutor._pad_weight(w1, rows=12, cols=64, dtype=np.float32)
    padded_w2 = TiledMlpExecutor._pad_weight(w2, rows=64, cols=32, dtype=np.float32)
    padded_b1 = TiledMlpExecutor._pad_bias(b1, size=64, dtype=np.float32)
    padded_b2 = TiledMlpExecutor._pad_bias(b2, size=32, dtype=np.float32)

    np.testing.assert_array_equal(padded_w1[:in_features, :hidden_features], w1)
    np.testing.assert_array_equal(padded_w2[:hidden_features, :out_features], w2)
    np.testing.assert_array_equal(padded_b1[:hidden_features], b1)
    np.testing.assert_array_equal(padded_b2[:out_features], b2)
    assert np.count_nonzero(padded_w1[in_features:, :]) == 0
    assert np.count_nonzero(padded_w1[:, hidden_features:]) == 0
    assert np.count_nonzero(padded_w2[hidden_features:, :]) == 0
    assert np.count_nonzero(padded_w2[:, out_features:]) == 0
    assert np.count_nonzero(padded_b1[hidden_features:]) == 0
    assert np.count_nonzero(padded_b2[out_features:]) == 0


def test_torch_helper_matches_numpy_reference() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 6), dtype=np.float32)
    w1 = rng.standard_normal((6, 8), dtype=np.float32)
    b1 = rng.standard_normal((8,), dtype=np.float32)
    w2 = rng.standard_normal((8, 5), dtype=np.float32)
    b2 = rng.standard_normal((5,), dtype=np.float32)

    expected = numpy_mlp_naive(x, w1, b1, w2, b2)
    actual = torch_mlp_fp32(x, w1, b1, w2, b2)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_int32_helpers_match_strict_reference() -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(-4, 4, size=(4, 6), dtype=np.int32)
    w1 = rng.integers(-4, 4, size=(6, 8), dtype=np.int32)
    b1 = rng.integers(-16, 16, size=(8,), dtype=np.int32)
    w2 = rng.integers(-4, 4, size=(8, 5), dtype=np.int32)
    b2 = rng.integers(-16, 16, size=(5,), dtype=np.int32)

    _validate_int32_mlp_contract(x, w1, b1, w2, b2)
    expected = reference_mlp_int32(x, w1, b1, w2, b2)
    actual = numpy_mlp_int32(x, w1, b1, w2, b2)

    np.testing.assert_array_equal(actual, expected)


def test_int32_contract_rejects_hidden_values_that_exceed_smul24_range() -> None:
    x = np.full((1, 1024), 256, dtype=np.int32)
    w1 = np.full((1024, 1024), 256, dtype=np.int32)
    b1 = np.zeros((1024,), dtype=np.int32)
    w2 = np.ones((1024, 32), dtype=np.int32)
    b2 = np.zeros((32,), dtype=np.int32)

    try:
        _validate_int32_mlp_contract(x, w1, b1, w2, b2)
    except ValueError as exc:
        assert "hidden activations" in str(exc)
    else:
        raise AssertionError("expected the int32 hidden-range contract to fail")


def test_torch_int32_helper_matches_reference_when_available() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(4)
    x = rng.integers(-4, 4, size=(3, 5), dtype=np.int32)
    w1 = rng.integers(-4, 4, size=(5, 7), dtype=np.int32)
    b1 = rng.integers(-8, 8, size=(7,), dtype=np.int32)
    w2 = rng.integers(-4, 4, size=(7, 6), dtype=np.int32)
    b2 = rng.integers(-8, 8, size=(6,), dtype=np.int32)

    _validate_int32_mlp_contract(x, w1, b1, w2, b2)
    expected = reference_mlp_int32(x, w1, b1, w2, b2)
    actual = torch_mlp_int32(x, w1, b1, w2, b2)

    np.testing.assert_array_equal(actual, expected)
