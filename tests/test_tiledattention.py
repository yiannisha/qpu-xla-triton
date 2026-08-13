import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "tiledattention.py"
SPEC = importlib.util.spec_from_file_location("tiledattention", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PreparedAttentionProblem = MODULE.PreparedAttentionProblem
TiledAttentionExecutorFP32 = MODULE.TiledAttentionExecutorFP32
numpy_attention_fp32 = MODULE.numpy_attention_fp32
numpy_attention_scores_fp32 = MODULE.numpy_attention_scores_fp32
numpy_attention_int32 = MODULE.numpy_attention_int32
numpy_attention_scores_int32 = MODULE.numpy_attention_scores_int32
numpy_sdpa_fp32 = MODULE.numpy_sdpa_fp32
reference_attention_int32 = MODULE.reference_attention_int32
reference_attention_scores_int32 = MODULE.reference_attention_scores_int32
torch = MODULE.torch
torch_attention_fp32 = MODULE.torch_attention_fp32
torch_attention_int32 = MODULE.torch_attention_int32
torch_attention_scores_fp32 = MODULE.torch_attention_scores_fp32
torch_attention_scores_int32 = MODULE.torch_attention_scores_int32
torch_attention_sdpa_fp32 = MODULE.torch_attention_sdpa_fp32
_validate_int32_attention_contract = MODULE._validate_int32_attention_contract


def test_numpy_attention_fp32_matches_explicit_matmul() -> None:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((7, 5), dtype=np.float32)
    k = rng.standard_normal((9, 5), dtype=np.float32)
    v = rng.standard_normal((9, 6), dtype=np.float32)

    expected = q.dot(k.T).dot(v)
    actual = numpy_attention_fp32(q, k, v)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_numpy_attention_score_stage_fp32_matches_explicit_matmul() -> None:
    rng = np.random.default_rng(5)
    q = rng.standard_normal((6, 4), dtype=np.float32)
    k = rng.standard_normal((8, 4), dtype=np.float32)

    expected = q.dot(k.T)
    actual = numpy_attention_scores_fp32(q, k)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_prepare_problem_padding_preserves_values() -> None:
    rng = np.random.default_rng(1)
    query_len = 19
    key_len = 23
    depth = 10
    value_dim = 18

    q = rng.standard_normal((query_len, depth), dtype=np.float32)
    k = rng.standard_normal((key_len, depth), dtype=np.float32)
    v = rng.standard_normal((key_len, value_dim), dtype=np.float32)

    executor = object.__new__(TiledAttentionExecutorFP32)
    executor.query_len = query_len
    executor.key_len = key_len
    executor.depth = depth
    executor.value_dim = value_dim
    executor.query_padded = 32
    executor.key_padded = 32
    executor.depth_padded = 12
    executor.value_padded = 32
    executor._q_dev = np.empty((32, 12), dtype=np.float32)
    executor._k_t_dev = np.empty((12, 32), dtype=np.float32)
    executor._v_dev = np.empty((32, 32), dtype=np.float32)

    prepared = TiledAttentionExecutorFP32.prepare_problem(executor, q, k, v)

    assert isinstance(prepared, PreparedAttentionProblem)
    assert prepared.q.shape == (32, 12)
    assert prepared.k_t.shape == (12, 32)
    assert prepared.v.shape == (32, 32)

    np.testing.assert_array_equal(prepared.q[:query_len, :depth], q)
    np.testing.assert_array_equal(prepared.k_t[:depth, :key_len], k.T)
    np.testing.assert_array_equal(prepared.v[:key_len, :value_dim], v)
    assert np.count_nonzero(prepared.q[query_len:, :]) == 0
    assert np.count_nonzero(prepared.q[:, depth:]) == 0
    assert np.count_nonzero(prepared.k_t[depth:, :]) == 0
    assert np.count_nonzero(prepared.k_t[:, key_len:]) == 0
    assert np.count_nonzero(prepared.v[key_len:, :]) == 0
    assert np.count_nonzero(prepared.v[:, value_dim:]) == 0


def test_torch_attention_fp32_matches_numpy_reference() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(2)
    q = rng.standard_normal((8, 6), dtype=np.float32)
    k = rng.standard_normal((10, 6), dtype=np.float32)
    v = rng.standard_normal((10, 7), dtype=np.float32)

    expected = numpy_attention_fp32(q, k, v)
    actual = torch_attention_fp32(q, k, v)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_torch_attention_score_stage_fp32_matches_numpy_reference() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(6)
    q = rng.standard_normal((7, 5), dtype=np.float32)
    k = rng.standard_normal((9, 5), dtype=np.float32)

    expected = numpy_attention_scores_fp32(q, k)
    actual = torch_attention_scores_fp32(q, k)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_torch_sdpa_fp32_matches_numpy_reference_when_available() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(7)
    q = rng.standard_normal((5, 4), dtype=np.float32)
    k = rng.standard_normal((6, 4), dtype=np.float32)
    v = rng.standard_normal((6, 3), dtype=np.float32)

    try:
        actual = torch_attention_sdpa_fp32(q, k, v)
    except RuntimeError:
        return

    expected = numpy_sdpa_fp32(q, k, v)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_int32_helpers_match_strict_reference() -> None:
    rng = np.random.default_rng(3)
    q = rng.integers(-4, 4, size=(5, 7), dtype=np.int32)
    k = rng.integers(-4, 4, size=(9, 7), dtype=np.int32)
    v = rng.integers(-4, 4, size=(9, 6), dtype=np.int32)

    _validate_int32_attention_contract(q, k, v)
    expected = reference_attention_int32(q, k, v)
    actual = numpy_attention_int32(q, k, v)

    np.testing.assert_array_equal(actual, expected)


def test_int32_score_stage_matches_strict_reference() -> None:
    rng = np.random.default_rng(8)
    q = rng.integers(-4, 4, size=(4, 6), dtype=np.int32)
    k = rng.integers(-4, 4, size=(7, 6), dtype=np.int32)

    expected = reference_attention_scores_int32(q, k)
    actual = numpy_attention_scores_int32(q, k)

    np.testing.assert_array_equal(actual, expected)


def test_int32_contract_rejects_scores_that_exceed_smul24_range() -> None:
    q = np.full((1, 1024), 4096, dtype=np.int32)
    k = np.full((32, 1024), 4096, dtype=np.int32)
    v = np.ones((32, 16), dtype=np.int32)

    try:
        _validate_int32_attention_contract(q, k, v)
    except ValueError as exc:
        assert "score matrix" in str(exc)
    else:
        raise AssertionError("expected the int32 score-range contract to fail")


def test_torch_int32_helper_matches_reference_when_available() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(4)
    q = rng.integers(-4, 4, size=(4, 5), dtype=np.int32)
    k = rng.integers(-4, 4, size=(7, 5), dtype=np.int32)
    v = rng.integers(-4, 4, size=(7, 6), dtype=np.int32)

    _validate_int32_attention_contract(q, k, v)
    expected = reference_attention_int32(q, k, v)
    actual = torch_attention_int32(q, k, v)

    np.testing.assert_array_equal(actual, expected)


def test_torch_int32_score_stage_matches_reference_when_available() -> None:
    if torch is None:
        return

    rng = np.random.default_rng(9)
    q = rng.integers(-4, 4, size=(3, 5), dtype=np.int32)
    k = rng.integers(-4, 4, size=(6, 5), dtype=np.int32)

    expected = reference_attention_scores_int32(q, k)
    actual = torch_attention_scores_int32(q, k)

    np.testing.assert_array_equal(actual, expected)
