import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "tiledconv2d.py"
SPEC = importlib.util.spec_from_file_location("tiledconv2d", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

numpy_conv2d_nchw = MODULE.numpy_conv2d_nchw
_pad_matrix_rows_cols = MODULE._pad_matrix_rows_cols
_prepare_conv_problem = MODULE._prepare_conv_problem
_reshape_gemm_output = MODULE._reshape_gemm_output
_reshape_weight_oihw_to_gemm = MODULE._reshape_weight_oihw_to_gemm
im2col_nchw = MODULE.im2col_nchw
pack_int16_pairs = MODULE.pack_int16_pairs
reference_conv2d_nchw = MODULE.reference_conv2d_nchw
torch_conv2d_nchw = MODULE.torch_conv2d_nchw
tiledconv2d_int16 = MODULE.tiledconv2d_int16
torch = MODULE.torch


def _unpack_int16_pairs(packed: np.ndarray) -> np.ndarray:
    words = packed.astype(np.uint32, copy=False)
    lo = (words & 0xFFFF).astype(np.uint16, copy=False).view(np.int16)
    hi = (words >> 16).astype(np.uint16, copy=False).view(np.int16)

    unpacked = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int16)
    unpacked[:, 0::2] = lo
    unpacked[:, 1::2] = hi
    return unpacked


def test_im2col_lowering_matches_reference_conv2d() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 7, 6), dtype=np.float32)
    weight = rng.standard_normal((5, 3, 3, 2), dtype=np.float32)

    cols = im2col_nchw(x, (3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 2))
    gemm_weight = _reshape_weight_oihw_to_gemm(weight)
    reference = reference_conv2d_nchw(x, weight, stride=(2, 1), padding=(1, 0), dilation=(1, 2))

    lowered = cols.dot(gemm_weight)
    actual = lowered.reshape(2, 4, 4, 5).transpose(0, 3, 1, 2)

    np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)


def test_pack_int16_pairs_round_trips_exactly() -> None:
    rng = np.random.default_rng(1)
    original = rng.integers(-512, 512, size=(7, 10), dtype=np.int16)

    packed = pack_int16_pairs(original)
    unpacked = _unpack_int16_pairs(packed)

    np.testing.assert_array_equal(unpacked, original)


def test_prepare_and_pad_conv_problem_preserves_original_submatrices() -> None:
    rng = np.random.default_rng(2)
    x = rng.integers(-8, 8, size=(1, 2, 5, 4), dtype=np.int32)
    weight = rng.integers(-8, 8, size=(7, 2, 3, 3), dtype=np.int32)

    a, b, output_shape = _prepare_conv_problem(x, weight, stride=1, padding=1, dilation=1)
    a_padded, b_padded, p, q, r = _pad_matrix_rows_cols(a, b, p_tile=16, q_tile=4, r_tile=16)

    assert output_shape == (1, 7, 5, 4)
    assert a.shape == (20, 18)
    assert b.shape == (18, 7)
    assert a_padded.shape == (32, 20)
    assert b_padded.shape == (20, 16)

    np.testing.assert_array_equal(a_padded[:p, :q], a)
    np.testing.assert_array_equal(b_padded[:q, :r], b)
    assert np.count_nonzero(a_padded[p:, :]) == 0
    assert np.count_nonzero(a_padded[:, q:]) == 0
    assert np.count_nonzero(b_padded[q:, :]) == 0
    assert np.count_nonzero(b_padded[:, r:]) == 0

    lowered = a.dot(b).reshape(1, 5, 4, 7).transpose(0, 3, 1, 2)
    reshaped = _reshape_gemm_output(a.dot(b), output_shape, p, r)
    np.testing.assert_array_equal(reshaped, lowered)


def test_numpy_and_torch_conv_helpers_match_reference() -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(-16, 16, size=(1, 3, 6, 5), dtype=np.int16)
    weight = rng.integers(-16, 16, size=(4, 3, 3, 2), dtype=np.int16)

    expected = reference_conv2d_nchw(
        x.astype(np.int32),
        weight.astype(np.int32),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 1),
    ).astype(np.int32)
    actual_numpy = numpy_conv2d_nchw(
        x,
        weight,
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 1),
        compute_dtype=np.int32,
        out_dtype=np.int32,
    )

    np.testing.assert_array_equal(actual_numpy, expected)

    if torch is not None:
        actual_torch = torch_conv2d_nchw(
            x.astype(np.int32),
            weight.astype(np.int32),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 1),
        )
        np.testing.assert_array_equal(actual_torch, expected)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_public_int16_qpu_path_matches_int32_accumulation_reference() -> None:
    rng = np.random.default_rng(4)
    x = rng.integers(-32, 32, size=(1, 2, 6, 5), dtype=np.int16)
    weight = rng.integers(-32, 32, size=(4, 2, 3, 3), dtype=np.int16)

    expected = reference_conv2d_nchw(
        x.astype(np.int32),
        weight.astype(np.int32),
        stride=1,
        padding=1,
        dilation=1,
    ).astype(np.int32)
    actual = tiledconv2d_int16(x, weight, stride=1, padding=1, dilation=1)

    np.testing.assert_array_equal(actual, expected)
