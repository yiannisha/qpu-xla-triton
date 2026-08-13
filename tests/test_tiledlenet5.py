import importlib.util
import sys
from pathlib import Path

import numpy as np

LENET5_PATH = Path(__file__).resolve().parents[1] / "examples" / "tiledlenet5.py"
LENET5_SPEC = importlib.util.spec_from_file_location("tiledlenet5", LENET5_PATH)
assert LENET5_SPEC is not None
assert LENET5_SPEC.loader is not None
LENET5_MODULE = importlib.util.module_from_spec(LENET5_SPEC)
sys.modules[LENET5_SPEC.name] = LENET5_MODULE
LENET5_SPEC.loader.exec_module(LENET5_MODULE)

CONV2D_PATH = Path(__file__).resolve().parents[1] / "examples" / "tiledconv2d.py"
CONV2D_SPEC = importlib.util.spec_from_file_location("tiledconv2d_for_lenet5_tests", CONV2D_PATH)
assert CONV2D_SPEC is not None
assert CONV2D_SPEC.loader is not None
CONV2D_MODULE = importlib.util.module_from_spec(CONV2D_SPEC)
sys.modules[CONV2D_SPEC.name] = CONV2D_MODULE
CONV2D_SPEC.loader.exec_module(CONV2D_MODULE)

build_matrix_conv_lowering_meta = LENET5_MODULE.build_matrix_conv_lowering_meta
build_matrix_pool_meta = LENET5_MODULE.build_matrix_pool_meta
build_nchw_conv_lowering_meta = LENET5_MODULE.build_nchw_conv_lowering_meta
int32_stage_bounds = LENET5_MODULE.int32_stage_bounds
make_lenet5_problem_fp32 = LENET5_MODULE.make_lenet5_problem_fp32
make_lenet5_problem_int32 = LENET5_MODULE.make_lenet5_problem_int32
make_torch_runner_fp32 = LENET5_MODULE.make_torch_runner_fp32
make_torch_runner_int32 = LENET5_MODULE.make_torch_runner_int32
numpy_avgpool2d_int64 = LENET5_MODULE.numpy_avgpool2d_int64
reference_lenet5_fp32 = LENET5_MODULE.reference_lenet5_fp32
reference_lenet5_int32 = LENET5_MODULE.reference_lenet5_int32
torch = LENET5_MODULE.torch
im2col_nchw = CONV2D_MODULE.im2col_nchw


def _gather_words(source: np.ndarray, meta: np.ndarray, *, base_addr: int) -> np.ndarray:
    flat = source.reshape(-1)
    out = np.zeros(meta.size, dtype=source.dtype)
    for idx, addr in enumerate(meta.reshape(-1)):
        if int(addr) == 0:
            out[idx] = 0
            continue
        flat_idx = (int(addr) - base_addr) // source.itemsize
        out[idx] = flat[flat_idx]
    return out


def test_nchw_lowering_meta_matches_im2col_order() -> None:
    x = np.arange(1 * 2 * 6 * 5, dtype=np.int32).reshape(1, 2, 6, 5)
    q_actual = 2 * 3 * 2
    q_padded = 16
    base_addr = 4
    meta = build_nchw_conv_lowering_meta(
        base_addr=base_addr,
        batch_stride=int(x.strides[0]),
        channel_stride=int(x.strides[1]),
        row_stride=int(x.strides[2]),
        col_stride=int(x.strides[3]),
        batch=1,
        in_channels=2,
        in_height=6,
        in_width=5,
        kernel_height=3,
        kernel_width=2,
        q_padded=q_padded,
        zero_addr=0,
    )

    lowered = _gather_words(x, meta.reshape(-1), base_addr=base_addr).reshape(-1, q_padded)
    expected = im2col_nchw(x, (3, 2))

    np.testing.assert_array_equal(lowered[:, :q_actual], expected)
    assert np.count_nonzero(lowered[:, q_actual:]) == 0


def test_matrix_lowering_meta_matches_nchw_im2col_after_nhwc_relayout() -> None:
    x = np.arange(2 * 3 * 6 * 5, dtype=np.int32).reshape(2, 3, 6, 5)
    matrix = np.ascontiguousarray(x.transpose(0, 2, 3, 1).reshape(-1, 3))
    q_actual = 3 * 3 * 2
    q_padded = 20
    base_addr = 8
    meta = build_matrix_conv_lowering_meta(
        base_addr=base_addr,
        row_stride=int(matrix.strides[0]),
        itemsize=matrix.itemsize,
        batch=2,
        in_channels=3,
        in_height=6,
        in_width=5,
        kernel_height=3,
        kernel_width=2,
        q_padded=q_padded,
        zero_addr=0,
    )

    lowered = _gather_words(matrix, meta.reshape(-1), base_addr=base_addr).reshape(-1, q_padded)
    expected = im2col_nchw(x, (3, 2))

    np.testing.assert_array_equal(lowered[:, :q_actual], expected)
    assert np.count_nonzero(lowered[:, q_actual:]) == 0


def test_matrix_pool_meta_matches_avgpool_in_matrix_layout() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(-9, 10, size=(2, 3, 4, 6), dtype=np.int32)
    matrix = np.zeros((2 * 4 * 6, 4), dtype=np.int32)
    matrix[:, :3] = x.transpose(0, 2, 3, 1).reshape(-1, 3)

    base_addr = 16
    meta, x_stride, y_stride = build_matrix_pool_meta(
        base_addr=base_addr,
        row_stride=int(matrix.strides[0]),
        itemsize=matrix.itemsize,
        batch=2,
        in_height=4,
        in_width=6,
        channels_actual=3,
        channels_padded=4,
        zero_addr=0,
    )

    flat = matrix.reshape(-1)
    pooled = np.zeros(meta.size, dtype=np.int32)
    x_stride_words = x_stride // matrix.itemsize
    y_stride_words = y_stride // matrix.itemsize
    for idx, addr in enumerate(meta.reshape(-1)):
        if int(addr) == 0:
            pooled[idx] = 0
            continue
        base_idx = (int(addr) - base_addr) // matrix.itemsize
        vals = np.array(
            [
                flat[base_idx],
                flat[base_idx + x_stride_words],
                flat[base_idx + y_stride_words],
                flat[base_idx + y_stride_words + x_stride_words],
            ],
            dtype=np.int64,
        )
        pooled[idx] = int(numpy_avgpool2d_int64(vals.reshape(1, 1, 2, 2))[0, 0, 0, 0])

    expected_nchw = numpy_avgpool2d_int64(x.astype(np.int64))
    expected_matrix = np.zeros((2 * 2 * 3, 4), dtype=np.int32)
    expected_matrix[:, :3] = expected_nchw.astype(np.int32).transpose(0, 2, 3, 1).reshape(-1, 3)

    np.testing.assert_array_equal(pooled.reshape(expected_matrix.shape), expected_matrix)


def test_reference_lenet5_fp32_matches_torch() -> None:
    if torch is None:
        return

    x, weights = make_lenet5_problem_fp32(batch=2, seed=0)
    expected = reference_lenet5_fp32(x, weights).output
    actual = make_torch_runner_fp32(x, weights)().cpu().numpy()

    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)


def test_reference_lenet5_int32_matches_torch_and_contract() -> None:
    if torch is None:
        return

    x, weights, reference = make_lenet5_problem_int32(batch=2, seed=0)
    actual = make_torch_runner_int32(x, weights)().cpu().numpy()

    np.testing.assert_array_equal(actual, reference.output)
    bounds = int32_stage_bounds(x, reference)
    assert all(value < (1 << 23) for value in bounds.values())
