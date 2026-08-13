"""QPU-backed reusable kernel specializations."""

from qpu_xla.kernels.copy import WORD_COPY_KERNEL, supports_word_copy
from qpu_xla.kernels.gemm import TILED_INT32_GEMM_KERNEL, supports_tiled_int32_gemm
from qpu_xla.kernels.gemm_fp32 import TILED_FP32_GEMM_KERNEL, supports_tiled_fp32_gemm
from qpu_xla.kernels.minmax import MAXIMUM_WORD_KERNEL, MINIMUM_WORD_KERNEL, supports_word_minmax
from qpu_xla.kernels.pool2d import (
    AVGPOOL2D_FP32_KERNEL,
    AVGPOOL2D_INT32_KERNEL,
    MAXPOOL2D_FP32_KERNEL,
    MAXPOOL2D_INT32_KERNEL,
    supports_pool2d_fp32,
    supports_pool2d_int32,
)

__all__ = [
    "MAXIMUM_WORD_KERNEL",
    "MAXPOOL2D_FP32_KERNEL",
    "MAXPOOL2D_INT32_KERNEL",
    "MINIMUM_WORD_KERNEL",
    "TILED_FP32_GEMM_KERNEL",
    "AVGPOOL2D_INT32_KERNEL",
    "AVGPOOL2D_FP32_KERNEL",
    "TILED_INT32_GEMM_KERNEL",
    "WORD_COPY_KERNEL",
    "supports_word_copy",
    "supports_tiled_int32_gemm",
    "supports_pool2d_int32",
    "supports_pool2d_fp32",
    "supports_tiled_fp32_gemm",
    "supports_word_minmax",
]
