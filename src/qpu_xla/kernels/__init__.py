"""QPU-backed reusable kernel specializations."""

from qpu_xla.kernels.argmax import ARGMAX_FP32_KERNEL, supports_argmax_fp32
from qpu_xla.kernels.bias_activation import (
    BIAS_ONLY_FP32_KERNEL,
    BIAS_ONLY_INT32_KERNEL,
    BIAS_RELU_FP32_KERNEL,
    BIAS_RELU_INT32_KERNEL,
    RELU_ONLY_FP32_KERNEL,
    RELU_ONLY_INT32_KERNEL,
    bias_activation_kernel,
    supports_bias_activation,
)
from qpu_xla.kernels.copy import WORD_COPY_KERNEL, supports_word_copy
from qpu_xla.kernels.depthwise_w8a8 import DEPTHWISE_W8A8_3X3_KERNEL
from qpu_xla.kernels.embedding import EMBEDDING_LOOKUP_FP32_KERNEL, supports_embedding_lookup_fp32
from qpu_xla.kernels.gemm import TILED_INT32_GEMM_KERNEL, supports_tiled_int32_gemm
from qpu_xla.kernels.gemm_fp32 import TILED_FP32_GEMM_KERNEL, supports_tiled_fp32_gemm
from qpu_xla.kernels.gemm_int8 import (
    TILED_W8A8_GEMM_DEQUANTIZE_KERNEL,
    TILED_W8A8_GEMM_KERNEL,
    pack_int8_gemm_operands,
    pack_int8_quads,
    supports_tiled_w8a8_gemm,
    supports_tiled_w8a8_gemm_dequantize,
)
from qpu_xla.kernels.gemv_fp32 import FP32_GEMV_KERNEL, supports_fp32_gemv
from qpu_xla.kernels.gemv_int8 import W8A8_GEMV_KERNEL, supports_w8a8_gemv
from qpu_xla.kernels.ggml_flash_attn import (
    GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL,
    GGML_GEMMA_FLASH_ATTN_F16_MX_KERNEL,
    ggml_flash_attn_ext_reference,
    supports_ggml_gemma_flash_attn_f16_m1,
    supports_ggml_gemma_flash_attn_f16_mx,
)
from qpu_xla.kernels.ggml_geglu_q8 import (
    GGML_GEGLU_Q8_0_KERNEL,
    GGML_GEGLU_Q8_0_SPLIT_KERNEL,
    ggml_geglu_q8_0_reference,
    ggml_gelu_fp16_table,
    supports_ggml_geglu_q8_0,
    supports_ggml_geglu_q8_0_split,
)
from qpu_xla.kernels.ggml_q4_0 import (
    GGML_Q4_0_Q8_0_LINEAR_KERNEL,
    GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL,
    ggml_q4_0_q8_0_reference,
    pack_ggml_q4_0_blocks,
    pack_ggml_q4_0_tiled_weights,
    pack_ggml_q8_0_blocks,
    supports_ggml_q4_0_q8_0_linear,
    supports_ggml_q4_0_q8_0_tiled_linear,
    unpack_ggml_q4_0_blocks,
    unpack_ggml_q8_0_blocks,
)
from qpu_xla.kernels.ggml_q4_k import (
    GGML_Q4_K_Q8_K_LINEAR_M4_KERNEL,
    ggml_q4_k_q8_k_reference,
    pack_ggml_q4_k_blocks,
    pack_ggml_q8_k_blocks,
    supports_ggml_q4_k_q8_k_linear_m4,
    unpack_ggml_q4_k_blocks,
    unpack_ggml_q8_k_blocks,
)
from qpu_xla.kernels.ggml_q6_k import (
    GGML_Q6_K_Q8_K_LINEAR_M4_KERNEL,
    ggml_q6_k_q8_k_reference,
    pack_ggml_q6_k_blocks,
    supports_ggml_q6_k_q8_k_linear_m4,
    unpack_ggml_q6_k_blocks,
)
from qpu_xla.kernels.ggml_q8_0 import (
    GGML_Q8_0_Q8_0_LINEAR_M4_KERNEL,
    ggml_q8_0_q8_0_reference,
    supports_ggml_q8_0_q8_0_linear_m4,
)
from qpu_xla.kernels.minmax import MAXIMUM_WORD_KERNEL, MINIMUM_WORD_KERNEL, supports_word_minmax
from qpu_xla.kernels.pool2d import (
    AVGPOOL2D_FP32_KERNEL,
    AVGPOOL2D_INT32_KERNEL,
    MAXPOOL2D_FP32_KERNEL,
    MAXPOOL2D_INT32_KERNEL,
    supports_pool2d_fp32,
    supports_pool2d_int32,
)
from qpu_xla.kernels.residual import RESIDUAL_ADD_FP32_KERNEL, supports_residual_add_fp32
from qpu_xla.kernels.rms_norm import RMS_NORM_FP32_KERNEL, supports_rms_norm_fp32
from qpu_xla.kernels.rope import ROPE_FP32_KERNEL, supports_rope_fp32
from qpu_xla.kernels.softmax import SOFTMAX_FP32_KERNEL, supports_softmax_fp32
from qpu_xla.kernels.swiglu import SWIGLU_FP32_KERNEL, supports_swiglu_fp32
from qpu_xla.kernels.w8a8_epilogue import W8A8_DEQUANTIZE_KERNEL, supports_w8a8_dequantize

__all__ = [
    "MAXIMUM_WORD_KERNEL",
    "ARGMAX_FP32_KERNEL",
    "BIAS_ONLY_FP32_KERNEL",
    "BIAS_ONLY_INT32_KERNEL",
    "BIAS_RELU_FP32_KERNEL",
    "BIAS_RELU_INT32_KERNEL",
    "DEPTHWISE_W8A8_3X3_KERNEL",
    "MAXPOOL2D_FP32_KERNEL",
    "MAXPOOL2D_INT32_KERNEL",
    "MINIMUM_WORD_KERNEL",
    "TILED_FP32_GEMM_KERNEL",
    "FP32_GEMV_KERNEL",
    "GGML_Q4_0_Q8_0_LINEAR_KERNEL",
    "GGML_Q4_K_Q8_K_LINEAR_M4_KERNEL",
    "GGML_Q6_K_Q8_K_LINEAR_M4_KERNEL",
    "GGML_Q8_0_Q8_0_LINEAR_M4_KERNEL",
    "GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL",
    "GGML_GEMMA_FLASH_ATTN_F16_MX_KERNEL",
    "GGML_GEGLU_Q8_0_KERNEL",
    "GGML_GEGLU_Q8_0_SPLIT_KERNEL",
    "GGML_Q4_0_Q8_0_TILED_LINEAR_KERNEL",
    "AVGPOOL2D_INT32_KERNEL",
    "AVGPOOL2D_FP32_KERNEL",
    "TILED_INT32_GEMM_KERNEL",
    "TILED_W8A8_GEMM_DEQUANTIZE_KERNEL",
    "TILED_W8A8_GEMM_KERNEL",
    "W8A8_GEMV_KERNEL",
    "W8A8_DEQUANTIZE_KERNEL",
    "SWIGLU_FP32_KERNEL",
    "WORD_COPY_KERNEL",
    "EMBEDDING_LOOKUP_FP32_KERNEL",
    "RELU_ONLY_FP32_KERNEL",
    "RELU_ONLY_INT32_KERNEL",
    "bias_activation_kernel",
    "supports_bias_activation",
    "supports_argmax_fp32",
    "supports_word_copy",
    "supports_embedding_lookup_fp32",
    "supports_tiled_int32_gemm",
    "supports_pool2d_int32",
    "supports_pool2d_fp32",
    "supports_tiled_fp32_gemm",
    "supports_fp32_gemv",
    "supports_ggml_q4_0_q8_0_linear",
    "supports_ggml_q4_k_q8_k_linear_m4",
    "supports_ggml_q6_k_q8_k_linear_m4",
    "supports_ggml_q8_0_q8_0_linear_m4",
    "supports_tiled_w8a8_gemm",
    "supports_tiled_w8a8_gemm_dequantize",
    "supports_w8a8_gemv",
    "supports_w8a8_dequantize",
    "supports_swiglu_fp32",
    "RMS_NORM_FP32_KERNEL",
    "RESIDUAL_ADD_FP32_KERNEL",
    "ROPE_FP32_KERNEL",
    "SOFTMAX_FP32_KERNEL",
    "supports_rms_norm_fp32",
    "supports_residual_add_fp32",
    "supports_rope_fp32",
    "supports_softmax_fp32",
    "supports_word_minmax",
    "pack_int8_gemm_operands",
    "pack_int8_quads",
    "pack_ggml_q4_0_blocks",
    "pack_ggml_q8_0_blocks",
    "pack_ggml_q4_0_tiled_weights",
    "unpack_ggml_q4_0_blocks",
    "unpack_ggml_q8_0_blocks",
    "ggml_q4_0_q8_0_reference",
    "pack_ggml_q4_k_blocks",
    "pack_ggml_q8_k_blocks",
    "unpack_ggml_q4_k_blocks",
    "unpack_ggml_q8_k_blocks",
    "ggml_q4_k_q8_k_reference",
    "pack_ggml_q6_k_blocks",
    "unpack_ggml_q6_k_blocks",
    "ggml_q6_k_q8_k_reference",
    "ggml_q8_0_q8_0_reference",
    "ggml_flash_attn_ext_reference",
    "ggml_gelu_fp16_table",
    "ggml_geglu_q8_0_reference",
    "supports_ggml_gemma_flash_attn_f16_m1",
    "supports_ggml_gemma_flash_attn_f16_mx",
    "supports_ggml_geglu_q8_0",
    "supports_ggml_geglu_q8_0_split",
    "supports_ggml_q4_0_q8_0_tiled_linear",
]
