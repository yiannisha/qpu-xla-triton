"""Supported tensor operators and their stable host-side contracts."""

from qpu_xla.ops.attention import attention_fp32, attention_int32
from qpu_xla.ops.attention_plan import AttentionInt32Plan
from qpu_xla.ops.bias_activation import bias_activation
from qpu_xla.ops.conv2d import conv2d_fp32, conv2d_int32
from qpu_xla.ops.conv2d_plan import Conv2dInt32Plan
from qpu_xla.ops.conv2d_w8a8 import Conv2dW8A8Plan
from qpu_xla.ops.elementwise import copy, maximum, minimum
from qpu_xla.ops.embedding import embedding_lookup_fp32
from qpu_xla.ops.linear_fp32 import PreparedFP32Linear
from qpu_xla.ops.matmul import (
    calibrate_matmul,
    hybrid_column_partitions,
    hybrid_matmul,
    hybrid_row_partitions,
    matmul,
    plan_matmul,
)
from qpu_xla.ops.mlp import mlp_int32
from qpu_xla.ops.mlp_fp32 import mlp_fp32
from qpu_xla.ops.mlp_plan import MlpInt32Plan
from qpu_xla.ops.pool2d import PreparedPool2DFP32, pool2d_fp32, pool2d_int32
from qpu_xla.ops.residual import residual_add_fp32
from qpu_xla.ops.rope import apply_rope_tables_fp32, rope_tables_fp32
from qpu_xla.ops.sampling import greedy_sample_fp32
from qpu_xla.ops.sdpa import scaled_dot_product_attention_fp32
from qpu_xla.ops.softmax import softmax_fp32
from qpu_xla.ops.swiglu import swiglu_fp32

__all__ = [
    "attention_int32",
    "attention_fp32",
    "bias_activation",
    "AttentionInt32Plan",
    "calibrate_matmul",
    "conv2d_int32",
    "conv2d_fp32",
    "Conv2dInt32Plan",
    "Conv2dW8A8Plan",
    "copy",
    "embedding_lookup_fp32",
    "hybrid_matmul",
    "hybrid_column_partitions",
    "hybrid_row_partitions",
    "matmul",
    "maximum",
    "minimum",
    "MlpInt32Plan",
    "PreparedFP32Linear",
    "PreparedPool2DFP32",
    "mlp_int32",
    "mlp_fp32",
    "plan_matmul",
    "pool2d_int32",
    "pool2d_fp32",
    "scaled_dot_product_attention_fp32",
    "greedy_sample_fp32",
    "swiglu_fp32",
    "apply_rope_tables_fp32",
    "rope_tables_fp32",
    "residual_add_fp32",
    "softmax_fp32",
]
