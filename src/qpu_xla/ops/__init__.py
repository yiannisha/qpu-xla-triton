"""Supported tensor operators and their stable host-side contracts."""

from qpu_xla.ops.attention import attention_fp32, attention_int32
from qpu_xla.ops.attention_plan import AttentionInt32Plan
from qpu_xla.ops.conv2d import conv2d_fp32, conv2d_int32
from qpu_xla.ops.conv2d_plan import Conv2dInt32Plan
from qpu_xla.ops.elementwise import copy, maximum, minimum
from qpu_xla.ops.matmul import calibrate_matmul, hybrid_matmul, matmul, plan_matmul
from qpu_xla.ops.mlp import mlp_int32
from qpu_xla.ops.mlp_plan import MlpInt32Plan
from qpu_xla.ops.pool2d import pool2d_fp32, pool2d_int32
from qpu_xla.ops.sdpa import scaled_dot_product_attention_fp32

__all__ = [
    "attention_int32",
    "attention_fp32",
    "AttentionInt32Plan",
    "calibrate_matmul",
    "conv2d_int32",
    "conv2d_fp32",
    "Conv2dInt32Plan",
    "copy",
    "hybrid_matmul",
    "matmul",
    "maximum",
    "minimum",
    "MlpInt32Plan",
    "mlp_int32",
    "plan_matmul",
    "pool2d_int32",
    "pool2d_fp32",
    "scaled_dot_product_attention_fp32",
]
