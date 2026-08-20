"""TinyLlama-oriented runtime building blocks with explicit experimental scope."""

from qpu_xla.models.tinyllama.accelerated import TinyLlamaW8A8Runtime, TinyLlamaW8A8Session
from qpu_xla.models.tinyllama.artifact import TinyLlamaArtifact
from qpu_xla.models.tinyllama.checkpoint import TinyLlamaCheckpoint, TinyLlamaConfig
from qpu_xla.models.tinyllama.functional import (
    embedding_lookup_fp32,
    greedy_sample_fp32,
    residual_add_fp32,
    rms_norm_fp32,
    rope_fp32,
    silu_gated_fp32,
)
from qpu_xla.models.tinyllama.generation import (
    GreedyVocabularyTokenizer,
    SentencePieceTokenizer,
    TinyLlamaGreedyGenerator,
    TinyLlamaTokenizer,
)
from qpu_xla.models.tinyllama.kv_cache import KvCacheFp32
from qpu_xla.models.tinyllama.quantization import (
    CalibratedW8A8Linear,
    PreparedW8A8Linear,
    QuantizedMatrixInt8,
    quantize_per_output_channel_int8,
    quantized_linear_fp32,
    quantized_linear_int8_gemm_fp32,
)
from qpu_xla.models.tinyllama.reference import (
    TinyLlamaForwardResult,
    TinyLlamaReferenceRuntime,
    TinyLlamaReferenceSession,
)

__all__ = [
    "CalibratedW8A8Linear",
    "KvCacheFp32",
    "GreedyVocabularyTokenizer",
    "QuantizedMatrixInt8",
    "PreparedW8A8Linear",
    "TinyLlamaCheckpoint",
    "TinyLlamaArtifact",
    "TinyLlamaConfig",
    "TinyLlamaForwardResult",
    "TinyLlamaGreedyGenerator",
    "TinyLlamaTokenizer",
    "TinyLlamaW8A8Runtime",
    "TinyLlamaW8A8Session",
    "SentencePieceTokenizer",
    "TinyLlamaReferenceRuntime",
    "TinyLlamaReferenceSession",
    "embedding_lookup_fp32",
    "greedy_sample_fp32",
    "quantize_per_output_channel_int8",
    "quantized_linear_int8_gemm_fp32",
    "quantized_linear_fp32",
    "rms_norm_fp32",
    "rope_fp32",
    "residual_add_fp32",
    "silu_gated_fp32",
]
