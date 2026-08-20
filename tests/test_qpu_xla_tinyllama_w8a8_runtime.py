from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.benchmark import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PartitionEvidence,
    PerformanceEvidence,
)
from qpu_xla.models.tinyllama import (
    GreedyVocabularyTokenizer,
    TinyLlamaCheckpoint,
    TinyLlamaConfig,
    TinyLlamaGreedyGenerator,
    TinyLlamaW8A8Runtime,
    quantize_per_output_channel_int8,
)
from qpu_xla.models.tinyllama.reference import _causal_attention, _rms_norm, _rope
from qpu_xla.scheduler import Placement


def _checkpoint() -> TinyLlamaCheckpoint:
    config = TinyLlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    rng = np.random.default_rng(2219)
    tensors = {
        name: np.ascontiguousarray(rng.standard_normal(shape, dtype=np.float32) * np.float32(0.08))
        for name, shape in config.expected_shapes().items()
    }
    return TinyLlamaCheckpoint(config, tensors)


def _linear(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    quantized_weight = quantize_per_output_channel_int8(weight)
    scales = np.maximum(
        np.max(np.abs(values), axis=1) / np.float32(127),
        np.float32(1 / 127),
    )
    quantized_values = np.rint(values / scales[:, None]).clip(-127, 127).astype(np.int32)
    accumulation = quantized_values @ quantized_weight.values.astype(np.int32).T
    return accumulation.astype(np.float32) * scales[:, None] * quantized_weight.scales[None, :]


def _reference(checkpoint: TinyLlamaCheckpoint, tokens: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    config = checkpoint.config
    tensors = checkpoint.tensors
    hidden = np.array(tensors["model.embed_tokens.weight"][tokens], dtype=np.float32, copy=True)
    positions = np.arange(tokens.size, dtype=np.int32)
    fixtures = []
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        normalized = _rms_norm(hidden, tensors[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
        query = _linear(normalized, tensors[f"{prefix}.self_attn.q_proj.weight"])
        key = _linear(normalized, tensors[f"{prefix}.self_attn.k_proj.weight"])
        value = _linear(normalized, tensors[f"{prefix}.self_attn.v_proj.weight"])
        query = _rope(
            query.reshape(tokens.size, config.num_attention_heads, config.head_dim),
            positions,
            config.rope_theta,
        )
        key = _rope(
            key.reshape(tokens.size, config.num_key_value_heads, config.head_dim),
            positions,
            config.rope_theta,
        )
        attention = _causal_attention(
            query,
            key,
            value.reshape(tokens.size, config.num_key_value_heads, config.head_dim),
        )
        projected = _linear(
            attention.reshape(tokens.size, config.hidden_size),
            tensors[f"{prefix}.self_attn.o_proj.weight"],
        )
        residual = hidden + projected
        normalized = _rms_norm(
            residual,
            tensors[f"{prefix}.post_attention_layernorm.weight"],
            config.rms_norm_eps,
        )
        gate = _linear(normalized, tensors[f"{prefix}.mlp.gate_proj.weight"])
        up = _linear(normalized, tensors[f"{prefix}.mlp.up_proj.weight"])
        activated = (gate / (np.float32(1) + np.exp(-gate))) * up
        hidden = residual + _linear(activated, tensors[f"{prefix}.mlp.down_proj.weight"])
        fixtures.append(np.array(hidden, copy=True))
    normalized = _rms_norm(hidden, tensors["model.norm.weight"], config.rms_norm_eps)
    return _linear(normalized, tensors["lm_head.weight"]), tuple(fixtures)


def _candidates() -> CandidateRegistry:
    records = []
    for index, shape in enumerate(("16x16x16", "16x16x8", "16x16x32", "16x32x16")):
        records.append(
            CandidateRecord(
                f"test-w8a8-{index}",
                "linear",
                "w8a8-i32-fp32",
                "row-major-packed-k4",
                shape,
                "test",
                CandidateStatus.SUPPORTED_WIN,
                CorrectnessEvidence("numpy", 1, True, 0.0, 0.0, 0.0, 0.0),
                PerformanceEvidence("numpy", (0.002,), (0.001,)),
            )
        )
    return CandidateRegistry(tuple(records))


def _hybrid_candidates() -> CandidateRegistry:
    records = []
    for index, shape in enumerate(("32x16x16", "32x16x8", "32x16x32", "32x32x16")):
        records.append(
            CandidateRecord(
                f"test-w8a8-hybrid-{index}",
                "linear",
                "w8a8-i32-fp32",
                "row-major-packed-k4",
                shape,
                "test",
                CandidateStatus.SUPPORTED_WIN,
                CorrectnessEvidence("numpy", 1, True, 0.0, 0.0, 0.0, 0.0),
                PerformanceEvidence("numpy-fp32", (0.002,), (0.001,)),
                placement="hybrid",
                partition=PartitionEvidence("rows", 16, 32, 16),
            )
        )
    return CandidateRegistry(tuple(records))


def test_tinyllama_w8a8_runtime_matches_independent_quantized_reference() -> None:
    checkpoint = _checkpoint()
    tokens = np.arange(16, dtype=np.int32)
    expected_logits, expected_fixtures = _reference(checkpoint, tokens)
    with (
        Device.fake() as device,
        TinyLlamaW8A8Runtime(
            device,
            checkpoint,
            max_batch=16,
            candidates=_candidates(),
        ) as runtime,
    ):
        actual = runtime.forward(tokens)

    np.testing.assert_allclose(actual.logits, expected_logits, atol=1e-6, rtol=1e-6)
    for fixture, expected in zip(actual.hidden_states, expected_fixtures, strict=True):
        np.testing.assert_allclose(fixture, expected, atol=1e-6, rtol=1e-6)


def test_tinyllama_w8a8_runtime_does_not_prepare_unpromoted_projection_weights() -> None:
    checkpoint = _checkpoint()
    with (
        Device.fake() as device,
        TinyLlamaW8A8Runtime(
            device,
            checkpoint,
            max_batch=16,
            candidates=CandidateRegistry(),
        ) as runtime,
    ):
        assert runtime._plans == {}


def test_tinyllama_w8a8_session_prefill_and_decode_match_full_causal_forward() -> None:
    checkpoint = _checkpoint()
    tokens = np.array([2, 5, 7, 11], dtype=np.int32)
    with (
        Device.fake() as device,
        TinyLlamaW8A8Runtime(
            device,
            checkpoint,
            max_batch=16,
            candidates=_candidates(),
        ) as runtime,
        runtime.session() as session,
    ):
        prefill_logits = session.prefill(tokens)
        next_logits = session.decode(3)
        expected_prefill = runtime.forward(tokens).logits
        expected_next = runtime.forward(np.concatenate((tokens, np.array([3], dtype=np.int32)))).logits[-1]
        assert session.length == 5
        session.reset()
        assert session.length == 0

    np.testing.assert_allclose(prefill_logits, expected_prefill, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(next_logits, expected_next, atol=1e-6, rtol=1e-6)


def test_greedy_generator_accepts_the_incremental_w8a8_runtime() -> None:
    checkpoint = _checkpoint()
    vocabulary = {"<unk>": 0} | {f"t{index}": index for index in range(1, checkpoint.config.vocab_size)}
    tokenizer = GreedyVocabularyTokenizer(vocabulary)
    with (
        Device.fake() as device,
        TinyLlamaW8A8Runtime(
            device,
            checkpoint,
            max_batch=16,
            candidates=_candidates(),
        ) as runtime,
    ):
        generator = TinyLlamaGreedyGenerator(runtime, tokenizer)
        generated = generator.generate_tokens(np.array([2, 5, 7], dtype=np.int32), max_new_tokens=2)

    assert generated.shape == (5,)
    np.testing.assert_array_equal(generated[:3], [2, 5, 7])


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_tinyllama_w8a8_runtime_dispatches_supported_projections_to_qpu() -> None:
    checkpoint = _checkpoint()
    tokens = np.arange(16, dtype=np.int32)
    with (
        Device.fake() as cpu_device,
        TinyLlamaW8A8Runtime(
            cpu_device,
            checkpoint,
            max_batch=16,
            candidates=_candidates(),
        ) as cpu_runtime,
    ):
        expected = cpu_runtime.forward(tokens)
    with (
        Device.open(data_area_size=16 * 1024 * 1024) as qpu_device,
        TinyLlamaW8A8Runtime(
            qpu_device,
            checkpoint,
            max_batch=16,
            candidates=_candidates(),
        ) as qpu_runtime,
    ):
        actual = qpu_runtime.forward(tokens)
        trace = qpu_runtime.queue.chrome_trace()["traceEvents"]

    np.testing.assert_allclose(actual.logits, expected.logits, atol=1e-6, rtol=1e-6)
    assert any(event.get("name") == "vc7.tiled_w8a8_gemm_dequantize" for event in trace)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_tinyllama_w8a8_runtime_executes_calibrated_hybrid_rows() -> None:
    checkpoint = _checkpoint()
    tokens = np.arange(32, dtype=np.int32)
    expected_logits, _ = _reference(checkpoint, tokens)
    with (
        Device.open(data_area_size=16 * 1024 * 1024) as qpu_device,
        TinyLlamaW8A8Runtime(
            qpu_device,
            checkpoint,
            max_batch=32,
            candidates=_hybrid_candidates(),
            placement=Placement.HYBRID,
        ) as runtime,
    ):
        actual = runtime.forward(tokens)
        cpu_trace = runtime.cpu_queue.chrome_trace()["traceEvents"]

    np.testing.assert_allclose(actual.logits, expected_logits, atol=1e-6, rtol=1e-6)
    assert any(event.get("name") == "w8a8_linear.hybrid_join" for event in cpu_trace)
