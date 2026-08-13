from __future__ import annotations

import numpy as np
import pytest

from qpu_xla.models.tinyllama import TinyLlamaCheckpoint, TinyLlamaConfig, TinyLlamaReferenceRuntime


def _runtime() -> TinyLlamaReferenceRuntime:
    config = TinyLlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    rng = np.random.default_rng(42)
    tensors = {
        name: (rng.standard_normal(shape).astype(np.float32) * np.float32(0.05))
        for name, shape in config.expected_shapes().items()
    }
    return TinyLlamaReferenceRuntime(TinyLlamaCheckpoint(config, tensors))


def test_tinyllama_reference_runtime_produces_causal_logits_and_layer_fixtures() -> None:
    runtime = _runtime()

    prefix = runtime.forward(np.array([2, 5], dtype=np.int32))
    extended = runtime.forward(np.array([2, 5, 7], dtype=np.int32))

    assert prefix.logits.shape == (2, 16)
    assert len(prefix.hidden_states) == 1
    assert prefix.hidden_states[0].shape == (2, 8)
    np.testing.assert_allclose(extended.logits[:2], prefix.logits, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(prefix.logits[0, :4], [0.013911, -0.000036, 0.003242, -0.010931], atol=1e-6)


def test_tinyllama_incremental_session_matches_full_prefill_and_decode() -> None:
    runtime = _runtime()
    tokens = np.array([2, 5, 7], dtype=np.int32)
    session = runtime.session()

    logits = session.prefill(tokens)
    next_logits = session.decode(3)

    np.testing.assert_allclose(logits, runtime.forward(tokens).logits, atol=1e-6, rtol=1e-6)
    expected_next = runtime.forward(np.array([2, 5, 7, 3], dtype=np.int32)).logits[-1]
    np.testing.assert_allclose(next_logits, expected_next, atol=1e-6)
    assert session.length == 4


@pytest.mark.parametrize(
    "tokens, offset, message",
    [
        (np.array([], dtype=np.int32), 0, "non-empty"),
        (np.array([16], dtype=np.int32), 0, "vocabulary"),
        (np.array([1], dtype=np.float32), 0, "integer"),
        (np.array([1], dtype=np.int32), -1, "non-negative"),
    ],
)
def test_tinyllama_reference_runtime_rejects_invalid_tokens_or_positions(
    tokens: np.ndarray, offset: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _runtime().forward(tokens, position_offset=offset)
