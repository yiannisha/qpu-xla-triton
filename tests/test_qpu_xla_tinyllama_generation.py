from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from qpu_xla.models.tinyllama import (
    GreedyVocabularyTokenizer,
    SentencePieceTokenizer,
    TinyLlamaCheckpoint,
    TinyLlamaConfig,
    TinyLlamaGreedyGenerator,
    TinyLlamaReferenceRuntime,
)


def _generator() -> TinyLlamaGreedyGenerator:
    config = TinyLlamaConfig(
        vocab_size=5,
        hidden_size=4,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )
    tensors = {name: np.zeros(shape, dtype=np.float32) for name, shape in config.expected_shapes().items()}
    tokenizer = GreedyVocabularyTokenizer({"<unk>": 0, "a": 1, "b": 2, "ab": 3, "z": 4})
    return TinyLlamaGreedyGenerator(TinyLlamaReferenceRuntime(TinyLlamaCheckpoint(config, tensors)), tokenizer)


def test_greedy_vocabulary_prefers_longest_token_and_round_trips_json(tmp_path) -> None:
    tokenizer = _generator().tokenizer
    assert tokenizer.encode("abx").tolist() == [3, 0]
    assert tokenizer.decode(np.array([3, 0], dtype=np.int32)) == "ab<unk>"
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"token_to_id": tokenizer.token_to_id}), encoding="utf-8")
    assert GreedyVocabularyTokenizer.from_json(path) == tokenizer


def test_tinyllama_greedy_generator_runs_end_to_end_with_eos_and_context_validation() -> None:
    generator = _generator()
    tokens = generator.generate_tokens(np.array([1], dtype=np.int32), max_new_tokens=4, eos_token_id=0)

    assert tokens.tolist() == [1, 0]
    assert generator.generate_text("a", max_new_tokens=1, eos_token_id=0) == "a<unk>"
    with pytest.raises(ValueError, match="max_position"):
        generator.generate_tokens(np.ones((8,), dtype=np.int32), max_new_tokens=1)


def test_vocabulary_and_generator_reject_incompatible_contracts() -> None:
    with pytest.raises(ValueError, match="dense"):
        GreedyVocabularyTokenizer({"<unk>": 0, "a": 2})
    with pytest.raises(ValueError, match="vocabulary size"):
        TinyLlamaGreedyGenerator(
            _generator().runtime,
            GreedyVocabularyTokenizer({"<unk>": 0, "a": 1, "b": 2, "z": 3}),
        )


def test_sentencepiece_tokenizer_adapter_uses_native_processor_without_fixture_vocabulary(
    monkeypatch, tmp_path
) -> None:
    class FakeProcessor:
        def __init__(self, *, model_file: str) -> None:
            assert model_file.endswith("tokenizer.model")

        def get_piece_size(self) -> int:
            return 5

        def bos_id(self) -> int:
            return 1

        def eos_id(self) -> int:
            return 2

        def encode(self, text: str, *, out_type: type[int]) -> list[int]:
            assert out_type is int
            return [3] if text == "hello" else []

        def decode(self, token_ids: list[int]) -> str:
            return "hello" if token_ids == [3] else ""

    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=FakeProcessor))
    tokenizer = SentencePieceTokenizer(tmp_path / "tokenizer.model")

    assert tokenizer.vocab_size == 5
    assert (tokenizer.bos_id, tokenizer.eos_id) == (1, 2)
    assert tokenizer.encode("hello").tolist() == [3]
    assert tokenizer.decode(np.array([3], dtype=np.int32)) == "hello"
