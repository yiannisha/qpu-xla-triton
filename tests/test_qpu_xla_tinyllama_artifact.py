from __future__ import annotations

import json
import sys
from dataclasses import asdict
from types import ModuleType

import numpy as np

from qpu_xla.models.tinyllama import TinyLlamaArtifact, TinyLlamaConfig


def _config() -> TinyLlamaConfig:
    return TinyLlamaConfig(
        vocab_size=5,
        hidden_size=4,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )


def test_tinyllama_artifact_loader_wires_huggingface_config_weights_and_tokenizer(monkeypatch, tmp_path) -> None:
    config = _config()
    values = {name: np.zeros(shape, dtype=np.float32) for name, shape in config.expected_shapes().items()}
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    package = ModuleType("safetensors")
    numpy_module = ModuleType("safetensors.numpy")
    setattr(numpy_module, "load_file", lambda _: values)
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.numpy", numpy_module)

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
            return [3]

        def decode(self, token_ids: list[int]) -> str:
            return "ok"

    sentencepiece = ModuleType("sentencepiece")
    setattr(sentencepiece, "SentencePieceProcessor", FakeProcessor)
    monkeypatch.setitem(sys.modules, "sentencepiece", sentencepiece)

    artifact = TinyLlamaArtifact.from_huggingface_directory(tmp_path)

    assert artifact.config == config
    assert artifact.tokenizer.vocab_size == 5
    assert artifact.generator().generate_text("prompt", max_new_tokens=0) == "ok"
