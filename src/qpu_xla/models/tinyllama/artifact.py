"""One explicit production-artifact contract for TinyLlama demo execution."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self

from qpu_xla.models.tinyllama.checkpoint import TinyLlamaCheckpoint, TinyLlamaConfig
from qpu_xla.models.tinyllama.generation import SentencePieceTokenizer, TinyLlamaGreedyGenerator
from qpu_xla.models.tinyllama.reference import TinyLlamaReferenceRuntime


@dataclass(frozen=True, slots=True)
class TinyLlamaArtifact:
    """Validated model configuration, weights, and matching production tokenizer."""

    config: TinyLlamaConfig
    checkpoint: TinyLlamaCheckpoint
    tokenizer: SentencePieceTokenizer

    @classmethod
    def from_huggingface_directory(
        cls: type[TinyLlamaArtifact],
        directory: str | PathLike[str],
        *,
        config_filename: str = "config.json",
        checkpoint_filename: str = "model.safetensors",
        tokenizer_filename: str = "tokenizer.model",
    ) -> TinyLlamaArtifact:
        """Load one unsharded Hugging Face-style TinyLlama directory with strict validation."""
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"TinyLlama artifact directory does not exist: {root}")
        config = TinyLlamaConfig.from_huggingface_json(root / config_filename)
        checkpoint_path = root / checkpoint_filename
        index_path = root / "model.safetensors.index.json"
        checkpoint = (
            TinyLlamaCheckpoint.from_safetensors(checkpoint_path, config)
            if checkpoint_path.exists()
            else TinyLlamaCheckpoint.from_safetensors_index(index_path, config)
        )
        tokenizer = SentencePieceTokenizer(root / tokenizer_filename)
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("TinyLlama tokenizer vocabulary size does not match checkpoint config")
        return cls(config, checkpoint, tokenizer)

    def generator(self: Self) -> TinyLlamaGreedyGenerator:
        """Build the cache-backed greedy generation reference for this artifact."""
        return TinyLlamaGreedyGenerator(TinyLlamaReferenceRuntime(self.checkpoint), self.tokenizer)
