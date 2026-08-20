"""Portable vocabulary and deterministic greedy generation for reference demos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

from qpu_xla.models.tinyllama.checkpoint import TinyLlamaCheckpoint


class TinyLlamaDecodeSession(Protocol):
    """Incremental session contract shared by reference and W8A8 runtimes."""

    def prefill(self: Self, token_ids: npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
        """Append a prompt and return one logits row per token."""

    def decode(self: Self, token_id: int) -> npt.NDArray[np.float32]:
        """Append one token and return its logits row."""


class TinyLlamaInferenceRuntime(Protocol):
    """Runtime surface required by deterministic generation."""

    checkpoint: TinyLlamaCheckpoint

    def session(self: Self) -> TinyLlamaDecodeSession:
        """Create one empty incremental session."""


class TinyLlamaTokenizer(Protocol):
    """The small tokenizer interface required by deterministic model generation."""

    @property
    def vocab_size(self: Self) -> int:
        """Return the model embedding-table size."""

    def encode(self: Self, text: str) -> npt.NDArray[np.int32]:
        """Encode text to one-dimensional dense token ids."""

    def decode(self: Self, token_ids: npt.NDArray[np.integer]) -> str:
        """Decode one-dimensional dense token ids to text."""


@dataclass(frozen=True, slots=True)
class GreedyVocabularyTokenizer:
    """A strict longest-token vocabulary codec for portable fixture/demo artifacts.

    This is deliberately not a SentencePiece implementation. A production
    TinyLlama artifact must provide a separately supported tokenizer revision;
    this codec makes the test/demo artifact contract explicit and reproducible.
    """

    token_to_id: dict[str, int]
    unknown_token: str = "<unk>"

    def __post_init__(self: Self) -> None:
        """Require a dense, injective non-empty vocabulary with an unknown token."""
        if not self.token_to_id or self.unknown_token not in self.token_to_id:
            raise ValueError("vocabulary must be non-empty and contain its unknown token")
        ids = list(self.token_to_id.values())
        if any(not isinstance(token, str) or not token for token in self.token_to_id) or any(
            not isinstance(identifier, int) or identifier < 0 for identifier in ids
        ):
            raise ValueError("vocabulary tokens must be non-empty strings with non-negative integer ids")
        if len(set(ids)) != len(ids) or set(ids) != set(range(len(ids))):
            raise ValueError("vocabulary ids must be unique and dense from zero")

    @property
    def vocab_size(self: Self) -> int:
        """Return the dense model-vocabulary size."""
        return len(self.token_to_id)

    @property
    def id_to_token(self: Self) -> tuple[str, ...]:
        """Return a stable id-indexed token table."""
        table = ["" for _ in range(self.vocab_size)]
        for token, identifier in self.token_to_id.items():
            table[identifier] = token
        return tuple(table)

    @classmethod
    def from_json(cls: type[GreedyVocabularyTokenizer], path: str | PathLike[str]) -> GreedyVocabularyTokenizer:
        """Load the portable ``{"token_to_id": {...}}`` JSON artifact without code execution."""
        location = Path(path)
        try:
            payload = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load vocabulary JSON {location}") from exc
        if not isinstance(payload, dict) or set(payload) - {"token_to_id", "unknown_token"}:
            raise ValueError("vocabulary JSON must contain only token_to_id and optional unknown_token")
        token_to_id = payload.get("token_to_id")
        unknown_token = payload.get("unknown_token", "<unk>")
        if not isinstance(token_to_id, dict) or not isinstance(unknown_token, str):
            raise ValueError("vocabulary JSON has invalid token_to_id or unknown_token")
        return cls(dict(token_to_id), unknown_token)

    def encode(self: Self, text: str) -> npt.NDArray[np.int32]:
        """Greedily encode the longest matching vocabulary token at each position."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        candidates = sorted(
            ((token, identifier) for token, identifier in self.token_to_id.items() if token != self.unknown_token),
            key=lambda candidate: (-len(candidate[0]), candidate[1]),
        )
        unknown_id = self.token_to_id[self.unknown_token]
        output: list[int] = []
        offset = 0
        while offset < len(text):
            match = next(
                ((token, identifier) for token, identifier in candidates if text.startswith(token, offset)), None
            )
            if match is None:
                output.append(unknown_id)
                offset += 1
            else:
                token, identifier = match
                output.append(identifier)
                offset += len(token)
        return np.asarray(output, dtype=np.int32)

    def decode(self: Self, token_ids: npt.NDArray[np.integer]) -> str:
        """Decode a one-dimensional dense token-id array back to vocabulary text."""
        if token_ids.ndim != 1 or not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must be a rank-1 integer array")
        table = self.id_to_token
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError("token ids are outside the vocabulary range")
        return "".join(table[int(identifier)] for identifier in token_ids)


class SentencePieceTokenizer:
    """Lazy adapter for a production SentencePiece tokenizer model artifact."""

    def __init__(self: Self, model_path: str | PathLike[str]) -> None:
        """Load a SentencePiece model only when its optional dependency is installed."""
        try:
            import sentencepiece as sentencepiece
        except ImportError as exc:
            raise RuntimeError(
                "SentencePiece support requires the optional 'sentencepiece' package; "
                "install it before loading a production TinyLlama tokenizer"
            ) from exc
        location = Path(model_path)
        try:
            self._processor = sentencepiece.SentencePieceProcessor(model_file=str(location))
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot load SentencePiece tokenizer model {location}") from exc
        if self._processor.get_piece_size() <= 0:
            raise ValueError("SentencePiece tokenizer must have a positive vocabulary size")

    @property
    def vocab_size(self: Self) -> int:
        """Return the tokenizer's model vocabulary size."""
        return int(self._processor.get_piece_size())

    @property
    def bos_id(self: Self) -> int:
        """Return the configured beginning-of-sequence token id, if any."""
        return int(self._processor.bos_id())

    @property
    def eos_id(self: Self) -> int:
        """Return the configured end-of-sequence token id, if any."""
        return int(self._processor.eos_id())

    def encode(self: Self, text: str) -> npt.NDArray[np.int32]:
        """Encode text using the model's native SentencePiece segmentation."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return np.asarray(self._processor.encode(text, out_type=int), dtype=np.int32)

    def decode(self: Self, token_ids: npt.NDArray[np.integer]) -> str:
        """Decode one-dimensional integer token ids through SentencePiece."""
        if token_ids.ndim != 1 or not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must be a rank-1 integer array")
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError("token ids are outside the vocabulary range")
        return str(self._processor.decode([int(identifier) for identifier in token_ids]))


class TinyLlamaGreedyGenerator:
    """End-to-end greedy prefill/decode reference over a portable vocabulary."""

    def __init__(self: Self, runtime: TinyLlamaInferenceRuntime, tokenizer: TinyLlamaTokenizer) -> None:
        """Require the tokenizer vocabulary to match the checkpoint embedding table."""
        if tokenizer.vocab_size != runtime.checkpoint.config.vocab_size:
            raise ValueError("tokenizer vocabulary size must match the TinyLlama checkpoint")
        self.runtime = runtime
        self.tokenizer = tokenizer

    def generate_tokens(
        self: Self,
        prompt_token_ids: npt.NDArray[np.integer],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> npt.NDArray[np.int32]:
        """Generate deterministic argmax tokens, recomputing the reference prefill each step."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if (
            prompt_token_ids.ndim != 1
            or prompt_token_ids.size == 0
            or not np.issubdtype(prompt_token_ids.dtype, np.integer)
        ):
            raise ValueError("prompt_token_ids must be a non-empty rank-1 integer array")
        config = self.runtime.checkpoint.config
        tokens = prompt_token_ids.astype(np.int32, copy=True)
        if eos_token_id is not None and not 0 <= eos_token_id < config.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary range")
        session = self.runtime.session()
        try:
            logits = session.prefill(tokens)
            for _ in range(max_new_tokens):
                if tokens.size >= config.max_position_embeddings:
                    raise ValueError("generation would exceed max_position_embeddings")
                next_token = np.asarray([np.argmax(logits[-1])], dtype=np.int32)
                tokens = np.concatenate((tokens, next_token))
                if eos_token_id is not None and int(next_token[0]) == eos_token_id:
                    break
                logits = session.decode(int(next_token[0]))[None, :]
        finally:
            close = getattr(session, "close", None)
            if close is not None:
                close()
        return tokens

    def generate_text(self: Self, prompt: str, *, max_new_tokens: int, eos_token_id: int | None = None) -> str:
        """Encode, greedily generate, and decode one text prompt through the fixture tokenizer."""
        token_ids = self.tokenizer.encode(prompt)
        if token_ids.size == 0:
            raise ValueError("prompt must encode to at least one token")
        generated = self.generate_tokens(token_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
        return self.tokenizer.decode(generated)
