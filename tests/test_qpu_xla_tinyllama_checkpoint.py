from __future__ import annotations

import json
import sys
from dataclasses import asdict
from types import ModuleType

import numpy as np
import pytest

from qpu_xla.models.tinyllama import TinyLlamaCheckpoint, TinyLlamaConfig


def _config() -> TinyLlamaConfig:
    return TinyLlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )


def _tensors(config: TinyLlamaConfig) -> dict[str, np.ndarray]:
    return {
        name: np.full(shape, index, dtype=np.float32)
        for index, (name, shape) in enumerate(config.expected_shapes().items())
    }


def test_tinyllama_checkpoint_loads_exact_npz_contract_and_estimates_memory(tmp_path) -> None:
    config = _config()
    values = _tensors(config)
    path = tmp_path / "tinyllama-toy.npz"
    np.savez(path, **values)

    checkpoint = TinyLlamaCheckpoint.from_npz(path, config)

    assert checkpoint.tensors.keys() == values.keys()
    assert checkpoint.tensors["model.embed_tokens.weight"].dtype == np.dtype(np.float32)
    assert config.head_dim == 4
    assert config.key_value_size == 4
    fp32_bytes = sum(array.nbytes for array in values.values())
    assert config.estimate_weight_bytes() == fp32_bytes
    assert 0 < config.estimate_weight_bytes(quantized=True) < fp32_bytes


def test_tinyllama_checkpoint_rejects_missing_or_wrongly_shaped_tensor() -> None:
    config = _config()
    values = _tensors(config)
    values.pop("lm_head.weight")
    with pytest.raises(ValueError, match="tensor names"):
        TinyLlamaCheckpoint(config, values)


def test_tinyllama_huggingface_config_and_safetensors_adapters_use_the_same_contract(monkeypatch, tmp_path) -> None:
    config = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    loaded_config = TinyLlamaConfig.from_huggingface_json(config_path)
    values = _tensors(config)
    package = ModuleType("safetensors")
    numpy_module = ModuleType("safetensors.numpy")
    setattr(numpy_module, "load_file", lambda _: values)
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.numpy", numpy_module)

    checkpoint = TinyLlamaCheckpoint.from_safetensors(tmp_path / "model.safetensors", loaded_config)

    assert checkpoint.config == config
    np.testing.assert_array_equal(checkpoint.tensors["lm_head.weight"], values["lm_head.weight"])


def test_tinyllama_safetensors_index_loads_validated_shards(monkeypatch, tmp_path) -> None:
    config = _config()
    values = _tensors(config)
    names = list(values)
    shard_one, shard_two = names[: len(names) // 2], names[len(names) // 2 :]
    weight_map = {name: "one.safetensors" for name in shard_one} | {name: "two.safetensors" for name in shard_two}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    package = ModuleType("safetensors")
    numpy_module = ModuleType("safetensors.numpy")
    setattr(
        numpy_module,
        "load_file",
        lambda path: {name: values[name] for name in (shard_one if path.endswith("one.safetensors") else shard_two)},
    )
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.numpy", numpy_module)

    checkpoint = TinyLlamaCheckpoint.from_safetensors_index(tmp_path / "model.safetensors.index.json", config)

    assert checkpoint.tensors.keys() == values.keys()

    values = _tensors(config)
    values["lm_head.weight"] = np.empty((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="lm_head.weight"):
        TinyLlamaCheckpoint(config, values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 7},
        {"num_key_value_heads": 3},
        {"rms_norm_eps": 0.0},
        {"rope_theta": 1.0},
    ],
)
def test_tinyllama_config_rejects_invalid_attention_or_numerical_contracts(kwargs: dict[str, int | float]) -> None:
    baseline = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
    }
    baseline.update(kwargs)
    with pytest.raises(ValueError):
        TinyLlamaConfig(**baseline)
