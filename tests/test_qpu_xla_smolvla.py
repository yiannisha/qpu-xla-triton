from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from qpu_xla.benchmark import (
    CandidateRecord,
    CandidateRegistry,
    CandidateStatus,
    CorrectnessEvidence,
    PerformanceEvidence,
)
from qpu_xla.device import Device
from qpu_xla.models.smolvla import (
    UPSTREAM_LEROBOT_REVISION,
    UPSTREAM_SMOLVLA_REVISION,
    SmolVLAArtifact,
    SmolVLACheckpoint,
    SmolVLAConfig,
    SmolVLANumerics,
    SmolVLAObservation,
    SmolVLAPlacementPolicy,
    SmolVLAReferenceRuntime,
    SmolVLAReplay,
    SmolVLARuntime,
    convert_smolvla_safetensors,
)
from qpu_xla.models.smolvla.checkpoint import (
    Weight,
    is_quantized_weight,
    quantize_per_output_channel,
)
from qpu_xla.scheduler import Placement


def _tiny_config() -> SmolVLAConfig:
    return SmolVLAConfig(
        image_keys=("observation.images.front",),
        input_image_height=16,
        input_image_width=24,
        image_size=32,
        patch_size=16,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_layers=2,
        vision_num_heads=2,
        pixel_shuffle_factor=1,
        vocab_size=32,
        vlm_hidden_size=64,
        vlm_intermediate_size=64,
        vlm_num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        tokenizer_max_length=4,
        state_dim=3,
        action_dim=3,
        max_state_dim=16,
        max_action_dim=16,
        expert_hidden_size=32,
        expert_intermediate_size=64,
        expert_num_layers=2,
        self_attn_every_n_layers=2,
        chunk_size=20,
        num_steps=2,
    )


def _fp32_checkpoint(config: SmolVLAConfig) -> SmolVLACheckpoint:
    generator = np.random.default_rng(11)
    tensors: dict[str, Weight] = {}
    for name, shape in config.expected_shapes().items():
        if (
            name.endswith("layernorm.weight")
            or name.endswith("layer_norm1.weight")
            or name.endswith("layer_norm2.weight")
            or name.endswith(".norm.weight")
        ):
            values = np.ones(shape, dtype=np.float32)
        elif name.endswith(".bias"):
            values = np.zeros(shape, dtype=np.float32)
        else:
            scale = np.float32(0.08 / np.sqrt(shape[-1])) if len(shape) >= 2 else np.float32(0.01)
            values = generator.normal(0.0, scale, shape).astype(np.float32)
        tensors[name] = np.ascontiguousarray(values)
    return SmolVLACheckpoint(config, SmolVLANumerics.FP32, tensors)


def _w8a8_checkpoint(source: SmolVLACheckpoint) -> SmolVLACheckpoint:
    tensors: dict[str, Weight] = {}
    for name, shape in source.config.expected_shapes().items():
        value = source.fp32(name)
        tensors[name] = quantize_per_output_channel(value) if is_quantized_weight(name, shape) else value
    return SmolVLACheckpoint(source.config, SmolVLANumerics.W8A8, tensors)


def _observation(config: SmolVLAConfig) -> SmolVLAObservation:
    generator = np.random.default_rng(29)
    image = generator.integers(
        0,
        256,
        size=(config.input_image_height, config.input_image_width, 3),
        dtype=np.uint8,
    )
    return SmolVLAObservation(
        images=(np.ascontiguousarray(image),),
        image_masks=np.ones((1,), dtype=np.bool_),
        language_tokens=np.array([1, 4, 7, 2], dtype=np.int64),
        language_mask=np.array([True, True, True, False], dtype=np.bool_),
        state=generator.normal(size=(config.state_dim,)).astype(np.float32),
        noise=generator.normal(size=(config.chunk_size, config.max_action_dim)).astype(np.float32),
    )


def _write_safetensors(path: Path, checkpoint: SmolVLACheckpoint) -> None:
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(checkpoint.tensors):
        values = checkpoint.fp32(name)
        data = values.astype("<f4", copy=False).tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def _assert_actions_close(
    actual: npt.NDArray[np.float32],
    expected: npt.NDArray[np.float32],
    *,
    atol: float,
) -> None:
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=atol)


def test_smolvla_policy_config_parser_locks_executed_inference_semantics(tmp_path: Path) -> None:
    payload = {
        "type": "smolvla",
        "n_obs_steps": 1,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "resize_imgs_with_padding": [512, 512],
        "chunk_size": 50,
        "num_steps": 10,
        "tokenizer_max_length": 48,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_vlm_layers": 16,
        "num_expert_layers": 0,
        "self_attn_every_n_layers": 2,
        "expert_width_multiplier": 0.75,
        "attention_mode": "cross_attn",
        "adapt_to_pi_aloha": False,
        "add_image_special_tokens": False,
        "use_cache": True,
        "pad_language_to": "max_length",
        "prefix_length": 0,
        "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = SmolVLAConfig.from_huggingface_json(path)

    assert config.prefix_length == 241
    assert config.rope_theta == 10_000.0
    assert len(config.expected_shapes()) == 500

    payload["add_image_special_tokens"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="policy semantics"):
        SmolVLAConfig.from_huggingface_json(path)


def test_smolvla_reference_executes_complete_fp32_and_w8a8_graphs() -> None:
    config = _tiny_config()
    fp32 = _fp32_checkpoint(config)
    observation = _observation(config)
    fp32_actions = SmolVLAReferenceRuntime(fp32).predict_action_chunk(observation)
    w8a8_actions = SmolVLAReferenceRuntime(_w8a8_checkpoint(fp32)).predict_action_chunk(observation)

    assert fp32_actions.actions.shape == (config.chunk_size, config.action_dim)
    assert set(fp32_actions.stage_ns) == {"prefix", "denoise", "total"}
    assert fp32_actions.stage_ns["total"] > 0
    assert np.all(np.isfinite(w8a8_actions.actions))


def test_smolvla_streaming_artifact_and_replay_round_trip(tmp_path: Path) -> None:
    config = _tiny_config()
    checkpoint = _fp32_checkpoint(config)
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, checkpoint)
    artifact_path = convert_smolvla_safetensors(
        source,
        tmp_path / "artifact",
        config,
        numerics=SmolVLANumerics.FP32,
    )
    artifact = SmolVLAArtifact.open(artifact_path)
    observation = _observation(config)
    expected = SmolVLAReferenceRuntime(artifact.checkpoint).predict_action_chunk(observation)
    replay = SmolVLAReplay((observation,), expected.actions[None])
    replay_path = tmp_path / "replay.npz"
    replay.save(replay_path, config)
    loaded = SmolVLAReplay.load(replay_path, config)

    assert loaded.model_revision == UPSTREAM_SMOLVLA_REVISION
    assert loaded.lerobot_revision == UPSTREAM_LEROBOT_REVISION
    assert loaded.upstream_actions is not None
    np.testing.assert_array_equal(loaded.observations[0].images[0], observation.images[0])
    np.testing.assert_array_equal(loaded.upstream_actions, expected.actions[None])


def test_smolvla_benchmark_runs_native_worker_in_an_isolated_process(tmp_path: Path) -> None:
    config = _tiny_config()
    checkpoint = _fp32_checkpoint(config)
    source = tmp_path / "model.safetensors"
    _write_safetensors(source, checkpoint)
    artifact_path = convert_smolvla_safetensors(
        source,
        tmp_path / "artifact",
        config,
        numerics=SmolVLANumerics.FP32,
    )
    replay = SmolVLAReplay((_observation(config),))
    replay_path = tmp_path / "replay.npz"
    replay.save(replay_path, config)
    output = tmp_path / "report.json"
    subprocess.run(
        (
            sys.executable,
            "examples/benchmark_qpu_xla_smolvla.py",
            "--fp32-artifact",
            str(artifact_path),
            "--replay",
            str(replay_path),
            "--modes",
            "native_cpu_fp32",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--output",
            str(output),
        ),
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["semantics"]["upstream_reference"] == "upstream_torch_cpu_fp32"
    assert report["reports"]["native_cpu_fp32"]["steady_state_seconds"]


def test_smolvla_auto_requires_exact_stage_block_and_full_replay_wins() -> None:
    config = _tiny_config()

    def winner(name: str, operation: str, shape_class: str) -> CandidateRecord:
        return CandidateRecord(
            name,
            operation,
            "fp32",
            "native-contiguous",
            shape_class,
            "source",
            CandidateStatus.SUPPORTED_WIN,
            CorrectnessEvidence("numpy-fp32", 8, False, 1e-6, 1e-7, 1e-6, 1e-5),
            PerformanceEvidence("upstream_torch_cpu_fp32", (0.002,), (0.001,)),
        )

    shape = "20x32x64"
    stage = winner("linear-stage", "smolvla.linear", shape)
    block = winner("linear-block", "smolvla.block.linear", shape)
    full = winner("full-replay", "smolvla.full_replay", config.model_shape_class)
    incomplete = SmolVLAPlacementPolicy.auto(CandidateRegistry((stage, block)))
    complete = SmolVLAPlacementPolicy.auto(CandidateRegistry((stage, block, full)))

    assert (
        incomplete.choose(
            "linear",
            dtype="fp32",
            shape_class=shape,
            model_dtype="fp32",
            model_shape_class=config.model_shape_class,
        )
        is Placement.CPU
    )
    assert (
        complete.choose(
            "linear",
            dtype="fp32",
            shape_class=shape,
            model_dtype="fp32",
            model_shape_class=config.model_shape_class,
        )
        is Placement.QPU
    )


@pytest.mark.hardware
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("numerics", [SmolVLANumerics.FP32, SmolVLANumerics.W8A8])
def test_smolvla_qpu_runtime_matches_its_complete_cpu_contract(
    numerics: SmolVLANumerics,
    hybrid: bool,
) -> None:
    config = _tiny_config()
    fp32 = _fp32_checkpoint(config)
    checkpoint = fp32 if numerics is SmolVLANumerics.FP32 else _w8a8_checkpoint(fp32)
    observation = _observation(config)
    expected = SmolVLAReferenceRuntime(checkpoint).predict_action_chunk(observation)
    policy = SmolVLAPlacementPolicy.hybrid() if hybrid else SmolVLAPlacementPolicy.forced_qpu()

    with Device.open(data_area_size=96 * 1024 * 1024) as device:
        with SmolVLARuntime(device, checkpoint, policy=policy) as runtime:
            actual = runtime.predict_action_chunk(observation)

    _assert_actions_close(actual.padded_actions, expected.padded_actions, atol=3e-4)
