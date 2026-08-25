from __future__ import annotations

import numpy as np
import pytest

from qpu_xla.device import Device
from qpu_xla.models.smolvla.reference import resize_with_top_left_padding_rgb
from qpu_xla.ops import (
    PreparedMultiHeadSDPAFP32,
    PreparedPatchifyFP32,
    PreparedPixelShuffleFP32,
    PreparedRGBResizeNormFP32,
    affine_fp32,
    apply_split_half_rope_tables_fp32,
    gelu_tanh_fp32,
    layer_norm_fp32,
    rms_norm_fp32,
    silu_fp32,
    split_half_rope_tables_fp32,
)
from qpu_xla.ops.gather import patchify_indices, pixel_shuffle_indices
from qpu_xla.scheduler import Placement


@pytest.mark.hardware
def test_smolvla_new_qpu_and_hybrid_stage_implementations_match_cpu() -> None:
    generator = np.random.default_rng(20260825)
    with (
        Device.open(data_area_size=48 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source_values = generator.normal(size=(10, 32)).astype(np.float32)
        weight_values = generator.uniform(0.8, 1.2, size=(32,)).astype(np.float32)
        bias_values = generator.normal(0.0, 0.1, size=(32,)).astype(np.float32)
        source = device.tensor(source_values.shape, np.float32)
        destination = device.tensor(source_values.shape, np.float32)
        weight = device.tensor(weight_values.shape, np.float32)
        bias = device.tensor(bias_values.shape, np.float32)
        source.numpy()[:] = source_values
        weight.numpy()[:] = weight_values
        bias.numpy()[:] = bias_values

        mean = np.mean(source_values, axis=1, keepdims=True, dtype=np.float32)
        centered = source_values - mean
        variance = np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float32)
        layer_expected = centered / np.sqrt(variance + np.float32(1e-6))
        layer_expected = layer_expected * weight_values + bias_values
        rms_expected = source_values / np.sqrt(
            np.mean(source_values * source_values, axis=1, keepdims=True, dtype=np.float32) + np.float32(1e-5)
        )
        rms_expected *= weight_values
        silu_expected = source_values / (np.float32(1.0) + np.exp(-source_values))
        gelu_inner = np.float32(np.sqrt(2.0 / np.pi)) * (source_values + np.float32(0.044715) * source_values**3)
        gelu_expected = np.float32(0.5) * source_values * (np.float32(1.0) + np.tanh(gelu_inner))
        affine_expected = source_values * np.float32(0.25) + bias_values

        for placement in (Placement.QPU, Placement.HYBRID):
            common = {
                "queue": qpu_queue,
                "cpu_queue": cpu_queue if placement is Placement.HYBRID else None,
                "placement": placement,
                "qpu_rows": 5 if placement is Placement.HYBRID else None,
            }
            layer_norm_fp32(
                destination,
                source,
                weight,
                bias,
                epsilon=1e-6,
                **common,
            ).wait()
            np.testing.assert_allclose(destination.numpy(), layer_expected, rtol=2e-6, atol=2e-6)
            rms_norm_fp32(destination, source, weight, epsilon=1e-5, **common).wait()
            np.testing.assert_allclose(destination.numpy(), rms_expected, rtol=2e-6, atol=2e-6)
            silu_fp32(destination, source, **common).wait()
            np.testing.assert_allclose(destination.numpy(), silu_expected, rtol=2e-6, atol=2e-6)
            gelu_tanh_fp32(destination, source, **common).wait()
            np.testing.assert_allclose(destination.numpy(), gelu_expected, rtol=2e-6, atol=2e-6)
            affine_fp32(destination, source, bias, scale=0.25, **common).wait()
            np.testing.assert_allclose(destination.numpy(), affine_expected, rtol=1e-6, atol=1e-6)

        positions = np.arange(5, dtype=np.int32)
        rope_values = generator.normal(size=(5, 2, 32)).astype(np.float32)
        rope_source = device.tensor((10, 32), np.float32)
        rope_destination = device.tensor((10, 32), np.float32)
        cosine_values, sine_values = split_half_rope_tables_fp32(
            positions,
            heads=2,
            head_dim=32,
            base=10_000.0,
        )
        cosine = device.tensor(cosine_values.shape, np.float32)
        sine = device.tensor(sine_values.shape, np.float32)
        rope_source.numpy()[:] = rope_values.reshape(10, 32)
        cosine.numpy()[:] = cosine_values
        sine.numpy()[:] = sine_values
        half = rope_values.shape[-1] // 2
        exponents = np.float32(2.0 / rope_values.shape[-1]) * np.arange(half, dtype=np.float32)
        radians = positions.astype(np.float32)[:, None] / np.power(np.float32(10_000.0), exponents)[None]
        rope_expected = np.empty_like(rope_values)
        rope_expected[..., :half] = (
            rope_values[..., :half] * np.cos(radians)[:, None]
            - rope_values[..., half:] * np.sin(radians)[:, None]
        )
        rope_expected[..., half:] = (
            rope_values[..., half:] * np.cos(radians)[:, None]
            + rope_values[..., :half] * np.sin(radians)[:, None]
        )
        for placement in (Placement.QPU, Placement.HYBRID):
            apply_split_half_rope_tables_fp32(
                rope_destination,
                rope_source,
                cosine,
                sine,
                queue=qpu_queue,
                cpu_queue=cpu_queue if placement is Placement.HYBRID else None,
                placement=placement,
                qpu_rows=5 if placement is Placement.HYBRID else None,
            ).wait()
            np.testing.assert_allclose(
                rope_destination.numpy().reshape(rope_values.shape),
                rope_expected,
                rtol=1e-6,
                atol=1e-6,
            )

        patch_source = device.tensor((3, 32, 32), np.float32)
        patch_destination = device.tensor((4, 768), np.float32)
        patch_values = generator.normal(size=patch_source.shape).astype(np.float32)
        patch_source.numpy()[:] = patch_values
        patch_expected = patch_values.reshape(-1)[patchify_indices(3, 32, 32, 16)]
        with PreparedPatchifyFP32(patch_source, patch_destination, patch_size=16) as patch_plan:
            patch_plan.execute(
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                placement=Placement.HYBRID,
                qpu_rows=2,
            ).wait()
            np.testing.assert_array_equal(patch_destination.numpy(), patch_expected)

        pixel_source = device.tensor((16, 16), np.float32)
        pixel_destination = device.tensor((4, 64), np.float32)
        pixel_values = generator.normal(size=pixel_source.shape).astype(np.float32)
        pixel_source.numpy()[:] = pixel_values
        pixel_expected = pixel_values.reshape(-1)[pixel_shuffle_indices(16, 16, 2)]
        with PreparedPixelShuffleFP32(pixel_source, pixel_destination, scale_factor=2) as pixel_plan:
            pixel_plan.execute(
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                placement=Placement.HYBRID,
                qpu_rows=2,
            ).wait()
            np.testing.assert_array_equal(pixel_destination.numpy(), pixel_expected)

        image = generator.integers(0, 256, size=(24, 40, 3), dtype=np.uint8)
        rgb_source = device.tensor((24, 40, 3), np.float32)
        rgb_destination = device.tensor((3, 32, 32), np.float32)
        rgb_expected = resize_with_top_left_padding_rgb(image, 32, 32)
        with PreparedRGBResizeNormFP32(rgb_source, rgb_destination) as rgb_plan:
            uploaded = rgb_plan.upload(np.ascontiguousarray(image), queue=qpu_queue)
            rgb_plan.execute(
                queue=qpu_queue,
                cpu_queue=cpu_queue,
                placement=Placement.HYBRID,
                qpu_rows=48,
                wait_for=(uploaded,),
            ).wait()
            np.testing.assert_allclose(rgb_destination.numpy(), rgb_expected, rtol=1e-6, atol=1e-6)

        query_values = generator.normal(size=(17, 4, 16)).astype(np.float32)
        key_values = generator.normal(size=(13, 2, 16)).astype(np.float32)
        value_values = generator.normal(size=(13, 2, 16)).astype(np.float32)
        mask = np.arange(13)[None, :] <= np.minimum(np.arange(17)[:, None], 12)
        expanded_key = np.repeat(key_values, 2, axis=1).transpose(1, 0, 2)
        expanded_value = np.repeat(value_values, 2, axis=1).transpose(1, 0, 2)
        scores = np.matmul(
            query_values.transpose(1, 0, 2),
            expanded_key.transpose(0, 2, 1),
            dtype=np.float32,
        )
        scores *= np.float32(1.0 / np.sqrt(16))
        scores[:] = np.where(mask[None], scores, np.finfo(np.float32).min)
        scores -= np.max(scores, axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=-1, keepdims=True, dtype=np.float32)
        attention_expected = np.matmul(scores, expanded_value, dtype=np.float32).transpose(1, 0, 2)
        query = device.tensor(query_values.shape, np.float32)
        key = device.tensor(key_values.shape, np.float32)
        value = device.tensor(value_values.shape, np.float32)
        attention_destination = device.tensor(query_values.shape, np.float32)
        query.numpy()[:] = query_values
        key.numpy()[:] = key_values
        value.numpy()[:] = value_values
        with PreparedMultiHeadSDPAFP32(
            device,
            query_length=17,
            key_length=13,
            query_heads=4,
            key_value_heads=2,
            head_dim=16,
        ) as attention_plan:
            for placement in (Placement.QPU, Placement.HYBRID):
                attention_plan.execute(
                    attention_destination,
                    query,
                    key,
                    value,
                    mask=np.ascontiguousarray(mask),
                    queue=qpu_queue,
                    cpu_queue=cpu_queue if placement is Placement.HYBRID else None,
                    placement=placement,
                    qpu_heads=2 if placement is Placement.HYBRID else None,
                ).wait()
                np.testing.assert_allclose(
                    attention_destination.numpy(),
                    attention_expected,
                    rtol=3e-5,
                    atol=3e-5,
                )
