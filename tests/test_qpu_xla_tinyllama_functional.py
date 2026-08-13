from __future__ import annotations

import numpy as np

from qpu_xla import Device
from qpu_xla.models.tinyllama import (
    embedding_lookup_fp32,
    greedy_sample_fp32,
    residual_add_fp32,
    rms_norm_fp32,
    rope_fp32,
    silu_gated_fp32,
)


def test_rms_norm_fp32_matches_explicit_reference() -> None:
    values = np.array([[1.0, -2.0, 3.0, -4.0], [-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    weight_values = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    expected = values / np.sqrt(np.mean(values * values, axis=1, keepdims=True) + np.float32(1e-5)) * weight_values
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(values.shape, np.float32)
        weight = device.tensor(weight_values.shape, np.float32)
        destination = device.tensor(values.shape, np.float32)
        source.numpy()[:] = values
        weight.numpy()[:] = weight_values
        rms_norm_fp32(destination, source, weight, queue=queue).wait()
        np.testing.assert_allclose(destination.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_rope_fp32_matches_pairwise_rotation_and_allows_in_place_output() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -3.0]], dtype=np.float32)
    positions_values = np.array([0, 3], dtype=np.int32)
    frequencies = np.power(np.float32(10_000.0), -np.arange(0, 4, 2, dtype=np.float32) / 4)
    angles = positions_values.astype(np.float32)[:, None] * frequencies[None, :]
    expected = np.empty_like(values)
    expected[:, 0::2] = values[:, 0::2] * np.cos(angles) - values[:, 1::2] * np.sin(angles)
    expected[:, 1::2] = values[:, 0::2] * np.sin(angles) + values[:, 1::2] * np.cos(angles)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(values.shape, np.float32)
        positions = device.tensor(positions_values.shape, np.int32)
        source.numpy()[:] = values
        positions.numpy()[:] = positions_values
        rope_fp32(source, source, positions, queue=queue).wait()
        np.testing.assert_allclose(source.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_embedding_lookup_and_silu_gated_fp32_match_numpy_reference() -> None:
    table_values = np.arange(20, dtype=np.float32).reshape(5, 4)
    ids_values = np.array([3, 1], dtype=np.int32)
    gate_values = np.array([[1.0, -2.0], [0.5, -0.5]], dtype=np.float32)
    up_values = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=np.float32)
    expected_gated = (gate_values / (1.0 + np.exp(-gate_values))) * up_values
    with Device.fake() as device, device.queue() as queue:
        table = device.tensor(table_values.shape, np.float32)
        ids = device.tensor(ids_values.shape, np.int32)
        embeddings = device.tensor((2, 4), np.float32)
        gate = device.tensor(gate_values.shape, np.float32)
        up = device.tensor(up_values.shape, np.float32)
        gated = device.tensor(gate_values.shape, np.float32)
        table.numpy()[:] = table_values
        ids.numpy()[:] = ids_values
        gate.numpy()[:] = gate_values
        up.numpy()[:] = up_values

        embedding_lookup_fp32(embeddings, ids, table, queue=queue).wait()
        silu_gated_fp32(gated, gate, up, queue=queue).wait()

        np.testing.assert_array_equal(embeddings.numpy(), table_values[ids_values])
        np.testing.assert_allclose(gated.numpy(), expected_gated, atol=1e-6, rtol=1e-6)


def test_residual_add_and_greedy_sample_fp32_match_numpy_reference() -> None:
    left_values = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    right_values = np.array([[0.5, 1.5], [-3.0, 2.0]], dtype=np.float32)
    logits_values = np.array([[0.0, 4.0, 3.0], [5.0, 5.0, 1.0]], dtype=np.float32)
    with Device.fake() as device, device.queue() as queue:
        left = device.tensor(left_values.shape, np.float32)
        right = device.tensor(right_values.shape, np.float32)
        residual = device.tensor(left_values.shape, np.float32)
        logits = device.tensor(logits_values.shape, np.float32)
        tokens = device.tensor((2,), np.int32)
        left.numpy()[:] = left_values
        right.numpy()[:] = right_values
        logits.numpy()[:] = logits_values

        add_event = residual_add_fp32(residual, left, right, queue=queue)
        greedy_sample_fp32(tokens, logits, queue=queue, wait_for=(add_event,)).wait()

        np.testing.assert_allclose(residual.numpy(), left_values + right_values, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(tokens.numpy(), np.array([1, 0], dtype=np.int32))
