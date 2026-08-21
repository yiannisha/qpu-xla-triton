from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.ggml_flash_attn import (
    GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL,
    ggml_flash_attn_ext_reference,
)


def test_flash_attention_reference_supports_gqa_mask_and_softcap() -> None:
    query = np.array(
        [
            [[[1.0], [0.5]]],
            [[[0.0], [1.0]]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
        ],
        dtype=np.float16,
    )
    value = np.array(
        [
            [[[2.0]], [[6.0]]],
            [[[4.0]], [[8.0]]],
            [[[1.0]], [[3.0]]],
        ],
        dtype=np.float16,
    )
    mask = np.zeros((2, 1, 1, 1), dtype=np.float16)
    actual = ggml_flash_attn_ext_reference(
        query,
        key,
        value,
        mask,
        scale=0.5,
        max_bias=0.0,
        logit_softcap=2.0,
    )
    assert actual.shape == (3, 2, 1, 1)
    for head in range(2):
        logits = key[:, :, 0, 0].astype(np.float32).T @ query[:, 0, head, 0]
        logits = 2.0 * np.tanh(logits * 0.25)
        weights = np.exp(logits - np.max(logits))
        weights /= np.sum(weights)
        expected = value[:, :, 0, 0].astype(np.float32) @ weights
        np.testing.assert_allclose(actual[:, head, 0, 0], expected, rtol=2e-7, atol=2e-7)


def test_flash_attention_reference_masks_values_and_accounts_for_sink_denominator() -> None:
    query = np.ones((1, 1, 1, 1), dtype=np.float32)
    key = np.ones((1, 2, 1, 1), dtype=np.float16)
    value = np.array([[[[4.0]], [[100.0]]]], dtype=np.float16)
    mask = np.array([[[[0.0]]], [[[-np.inf]]]], dtype=np.float16)
    actual = ggml_flash_attn_ext_reference(
        query,
        key,
        value,
        mask,
        scale=1.0,
        max_bias=0.0,
        logit_softcap=0.0,
        sinks=np.array([1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(actual, np.array([[[[2.0]]]], dtype=np.float32))


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII unavailable")
def test_fused_gemma_flash_attention_m1_matches_oracle() -> None:
    rng = np.random.default_rng(4961)
    heads, kv_rows, head_dim = 4, 32, 256
    query_host = rng.normal(0.0, 0.15, size=(heads, head_dim)).astype(np.float32)
    key_host = rng.normal(0.0, 0.15, size=(kv_rows, head_dim)).astype(np.float16)
    value_host = rng.normal(0.0, 0.2, size=(kv_rows, head_dim)).astype(np.float16)
    mask_host = np.zeros(kv_rows, dtype=np.float16)
    mask_host[-7:] = -np.inf
    scale = float(head_dim**-0.5)
    expected = ggml_flash_attn_ext_reference(
        query_host.T[:, None, :, None],
        key_host.T[:, :, None, None],
        value_host.T[:, :, None, None],
        mask_host[:, None, None, None],
        scale=scale,
        max_bias=0.0,
        logit_softcap=0.0,
    )[:, :, 0, 0].T
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        query = device.tensor(query_host.shape, np.float32)
        key = device.tensor(key_host.shape, np.float16)
        value = device.tensor(value_host.shape, np.float16)
        mask = device.tensor(mask_host.shape, np.float16)
        destination = device.tensor(expected.shape, np.float32)
        query.numpy()[:] = query_host
        key.numpy()[:] = key_host
        value.numpy()[:] = value_host
        mask.numpy()[:] = mask_host
        destination.numpy()[:] = np.nan
        queue.submit(
            GGML_GEMMA_FLASH_ATTN_F16_M1_KERNEL,
            (query, key, value, mask, destination, scale),
            grid=(heads, 1, 1),
        ).wait()
        actual = np.array(destination.numpy(), copy=True)
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
