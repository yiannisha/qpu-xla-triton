from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import apply_rope_tables_fp32, rope_tables_fp32
from qpu_xla.scheduler import Placement


def _reference(
    source: np.ndarray,
    cosine: np.ndarray,
    signed_sine: np.ndarray,
) -> np.ndarray:
    adjacent = source.reshape(-1, 2)[:, ::-1].reshape(source.shape)
    return source * cosine + adjacent * signed_sine


def test_rope_tables_encode_pair_signs_and_cpu_reference() -> None:
    positions = np.array([0, 3], dtype=np.int32)
    cosine_value, signed_sine_value = rope_tables_fp32(positions, 16)
    rng = np.random.default_rng(44)
    source_value = rng.standard_normal((2, 16), dtype=np.float32)
    expected = _reference(source_value, cosine_value, signed_sine_value)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        cosine = device.tensor(cosine_value.shape, np.float32)
        signed_sine = device.tensor(signed_sine_value.shape, np.float32)
        source.numpy()[:] = source_value
        cosine.numpy()[:] = cosine_value
        signed_sine.numpy()[:] = signed_sine_value

        apply_rope_tables_fp32(source, source, cosine, signed_sine, queue=queue).wait()

        np.testing.assert_allclose(source.numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", ((1, 16), (1, 64), (16, 64), (17, 128)))
def test_cached_rope_qpu_matches_numpy_in_place(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(600 + shape[0])
    source_value = rng.standard_normal(shape, dtype=np.float32)
    positions = np.arange(shape[0], dtype=np.int32) + 7
    cosine_value, signed_sine_value = rope_tables_fp32(positions, shape[1])
    expected = _reference(source_value, cosine_value, signed_sine_value)
    with Device.open(data_area_size=2 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(shape, np.float32)
        cosine = device.tensor(shape, np.float32)
        signed_sine = device.tensor(shape, np.float32)
        source.numpy()[:] = source_value
        cosine.numpy()[:] = cosine_value
        signed_sine.numpy()[:] = signed_sine_value

        event = apply_rope_tables_fp32(
            source,
            source,
            cosine,
            signed_sine,
            queue=queue,
            placement=Placement.QPU,
        )
        event.wait()
        actual = np.array(source.numpy(), copy=True)
        assert event.name == "vc7.rope_fp32"

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_cached_rope_hybrid_row_split_matches_numpy() -> None:
    rng = np.random.default_rng(777)
    source_value = rng.standard_normal((17, 64), dtype=np.float32)
    cosine_value, signed_sine_value = rope_tables_fp32(np.arange(17, dtype=np.int32), 64)
    expected = _reference(source_value, cosine_value, signed_sine_value)
    with (
        Device.open(data_area_size=2 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        tensors = [device.tensor(source_value.shape, np.float32) for _ in range(4)]
        source, cosine, signed_sine, destination = tensors
        source.numpy()[:] = source_value
        cosine.numpy()[:] = cosine_value
        signed_sine.numpy()[:] = signed_sine_value

        event = apply_rope_tables_fp32(
            destination,
            source,
            cosine,
            signed_sine,
            queue=qpu_queue,
            cpu_queue=cpu_queue,
            placement=Placement.HYBRID,
            qpu_rows=8,
        )
        event.wait()
        actual = np.array(destination.numpy(), copy=True)
        assert event.name == "rope_fp32.hybrid_join"

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
