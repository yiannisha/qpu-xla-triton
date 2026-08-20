from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.swiglu import _workgroup_count
from qpu_xla.ops import swiglu_fp32
from qpu_xla.scheduler import Placement


def _reference(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return (gate / (np.float32(1.0) + np.exp(-gate))) * up


def test_swiglu_fp32_cpu_reference_supports_arbitrary_shapes() -> None:
    rng = np.random.default_rng(20260819)
    with Device.fake() as device:
        gate = device.tensor((3, 19), np.float32)
        up = device.tensor(gate.shape, np.float32)
        destination = device.tensor(gate.shape, np.float32)
        gate.numpy()[:] = rng.standard_normal(gate.shape, dtype=np.float32)
        up.numpy()[:] = rng.standard_normal(up.shape, dtype=np.float32)

        swiglu_fp32(destination, gate, up).wait()

        np.testing.assert_array_equal(destination.numpy(), _reference(gate.numpy(), up.numpy()))


def test_swiglu_fp32_rejects_misaligned_shapes() -> None:
    with Device.fake() as device:
        gate = device.tensor((4, 16), np.float32)
        up = device.tensor((4, 8), np.float32)
        destination = device.tensor((4, 16), np.float32)
        with pytest.raises(ValueError, match="equal shapes"):
            swiglu_fp32(destination, gate, up)


def test_swiglu_workgroups_choose_widest_exact_partition() -> None:
    assert _workgroup_count(1) == 1
    assert _workgroup_count(352) == 11
    assert _workgroup_count(1_536) == 12
    assert _workgroup_count(45_056) == 11


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_swiglu_fp32_qpu_matches_numpy_across_activation_range() -> None:
    rng = np.random.default_rng(7719)
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        gate = device.tensor((16, 1536), np.float32)
        up = device.tensor(gate.shape, np.float32)
        destination = device.tensor(gate.shape, np.float32)
        gate.numpy()[:] = rng.standard_normal(gate.shape, dtype=np.float32) * np.float32(3.0)
        up.numpy()[:] = rng.standard_normal(up.shape, dtype=np.float32)
        expected = _reference(gate.numpy(), up.numpy())

        event = swiglu_fp32(destination, gate, up, queue=queue)
        event.wait()

        np.testing.assert_allclose(destination.numpy(), expected, atol=5e-6, rtol=2e-6)
        assert event.name == "vc7.swiglu_fp32"

        scalar_gate = device.tensor((1, 32), np.float32)
        scalar_up = device.tensor((1, 32), np.float32)
        scalar_destination = device.tensor((1, 32), np.float32)
        scalar_gate.numpy()[:] = rng.standard_normal((1, 32), dtype=np.float32)
        scalar_up.numpy()[:] = rng.standard_normal((1, 32), dtype=np.float32)

        swiglu_fp32(scalar_destination, scalar_gate, scalar_up, queue=queue).wait()
        np.testing.assert_allclose(
            scalar_destination.numpy(),
            _reference(scalar_gate.numpy(), scalar_up.numpy()),
            atol=5e-6,
            rtol=2e-6,
        )


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_swiglu_fp32_hybrid_row_split_matches_numpy() -> None:
    rng = np.random.default_rng(331)
    gate_value = rng.standard_normal((17, 64), dtype=np.float32)
    up_value = rng.standard_normal((17, 64), dtype=np.float32)
    expected = _reference(gate_value, up_value)
    with (
        Device.open(data_area_size=2 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        gate = device.tensor(gate_value.shape, np.float32)
        up = device.tensor(up_value.shape, np.float32)
        destination = device.tensor(gate_value.shape, np.float32)
        gate.numpy()[:] = gate_value
        up.numpy()[:] = up_value

        event = swiglu_fp32(
            destination,
            gate,
            up,
            queue=qpu_queue,
            cpu_queue=cpu_queue,
            placement=Placement.HYBRID,
            qpu_rows=8,
        )
        event.wait()
        actual = np.array(destination.numpy(), copy=True)
        assert event.name == "swiglu_fp32.hybrid_join"

    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=2e-6)
