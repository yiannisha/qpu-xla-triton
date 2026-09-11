from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from qpu_xla import Device
from qpu_xla.models.vision import VisionExecutionContext, VisionMode, instrument_torch_convolutions
from qpu_xla.ops import Conv2dFP32Plan
from qpu_xla.ops.conv2d import _im2col_nchw
from qpu_xla.scheduler import Placement


def test_cpu_fp32_instrumented_graph_matches_original_torch_graph() -> None:
    torch.manual_seed(10)
    original = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1, bias=True),
        torch.nn.BatchNorm2d(8),
        torch.nn.SiLU(),
        torch.nn.Conv2d(8, 16, 1, bias=False),
    ).eval()
    source = torch.randn(1, 3, 7, 7)
    with torch.inference_mode():
        expected = original(source).numpy()
    context = VisionExecutionContext(VisionMode.CPU_FP32)
    instrument_torch_convolutions(original, context)
    with torch.inference_mode():
        actual = original(source).numpy()
    context.close()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert context.telemetry.eligible_layers == 2
    assert context.telemetry.cpu_calls == 2


def test_cpu_w8a8_instrumented_graph_is_deterministic_and_finite() -> None:
    torch.manual_seed(20)
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.ReLU()).eval()
    source = torch.randn(1, 4, 5, 5)
    context = VisionExecutionContext(VisionMode.CPU_W8A8)
    instrument_torch_convolutions(model, context)
    with torch.inference_mode():
        first = model(source).numpy()
        second = model(source).numpy()
    context.close()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 16, 5, 5)
    assert np.all(np.isfinite(first))
    assert context.telemetry.cpu_calls == 2


def test_prepared_fp32_convolution_cpu_placement_matches_lowered_reference() -> None:
    rng = np.random.default_rng(30)
    source_shape = (1, 4, 5, 5)
    source_value = rng.standard_normal(source_shape, dtype=np.float32)
    weight_value = np.ascontiguousarray(rng.standard_normal((8, 4, 3, 3), dtype=np.float32))
    columns = cast(
        npt.NDArray[np.float32],
        _im2col_nchw(
            source_value,
            3,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        ),
    )
    expected = (columns @ weight_value.reshape(8, -1).T).reshape(1, 5, 5, 8).transpose(0, 3, 1, 2)
    with Device.fake() as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with Conv2dFP32Plan(
            device,
            source_shape=source_shape,
            weight=weight_value,
            padding=1,
        ) as plan:
            plan.execute(destination, source, queue=queue, placement=Placement.CPU).wait()
            actual = np.array(destination.numpy(), copy=True)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_qpu_mode_requires_an_explicit_open_device() -> None:
    for mode in (VisionMode.QPU_FP32, VisionMode.HYBRID_FP32, VisionMode.QPU_W8A8, VisionMode.HYBRID_W8A8):
        try:
            VisionExecutionContext(mode)
        except ValueError as exc:
            assert "Device" in str(exc)
        else:
            raise AssertionError(f"{mode.value} accepted no device")


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("placement", (Placement.QPU, Placement.HYBRID))
def test_prepared_fp32_convolution_hardware_matches_torch(placement: Placement) -> None:
    rng = np.random.default_rng(44)
    source_shape = (1, 4, 5, 5)
    source_value = rng.standard_normal(source_shape, dtype=np.float32)
    weight_value = np.ascontiguousarray(rng.standard_normal((16, 4, 3, 3), dtype=np.float32))
    expected = torch.nn.functional.conv2d(
        torch.from_numpy(source_value), torch.from_numpy(weight_value), padding=1
    ).numpy()
    with (
        Device.open(data_area_size=16 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with Conv2dFP32Plan(
            device,
            source_shape=source_shape,
            weight=weight_value,
            padding=1,
        ) as plan:
            plan.execute(
                destination,
                source,
                queue=qpu_queue,
                placement=placement,
                cpu_queue=cpu_queue if placement is Placement.HYBRID else None,
                qpu_rows=16 if placement is Placement.HYBRID else None,
            ).wait()
            actual = np.array(destination.numpy(), copy=True)
    max_error = float(np.max(np.abs(actual - expected)))
    assert max_error < 2e-4, f"FP32 {placement.value} max error was {max_error}"
