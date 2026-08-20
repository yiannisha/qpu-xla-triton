from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.models.tinyllama import quantize_per_output_channel_int8
from qpu_xla.ops import Conv2dW8A8Plan
from qpu_xla.ops.conv2d import _im2col_nchw


def _dynamic_group_reference(
    source: np.ndarray,
    weight: np.ndarray,
    *,
    groups: int,
    stride: int = 1,
    padding: int = 1,
) -> np.ndarray:
    batch, channels, height, width = source.shape
    outputs, _, kernel_height, kernel_width = weight.shape
    output_height = (height + 2 * padding - kernel_height) // stride + 1
    output_width = (width + 2 * padding - kernel_width) // stride + 1
    rows = batch * output_height * output_width
    input_group = channels // groups
    output_group = outputs // groups
    lowered = _im2col_nchw(
        source,
        kernel_height,
        kernel_width,
        stride=(stride, stride),
        padding=(padding, padding),
        dilation=(1, 1),
    ).reshape(rows, channels, kernel_height, kernel_width)
    group_results = []
    for group in range(groups):
        source_matrix = lowered[:, group * input_group : (group + 1) * input_group].reshape(rows, -1)
        weight_matrix = weight[group * output_group : (group + 1) * output_group].reshape(output_group, -1)
        quantized_weight = quantize_per_output_channel_int8(np.ascontiguousarray(weight_matrix))
        source_scales = np.maximum(
            np.max(np.abs(source_matrix), axis=1) / np.float32(127),
            np.float32(1 / 127),
        )
        quantized_source = np.rint(source_matrix / source_scales[:, None]).clip(-127, 127).astype(np.int32)
        accumulation = quantized_source @ quantized_weight.values.astype(np.int32).T
        group_results.append(
            accumulation.astype(np.float32) * source_scales[:, None] * quantized_weight.scales[None, :]
        )
    matrix = np.concatenate(group_results, axis=1)
    return matrix.reshape(batch, output_height, output_width, outputs).transpose(0, 3, 1, 2)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("groups", (1, 2, 4))
def test_conv2d_w8a8_plan_matches_dynamic_group_reference(groups: int) -> None:
    rng = np.random.default_rng(1200 + groups)
    source_value = rng.standard_normal((1, 4, 4, 4), dtype=np.float32)
    weight_value = np.ascontiguousarray(rng.standard_normal((4, 4 // groups, 3, 3), dtype=np.float32))
    expected = _dynamic_group_reference(source_value, weight_value, groups=groups)
    with Device.open(data_area_size=8 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with Conv2dW8A8Plan(
            device,
            source_shape=source_value.shape,
            weight=weight_value,
            padding=1,
            groups=groups,
        ) as plan:
            plan.execute(destination, source, queue=queue).wait()
            actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize(
    ("channels", "outputs", "kernel", "stride", "groups", "size"),
    (
        (4, 32, 1, 1, 1, 5),
        (4, 32, 3, 2, 1, 12),
        (4, 64, 3, 1, 2, 5),
        (16, 16, 3, 1, 16, 5),
    ),
)
def test_conv2d_w8a8_hybrid_partitions_match_dynamic_reference(
    channels: int,
    outputs: int,
    kernel: int,
    stride: int,
    groups: int,
    size: int,
) -> None:
    rng = np.random.default_rng(3100 + channels + outputs + kernel + stride + groups + size)
    source_value = rng.standard_normal((1, channels, size, size), dtype=np.float32)
    weight_value = np.ascontiguousarray(
        rng.standard_normal((outputs, channels // groups, kernel, kernel), dtype=np.float32)
    )
    padding = kernel // 2
    expected = _dynamic_group_reference(
        source_value,
        weight_value,
        groups=groups,
        stride=stride,
        padding=padding,
    )
    with (
        Device.open(data_area_size=16 * 1024 * 1024) as device,
        device.queue() as qpu_queue,
        device.queue() as cpu_queue,
    ):
        source = device.tensor(source_value.shape, np.float32)
        destination = device.tensor(expected.shape, np.float32)
        source.numpy()[:] = source_value
        with Conv2dW8A8Plan(
            device,
            source_shape=source_value.shape,
            weight=weight_value,
            stride=stride,
            padding=padding,
            groups=groups,
        ) as plan:
            plan.execute_hybrid(
                destination,
                source,
                qpu_queue=qpu_queue,
                cpu_queue=cpu_queue,
                qpu_rows=16,
            ).wait()
            row_actual = np.array(destination.numpy(), copy=True)
            if plan.output_split_alignment is None:
                with pytest.raises(ValueError, match="unsupported"):
                    plan.execute_hybrid(
                        destination,
                        source,
                        qpu_queue=qpu_queue,
                        cpu_queue=cpu_queue,
                        qpu_outputs=16,
                    )
                output_actual = None
            else:
                plan.execute_hybrid(
                    destination,
                    source,
                    qpu_queue=qpu_queue,
                    cpu_queue=cpu_queue,
                    qpu_outputs=plan.output_split_alignment,
                ).wait()
                output_actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_allclose(row_actual, expected, atol=1e-6, rtol=1e-6)
    if output_actual is not None:
        np.testing.assert_allclose(output_actual, expected, atol=1e-6, rtol=1e-6)
