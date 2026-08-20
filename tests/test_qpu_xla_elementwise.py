from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import DependencyError, Device
from qpu_xla.ops import copy, maximum, minimum


def test_copy_minimum_and_maximum_use_tensor_contracts() -> None:
    with Device.fake() as device, device.queue() as queue:
        left = device.tensor((2, 3), np.int32)
        right = device.tensor((2, 3), np.int32)
        output = device.tensor((2, 3), np.int32)
        copied = device.tensor((2, 3), np.int32)
        left.numpy()[:] = [[4, -2, 7], [1, 5, 0]]
        right.numpy()[:] = [[3, 8, 6], [1, -4, 9]]

        minimum(output, left, right, queue=queue).wait()
        np.testing.assert_array_equal(output.numpy(), [[3, -2, 6], [1, -4, 0]])

        maximum(output, left, right, queue=queue).wait()
        np.testing.assert_array_equal(output.numpy(), [[4, 8, 7], [1, 5, 9]])

        copy(copied, output, queue=queue).wait()
        np.testing.assert_array_equal(copied.numpy(), output.numpy())


def test_elementwise_rejects_shape_dtype_and_device_mismatches() -> None:
    with Device.fake() as first, Device.fake() as second:
        output = first.tensor((2,), np.int32)
        wrong_shape = first.tensor((3,), np.int32)
        wrong_dtype = first.tensor((2,), np.float32)
        foreign = second.tensor((2,), np.int32)

        with pytest.raises(ValueError, match="shapes"):
            minimum(output, output, wrong_shape)
        with pytest.raises(ValueError, match="dtypes"):
            maximum(output, output, wrong_dtype)
        with pytest.raises(DependencyError, match="same device"):
            copy(output, foreign)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_copy_uses_the_qpu_word_kernel_when_its_contract_is_met() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        source = device.tensor((32,), np.int32)
        destination = device.tensor((32,), np.int32)
        source.numpy()[:] = np.arange(32, dtype=np.int32) * -3
        destination.numpy()[:] = 0

        event = copy(destination, source, queue=queue)
        event.wait()

        np.testing.assert_array_equal(destination.numpy(), source.numpy())

        vectorized_length = 12 * 64
        vectorized_source = device.tensor((vectorized_length,), np.float32)
        vectorized_destination = device.tensor((vectorized_length,), np.float32)
        vectorized_source.numpy()[:] = np.linspace(-5.0, 7.0, vectorized_length, dtype=np.float32)

        copy(vectorized_destination, vectorized_source, queue=queue).wait()

        np.testing.assert_array_equal(vectorized_destination.numpy(), vectorized_source.numpy())


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_minimum_and_maximum_use_qpu_word_kernels() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        left = device.tensor((32,), np.int32)
        right = device.tensor((32,), np.int32)
        destination = device.tensor((32,), np.int32)
        left.numpy()[:] = np.arange(-16, 16, dtype=np.int32)
        right.numpy()[:] = np.arange(16, -16, -1, dtype=np.int32)

        minimum(destination, left, right, queue=queue).wait()
        np.testing.assert_array_equal(destination.numpy(), np.minimum(left.numpy(), right.numpy()))

        maximum(destination, left, right, queue=queue).wait()
        np.testing.assert_array_equal(destination.numpy(), np.maximum(left.numpy(), right.numpy()))

        vectorized_length = 12 * 64
        float_left = device.tensor((vectorized_length,), np.float32)
        float_right = device.tensor((vectorized_length,), np.float32)
        float_destination = device.tensor((vectorized_length,), np.float32)
        float_left.numpy()[:] = np.linspace(-2.0, 2.0, num=vectorized_length, dtype=np.float32)
        float_right.numpy()[:] = np.linspace(2.0, -2.0, num=vectorized_length, dtype=np.float32)

        minimum(float_destination, float_left, float_right, queue=queue).wait()
        np.testing.assert_allclose(float_destination.numpy(), np.minimum(float_left.numpy(), float_right.numpy()))

        maximum(float_destination, float_left, float_right, queue=queue).wait()
        np.testing.assert_allclose(float_destination.numpy(), np.maximum(float_left.numpy(), float_right.numpy()))
