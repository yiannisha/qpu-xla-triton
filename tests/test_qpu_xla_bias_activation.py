from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.ops import bias_activation


def test_bias_activation_cpu_reference_supports_tiled_and_untiled_shapes() -> None:
    with Device.fake() as device, device.queue() as queue:
        for shape in ((3, 5), (32, 32)):
            source = device.tensor(shape, np.float32)
            bias = device.tensor((shape[1],), np.float32)
            destination = device.tensor(shape, np.float32)
            source_value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 20.0
            bias_value = np.linspace(-3.0, 3.0, shape[1], dtype=np.float32)
            source.numpy()[:] = source_value
            bias.numpy()[:] = bias_value

            bias_activation(destination, source, bias, relu=True, queue=queue).wait()

            np.testing.assert_array_equal(destination.numpy(), np.maximum(source_value + bias_value, 0.0))


def test_bias_activation_rejects_invalid_bias_shape() -> None:
    with Device.fake() as device:
        source = device.tensor((16, 16), np.float32)
        destination = device.tensor((16, 16), np.float32)
        bias = device.tensor((15,), np.float32)
        with pytest.raises(ValueError, match="column dimension"):
            bias_activation(destination, source, bias)


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_bias_activation_qpu_fp32_and_int32_match_reference() -> None:
    with Device.open(data_area_size=1024 * 1024) as device, device.queue() as queue:
        for dtype in (np.float32, np.int32):
            source = device.tensor((32, 32), dtype)
            bias = device.tensor((32,), dtype)
            destination = device.tensor((32, 32), dtype)
            if dtype == np.dtype(np.float32):
                source_value = np.arange(1024, dtype=np.float32).reshape(32, 32) - 500.0
                bias_value = np.linspace(-3.0, 3.0, 32, dtype=np.float32)
            else:
                source_value = np.arange(1024, dtype=np.int32).reshape(32, 32) - 500
                bias_value = np.arange(32, dtype=np.int32) - 16
            source.numpy()[:] = source_value
            bias.numpy()[:] = bias_value

            bias_activation(destination, source, bias, relu=True, queue=queue).wait()
            actual = np.array(destination.numpy(), copy=True)

            np.testing.assert_array_equal(actual, np.maximum(source_value + bias_value, 0))

            bias_activation(destination, source, bias, queue=queue).wait()
            np.testing.assert_array_equal(destination.numpy(), source_value + bias_value)

            bias_activation(destination, source, relu=True, queue=queue).wait()
            np.testing.assert_array_equal(destination.numpy(), np.maximum(source_value, 0))

            scalar_source = device.tensor((16, 16), dtype)
            scalar_destination = device.tensor((16, 16), dtype)
            scalar_value = np.arange(256, dtype=dtype).reshape(16, 16) - 128
            scalar_source.numpy()[:] = scalar_value

            bias_activation(scalar_destination, scalar_source, relu=True, queue=queue).wait()
            np.testing.assert_array_equal(scalar_destination.numpy(), np.maximum(scalar_value, 0))
