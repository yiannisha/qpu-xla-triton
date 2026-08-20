from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpu_xla import Device
from qpu_xla.kernels.gemm_int8 import pack_int8_gemm_operands
from qpu_xla.kernels.gemv_int8 import W8A8_GEMV_KERNEL


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
@pytest.mark.parametrize("shape", ((16, 16), (512, 512), (1536, 512)))
def test_w8a8_gemv_matches_exact_int32_reference(shape: tuple[int, int]) -> None:
    reduction, outputs = shape
    rng = np.random.default_rng(190 + reduction)
    source_value = rng.integers(-127, 128, size=(1, reduction), dtype=np.int8)
    weight_value = rng.integers(-127, 128, size=(reduction, outputs), dtype=np.int8)
    expected = source_value.astype(np.int32) @ weight_value.astype(np.int32)
    packed_source, packed_weight = pack_int8_gemm_operands(source_value, weight_value)
    with Device.open(data_area_size=4 * 1024 * 1024) as device, device.queue() as queue:
        source = device.tensor(packed_source.shape, np.uint32)
        weight = device.tensor(packed_weight.shape, np.uint32)
        destination = device.tensor(expected.shape, np.int32)
        source.numpy()[:] = packed_source
        weight.numpy()[:] = packed_weight

        queue.submit(W8A8_GEMV_KERNEL, (source, weight, destination), grid=(outputs // 16, 1, 1)).wait()
        actual = np.array(destination.numpy(), copy=True)

    np.testing.assert_array_equal(actual, expected)
