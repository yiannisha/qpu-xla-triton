from __future__ import annotations

from pathlib import Path
from threading import Event as ThreadEvent

import numpy as np
import pytest

from qpu_xla import (
    AccessMode,
    AllocationError,
    BufferClosedError,
    DependencyError,
    Device,
    DeviceClosedError,
    EventCancelledError,
    EventStatus,
    EventTimeoutError,
    Kernel,
    KernelError,
)


def test_tensor_is_zero_copy_and_uses_the_buffer_address() -> None:
    with Device.fake() as device:
        tensor = device.tensor((2, 3), np.int32, alignment=16)
        tensor.numpy()[:] = [[1, 2, 3], [4, 5, 6]]

        assert tensor.address % 16 == 0
        assert tensor.nbytes == 24
        np.testing.assert_array_equal(tensor.numpy(), [[1, 2, 3], [4, 5, 6]])


def test_buffer_slice_and_tensor_slice_share_the_same_allocation() -> None:
    with Device.fake() as device:
        buffer = device.allocate(64, alignment=8)
        tensor = buffer.tensor((4, 4), np.int32)
        tensor.numpy()[:] = np.arange(16, dtype=np.int32).reshape(4, 4)

        view = tensor.slice((slice(1, 3), slice(2, 4)))
        np.testing.assert_array_equal(view.numpy(), [[6, 7], [10, 11]])
        assert view.address == tensor.address + 24

        byte_slice = buffer.slice(16, 16)
        assert byte_slice.address == buffer.address + 16
        np.testing.assert_array_equal(byte_slice.tensor((4,), np.int32).numpy(), [4, 5, 6, 7])


def test_invalid_buffer_and_tensor_layouts_are_rejected() -> None:
    with Device.fake() as device:
        with pytest.raises(AllocationError, match="positive"):
            device.allocate(0)
        with pytest.raises(AllocationError, match="power of two"):
            device.allocate(16, alignment=3)

        buffer = device.allocate(16)
        with pytest.raises(AllocationError, match="exceeds"):
            buffer.tensor((5,), np.int32)
        with pytest.raises(AllocationError, match="unit-step"):
            buffer.tensor((4,), np.int32).slice((slice(None, None, 2),))


def test_closing_a_buffer_invalidates_all_derived_views() -> None:
    with Device.fake() as device:
        buffer = device.allocate(16)
        tensor = buffer.tensor((4,), np.int32)
        buffer.close()

        with pytest.raises(BufferClosedError):
            _ = tensor.address
        with pytest.raises(BufferClosedError):
            tensor.numpy()


def test_closed_allocations_are_reused_without_reviving_stale_views() -> None:
    with Device.fake() as device:
        first = device.allocate(64, alignment=16)
        stale = first.tensor((16,), np.int32)
        first_address = first.address
        first.close()

        reused = device.allocate(32, alignment=16)
        assert reused.address == first_address
        assert reused.nbytes == 32
        with pytest.raises(BufferClosedError):
            stale.numpy()


@pytest.mark.hardware
@pytest.mark.skipif(not Path("/dev/dri/renderD128").exists(), reason="VideoCore VII render node is unavailable")
def test_hardware_pool_reuses_the_same_qpu_visible_address() -> None:
    with Device.open(data_area_size=1024 * 1024) as device:
        first = device.allocate(64)
        address = first.address
        first.close()
        assert device.allocate(64).address == address


def test_queue_runs_host_tasks_in_order_and_tracks_timestamps() -> None:
    with Device.fake() as device, device.queue() as queue:
        order: list[int] = []
        first = queue.host_task(lambda: order.append(1))
        second = queue.host_task(lambda: order.append(2))
        second.wait()

        assert order == [1, 2]
        assert first.status is EventStatus.SUCCEEDED
        assert second.started_ns is not None
        assert second.finished_ns is not None
        assert second.submitted_ns <= second.started_ns <= second.finished_ns


def test_queue_waits_for_explicit_dependencies() -> None:
    with Device.fake() as device, device.queue() as queue:
        gate = ThreadEvent()
        observed: list[str] = []
        first = queue.host_task(lambda: (gate.wait(), observed.append("first")))
        second = queue.host_task(lambda: observed.append("second"), wait_for=(first,))

        with pytest.raises(EventTimeoutError):
            second.wait(timeout=0.01)
        gate.set()
        second.wait()
        assert observed == ["first", "second"]


def test_queue_exports_chrome_trace_for_delay_work_and_dependencies(tmp_path) -> None:
    with Device.fake() as device, device.queue() as queue:
        first = queue.host_task(lambda: None, name="prepare")
        second = queue.host_task(lambda: None, wait_for=(first,), name="finish")
        second.wait()

        trace = queue.chrome_trace()
        execution = [entry for entry in trace["traceEvents"] if entry["ph"] == "X"]
        flows = [entry for entry in trace["traceEvents"] if entry["cat"] == "sync"]
        assert {entry["name"] for entry in execution} >= {"prepare", "finish"}
        assert {entry["ph"] for entry in flows} == {"s", "f"}

        trace_path = tmp_path / "queue-trace.json"
        queue.write_chrome_trace(trace_path)
        assert '"traceEvents"' in trace_path.read_text(encoding="utf-8")


def test_queue_propagates_host_task_failures_and_cancellation() -> None:
    with Device.fake() as device, device.queue() as queue:
        failed = queue.host_task(lambda: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(ValueError, match="boom"):
            failed.wait()
        assert failed.status is EventStatus.FAILED

        gate = ThreadEvent()
        blocking = queue.host_task(gate.wait)
        cancelled = queue.host_task(lambda: None)
        assert cancelled.cancel()
        gate.set()
        blocking.wait()
        with pytest.raises(EventCancelledError):
            cancelled.wait()
        assert cancelled.status is EventStatus.CANCELLED


def test_declared_conflicting_buffer_accesses_are_serialized() -> None:
    with Device.fake() as device, device.queue() as queue:
        tensor = device.tensor((4,), np.int32)
        first = queue.host_task(
            lambda: tensor.numpy().__setitem__(slice(None), 7),
            buffers=(tensor.access(AccessMode.WRITE),),
        )
        second = queue.host_task(
            lambda: tensor.numpy().__setitem__(0, tensor.numpy()[0] + 1),
            buffers=(tensor.access(AccessMode.READ_WRITE),),
        )
        second.wait()

        first.wait()
        np.testing.assert_array_equal(tensor.numpy(), [8, 7, 7, 7])


def test_queue_rejects_foreign_device_dependencies_and_accesses() -> None:
    with Device.fake() as first_device, Device.fake() as second_device:
        with first_device.queue() as first_queue, second_device.queue() as second_queue:
            event = second_queue.host_task(lambda: None)
            foreign = second_device.allocate(8)
            with pytest.raises(DependencyError, match="different device"):
                first_queue.host_task(lambda: None, wait_for=(event,))
            with pytest.raises(DependencyError, match="different device"):
                first_queue.host_task(lambda: None, buffers=(foreign.access(AccessMode.READ),))
            event.wait()


def test_kernel_submission_uses_the_backend_and_wraps_errors() -> None:
    with Device.fake() as device, device.queue() as queue:
        calls: list[tuple[tuple[object, ...], tuple[int, int, int]]] = []
        kernel = Kernel("record", lambda backend, args, grid: calls.append((args, grid)))
        event = queue.submit(kernel, ("argument",), grid=(2, 3, 4))
        event.wait()
        assert calls == [(("argument",), (2, 3, 4))]

        bad_kernel = Kernel("bad", lambda backend, args, grid: (_ for _ in ()).throw(RuntimeError("bad")))
        failed = queue.submit(bad_kernel)
        with pytest.raises(KernelError, match="bad"):
            failed.wait()


def test_closed_device_and_queue_reject_new_work() -> None:
    device = Device.fake()
    queue = device.queue()
    device.close()

    with pytest.raises(DeviceClosedError):
        device.allocate(4)
    with pytest.raises(DeviceClosedError):
        queue.host_task(lambda: None)
