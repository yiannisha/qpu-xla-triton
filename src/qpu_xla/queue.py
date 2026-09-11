"""In-order asynchronous queue and event primitives."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from queue import Queue as ThreadQueue
from threading import Condition, Lock, Thread
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, Self

from qpu_xla.errors import DependencyError, DeviceClosedError, EventCancelledError, EventTimeoutError
from qpu_xla.kernel import Kernel, LaunchConfig
from qpu_xla.memory import BufferAccess

if TYPE_CHECKING:
    from qpu_xla.device import Device


class EventStatus(Enum):
    """The lifecycle states of a submitted runtime operation."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Event:
    """Completion state, failure propagation, and timestamps for one operation."""

    def __init__(
        self: Self,
        queue: Queue,
        *,
        sequence: int,
        name: str,
        category: str,
        dependencies: tuple[Event, ...],
    ) -> None:
        """Create a pending event owned by one queue."""
        self._queue = queue
        self._sequence = sequence
        self._name = name
        self._category = category
        self._dependencies = dependencies
        self._condition = Condition()
        self._status = EventStatus.PENDING
        self._exception: BaseException | None = None
        self._submitted_ns = monotonic_ns()
        self._started_ns: int | None = None
        self._finished_ns: int | None = None

    @property
    def queue(self: Self) -> Queue:
        """Return the queue that owns this event."""
        return self._queue

    @property
    def name(self: Self) -> str:
        """Return the operation label used by profiling and trace export."""
        return self._name

    @property
    def status(self: Self) -> EventStatus:
        """Return the current event status."""
        with self._condition:
            return self._status

    @property
    def submitted_ns(self: Self) -> int:
        """Return the monotonic submission timestamp."""
        return self._submitted_ns

    @property
    def started_ns(self: Self) -> int | None:
        """Return the start timestamp when execution has begun."""
        with self._condition:
            return self._started_ns

    @property
    def finished_ns(self: Self) -> int | None:
        """Return the completion timestamp when execution has ended."""
        with self._condition:
            return self._finished_ns

    @property
    def exception(self: Self) -> BaseException | None:
        """Return the operation failure, if any."""
        with self._condition:
            return self._exception

    @property
    def done(self: Self) -> bool:
        """Whether the event reached a terminal state."""
        with self._condition:
            return self._status in {EventStatus.SUCCEEDED, EventStatus.FAILED, EventStatus.CANCELLED}

    def cancel(self: Self) -> bool:
        """Cancel a queued operation if the worker has not started it."""
        with self._condition:
            if self._status is not EventStatus.PENDING:
                return False
            self._status = EventStatus.CANCELLED
            self._finished_ns = monotonic_ns()
            self._condition.notify_all()
            return True

    def wait(self: Self, timeout: float | None = None) -> None:
        """Wait for completion and re-raise cancellation or task failures."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        with self._condition:
            if not self.done:
                completed = self._condition.wait_for(lambda: self.done, timeout=timeout)
                if not completed:
                    raise EventTimeoutError("event did not complete before the timeout")
            if self._status is EventStatus.CANCELLED:
                raise EventCancelledError("event was cancelled")
            if self._status is EventStatus.FAILED:
                assert self._exception is not None
                raise self._exception

    def _start(self: Self) -> bool:
        """Mark execution started, returning false for a cancelled event."""
        with self._condition:
            if self._status is EventStatus.CANCELLED:
                return False
            if self._status is not EventStatus.PENDING:
                raise RuntimeError("event cannot be started more than once")
            self._status = EventStatus.RUNNING
            self._started_ns = monotonic_ns()
            return True

    def _finish(self: Self, exception: BaseException | None = None) -> None:
        """Mark execution complete and retain a failure for waiters."""
        with self._condition:
            if self._status is EventStatus.CANCELLED:
                return
            self._exception = exception
            self._status = EventStatus.FAILED if exception is not None else EventStatus.SUCCEEDED
            self._finished_ns = monotonic_ns()
            self._condition.notify_all()


@dataclass(slots=True)
class _Submission:
    """One pending operation consumed by the queue worker."""

    event: Event
    action: Callable[[], None]
    dependencies: tuple[Event, ...]


class Queue:
    """A single-worker, in-order queue for QPU and explicitly declared host work."""

    def __init__(self: Self, device: Device) -> None:
        """Start a dedicated worker for one device's submissions."""
        self._device = device
        self._submissions: ThreadQueue[_Submission | None] = ThreadQueue()
        self._lock = Lock()
        self._closed = False
        self._tail: Event | None = None
        self._events: list[Event] = []
        self._next_sequence = 0
        self._live_accesses: list[tuple[BufferAccess, Event]] = []
        self._worker = Thread(target=self._run, name="qpu-xla-submit", daemon=True)
        self._worker.start()

    @property
    def device(self: Self) -> Device:
        """Return the device associated with this queue."""
        return self._device

    @property
    def closed(self: Self) -> bool:
        """Whether this queue no longer accepts new work."""
        with self._lock:
            return self._closed

    def submit(
        self: Self,
        kernel: Kernel,
        args: tuple[Any, ...] = (),
        *,
        grid: tuple[int, int, int] = (1, 1, 1),
        wait_for: Iterable[Event] = (),
        buffers: Iterable[BufferAccess] = (),
    ) -> Event:
        """Queue a kernel launch and return its completion event.

        The v0 queue is deliberately in-order. The single worker preserves
        submission order without turning a preceding failure into an implicit
        dependency failure. Explicit dependencies are preserved so callers can
        compose work across queues, and conflicting declared buffer ranges are
        recorded for a future out-of-order implementation.
        """
        launch = LaunchConfig(grid)
        return self._enqueue(
            lambda: kernel.execute(self._device.backend, args, launch),
            wait_for=wait_for,
            buffers=buffers,
            name=kernel.name,
            category="qpu",
        )

    def host_task(
        self: Self,
        fn: Callable[[], None],
        *,
        wait_for: Iterable[Event] = (),
        buffers: Iterable[BufferAccess] = (),
        name: str | None = None,
    ) -> Event:
        """Queue host work with the same dependency and hazard declarations."""
        task_name = name if name is not None else str(getattr(fn, "__qualname__", "host_task"))
        return self._enqueue(fn, wait_for=wait_for, buffers=buffers, name=task_name, category="host")

    def _enqueue(
        self: Self,
        action: Callable[[], None],
        *,
        wait_for: Iterable[Event],
        buffers: Iterable[BufferAccess],
        name: str,
        category: str,
    ) -> Event:
        """Validate a submission and place it behind its computed dependencies."""
        declared = tuple(buffers)
        for access in declared:
            if access.buffer.device is not self._device:
                raise DependencyError("a buffer access belongs to a different device")
        with self._lock:
            if self._closed:
                raise DeviceClosedError("queue is closed")
            dependencies = self._normalize_dependencies(wait_for)
            self._live_accesses = [(access, event) for access, event in self._live_accesses if not event.done]
            for access, event in self._live_accesses:
                if any(access.conflicts_with(previous) for previous in declared):
                    # The single worker already serializes these operations.
                    # Keep the event in the hazard ledger so a later
                    # out-of-order queue can turn this into a dependency.
                    continue
            event = Event(
                self,
                sequence=self._next_sequence,
                name=name,
                category=category,
                dependencies=dependencies,
            )
            self._next_sequence += 1
            self._tail = event
            self._events.append(event)
            self._live_accesses.extend((access, event) for access in declared)
            self._submissions.put(_Submission(event, action, tuple(dependencies)))
            return event

    def _normalize_dependencies(self: Self, wait_for: Iterable[Event]) -> tuple[Event, ...]:
        """Reject foreign events and collapse duplicate dependencies."""
        dependencies: list[Event] = []
        for event in wait_for:
            if not isinstance(event, Event):
                raise DependencyError("wait_for must contain Event instances")
            if event.queue.device is not self._device:
                raise DependencyError("event dependency belongs to a different device")
            if event not in dependencies:
                dependencies.append(event)
        return tuple(dependencies)

    def chrome_trace(self: Self) -> dict[str, list[dict[str, object]]]:
        """Return a Chrome Trace Event payload for completed queue activity.

        The payload has a queue-delay slice and an execution slice per started
        event. Explicit dependencies are represented as Chrome flow arrows,
        allowing trace viewers to display synchronization between submissions.
        Timestamps are monotonic-clock microseconds, which Chrome accepts as a
        common relative time base.
        """
        with self._lock:
            events = tuple(self._events)
        trace_events: list[dict[str, object]] = []
        for event in events:
            started_ns = event.started_ns
            finished_ns = event.finished_ns
            if started_ns is None:
                continue
            submitted_us = event.submitted_ns / 1_000
            started_us = started_ns / 1_000
            if started_ns > event.submitted_ns:
                trace_events.append(
                    {
                        "name": f"queue delay: {event.name}",
                        "cat": "queue",
                        "ph": "X",
                        "pid": "qpu-xla",
                        "tid": "submission",
                        "ts": submitted_us,
                        "dur": started_us - submitted_us,
                        "args": {"event": event._sequence},
                    }
                )
            if finished_ns is not None:
                trace_events.append(
                    {
                        "name": event.name,
                        "cat": event._category,
                        "ph": "X",
                        "pid": "qpu-xla",
                        "tid": event._category,
                        "ts": started_us,
                        "dur": max(0, finished_ns - started_ns) / 1_000,
                        "args": {"event": event._sequence, "status": event.status.value},
                    }
                )
            for dependency in event._dependencies:
                dependency_finished_ns = dependency.finished_ns
                if dependency_finished_ns is None:
                    continue
                flow_id = f"{dependency._sequence}->{event._sequence}"
                trace_events.extend(
                    (
                        {
                            "name": "dependency",
                            "cat": "sync",
                            "ph": "s",
                            "pid": "qpu-xla",
                            "tid": dependency._category,
                            "ts": dependency_finished_ns / 1_000,
                            "id": flow_id,
                        },
                        {
                            "name": "dependency",
                            "cat": "sync",
                            "ph": "f",
                            "pid": "qpu-xla",
                            "tid": event._category,
                            "ts": started_us,
                            "id": flow_id,
                        },
                    )
                )
        return {"traceEvents": trace_events}

    def consume_completed_event_counts(self: Self) -> dict[str, int]:
        """Count and release completed profiling events by execution category.

        Long-running model qualification can submit millions of small kernels.
        Callers that need aggregate dispatch coverage rather than a full Chrome
        trace can consume completed events at safe synchronization boundaries
        to keep profiling memory bounded.
        """
        with self._lock:
            completed: list[Event] = []
            retained: list[Event] = []
            for event in self._events:
                (completed if event.done else retained).append(event)
            self._events = retained
            self._live_accesses = [
                (access, event) for access, event in self._live_accesses if not event.done
            ]
        counts: dict[str, int] = {}
        for event in completed:
            counts[event._category] = counts.get(event._category, 0) + 1
        return counts

    def write_chrome_trace(self: Self, path: str | PathLike[str]) -> None:
        """Serialize :meth:`chrome_trace` to a Chrome Trace Event JSON file."""
        with open(path, "w", encoding="utf-8") as trace_file:
            json.dump(self.chrome_trace(), trace_file, separators=(",", ":"))

    def _run(self: Self) -> None:
        """Execute submissions serially, keeping failures attached to events."""
        while True:
            submission = self._submissions.get()
            if submission is None:
                self._submissions.task_done()
                return
            try:
                if submission.event._start():
                    for dependency in submission.dependencies:
                        dependency.wait()
                    submission.action()
            except BaseException as exc:
                submission.event._finish(exc)
            else:
                submission.event._finish()
            finally:
                self._submissions.task_done()

    def close(self: Self) -> None:
        """Stop accepting work and wait until all already-submitted work finishes."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._submissions.put(None)
        self._worker.join()

    def __enter__(self: Self) -> Self:
        """Enter a context-managed queue lifetime."""
        if self.closed:
            raise DeviceClosedError("queue is closed")
        return self

    def __exit__(self: Self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the queue at context exit."""
        self.close()
