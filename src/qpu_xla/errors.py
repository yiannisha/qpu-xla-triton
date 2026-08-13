"""Exception hierarchy for the public QPU-XLA runtime."""


class QpuXlaError(Exception):
    """Base class for all QPU-XLA failures."""


class DeviceClosedError(QpuXlaError):
    """Raised when an operation is attempted after a device is closed."""


class BufferClosedError(QpuXlaError):
    """Raised when an operation is attempted on a closed allocation."""


class AllocationError(QpuXlaError):
    """Raised when a buffer allocation or layout is invalid."""


class DependencyError(QpuXlaError):
    """Raised when an event dependency cannot be accepted."""


class EventTimeoutError(QpuXlaError):
    """Raised when waiting for an event exceeds the requested timeout."""


class EventCancelledError(QpuXlaError):
    """Raised when waiting on an event that was cancelled."""


class KernelError(QpuXlaError):
    """Raised when a kernel cannot be launched or fails during execution."""


class DslCompileError(QpuXlaError):
    """Raised when a restricted QPU-XLA DSL kernel cannot be compiled safely."""
