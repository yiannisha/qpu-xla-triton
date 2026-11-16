"""Keep legacy device tests out of CPU-only runs."""

from pathlib import Path

import pytest

# These legacy modules open Driver/DRM_V3D directly in every test. New tests
# should carry explicit @pytest.mark.hardware decorators at the test itself.
LEGACY_HARDWARE_MODULES = {
    "test_alu.py",
    "test_branch.py",
    "test_condition_codes.py",
    "test_driver.py",
    "test_drm.py",
    "test_guard.py",
    "test_labels.py",
    "test_parallel.py",
    "test_raw.py",
    "test_registers.py",
    "test_signals.py",
    "test_tmu.py",
    "test_unifa.py",
    "test_v3d.py",
}


def has_vc7_device():
    """Identify the target board instead of an unrelated desktop render node."""
    model = Path("/proc/device-tree/model")
    return model.exists() and "Raspberry Pi 5" in model.read_text() and Path("/dev/dri/renderD128").exists()


def pytest_ignore_collect(collection_path):
    """Avoid importing Pi-only C register access on portable hosts."""
    if collection_path.name in LEGACY_HARDWARE_MODULES and not has_vc7_device():
        return True
    return None


def pytest_collection_modifyitems(items):
    """Classify known legacy modules and skip device tests on portable hosts."""
    available = has_vc7_device()
    for item in items:
        if item.path.name in LEGACY_HARDWARE_MODULES:
            item.add_marker(pytest.mark.hardware)
        if item.get_closest_marker("hardware") and not available:
            item.add_marker(pytest.mark.skip(reason="Raspberry Pi 5 VideoCore VII device is unavailable"))
