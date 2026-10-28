"""Keep explicit benchmark paths from causing evaluator self-matches."""

import os
from types import SimpleNamespace

import pytest

from scripts.llama_cpp_common import filter_competing_processes, gpu_error_lines
from scripts.run_llama_cpp_qpu_ffn_island_eval import require_no_competing_workloads


def test_gpu_errors_survive_successful_process_status():
    log = "v3d 1002000000.v3d: Resetting GPU for hang.\nv3d: MMU error from client PTB (1)\n"
    assert gpu_error_lines(log) == log.splitlines()


def test_normal_gpu_initialization_is_not_a_timing_rejection():
    assert gpu_error_lines("v3d: initialized render node\nusb: unrelated *ERROR*\n") == []


def test_filter_preserves_other_processes_and_unparseable_lines():
    output = "123 python evaluator --llama-bench /bin/llama-bench\n456 llama-server\nunknown record\n"
    assert filter_competing_processes(output, exclude_pids={123}) == "456 llama-server\nunknown record"


def test_guard_ignores_own_arguments(monkeypatch):
    result = SimpleNamespace(
        returncode=0, stdout=f"{os.getpid()} python evaluator --graph-smoke qpu_graph_smoke\n", stderr=""
    )
    monkeypatch.setattr("scripts.run_llama_cpp_qpu_ffn_island_eval.subprocess.run", lambda *a, **k: result)
    require_no_competing_workloads("test")


def test_guard_still_rejects_an_actual_competing_process(monkeypatch):
    result = SimpleNamespace(
        returncode=0, stdout=f"{os.getpid()} python evaluator\n99999999 llama-server\n", stderr=""
    )
    monkeypatch.setattr("scripts.run_llama_cpp_qpu_ffn_island_eval.subprocess.run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="99999999 llama-server"):
        require_no_competing_workloads("test")
