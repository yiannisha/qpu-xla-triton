"""Per-question MMLU comparison for the patched llama.cpp evaluator."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from qpu_xla.quality.metrics import paired_bootstrap_delta


@dataclass(frozen=True, slots=True)
class MMLURecord:
    """One normalized-log-probability multiple-choice decision."""

    task_id: int
    correct_index: int
    predicted_index: int
    log_probs: tuple[float, ...]

    @classmethod
    def from_dict(cls: type[MMLURecord], payload: dict[str, Any]) -> MMLURecord:
        """Validate one llama.cpp JSONL record."""
        probabilities = tuple(float(value) for value in payload["log_probs"])
        result = cls(
            int(payload["task_id"]),
            int(payload["correct_index"]),
            int(payload["predicted_index"]),
            probabilities,
        )
        if len(probabilities) < 2 or not all(np.isfinite(probabilities)):
            raise ValueError("MMLU log probabilities must be finite and contain at least two options")
        if not 0 <= result.correct_index < len(probabilities) or not 0 <= result.predicted_index < len(probabilities):
            raise ValueError("MMLU answer index is outside its options")
        return result


def load_mmlu_jsonl(path: str | PathLike[str]) -> tuple[MMLURecord, ...]:
    """Load stable per-item output from the llama.cpp qualification patch."""
    records = tuple(
        MMLURecord.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records or len({record.task_id for record in records}) != len(records):
        raise ValueError("MMLU result must contain unique task IDs")
    return records


def compare_mmlu(baseline: tuple[MMLURecord, ...], candidate: tuple[MMLURecord, ...]) -> dict[str, Any]:
    """Report accuracy and option-score differences on exactly matched tasks."""
    baseline_by_id = {record.task_id: record for record in baseline}
    candidate_by_id = {record.task_id: record for record in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("MMLU baseline and candidate task IDs differ")
    ordered = sorted(baseline_by_id)
    before = [baseline_by_id[index] for index in ordered]
    after = [candidate_by_id[index] for index in ordered]
    if any(left.correct_index != right.correct_index for left, right in zip(before, after, strict=True)):
        raise ValueError("MMLU baseline and candidate correct answers differ")
    baseline_correct = np.asarray([item.predicted_index == item.correct_index for item in before])
    candidate_correct = np.asarray([item.predicted_index == item.correct_index for item in after])
    score_errors = np.asarray(
        [
            np.max(np.abs(np.asarray(right.log_probs) - np.asarray(left.log_probs)))
            for left, right in zip(before, after, strict=True)
        ],
        dtype=np.float64,
    )
    return {
        "questions": len(ordered),
        "baseline_accuracy": float(np.mean(baseline_correct)),
        "candidate_accuracy": float(np.mean(candidate_correct)),
        "accuracy_delta": paired_bootstrap_delta(baseline_correct, candidate_correct).to_dict(),
        "answer_agreement": float(
            np.mean([left.predicted_index == right.predicted_index for left, right in zip(before, after, strict=True)])
        ),
        "cpu_correct_candidate_wrong": int(np.count_nonzero(baseline_correct & ~candidate_correct)),
        "cpu_wrong_candidate_correct": int(np.count_nonzero(~baseline_correct & candidate_correct)),
        "option_logprob_max_abs_error": float(np.max(score_errors, initial=0.0)),
        "option_logprob_mean_max_abs_error": float(np.mean(score_errors)),
    }


def run_llama_mmlu(
    binary: str | PathLike[str],
    model: str | PathLike[str],
    dataset: str | PathLike[str],
    output: str | PathLike[str],
    *,
    tasks: int = 1_000,
    threads: int = 4,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the pinned evaluator in a fresh process and require complete JSONL."""
    if tasks <= 0 or threads <= 0:
        raise ValueError("MMLU task and thread counts must be positive")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(binary).resolve()),
        "-m",
        str(Path(model).resolve()),
        "-f",
        str(Path(dataset).resolve()),
        "--multiple-choice",
        "--multiple-choice-tasks",
        str(tasks),
        "--multiple-choice-jsonl",
        str(output_path.resolve()),
        "-t",
        str(threads),
    ]
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=os.environ.copy() if environment is None else environment,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"llama.cpp MMLU evaluation failed ({process.returncode}): {process.stderr[-4000:]}")
    records = load_mmlu_jsonl(output_path)
    if len(records) != tasks:
        raise RuntimeError(f"llama.cpp emitted {len(records)} MMLU records, expected {tasks}")
    return process


__all__ = ["MMLURecord", "compare_mmlu", "load_mmlu_jsonl", "run_llama_mmlu"]
