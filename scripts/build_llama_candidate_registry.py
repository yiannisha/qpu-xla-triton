"""Build the calibrated dense-Llama placement registry from retained evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from qpu_xla.benchmark import CandidateRecord, CandidateRegistry, CandidateStatus
from qpu_xla.workloads import LLAMA_DENSE_V1

BASE = Path("experiment_logs/20260819-qpu-xla-w8a8")
LINEAR_STEMS = (
    "prefill-h512-t16-fused",
    "prefill-h1024-t64-fused",
    "prefill-h2048-t128-fused",
    "decode-h2048-c512-final",
    "decode-h2048-c2048-final",
    "prefill-h2048-t256-fused",
    "prefill-h4096-t16-hidden-final",
    "prefill-h4096-t16-kv-final",
    "prefill-h4096-t16-up-final",
    "prefill-h4096-t16-down-final",
    "prefill-h4096-t16-lm-final",
    "decode-h3072-c4096-hidden-final",
    "decode-h3072-c4096-kv-final",
    "decode-h3072-c4096-up-final",
    "decode-h3072-c4096-down-final",
    "decode-h3072-c4096-lm-final",
)
SWIGLU_STEMS = tuple(
    f"{case.name}-swiglu-final"
    for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)
)
RUNTIME_STEMS = (
    "prefill-h512-t16-runtime-final",
    "prefill-h1024-t64-runtime-final",
    "prefill-h2048-t128-runtime-final",
    "decode-h2048-c512-runtime-final",
    "decode-h2048-c2048-runtime-final",
    "prefill-h2048-t256-runtime-final",
    "decode-h3072-c4096-runtime-final",
)


def _runtime_speedups() -> dict[str, float]:
    speedups: dict[str, float] = {}
    for stem in RUNTIME_STEMS:
        with (BASE / f"{stem}.json").open(encoding="utf-8") as source:
            payload = cast(dict[str, object], json.load(source))
        case = cast(dict[str, object], payload["case"])
        timings = cast(dict[str, object], payload["timings"])
        raw_speedup = timings.get("speedup_over_fp32_cpu", timings.get("speedup_over_fp32_numpy"))
        if not isinstance(raw_speedup, int | float):
            raise ValueError(f"{stem} does not contain an FP32 model speedup")
        speedups[str(case["name"])] = float(raw_speedup)
    return speedups


def _case_name(record: CandidateRecord) -> str:
    matches: list[str] = [
        str(case.name)
        for case in (*LLAMA_DENSE_V1.tuning, *LLAMA_DENSE_V1.holdout)
        if case.name in record.name
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve one workload case for {record.name}")
    return matches[0]


def main() -> None:
    """Combine stage evidence and demote candidates that regress their full model case."""
    registries = [
        CandidateRegistry.load(BASE / f"{stem}.candidates.json")
        for stem in (*LINEAR_STEMS, *SWIGLU_STEMS)
    ]
    speedups = _runtime_speedups()
    records: list[CandidateRecord] = []
    for record in CandidateRegistry.combine(*registries).records:
        case_name = _case_name(record)
        model_speedup = speedups.get(case_name)
        if (
            record.status is CandidateStatus.SUPPORTED_WIN
            and model_speedup is not None
            and model_speedup < 1.05
        ):
            record = replace(
                record,
                status=CandidateStatus.EXPERIMENTAL_CORRECT_SLOWER,
                reason=(
                    f"standalone stage passed, but the calibrated one-layer model achieved "
                    f"only {model_speedup:.3f}x versus FP32 CPU"
                ),
            )
        records.append(record)
    registry = CandidateRegistry(tuple(records))
    output = BASE / "llama-calibrated.candidates.json"
    registry.save(output)
    print(f"saved {output}: {len(registry.records)} records, {len(registry.supported)} supported")


if __name__ == "__main__":
    main()
