"""Protect evidence integrity and bounded artifact execution."""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.artifact import run_profile
from scripts.artifact_common import contained, read_record
from scripts.summarize_rebuttal_remaining import summarize


def test_archived_reader_checks_original_content_hash(tmp_path):
    data = b'{"completed": true}'
    (tmp_path / "record.json.gz").write_bytes(gzip.compress(data))
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_path": "old-workspace/todo3/ablation.json",
                        "archive_path": "record.json.gz",
                        "sha256_uncompressed": hashlib.sha256(data).hexdigest(),
                    }
                ]
            }
        )
    )
    assert read_record(tmp_path, "todo3/ablation.json")["completed"]
    (tmp_path / "record.json.gz").write_bytes(gzip.compress(b'{"completed": false}'))
    with pytest.raises(ValueError, match="provenance mismatch"):
        read_record(tmp_path, "todo3/ablation.json")


def test_inventory_cannot_escape_package(tmp_path):
    with pytest.raises(ValueError, match="unsafe artifact path"):
        contained(tmp_path, "../outside")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes package"):
        contained(tmp_path, "escape/outside")


def test_archived_rebuttal_regeneration_preserves_rejections_and_boundaries():
    archive = Path(__file__).resolve().parents[1] / "experiment_logs/20261028-rebuttal"
    result = summarize(archive)
    retained = json.loads((archive / "summary.json").read_text())
    assert result == retained
    assert "unretained" in result["todo5"]["depthwise_status"]
    assert result["todo8"]["configurations"] == 112


@pytest.mark.parametrize("only, expected", [(None, ["ordinary"]), ("diagnostic", ["diagnostic"])])
def test_diagnostic_stage_requires_explicit_selection(tmp_path, only, expected):
    config = tmp_path / "profiles.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "quality": {
                        "stages": [
                            {"id": "ordinary", "argv": ["{python}", "-V"]},
                            {"id": "diagnostic", "enabled": False, "argv": ["{python}", "-V"]},
                        ]
                    }
                }
            }
        )
    )
    args = argparse.Namespace(config=config, profile="quality", only=only, execute=False, output=tmp_path / "run")
    result = run_profile(args)
    assert not result["executed"]
    assert [s["id"] for s in result["stages"]] == expected
    assert not args.output.exists()


def test_paper_profile_uses_the_selected_isolated_native_build(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    build = tmp_path / "isolated-build"
    monkeypatch.setenv("BUILD_ROOT", str(build))
    monkeypatch.delenv("INTEGRATION_BUILD", raising=False)
    monkeypatch.setenv("LLAMA_ROOT", str(tmp_path / "llama"))
    monkeypatch.setenv("GEMMA_MODEL", str(tmp_path / "model.gguf"))
    args = argparse.Namespace(
        config=root / "artifact/profiles.json", profile="paper", only=None, execute=False, output=tmp_path / "run"
    )
    result = run_profile(args)
    for stage in result["stages"]:
        for flag in ("--plugin", "--stride-smoke", "--graph-smoke", "--model-smoke"):
            selected = Path(stage["argv"][stage["argv"].index(flag) + 1])
            assert selected.is_relative_to(build)


def test_profile_rejects_failed_control_gate_even_when_child_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.artifact.check_environment", lambda hardware: {"ok": True})
    worker = tmp_path / "run_rebuttal_diagnostics.py"
    worker.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "out=Path(sys.argv[2]);out.mkdir(parents=True)\n"
        "(out/'process.json').write_text(json.dumps({'completed':True,"
        "'retained_for_latency':False,'kernel_error_lines':['MMU error']}))\n"
    )
    config = tmp_path / "profiles.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "paper": {
                        "stages": [
                            {"id": "prompt", "argv": [sys.executable, str(worker), "--output", "{output}/control"]}
                        ]
                    }
                }
            }
        )
    )
    args = argparse.Namespace(config=config, profile="paper", only=None, execute=True, output=tmp_path / "run")
    with pytest.raises(ValueError, match="stage prompt failed"):
        run_profile(args)
    record = json.loads((args.output / "campaign.json").read_text())
    assert record["completed"] is False
    assert record["failed_stage"] == "prompt"
    assert record["stages"][0]["returncode"] == 0
    assert record["stages"][0]["control"]["retained_for_latency"] is False
