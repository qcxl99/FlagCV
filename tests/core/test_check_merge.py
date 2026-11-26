"""Verify acceptance CLI forwarding and report semantics without a GPU."""

import json
import sys
from pathlib import Path

import pytest

from helpers.baseline import comparison
from tools import check_merge
from workloads.core import OPERATORS, python_cases


@pytest.mark.parametrize("correct", [True, False])
def test_forward_target_profile_and_summarize_cases(tmp_path, monkeypatch, correct):
    output, target, base = (tmp_path / name for name in ("status", "target.json", "base.json"))
    commands = []
    states = iter([
        ("passed", 0.10, 9.0, None, None),
        ("unavailable", None, 0.20, None, None),
        ("unavailable", None, None, 6.0, None),
        ("unavailable", None, None, None, "external timing protocol differs"),
        ("failed", None, 9.0, None, None),
    ])

    def run(command, **kwargs):
        commands.append(command)
        if "--install" in command:
            (output / "install-before-relocation").mkdir()
        if "--target-profile" in command:
            report = {}
            for op in OPERATORS:
                rows = []
                for _, row in python_cases(op):
                    status, native, equivalent, raw, reason = next(states, ("unavailable", None, None, None, None))
                    formal = comparison(status, "fixture", native, dict(
                        compared_speedup=equivalent, base_raw_speedup=raw, base_miss_reason=reason))
                    rows.append(dict(row, accuracy_checked=correct, native_status=status,
                                     kernel_dispatch={"variant": "fixture"},
                                     latency=1.0, latency_samples_ms=[1.0], **formal))
                report[op] = {"result": "passed", "details": [{"result": rows}]}
            (output / "python-core.json").write_text(json.dumps(report))
        return check_merge.subprocess.CompletedProcess(command, 0)

    def check_output(command, **kwargs):
        if command[0] == "git":
            return "fixture\n"
        return f"libflagcv.so => {output}/installed/lib/libflagcv.so (0x1)\n"

    monkeypatch.setattr(check_merge.subprocess, "run", run)
    monkeypatch.setattr(check_merge.subprocess, "check_output", check_output)
    monkeypatch.setattr(check_merge.shutil, "copytree", lambda *args: None)
    monkeypatch.setattr(check_merge.shutil, "copy", lambda *args: None)
    monkeypatch.setattr(check_merge, "read_cpp", lambda path: (
        {}, {"key": {"native_status": "unavailable"}}, range(3)))
    monkeypatch.setattr(sys, "argv", ["check_merge.py", "--profile", "hygon", "--output", str(output),
                                    "--opencv-dir", str(tmp_path), "--base-data", str(base),
                                    "--target-profile", str(target)])
    assert check_merge.main() == (0 if correct else 1)
    [benchmark] = [command for command in commands if "--target-profile" in command]
    assert benchmark[-4:] == ["--base-data", str(base), "--target-profile", str(target)]
    acceptance = json.loads((output / "acceptance.json").read_text())
    assert acceptance["target_profile"] == str(target)
    if not correct:
        assert acceptance["status"] == "failed"
        return
    assert acceptance["status"] == "passed"
    assert acceptance["cpp_native_statuses"] == {"unavailable": 1}
    assert acceptance["python_comparisons"] == {
        "cases": 92,
        "statuses": {"passed": 2, "raw_only": 1, "incomparable": 1, "failed": 1, "missing": 87},
        "native_statuses": {"passed": 1, "unavailable": 90, "failed": 1},
        "comparisons": {"origin": {"count": 1, "mean": 0.10},
                        "nv_baseline": {"count": 1, "mean": 0.20}},
    }
