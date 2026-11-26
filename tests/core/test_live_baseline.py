"""Validate the live comparison contract without requiring a device SDK."""

from types import SimpleNamespace

import pytest

from helpers import benchmark
from workloads.core import python_cases

_, CASE = next(python_cases('add'))


def config(baseline="opencv-cuda"):
    return SimpleNamespace(
        getoption={"baseline": baseline, "benchmark_samples": 3, "benchmark_profile": "default"}.get,
        _flagcv_runtime=SimpleNamespace(vendor="nvidia"),
        _evaluation=SimpleNamespace(timings=0),
    )


@pytest.mark.parametrize("broken", ["candidate", "baseline", "mutation"])
def test_invalid_output_fails_before_timing(monkeypatch, broken):
    torch = pytest.importorskip("torch")
    from reference.add import reference

    args = (torch.ones(2, 3), torch.ones(2, 3), torch.empty(2, 3))
    baseline_args = (*args[:2], torch.empty_like(args[2]))
    closed = []
    timings = []

    class Baseline:
        def __call__(self, *values):
            if broken == "baseline":
                values[2].zero_()
                return values[2]
            return reference(*values)

        def close(self):
            closed.append(True)

    def candidate(a, b, dst):
        if broken == "candidate":
            return dst.zero_()
        result = reference(a, b, dst)
        if broken == "mutation":
            a.zero_()
        return result

    monkeypatch.setattr(
        benchmark, "_prepare_baseline", lambda *a: (Baseline(), baseline_args)
    )
    def timer(function, operands, **kwargs):
        function(*operands)
        timings.append(True)
        return 1.0

    monkeypatch.setattr(benchmark, "_measure_ms", timer)
    if broken == "baseline":
        result = benchmark._time_case(config(), "add", candidate, args, CASE)
        assert result["native_status"] == result["formal_status"] == "failed"
        assert timings == [True] * 3
        assert closed == [True]
    else:
        with pytest.raises(AssertionError):
            benchmark._time_case(config(), "add", candidate, args, CASE)
        assert not timings and not closed


def test_paired_samples_alternate_and_check_both_timers(monkeypatch):
    calls = []
    closed = []

    class Baseline:
        version = "test"

        def __call__(self, *args):
            calls.append("baseline")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(benchmark, "_prepare_baseline", lambda *a: (Baseline(), ()))
    monkeypatch.setattr(benchmark, "_validate_outputs", lambda *a: None)

    def measure(function, args, *, warmup_ms, measurement_ms):
        function(*args)
        return 2.0 if calls[-1] == "candidate" else 8.0

    monkeypatch.setattr(benchmark, "_measure_ms", measure)
    result = benchmark._time_case(
        config(), "add", lambda *a: calls.append("candidate"), (), CASE
    )
    assert calls == [
        "candidate",
        "baseline",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert result["speedup"] == result["formal_speedup"] == 4.0
    assert result["compared_speedup"] is None
    assert closed == [True]


def test_rejects_non_nvidia_baseline():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="requires NVIDIA"):
        benchmark._prepare_baseline("add", (), "hygon")
