"""Host controls for the bounded ATen HIP comparative path."""

from types import SimpleNamespace

import pytest

from adapters import aten_hip
from helpers import benchmark
from workloads import bitwise_not
from workloads.core import python_cases


@pytest.mark.parametrize("op,left,right,wrapped", [
    ("add", 250, 10, 4), ("subtract", 0, 1, 255), ("multiply", 200, 2, 144),
])
def test_plain_aten_uint8_arithmetic_is_not_saturating(op, left, right, wrapped):
    torch = pytest.importorskip("torch")
    operands = tuple(torch.tensor(number, dtype=torch.uint8) for number in (left, right))
    assert getattr(torch, op)(*operands, out=torch.empty_like(operands[0])).item() == wrapped
    for vendor in ("amd", "hygon"):
        assert aten_hip.qualification(op, (), "torch.uint8", vendor)[0] == "not_equivalent"
        assert aten_hip.qualification(op, (), "torch.float32", vendor)[0] == "pending"


def test_platform_gate_excludes_unverified_targets_not_versions(monkeypatch):
    monkeypatch.delenv("OPTEST_TARGET_CHIP", raising=False)
    properties = SimpleNamespace(name="BW", gcnArchName="gfx936:other", multi_processor_count=80)
    version = SimpleNamespace(hip="6.3.26045", git_version="275d08c233b2cf705e4b66d9f98f7590daae5cba")
    torch = SimpleNamespace(__version__="2.9.0", version=version,
                            cuda=SimpleNamespace(get_device_properties=lambda device: properties))
    cuda = SimpleNamespace(type="cuda")
    assert aten_hip._qualified_platform("hygon", torch, cuda)
    assert not aten_hip._qualified_platform("amd", torch, cuda)
    for arch in ("gfx938", "gfx1100"):
        properties.gcnArchName = arch
        assert not aten_hip._qualified_platform("hygon", torch, cuda)
    properties.gcnArchName = "gfx936"
    version.git_version = "different-build"
    assert aten_hip._qualified_platform("hygon", torch, cuda)
    monkeypatch.setenv("OPTEST_TARGET_CHIP", "BW1101")
    assert not aten_hip._qualified_platform("hygon", torch, cuda)


def test_nonstandard_layout_stays_pending(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(aten_hip, "_qualified_platform", lambda *args: True)
    src, dst = [torch.empty((63, 129), dtype=torch.uint8) for _ in range(2)]
    assert aten_hip.qualification("bitwise_not", (src, dst), "torch.uint8", "hygon")[0] == "eligible"
    for operands in ((src.T, dst), (src, src), (src, dst, dst)):
        assert aten_hip.qualification("bitwise_not", operands, "torch.uint8", "hygon")[0] == "pending"
    assert aten_hip.qualification("flip", (), "torch.uint8", "hygon")[0] == "not_equivalent"
    lhs, rhs, out = [torch.empty_like(src, dtype=torch.float32) for _ in range(3)]
    assert aten_hip.qualification("multiply", (lhs, rhs, out, 1.0), "torch.float32", "hygon")[0] == "eligible"
    assert aten_hip.qualification("multiply", (lhs, rhs, out, 2.0), "torch.float32", "hygon")[0] == "pending"


def test_external_candidate_uses_verified_launch_boundary_not_source(monkeypatch):
    from helpers.launch_boundary import CURRENT_STREAM_OUT

    monkeypatch.setattr(benchmark, "_check_stream_allocation", lambda fn, args: True)
    assert not benchmark._aten_candidate_boundary(SimpleNamespace(source="override"), ())
    assert not benchmark._aten_candidate_boundary(SimpleNamespace(source="builtin"), ())
    assert benchmark._aten_candidate_boundary(SimpleNamespace(
        source="override", execution_boundary=CURRENT_STREAM_OUT), ())


@pytest.mark.parametrize("native_status", ["passed", "failed", "pending", "disabled"])
def test_native_path_preserves_latency_and_formal_status(monkeypatch, native_status):
    torch = pytest.importorskip("torch")
    case, row = next(python_cases("bitwise_not"))
    args = bitwise_not.make_inputs(case, "cpu")
    monkeypatch.setattr(aten_hip, "_qualified_platform", lambda *args: native_status != "pending")
    config = SimpleNamespace(
        getoption={"baseline": "none" if native_status == "disabled" else "auto",
                   "benchmark_samples": 3, "benchmark_profile": "default"}.get,
        _flagcv_runtime=SimpleNamespace(vendor="hygon"), _evaluation=SimpleNamespace(timings=0),
    )

    def candidate(src, dst):
        return torch.bitwise_not(src, out=dst)

    candidate.source = "override"
    monkeypatch.setattr(benchmark, "_aten_candidate_boundary", lambda *args: True)
    monkeypatch.setattr(benchmark, "_check_stream_allocation", lambda *args: True)

    def time(function, operands, **kwargs):
        if operands[-1] is not args[-1] and native_status == "failed":
            raise RuntimeError("native timing failed")
        assert function(*operands) is operands[-1]
        return 1.0 if operands[-1] is args[-1] else 2.0

    monkeypatch.setattr(benchmark, "_measure_ms", time)
    result = benchmark._time_case(config, "bitwise_not", candidate, args, row)
    assert result["native_status"] == native_status
    assert result["latency"] == 1.0
    if native_status == "passed":
        assert result["native_comparator"] == result["baseline_source"] == "aten_hip_out"
        assert result["formal_source"] == "origin" and result["formal_speedup"] == 2.0
    else:
        assert result["baseline_source"] is None and result["formal_speedup"] is None
        assert result["formal_status"] == ("failed" if native_status == "failed" else "missing")
