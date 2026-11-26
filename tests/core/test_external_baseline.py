"""Verify case matching and native priority without requiring either GPU."""

import json

import pytest

from helpers.baseline import comparison, external_result, summarize
from helpers.benchmark_protocol import PROFILES
from workloads.core import case_id, python_cases, shape_key

PROTOCOL = PROFILES["default"].name


@pytest.fixture
def fixture_data(tmp_path):
    cases = [row for _, row in python_cases("flip")][:3]
    rows = {}
    for index, row in enumerate(cases):
        cfg = row["config"]
        rows[shape_key(cfg)] = {row["dtype"]: {
            "config": cfg, "latency_ms": 0.004 + 0.001 * index, "timing_protocol": PROTOCOL,
            # The shorter kernel is compute-heavy. The weighted aggregate is memory-bound.
            "per_kernel": [{"T_us": 3, "U_mem": 90, "U_cuda": 40, "U_tensor": 0},
                           {"T_us": 1, "U_mem": 20, "U_cuda": 90, "U_tensor": 0}],
        }}
    data = {
        "flip": {"shapes": rows},
        "_environment": {"reference_chip": "H20", "hardware": {"gpu": {"name": "NVIDIA H20"}}},
        "_scaling_factors": {"reference_chip": "H20", "source_specs": "test-specs.json", "chips": [
            {"chip": "H20", "vendor": "nvidia", "measured": {"bandwidth_gbps": 1.0}},
            {"chip": "BW1101", "vendor": "hygon", "measured": {"bandwidth_gbps": 0.50}},
        ]},
    }
    return tmp_path / "base.json", data, cases


def test_equal_shapes_keep_distinct_codes_and_configs(fixture_data):
    path, data, cases = fixture_data
    path.write_text(json.dumps(data))
    assert len({row["case_id"] for row in cases}) == 3
    assert len({shape_key(row["config"]) for row in cases}) == 3
    matched = [external_result(path, row, 0.002, PROTOCOL) for row in cases]
    assert [r["base_raw_speedup"] for r in matched] == pytest.approx([2.0, 2.5, 3.0])
    cfg = cases[0]["config"]
    assert all(case_id(candidate) != case_id(cfg) and shape_key(candidate) != shape_key(cfg)
               for candidate in ({**cfg, "layout": "pitched"},
                                 {**cfg, "params": {**cfg["params"], "mask": "mixed"}}))
    assert external_result(path, cases[0], 0.002, "long")["base_miss_reason"] == (
        "external timing protocol differs"
    )


@pytest.mark.parametrize("mismatch", ["none", "denominator", "precision", "config"])
def test_requires_matched_measured_factor(fixture_data, mismatch):
    path, data, cases = fixture_data
    row = cases[0]
    base = data["flip"]["shapes"][shape_key(row["config"])][row["dtype"]]
    if mismatch == "denominator":
        data["_scaling_factors"]["reference_chip"] = "H800"
    elif mismatch == "config":
        base["config"] = {**row["config"], "output": "allocated"}
    resource = {"T_us": 5, "U_mem": 90, "U_cuda": 10, "U_tensor": 0}
    if mismatch == "precision":
        resource.update(U_mem=10, U_cuda=90)
    path.write_text(json.dumps(data))
    target = {"vendor": "hygon", "chip": "BW1101"}
    profile = {**target, "per_kernel": [resource], "timing_protocol": PROTOCOL, "config": row["config"]}
    result = external_result(path, row, 0.002, PROTOCOL, target, profile)
    if mismatch == "none":
        assert result["compared_speedup"] == 4.0
        assert result["base_miss_reason"] is None
    else:
        assert result["compared_speedup"] is None
        assert result["base_miss_reason"]
        assert result["base_raw_speedup"] == (None if mismatch == "config" else 2.0)


def test_successful_native_wins_and_failure_survives_display_fallback():
    external = {"compared_speedup": 2.0, "base_raw_speedup": 3.0, "base_miss_reason": None}
    native = comparison("passed", None, 0.8, external)
    fallback = comparison("failed", "native launch failed", None, external)
    equivalent = comparison("unavailable", "no native", None, external)
    without_scale = comparison("unavailable", "no native", None, {**external, "compared_speedup": None})
    assert native["formal_speedup"] == native["display_speedup"] == 0.8
    assert fallback["formal_status"] == "failed" and fallback["formal_speedup"] is None
    assert fallback["display_speedup"] == 2.0 and fallback["speedup_fallback_reason"] == "native launch failed"
    assert equivalent["formal_speedup"] == 2.0
    assert without_scale["formal_status"] == "raw_only"
    assert summarize([native, fallback, equivalent, without_scale]) == {
        "cases": 4, "statuses": {"passed": 2, "failed": 1, "raw_only": 1}, "native_statuses": {},
        "comparisons": {"origin": {"count": 1, "mean": 0.8}, "nv_baseline": {"count": 1, "mean": 2.0}},
    }
