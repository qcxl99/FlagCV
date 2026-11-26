"""Match Optest records and report native and equivalent performance."""

import math
from functools import lru_cache
from pathlib import Path
from statistics import fmean
import json

from workloads.core import canonical, shape_key


@lru_cache(maxsize=None)
def read_json(path):
    return json.loads(Path(path).read_text())


def _positive(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _chip(name):
    return name.removeprefix("NVIDIA ") if isinstance(name, str) else None


def _resource(profile):
    """Select a unique resource using duration-weighted profiling counters."""
    kernels = profile.get("per_kernel", [])
    if not kernels or any(not _positive(kernel.get("T_us")) for kernel in kernels):
        return None
    fields = {"cuda": "U_cuda", "tensor": "U_tensor", "mem": "U_mem"}
    durations = [kernel["T_us"] for kernel in kernels]
    utilization = {}
    for resource, field in fields.items():
        metrics = [kernel.get(field) for kernel in kernels]
        if any(not isinstance(u, (int, float)) or not math.isfinite(u) or not 0 <= u <= 100
               for u in metrics):
            return None
        utilization[resource] = sum(t * u for t, u in zip(durations, metrics)) / sum(durations)
    ranked = sorted(utilization, key=utilization.get, reverse=True)
    first, second = ranked[:2]
    if utilization[first] == 0 or math.isclose(utilization[first], utilization[second]):
        return None
    precision = profile.get("resource_precision")
    return (first, precision) if first == "mem" or precision else None


def _scale(data, baseline, config, protocol, target, target_profile):
    environment = data.get("_environment", {})
    gpu = environment.get("hardware", {}).get("gpu", {})
    baseline_chip = _chip(gpu.get("name"))
    factors = data.get("_scaling_factors", {})
    if not str(gpu.get("name", "")).startswith("NVIDIA ") or not baseline_chip:
        return None, "missing actual NVIDIA baseline chip"
    if (baseline_chip != factors.get("reference_chip")
            or baseline_chip != environment.get("reference_chip")):
        return None, "scaling denominator differs from the actual NVIDIA baseline chip"
    if not factors.get("source_specs"):
        return None, "missing measured factor provenance"
    if not target or not target_profile:
        return None, "missing target resource measurements"
    if (target_profile.get("chip") != target["chip"]
            or target_profile.get("vendor") != target["vendor"]
            or target_profile.get("timing_protocol") != protocol
            or canonical(target_profile.get("config")) != canonical(config)):
        return None, "target resource measurements do not match this chip and case"
    resource = _resource(baseline)
    if resource is None or _resource(target_profile) != resource:
        return None, "matching bottleneck resource and precision could not be established"

    def select(vendor, chip):
        records = [row["measured"] for row in factors.get("chips", [])
                   if row.get("vendor") == vendor and row.get("chip") == chip and row.get("measured")]
        return records[0] if len(records) == 1 else {}

    def value(measurements):
        kind, precision = resource
        if kind == "mem":
            return measurements.get("bandwidth_gbps")
        return measurements.get("vector" if kind == "cuda" else "tensor", {}).get(precision)

    denominator = value(select("nvidia", baseline_chip))
    numerator = value(select(target["vendor"], target["chip"]))
    if denominator != 1.0 or not _positive(numerator):
        return None, "missing measured factor for the matched resource and precision"
    return numerator, None


def external_result(path, row, latency, protocol, target=None, target_profile=None):
    config, dtype = row["config"], row["dtype"]
    key = shape_key(config)
    result = dict(base_latency_ms=None, base_raw_speedup=None, base_scaling_factor=None,
                  compared_speedup=None, base_data_source=str(path) if path else None,
                  base_shape_key=key, base_miss_reason=None)
    if path is None:
        result["base_miss_reason"] = "external baseline not configured"
        return result
    data = read_json(path)
    record = data.get(config["op_name"], {}).get("shapes", {}).get(key, {}).get(dtype, {})
    if not record:
        result["base_miss_reason"] = "external case missing"
    elif canonical(record.get("config")) != canonical(config):
        result["base_miss_reason"] = "external case configuration differs"
    elif record.get("timing_protocol") != protocol:
        result["base_miss_reason"] = "external timing protocol differs"
    elif not _positive(record.get("latency_ms")) or record.get("status", "passed") != "passed":
        result["base_miss_reason"] = "external measurement unavailable"
    else:
        base = record["latency_ms"]
        scale, reason = _scale(data, record, config, protocol, target, target_profile)
        raw = base / latency
        result.update(base_latency_ms=base, base_raw_speedup=raw, base_scaling_factor=scale,
                      compared_speedup=raw / scale if scale is not None else None,
                      base_miss_reason=reason)
    return result


def comparison(native_status, native_reason, native_speedup, external):
    equivalent = external["compared_speedup"]
    raw = external["base_raw_speedup"]
    if native_status == "passed":
        status, source, speedup = "passed", "origin", native_speedup
    elif native_status == "failed":
        status, source, speedup = "failed", "origin", None
    elif equivalent is not None:
        status, source, speedup = "passed", "nv_baseline", equivalent
    else:
        incompatible = {"external case configuration differs", "external timing protocol differs"}
        status = ("raw_only" if raw is not None
                  else "incomparable" if external["base_miss_reason"] in incompatible else "missing")
        source, speedup = "nv_baseline", None
    target = 0.90 if source == "origin" else 0.80

    displayed = speedup
    displayed_source = source if displayed is not None else None
    fallback_reason = None
    if displayed is None and equivalent is not None:
        displayed, displayed_source = equivalent, "nv_baseline"
        fallback_reason = native_reason
    return {
        "formal_status": status, "formal_speedup": speedup, "formal_source": source,
        "reference_target": target, "reference_gap": speedup - target if speedup is not None else None,
        "display_speedup": displayed, "speedup_source": displayed_source,
        "speedup_fallback_reason": fallback_reason,
    }


def summarize(rows):
    status = {}
    native_status = {}
    ratio = {"origin": [], "nv_baseline": []}
    for row in rows:
        state = row["formal_status"]
        status[state] = status.get(state, 0) + 1
        if (native_state := row.get("native_status")) is not None:
            native_status[native_state] = native_status.get(native_state, 0) + 1
        if row["formal_speedup"] is not None:
            ratio[row["formal_source"]].append(row["formal_speedup"])
    return {"cases": len(rows), "statuses": status, "native_statuses": native_status,
            "comparisons": {k: {"count": len(v), "mean": fmean(v) if v else None}
                            for k, v in ratio.items()}}
