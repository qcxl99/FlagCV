"""Match Optest records and report native and equivalent performance."""

import json
import math
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from workloads.core import canonical, shape_key
from runtime.identity import cuda_chip


@lru_cache(maxsize=None)
def read_json(path):
    return json.loads(Path(path).read_text())


def _positive(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


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
        if any(not isinstance(u, (int, float)) or isinstance(u, bool)
               or not math.isfinite(u) or not 0 <= u <= 100
               for u in metrics):
            return None
        utilization[resource] = sum(t * u for t, u in zip(durations, metrics)) / sum(durations)
    ranked = sorted(utilization, key=utilization.get, reverse=True)
    first, second = ranked[:2]
    if utilization[first] <= 0 or math.isclose(utilization[first], utilization[second]):
        return None
    precision = profile.get("resource_precision")
    return (first, None) if first == "mem" else (first, precision) if precision else None


def _scale(data, baseline, config, protocol, target, target_profile):
    environment = data.get("_environment", {})
    gpu = environment.get("hardware", {}).get("gpu", {})
    baseline_chip = cuda_chip(gpu.get("name"))
    factors = data.get("_scaling_factors", {})
    provenance = {"basis": "dual_profile" if target_profile is not None else "nvidia_reference",
                  "reference_chip": baseline_chip, "target": target,
                  "factor_source": factors.get("source_specs"),
                  "reference_profile": baseline.get("resource_profile")}

    def reject(reason):
        return None, reason, provenance

    if not baseline_chip:
        return reject("missing actual NVIDIA baseline chip")
    if (baseline_chip != factors.get("reference_chip")
            or baseline_chip != environment.get("reference_chip")):
        return reject("scaling denominator differs from the actual NVIDIA baseline chip")
    if not isinstance(factors.get("source_specs"), str) or not factors["source_specs"]:
        return reject("missing measured factor provenance")
    if not target or not target.get("chip"):
        return reject((target or {}).get("identity_basis") or "target chip identity not established")
    profile = baseline.get("resource_profile") or {}
    if profile.get("exclusion"):
        return reject(profile["exclusion"])
    if (profile.get("source") != "ncu" or profile.get("cache_control") != "all"
            or not profile.get("metrics") or profile.get("repetitions") != 1):
        return reject("missing NVIDIA resource counters or cold-cache provenance")
    duration = profile.get("kernel_duration_us")
    if not _positive(duration):
        return reject("missing independently measured NVIDIA kernel duration")
    resource = _resource(baseline)
    if resource is None:
        return reject("no dominant NVIDIA resource with verified precision")
    kind, precision = resource
    provenance.update(resource=kind, precision=precision)
    if kind == "mem":
        if not _positive(profile.get("dram_bytes")):
            return reject("missing measured DRAM traffic for the NVIDIA invocation")
        # A cold invocation can access DRAM even when its working set fits L2.
        working_set = profile.get("working_set_bytes")
        for label, size in (("reference", profile.get("reference_l2_bytes")),
                            ("target", target.get("l2_bytes"))):
            if _positive(working_set) and _positive(size):
                provenance[f"working_set_fits_{label}_l2"] = working_set <= size
    elif (kind != "cuda" or precision != "fp32" or not profile.get("precision_source")
          or not _positive(profile.get("cuda_flops", {}).get("fp32"))
          or profile.get("cuda_flops", {}).get("fp16") != 0):
        return reject("no validated arithmetic precision for the dominant resource")
    if target_profile is not None:
        target_measurement = target_profile.get("resource_profile") or {}
        provenance["target_profile"] = target_measurement
        if (target_profile.get("chip") != target["chip"]
                or target_profile.get("vendor") != target["vendor"]
                or target_profile.get("timing_protocol") != protocol
                or canonical(target_profile.get("config")) != canonical(config)):
            return reject("target resource measurements do not match this chip and case")
        if target_measurement.get("exclusion"):
            return reject(target_measurement["exclusion"])
        if not target_measurement.get("source") or target_measurement.get("cache_control") != "all":
            return reject("target diagnostic lacks counter or matching cache provenance")
        if "kernel_duration_us" in target_measurement:
            target_duration = target_measurement["kernel_duration_us"]
            if not _positive(target_duration) or target_duration < 5:
                return reject("target diagnostic indicates short or invalid kernel duration")
        if kind == "cuda" and not target_measurement.get("precision_source"):
            return reject("target diagnostic lacks measured arithmetic precision provenance")
        if _resource(target_profile) != resource:
            return reject("target bottleneck or precision differs from the NVIDIA reference")

    def select(vendor, chip):
        records = [row["measured"] for row in factors.get("chips", [])
                   if row.get("vendor") == vendor and row.get("chip") == chip and row.get("measured")]
        return records[0] if len(records) == 1 else {}

    def value(measurements):
        if kind == "mem":
            return measurements.get("bandwidth_gbps")
        return measurements.get("vector" if kind == "cuda" else "tensor", {}).get(precision)

    denominator = value(select("nvidia", baseline_chip))
    numerator = value(select(target["vendor"], target["chip"]))
    provenance.update(reference_factor=denominator, target_factor=numerator)
    if not _positive(denominator) or denominator != 1.0 or not _positive(numerator):
        return reject("missing measured factor for the matched resource and precision")
    return numerator, None, provenance


def external_result(path, row, latency, protocol, target=None, target_profile=None):
    config, dtype = row["config"], row["dtype"]
    key = shape_key(config)
    result = dict(base_latency_ms=None, base_raw_speedup=None, base_scaling_factor=None,
                  compared_speedup=None, base_data_source=str(path) if path else None,
                  base_shape_key=key, base_miss_reason=None, normalization=None)
    if path is None:
        result["base_miss_reason"] = "external baseline not configured"
        return result
    data = read_json(path)
    entry = data.get(config["op_name"], {})
    record = entry.get("shapes", {}).get(key, {}).get(dtype, {})
    if not record:
        result["base_miss_reason"] = "external case missing"
    elif canonical(record.get("config")) != canonical(config):
        result["base_miss_reason"] = "external case configuration differs"
    elif record.get("timing_protocol") != protocol:
        result["base_miss_reason"] = "external timing protocol differs"
    elif not _positive(record.get("latency_ms")) or record.get("status", "passed") != "passed":
        result["base_miss_reason"] = "external measurement unavailable"
    elif not _positive(latency) or not _positive(record["latency_ms"] / latency):
        result["base_miss_reason"] = "invalid raw latency ratio"
    else:
        base = record["latency_ms"]
        scale, reason, provenance = _scale(data, record, config, protocol, target, target_profile)
        raw = base / latency
        if scale is not None and not _positive(raw / scale):
            scale, reason = None, "invalid normalized latency ratio"
        provenance.update(reference_comparator=entry.get("native_api"),
                          reference_version=record.get("baseline_version"))
        result.update(base_latency_ms=base, base_raw_speedup=raw, base_scaling_factor=scale,
                      compared_speedup=raw / scale if scale is not None else None,
                      base_miss_reason=reason, normalization=provenance)
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
