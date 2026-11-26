"""Validate Core benchmark coverage and compare cases with the same protocol."""

import csv
import json
import math
from pathlib import Path
from statistics import median

from workloads.core import OPERATORS, canonical, case_id, cpp_cases, python_cases


def _inventory(actual, expected, label):
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"{label}: missing={missing[:5]} extra={extra[:5]}")


def _check_ms(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid measured latency: {value}")
    return value


def read_cpp(path):
    with Path(path).open(newline="") as stream:
        header = stream.readline()
        if not header.startswith("# "):
            raise ValueError("missing C++ run metadata")
        environment = dict(item.split("=", 1) for item in header[2:].strip().split(","))
        if environment["protocol"] != "flagcv-cpp-single-stream-event-and-api/1":
            raise ValueError("C++ timing protocol differs")
        expected = {case_id(config): tuple(legacy) for legacy, config in cpp_cases()}
        fields = ("op", "height", "width", "depth", "channels", "layout", "flip_code", "input_domain")
        explicit, convenience = {}, {}
        for row in csv.DictReader(stream):
            key = row["case_id"]
            if tuple(row[field] for field in fields) != expected.get(key):
                raise ValueError(f"{key}: C++ workload configuration differs")
            if row["validated_against_opencv_cpu"] != "true":
                raise ValueError(f"{key}: C++ correctness failed")
            samples = int(environment["samples"])
            for field in ("device", "api"):
                values = [_check_ms(x) for x in row[f"{field}_samples_us"].split(";")]
                if len(values) != samples or median(values) != _check_ms(row[f"{field}_us"]):
                    raise ValueError(f"{key}: C++ {field} samples do not match the reported median")
            destinations = {"explicit": explicit, "setDevice": convenience}
            rows = destinations[row["api"]]
            if key in rows:
                raise ValueError(f"{key}: duplicate C++ {row['api']} record")
            rows[key] = row
    _inventory(explicit, expected, "C++ Core")
    return environment, explicit, convenience


def read_python(path):
    report = json.loads(Path(path).read_text())
    expected = {row["case_id"]: row for op in OPERATORS for _, row in python_cases(op)}
    rows = {}
    for op in OPERATORS:
        suite = report[op]
        if suite["result"] != "passed":
            raise ValueError(f"{op}: Python benchmark failed")
        for group in suite["details"]:
            for row in group["result"]:
                key = row["case_id"]
                if key in rows:
                    raise ValueError(f"{key}: duplicate Python record")
                if canonical(row["config"]) != canonical(expected[key]["config"]):
                    raise ValueError(f"{key}: Python workload configuration differs")
                if row.get("accuracy_checked") is not True or not row.get("kernel_dispatch"):
                    raise ValueError(f"{key}: missing delivery dispatch or correctness evidence")
                values = [_check_ms(x) for x in row["latency_samples_ms"]]
                if not values or median(values) != _check_ms(row["latency"]):
                    raise ValueError(f"{key}: Python samples do not match the reported median")
                rows[key] = row
    _inventory(rows, expected, "Python Core")
    return rows


def cpp_deltas(current, previous):
    env, current, _ = read_cpp(current)
    old_env, previous, _ = read_cpp(previous)
    if env != old_env:
        raise ValueError("C++ measurements use different devices, toolchains or protocols")
    return [{"case_id": key, "op": row["op"],
             **{m + "_change_pct": 100 * (float(row[m]) / float(previous[key][m]) - 1)
                for m in ("device_us", "api_us")}}
            for key, row in current.items()]
