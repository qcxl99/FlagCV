"""Check FlagCV's frozen NVIDIA workloads and summarize paired performance."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, mean, median, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES

OPERATORS = ("add", "subtract", "multiply", "bitwise_and", "bitwise_not", "flip")
CPP_FIELDS = ("op", "height", "width", "depth", "channels", "layout", "flip_code", "input_domain")
CPP_PROTOCOL = "flagcv-cpp-opencv-ticks-stream-wait/2"
CPP_MODES = ("host_single", "host_batch")
MANIFEST = Path(__file__).with_name("nvidia_cases.json")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(path):
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid latency {value}")
    return value


def sample_stats(values, center, count):
    values = [positive(value) for value in values]
    if len(values) != count or count < 2:
        raise ValueError("missing timing samples")
    if not math.isclose(median(values), positive(center), rel_tol=1e-12, abs_tol=1e-5):
        raise ValueError("reported median disagrees with timing samples")
    return {"samples": count, "min": min(values), "median": median(values),
            "mean": mean(values), "stddev": pstdev(values), "gmean": geometric_mean(values)}


def add_case(rows, key, metrics):
    if key in rows:
        raise ValueError(f"duplicate workload: {key}")
    rows[key] = metrics


def check_inventory(rows, expected):
    if rows.keys() != expected:
        missing = sorted(expected - rows.keys())
        extra = sorted(rows.keys() - expected)
        raise ValueError(f"workload manifest mismatch: missing={missing[:3]}, extra={extra[:3]}")


def read_cpp(path, manifest):
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        header = stream.readline()
        if not header.startswith("# device=NVIDIA "):
            raise ValueError("missing NVIDIA C++ environment metadata")
        info = dict(part.split("=", 1) for part in header[2:].strip().split(","))
        if info.get("protocol") != CPP_PROTOCOL:
            raise ValueError("C++ timing protocol mismatch")
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in CPP_FIELDS)
            count = int(row["samples"])
            if (row["validated_against_opencv_cpu"] != "true" or count < 3
                    or int(row["iterations"]) < 1):
                raise ValueError(f"{key} is missing correctness or timing evidence")
            metrics = {}
            for mode in CPP_MODES:
                metrics[mode] = {
                    side: sample_stats(row[f"{side}_{mode}_samples_us"].split(";"),
                                       row[f"{side}_{mode}_us"], count)
                    for side in ("flagcv", "opencv")
                }
            for mode in ("stream", "submit"):
                metrics[mode] = {side: {"median": positive(row[f"{side}_{mode}_us"])}
                                 for side in ("flagcv", "opencv")}
            add_case(rows, key, metrics)
    check_inventory(rows, {tuple(case) for case in manifest})
    return rows, info


def python_key(operator, case_id, dtype, shape, params):
    return operator, case_id, canonical([dtype, shape, params])


def read_python(path, manifest, protocol):
    rows = {}
    versions = set()
    report = load_json(path)
    for operator in OPERATORS:
        suite = report[operator]
        if suite["result"] != "passed" or report[suite["test_case"]]["result"] != "passed":
            raise ValueError(f"{operator} Python benchmark failed")
        for group in suite["details"]:
            for row in group["result"]:
                key = python_key(operator, row["case_id"], group["dtype"],
                                 row["shape_detail"], row["params"])
                if (
                    row["baseline_source"] != "opencv_cuda_live"
                    or not row["baseline_version"]
                    or row["accuracy_checked"] is not True
                    or row["candidate_source"] != "builtin"
                    or (row["timing_protocol"], row["warmup_ms"], row["measurement_ms"])
                       != (protocol.name, protocol.warmup_ms, MEASUREMENT_MS)
                ):
                    raise ValueError(f"{key} is missing live comparison or timing protocol evidence")
                versions.add(row["baseline_version"])
                count = protocol.samples
                metrics = {"kernel": {
                    "flagcv": sample_stats(row["latency_samples_ms"], row["latency"], count),
                    "opencv": sample_stats(row["base_latency_samples_ms"], row["base_latency_ms"], count),
                }}
                add_case(rows, key, metrics)
    check_inventory(rows, {python_key(*case) for case in manifest})
    if len(versions) != 1:
        raise ValueError("Python rows disagree on baseline version")
    return rows, versions.pop()


def pci_location(address):
    if not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", address):
        raise ValueError(f"invalid PCI address: {address}")
    return tuple(int(field, 16) for field in re.split(r"[:.]", address))


def bind_identity(path, cpp, python, metadata, protocol=None):
    identity = load_json(path)
    if identity["schema"] != "flagcv-nvidia-evidence/2":
        raise ValueError("unsupported performance evidence version")
    if identity["artifacts"] != {"cpp": digest(cpp), "python": digest(python)}:
        raise ValueError("timing files do not match their performance evidence")
    info = identity["environment"]
    if protocol and (info["protocol"]["python"], info["protocol"]["python_samples"]) != (
        protocol.name, protocol.samples
    ):
        raise ValueError("Python timing profile disagrees with performance evidence")
    if pci_location(info["device"]["pci_bus"]) != pci_location(metadata["pci_bus"]):
        raise ValueError("C++ physical GPU disagrees with performance evidence")
    if not info["device"]["uuid"] or not info["opencv_libraries"] or not identity["candidate"]:
        raise ValueError("incomplete performance evidence identity")
    if protocol and info["workload_sha256"] != digest(MANIFEST):
        raise ValueError("performance evidence belongs to another workload manifest")
    return identity


def check_comparable(current, previous, current_cpp, previous_cpp):
    if current["environment"]["device"] != previous["environment"]["device"]:
        raise ValueError("performance records are from different physical GPUs or drivers")
    if current["environment"] != previous["environment"]:
        raise ValueError("toolchain, baseline, workload or protocol differs")
    if any(current_cpp[key] != previous_cpp[key]
           for key in ("pci_bus", "driver_api", "cuda_runtime", "opencv", "protocol")):
        raise ValueError("C++ devices, runtime or protocol differ")


def summarize(name, rows, previous=None):
    results = defaultdict(list)
    details = []
    for key, metrics in rows.items():
        for mode, pair in metrics.items():
            candidate = pair["flagcv"]["median"]
            ratio = pair["opencv"]["median"] / candidate
            results[(key[0], mode)].append(ratio)
            row = {"case": key, "metric": mode, "timing": pair, "speedup": ratio}
            if previous is not None:
                old = previous[key][mode]["flagcv"]["median"]
                row["latency_change_pct"] = (candidate / old - 1) * 100
            details.append(row)
    for (operator, mode), speedups in results.items():
        print(f"{name} {operator:12s} {mode:11s} {len(speedups):2d} cases  "
              f"min {min(speedups):.3f}x  slower than OpenCV CUDA: "
              f"{sum(ratio < 1.0 for ratio in speedups)}")
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--imgproc-evidence", type=Path, help="root of the archived H20 runs")
    parser.add_argument("--require-imgproc-scope", action="append",
                        choices=("all", "first-batch", "geometry-small", "geometry-large", "arithmetic-filters"))
    parser.add_argument("--python-benchmark-profile", choices=tuple(PROFILES), default="default")
    parser.add_argument("--previous-cpp", type=Path)
    parser.add_argument("--previous-python", type=Path)
    parser.add_argument("--previous-identity", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.require_imgproc_scope and not args.imgproc_evidence:
        parser.error("--require-imgproc-scope needs --imgproc-evidence")
    if args.imgproc_evidence is not None:
        if any((args.cpp, args.python, args.identity, args.previous_cpp,
                args.previous_python, args.previous_identity)):
            parser.error("Imgproc archive intake uses its own evidence, without live comparison arguments")
        from imgproc_archive import load_archive, qualify_archive
        try:
            report = load_archive(args.imgproc_evidence)
            if args.require_imgproc_scope:
                report["scope_check"] = qualify_archive(report, args.require_imgproc_scope)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        print(json.dumps(report["counts"], indent=2))
        if "scope_check" in report:
            print(json.dumps(report["scope_check"], indent=2))
        if args.summary:
            args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if report.get("scope_check", {}).get("status") == "FAIL":
            raise SystemExit(1)
        return
    if not args.cpp or not args.python:
        parser.error("--cpp and --python are required for the live NVIDIA checker")
    protocol = PROFILES[args.python_benchmark_profile]
    try:
        cases = load_json(MANIFEST)
        cpp, metadata = read_cpp(args.cpp, cases["cpp"])
        python, python_baseline = read_python(args.python, cases["python"], protocol)
        if metadata["opencv"] != python_baseline:
            raise ValueError("C++ and Python OpenCV baseline versions differ")
        identity = bind_identity(args.identity, args.cpp, args.python, metadata, protocol) if args.identity else None
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    previous_cpp = previous_python = None
    previous_error = None
    if any((args.previous_cpp, args.previous_python, args.previous_identity)):
        try:
            if not all((identity, args.previous_identity, args.previous_cpp, args.previous_python)):
                raise ValueError("current and previous performance pairs need their identity files")
            old_cpp, old_metadata = read_cpp(args.previous_cpp, cases["cpp"])
            old_python, old_baseline = read_python(args.previous_python, cases["python"], protocol)
            old_identity = bind_identity(args.previous_identity, args.previous_cpp,
                                         args.previous_python, old_metadata, protocol)
            if old_metadata["opencv"] != old_baseline:
                raise ValueError("previous C++ and Python OpenCV baseline versions differ")
            check_comparable(identity, old_identity, metadata, old_metadata)
            previous_cpp, previous_python = old_cpp, old_python
        except (OSError, ValueError, KeyError, TypeError) as error:
            previous_error = str(error)
            print(f"Previous comparison unavailable: {previous_error}")
    summary = {"schema": "flagcv-nvidia-performance-summary/2",
               "previous_compared": previous_cpp is not None,
               "previous_comparison_unavailable": previous_error,
               "cpp_us": summarize("C++", cpp, previous_cpp),
               "python_ms": summarize("Python", python, previous_python)}
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
