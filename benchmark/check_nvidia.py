"""Verify complete NVIDIA reports against the FlagCV per-workload target."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MIN_SPEEDUP = 0.90
MAX_REGRESSION = 1.05
OPERATORS = ("add", "subtract", "multiply", "bitwise_and", "bitwise_not", "flip")
CPP_COUNTS = dict(zip(OPERATORS, (17, 17, 17, 7, 7, 47)))
PYTHON_COUNTS = dict(zip(OPERATORS, (8, 14, 14, 7, 7, 42)))


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid latency {value}")
    return value


def insert(rows, key, candidate, baseline):
    if key in rows:
        raise ValueError(f"duplicate workload {key}")
    rows[key] = positive(candidate), positive(baseline)


def check_counts(rows, expected):
    counts = Counter(key[0] for key in rows)
    if counts != expected:
        raise ValueError(f"expected case counts {expected}; received {dict(counts)}")


def read_cpp(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        environment = stream.readline().strip()
        if not (
            environment.startswith("# device=NVIDIA ") and ",opencv=" in environment
        ):
            raise ValueError(f"{path} is missing NVIDIA/OpenCV environment metadata")
        for row in csv.DictReader(stream):
            key = tuple(
                row[field] for field in
                ("op", "height", "width", "depth", "channels", "layout", "flip_code",
                 "input_domain")
            )
            if row["validated_against_opencv_cpu"] != "true" or int(row["samples"]) < 3:
                raise ValueError(f"{key} is missing validation or samples")
            insert(rows, key, row["flagcv_stream_us"], row["opencv_stream_us"])
    check_counts(rows, CPP_COUNTS)
    return rows


def read_python(path):
    rows = {}
    report = json.loads(path.read_text(encoding="utf-8"))
    for operator in OPERATORS:
        suite = report[operator]
        if suite["result"] != "passed":
            raise ValueError(f"{operator} pytest result is {suite['result']!r}")
        test_case = report[suite["test_case"]]
        if test_case["result"] != "passed":
            raise ValueError(f"{suite['test_case']} failed")
        for group in suite["details"]:
            for row in group["result"]:
                workload = json.dumps(
                    (group["dtype"], row["shape_detail"], row["params"]), sort_keys=True
                )
                key = operator, row["case_id"], workload
                samples = row["latency_samples_ms"]
                baseline_samples = row["base_latency_samples_ms"]
                if (
                    row["baseline_source"] != "opencv_cuda_live"
                    or not row["baseline_version"]
                    or not row["accuracy_checked"]
                    or row["candidate_source"] != "builtin"
                    or len(samples) < 3
                    or len(baseline_samples) != len(samples)
                ):
                    raise ValueError(f"{key} is missing live CUDA comparison evidence")
                insert(rows, key, row["latency"], row["base_latency_ms"])
    check_counts(rows, PYTHON_COUNTS)
    return rows


def evaluate(name, rows, previous=None):
    errors = []
    if previous is not None and rows.keys() != previous.keys():
        return [f"{name} workloads differ from the comparison report"]
    results = defaultdict(list)
    for key, (candidate, baseline) in rows.items():
        speedup = baseline / candidate
        results[key[0]].append(speedup)
        if speedup < MIN_SPEEDUP:
            errors.append(
                f"{name} {key}: {speedup:.3f}x < {MIN_SPEEDUP:.2f}x OpenCV CUDA"
            )
        if previous is not None and candidate / previous[key][0] > MAX_REGRESSION:
            change = (candidate / previous[key][0] - 1) * 100
            errors.append(f"{name} {key}: candidate latency increased by {change:.1f}%")
    for operator, speedups in results.items():
        print(
            f"{name} {operator:12s} {len(speedups):2d} cases  min {min(speedups):.3f}x"
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--previous-cpp", type=Path)
    parser.add_argument("--previous-python", type=Path)
    args = parser.parse_args()
    try:
        failures = evaluate(
            "C++", read_cpp(args.cpp),
            read_cpp(args.previous_cpp) if args.previous_cpp else None,
        )
        failures += evaluate(
            "Python", read_python(args.python),
            read_python(args.previous_python) if args.previous_python else None,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
