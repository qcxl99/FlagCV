"""Replay already classified Imgproc records without re-attributing timing."""
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median


MANIFEST = Path(__file__).with_name("imgproc_archive_cases.json")
CLASSES = {
    "COMPARABLE": "comparable",
    "SEMANTICS_DIFFER": "semantic_diagnostic",
    "NATIVE_REFUSED": "unsupported",
    "NATIVE_LAYOUT_CHANGED": "layout_diagnostic",
    "CANDIDATE_CPU_DIFFER": "candidate_cpu_diff",
}
ENTRIES = ("cpp", "python")


def read_bound(root, relative, expected):
    path = root / relative
    if not path.is_file():
        return None
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"archive identity mismatch: {relative}")
    return payload.decode()


def add_row(rows, key, row):
    if key in rows:
        raise ValueError(f"duplicate archive record: {key}")
    rows[key] = row


def normalize(raw_class, ratios, relative, source, background=None, record=None):
    if raw_class not in CLASSES.values():
        raise ValueError(f"unknown archive classification: {raw_class}")
    formal = {}
    if raw_class == "comparable":
        for metric, ratio in ratios.items():
            value = float(ratio)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid archived speed ratio: {relative}")
            formal[metric] = value
    result = dict(classification=raw_class, formal_ratios=formal or None,
                  evidence=relative, source=source)
    cpu = "fail" if raw_class == "candidate_cpu_diff" else "pass"
    baseline = {
        "comparable": "matches_cpu",
        "semantic_diagnostic": "differs_from_cpu",
        "unsupported": "refused",
        "layout_diagnostic": "changed_layout",
        "candidate_cpu_diff": "not_run",
    }[raw_class]
    result.update(cpu_reference={"status": cpu},
                  opencv_cuda={"status": baseline})
    if record:
        if record.get("error"):
            result["opencv_cuda"]["diagnostic"] = record["error"]
        for name in ("return_contract", "event_us", "native_sync_wall_us", "median",
                     "flagcv_us", "opencv_cuda_us", "sample_extent",
                     "flagcv_launches", "opencv_cuda_launches"):
            if name in record:
                result[name] = record[name]
    if raw_class == "comparable" and not formal:
        result["qualification"] = "matching_output_without_comparable_timing"
    if background:
        result["background_protocols"] = background
    return result


def read_first_batch(payload, relative, restored_python):
    entries = {side: {} for side in ENTRIES}
    if payload is None:
        return entries
    for row in json.loads(payload):
        for side, target in entries.items():
            record = row.get(side)
            if record is None:
                continue
            metrics = {"event": record["ratio"]} if record.get("ratio") is not None else {}
            source = row.get(f"{side}_measurement_source")
            if side == "python" and source is None:
                source = restored_python.get(row["id"])
            add_row(target, row["id"], normalize(
                record["class"], metrics, relative, source,
                list(record.get("candidate_only", {}).get("candidate_samples_us",
                                                         record.get("candidate_only", {}))), record))
    return entries


def read_jsonl(payload):
    values = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not values or "id" in values[0]:
        raise ValueError("missing Python archive run header")
    count = values[0].get("cases", values[0].get("rows"))
    if count != len(values) - 1:
        raise ValueError("Python archive row count differs from run header")
    return values[1:]


def geometry_large(payload):
    groups = defaultdict(list)
    for line in payload.splitlines():
        if line.startswith("PERF,"):
            _, case, category, background, index, event, wall, native_event, native_wall = line.split(",")
            groups[case, category, background].append(
                (int(index), float(event), float(wall), native_event, native_wall))
    for (case, category, background), samples in groups.items():
        if [item[0] for item in samples] != list(range(11)):
            raise ValueError(f"bad C++ sample count or order: {case}")
        ratios = {}
        if category == "COMPARABLE":
            for i, metric in ((1, "event"), (2, "sync_wall")):
                ratios[metric] = median(float(item[i + 2]) for item in samples) / median(
                    item[i] for item in samples)
        yield dict(case=case, classification=category, protocol=background, ratios=ratios)


def batch_cpp(payload):
    by_id = defaultdict(list)
    for row in csv.DictReader(io.StringIO(payload)):
        by_id[row["id"]].append(row)
    for case, rows in by_id.items():
        if [int(row["sample"]) for row in rows] != list(range(11)):
            raise ValueError(f"bad C++ batch sample count: {case}")
        identical = all(float(row["cuda_cpu_max_abs"]) == 0.0 for row in rows)
        ratios = {
            metric: median(float(row[f"cuda_{metric}_us"]) / float(row[f"flagcv_{metric}_us"])
                           for row in rows)
            for metric in ("event", "wall", "batch_wall")
        } if identical else {}
        yield dict(case=case, classification="COMPARABLE" if identical else "SEMANTICS_DIFFER", ratios=ratios)


def read_cpp_group(group_id, payload):
    if group_id == "resize-batch":
        return batch_cpp(payload)
    if group_id == "geometry-large":
        return geometry_large(payload)
    summary = json.loads(payload)
    return summary["cpp"]["rows"] if group_id == "arithmetic-filters" else summary["rows"]


def read_group(group, payloads):
    if "shared" in group:
        path = group["shared"]
        identity_path = group["source_identity"]
        sources = group["restored_python_sources"] if payloads[identity_path] is not None else {}
        return read_first_batch(payloads[path], path, sources)

    entries = {side: {} for side in ENTRIES}
    cpp = group["cpp"]
    if payloads[cpp] is not None:
        rows = list(read_cpp_group(group["id"], payloads[cpp]))
        resets = {row["case"] for row in rows if row.get("protocol") == "reset_included"}
        for row in rows:
            if row.get("protocol") == "reset_included":
                continue
            metrics = row.get("ratios") or {}
            if row.get("ratio") is not None:
                metrics = {"event": row["ratio"]}
            add_row(entries["cpp"], row["case"], normalize(
                CLASSES[row["classification"]], metrics, cpp, group["source"],
                ["repeat", "reset_included"] if row["case"] in resets else None, row))
    for path in group["python"]:
        if payloads[path] is None:
            continue
        if group["id"] == "resize-batch":
            batch_count = {row["python"]:row["parameters"]["count"] for row in group["cases"]}
            observations = [json.loads(line) for line in payloads[path].splitlines() if line.strip()]
            for row in observations:
                if (row["flagcv_launches"] != 1 or row["opencv_cuda_launches"] != batch_count[row["id"]]
                        or len(row["pairs"]) != 2):
                    raise ValueError(f"invalid resizeBatch launch evidence: {row['id']}")
                row["classification"] = "COMPARABLE" if row["cuda_cpu_max_abs"] == 0 else "SEMANTICS_DIFFER"
                row["ratio"] = row["opencv_cuda_us"] / row["flagcv_us"]
        else:
            observations = read_jsonl(payloads[path])
        for row in observations:
            metrics = {"event": row["ratio"]} if row.get("ratio") is not None else {}
            add_row(entries["python"], row["id"], normalize(
                CLASSES[row["classification"]], metrics, path, group["source"],
                ["repeat", "reset_included"] if "reset_included" in row else None, row))
    return entries


def load_archive(root):
    if not root.is_dir():
        raise ValueError("archive root must be a directory")
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "flagcv-imgproc-archive/2":
        raise ValueError("unsupported Imgproc archive schema")
    results = []
    counts = defaultdict(lambda: defaultdict(int))
    original_jobs = {}
    for group in manifest["groups"]:
        payloads = {name: read_bound(root, name, sha) for name, sha in group["files"].items()}
        rows = read_group(group, payloads)
        group_id = group["id"]
        original_jobs[group_id] = {}
        missing = {}
        for side in ENTRIES:
            jobs = {}
            for path in group["jobs"][side]:
                if payloads[path] is not None:
                    job = json.loads(payloads[path])
                    if job["gpu_uuid"] != group["gpu"] or job["status"] != 0:
                        raise ValueError(f"invalid original job identity: {path}")
                    jobs[path] = job
            original_jobs[group_id][side] = jobs
            missing[side] = [path for path in group["jobs"][side] + group["original_samples"][side]
                             if payloads[path] is None]
        for side in ENTRIES:
            ids = {item[side] for item in group["cases"]}
            if len(ids) != len(group["cases"]) or not rows[side].keys() <= ids:
                raise ValueError(f"invalid {side} workload inventory for {group['id']}")
        for item in group["cases"]:
            for side in ENTRIES:
                case = item[side]
                observation = rows[side].get(case, {"classification": "missing", "formal_ratios": None,
                                                     "cpu_reference": {"status": "missing"},
                                                     "opencv_cuda": {"status": "missing"}})
                observed = observation["classification"] != "missing"
                identity_verified = observed and not missing[side] and bool(observation.get("source"))
                observation["identity_status"] = ("verified" if identity_verified else
                                                  "unverified" if observed else "missing")
                if observed and not identity_verified:
                    observation["missing_original_evidence"] = missing[side]
                    observation["historical_ratios"] = observation["formal_ratios"]
                    observation["formal_ratios"] = None
                counts[side][observation["classification"]] += 1
                results.append(dict(case=case, group=group["id"], operator=item["operator"],
                                    entry=side, physical_gpu=group["gpu"], parameters=item.get("parameters"),
                                    **observation))
    return dict(schema="flagcv-imgproc-archive-summary/2", scope="archived",
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                archived_inputs={group["id"]:group["files"] for group in manifest["groups"]},
                original_jobs=original_jobs, counts=counts, rows=results)


def qualify_archive(report, groups):
    """Check completeness and identity of a declared historical scope."""
    selected = set(groups)
    available = {row["group"] for row in report["rows"]}
    if "all" in selected:
        selected = available
    if not selected or not selected <= available:
        raise ValueError("unknown or empty Imgproc archive scope")
    outcomes = defaultdict(int)
    for row in report["rows"]:
        if row["group"] not in selected:
            continue
        if row["classification"] == "missing":
            outcomes["missing"] += 1
        elif row["identity_status"] != "verified":
            outcomes["identity_unverified"] += 1
        elif row["cpu_reference"]["status"] != "pass":
            outcomes["candidate_cpu_failed"] += 1
    return dict(groups=sorted(selected), status="FAIL" if outcomes else "PASS",
                evidence_scope="historical", failures=dict(outcomes))
