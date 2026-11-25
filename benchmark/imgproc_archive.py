"""Replay already classified Imgproc records without re-attributing timing."""
from collections import defaultdict
import hashlib
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


def normalize(raw_class, ratios, relative, source, background=None):
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
    if raw_class == "comparable" and not formal:
        result["qualification"] = "matching_output_without_comparable_timing"
    if background:
        result["background_protocols"] = background
    return result


def read_first_batch(payload, relative):
    entries = {side: {} for side in ENTRIES}
    if payload is None:
        return entries
    for row in json.loads(payload):
        for side, target in entries.items():
            record = row.get(side)
            if record is None:
                continue
            metrics = {"event": record["ratio"]} if record.get("ratio") is not None else {}
            add_row(target, row["id"], normalize(
                record["class"], metrics, relative, row.get(f"{side}_measurement_source"),
                list(record.get("candidate_only", {}).get("candidate_samples_us",
                                                         record.get("candidate_only", {})))))
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


def read_cpp_group(group_id, payload):
    if group_id == "geometry-large":
        return geometry_large(payload)
    summary = json.loads(payload)
    return summary["cpp"]["rows"] if group_id == "arithmetic-filters" else summary["rows"]


def read_group(group, root):
    payloads = {name: read_bound(root, name, sha) for name, sha in group["files"].items()}
    if "shared" in group:
        path = group["shared"]
        return read_first_batch(payloads[path], path)

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
                ["repeat", "reset_included"] if row["case"] in resets else None))
    for path in group["python"]:
        if payloads[path] is None:
            continue
        for row in read_jsonl(payloads[path]):
            metrics = {"event": row["ratio"]} if row.get("ratio") is not None else {}
            add_row(entries["python"], row["id"], normalize(
                CLASSES[row["classification"]], metrics, path, group["source"],
                ["repeat", "reset_included"] if "reset_included" in row else None))
    return entries


def load_archive(root):
    if not root.is_dir():
        raise ValueError("archive root must be a directory")
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "flagcv-imgproc-archive/1":
        raise ValueError("unsupported Imgproc archive schema")
    results = []
    counts = defaultdict(lambda: defaultdict(int))
    for group in manifest["groups"]:
        rows = read_group(group, root)
        for side in ENTRIES:
            ids = {item[side] for item in group["cases"]}
            if len(ids) != len(group["cases"]) or not rows[side].keys() <= ids:
                raise ValueError(f"invalid {side} workload inventory for {group['id']}")
        for item in group["cases"]:
            for side in ENTRIES:
                case = item[side]
                observation = rows[side].get(case, {"classification": "missing", "formal_ratios": None})
                counts[side][observation["classification"]] += 1
                results.append(dict(case=case, group=group["id"], operator=item["operator"],
                                    entry=side, physical_gpu=group["gpu"], parameters=item.get("parameters"),
                                    **observation))
    return dict(schema="flagcv-imgproc-archive-summary/1", scope="archived",
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                archived_inputs={group["id"]:group["files"] for group in manifest["groups"]},
                counts=counts, rows=results)
