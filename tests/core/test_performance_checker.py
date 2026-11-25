"""Regression checks for proofs used by the NVIDIA merge checker."""
import csv
import json
from pathlib import Path
import runpy

import pytest

CHECKER = runpy.run_path(str(Path(__file__).parents[2] / "benchmark/check_nvidia.py"))
CASE = ("add", "64", "64", "8U", "1", "packed", "0", "default")


def test_replacing_case_is_rejected_even_with_the_same_count(tmp_path):
    report = tmp_path / "cpp.csv"
    row = dict(zip(CHECKER["CPP_FIELDS"], CASE))
    row.update(validated_against_opencv_cpu="true", samples="3", iterations="200")
    for side in ("flagcv", "opencv"):
        for mode in CHECKER["CPP_MODES"]:
            row[f"{side}_{mode}_samples_us"] = "1;2;3"
            row[f"{side}_{mode}_us"] = "2"
        row.update({f"{side}_stream_us": "1", f"{side}_submit_us": "1"})

    def write():
        with report.open("w", newline="") as out:
            out.write(f"# device=NVIDIA H20,protocol={CHECKER['CPP_PROTOCOL']}\n")
            writer = csv.DictWriter(out, row)
            writer.writeheader()
            writer.writerow(row)

    write()
    results, _ = CHECKER["read_cpp"](report, [CASE])
    assert results[CASE]["host_single"]["flagcv"]["median"] == 2
    row["width"] = "65001"
    write()
    with pytest.raises(ValueError, match="workload manifest mismatch"):
        CHECKER["read_cpp"](report, [CASE])


def test_comparison_requires_same_device_and_baseline():
    metadata = {"pci_bus": "0000:C3:00.0", "driver_api": "12080", "cuda_runtime": "12080",
                "opencv": "5.0.0", "protocol": CHECKER["CPP_PROTOCOL"]}

    def record(uuid, baseline, source):
        return {"environment": {"device": {"uuid": uuid}, "opencv_libraries": baseline},
                "candidate": source}

    current = record("card-a", "opencv-build-a", "new")
    old = record("card-a", "opencv-build-a", "old")
    CHECKER["check_comparable"](current, old, metadata, metadata)
    with pytest.raises(ValueError, match="different physical GPUs"):
        CHECKER["check_comparable"](current, record("card-b", "opencv-build-a", "old"),
                                     metadata, metadata)
    with pytest.raises(ValueError, match="baseline"):
        CHECKER["check_comparable"](current, record("card-a", "opencv-build-b", "old"),
                                     metadata, metadata)


def test_identity_is_bound_to_the_timing_files(tmp_path):
    cpp, python, identity = (tmp_path / name for name in ("cpp.csv", "python.json", "identity.json"))
    cpp.write_text("cpp")
    python.write_text("python")
    identity.write_text(json.dumps({
        "schema": "flagcv-nvidia-evidence/2",
        "environment": {"device": {"uuid": "gpu-a", "pci_bus": "00000000:C3:00.0"},
                        "opencv_libraries": {"libopencv_core.so": "hash"}},
        "candidate": {"source": "candidate"},
        "artifacts": {key: CHECKER["digest"](path)
                      for key, path in (("cpp", cpp), ("python", python))},
    }))
    metadata = {"pci_bus": "0000:c3:00.0"}
    CHECKER["bind_identity"](identity, cpp, python, metadata)
    with pytest.raises(ValueError, match="C\\+\\+ physical GPU"):
        CHECKER["bind_identity"](identity, cpp, python, {"pci_bus": "0001:c3:00.0"})
    python.write_text("changed after collection")
    with pytest.raises(ValueError, match="timing files do not match"):
        CHECKER["bind_identity"](identity, cpp, python, metadata)


@pytest.mark.parametrize("values, center", [([1, 2], 2), ([1, float("nan"), 3], 2),
                                            ([1, 2, 3], 8)])
def test_bad_samples_fail(values, center):
    with pytest.raises(ValueError):
        CHECKER["sample_stats"](values, center, 3)
