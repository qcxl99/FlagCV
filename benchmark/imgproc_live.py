"""Execute installed Imgproc C++ and Triton adapters for selected cases."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory

from helpers.benchmark_protocol import PROFILES


def output(command, **options):
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **options).stdout


def library_paths(binary, env):
    paths = {}
    for line in output(["ldd", str(binary)], env=env).splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "=>" and fields[2].startswith("/"):
            paths[fields[0]] = Path(fields[2]).resolve()
    return paths


def verify_identity(prefix, binary, env):
    paths = library_paths(binary, env)
    runtime = next((p for name,p in paths.items() if name.startswith("libflagcv.so")), None)
    if runtime is None or not runtime.is_relative_to(prefix):
        raise ValueError(f"did not load relocated FlagCV from {binary}")
    opencv = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in paths.items() if name.startswith("libopencv_")}
    return dict(runtime=str(runtime), runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
                executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), opencv=opencv)


def cpp_samples(family, case, raw):
    if family == "geometry":
        samples = []
        for line in raw.splitlines():
            if not line.startswith("PERF,"):
                continue
            _, case_id, classification, background, index, event, wall, native_event, native_wall = line.split(",")
            samples.append(dict(case=case_id, classification=classification, background=background,
                                sample=index, flagcv_event_us=event, flagcv_sync_wall_us=wall,
                                native_event_us=native_event, native_sync_wall_us=native_wall))
    else:
        rows = list(csv.DictReader(io.StringIO(raw)))
        samples = [row for row in rows if row["sample"].isdigit()]
    if not samples or any(row.get("case", row.get("id")) != case for row in samples):
        raise ValueError(f"{case} missing public C++ samples")
    groups = {}
    for row in samples:
        groups.setdefault(row.get("background", "repeat"), []).append(row)
    result = {}
    for name, rows in groups.items():
        state = rows[0].get("classification", rows[0].get("status"))
        if state not in ("COMPARABLE", "SEMANTICS_DIFFER", "NATIVE_REFUSED", "CANDIDATE_ONLY"):
            raise ValueError(f"{case} failed public C++ verification: {state}")
        if [int(row["sample"]) for row in rows] != list(range(11)):
            raise ValueError(f"{case} missing public C++ timing samples")
        ratios = {}
        if state == "COMPARABLE":
            for metric in ("event", "wall"):
                flag_names = [f"flagcv_{metric}_us", "flagcv_sync_wall_us" if metric == "wall" else ""]
                cuda_names = [f"cuda_{metric}_us",
                              "native_sync_wall_us" if metric == "wall" else "native_event_us"]
                flag = next((key for key in flag_names if rows[0].get(key)), None)
                native = next((key for key in cuda_names if rows[0].get(key)), None)
                if flag and native:
                    ratios[metric] = median(float(row[native]) for row in rows) / median(float(row[flag]) for row in rows)
        result[name] = dict(classification=state, formal_ratios=ratios or None, samples=rows)
    return result


def python_samples(case, raw, profile):
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [row for row in records if row.get("id") == case]
    if len(rows) != 1:
        raise ValueError(f"{case} missing or duplicate installed Python result")
    row = rows[0]
    state = row.get("classification")
    if state not in ("COMPARABLE", "SEMANTICS_DIFFER", "NATIVE_REFUSED") or not row.get("launch"):
        raise ValueError(f"{case} failed independent Python CPU verification: {row.get('error', state)}")
    if len(row.get("pairs") or row.get("samples") or []) != PROFILES[profile].samples:
        raise ValueError(f"{case} missing installed Python timing samples")
    return row


def run_live(prefix, cases, profile, summary):
    prefix = prefix.resolve()
    packages = list(prefix.glob("**/flagcv/benchmark/imgproc/live-install.json"))
    if len(packages) != 1:
        raise ValueError("expected one installed Imgproc benchmark package")
    package = packages[0].parent
    data = package.parents[1]
    install = json.loads(packages[0].read_text())
    data_cases = json.loads((package / "cases.json").read_text())
    registry = {row["id"]: row for row in data_cases["cases"]}
    selected = cases or data_cases["default"]
    if len(set(selected)) != len(selected) or not set(selected) <= registry.keys():
        raise ValueError("duplicate or unknown live Imgproc case")
    if not re.fullmatch(r"[0-9a-f]{40}", install["revision"]):
        raise ValueError("installed Imgproc revision is missing")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", visible):
        raise ValueError("CUDA_VISIBLE_DEVICES must select a single physical NVIDIA UUID")
    identity = output(["nvidia-smi", "--id", visible,
                       "--query-gpu=uuid,name,pci.bus_id,driver_version,compute_cap",
                       "--format=csv,noheader,nounits"]).strip().split(", ")
    if len(identity) != 5 or identity[0] != visible or identity[-1] != "9.0":
        raise ValueError("the installed Imgproc adapters currently require a single SM90 GPU")
    env = dict(os.environ)
    env.pop("FLAGCV_PLUGIN_DIR", None)
    env.update(FLAGCV_DATA_DIR=str(data), FLAGCV_PROJECTIVE_DATA=str(data),
               FLAGCV_PROJECTIVE_KERNELS=str(data/"kernels"), FLAGCV_PYTHON=sys.executable)
    env["PYTHONPATH"] = os.pathsep.join([str(data), str(data/"compiler"), env.get("PYTHONPATH", "")])
    runners = prefix / install["runners"]
    env["FLAGCV_IMGP_FIRST_BRIDGE"] = str(runners/"libflagcv_imgproc_first_bridge.so")
    env["FLAGCV_IMGP_BATCH_BRIDGE"] = str(runners/"libflagcv_imgproc_batch_bridge.so")
    env["FLAGCV_IMGP_SPATIAL_BRIDGE"] = str(runners/"libflagcv_imgproc_spatial_bridge.so")
    import cv2
    modules = list(Path(cv2.__file__).parent.glob("**/cv2*.so"))
    if len(modules) != 1:
        raise ValueError("unable to identify the OpenCV Python extension")
    python_opencv = library_paths(modules[0], env)
    provider = data/"backends/nvidia/compiler.py"
    result = dict(schema="flagcv-imgproc-live/1", scope="current_install",
                  revision=install["revision"], physical_device=identity,
                  provider={"path": str(provider), "sha256": hashlib.sha256(provider.read_bytes()).hexdigest()},
                  case_manifest_sha256=hashlib.sha256((package/"cases.json").read_bytes()).hexdigest(),
                  python_profile=profile, rows=[])
    failed = []
    with TemporaryDirectory(prefix="flagcv-imgproc-") as work:
        env["FLAGCV_CACHE_DIR"] = str(Path(work)/"flagcv-cache")
        env["TRITON_CACHE_DIR"] = str(Path(work)/"triton-cache")
        manifest = Path(work)/"selection.json"
        for case in selected:
            fixture = registry[case]
            family = fixture["family"]
            binary = runners/f"flagcv_imgproc_{family}"
            cpp_identity = verify_identity(prefix, binary, env)
            for module, details in cpp_identity["opencv"].items():
                if module not in python_opencv or hashlib.sha256(python_opencv[module].read_bytes()).hexdigest() != details["sha256"]:
                    raise ValueError(f"C++ and Python resolve different {module}")
            for bridge in ("FLAGCV_IMGP_FIRST_BRIDGE", "FLAGCV_IMGP_SPATIAL_BRIDGE", "FLAGCV_IMGP_BATCH_BRIDGE"):
                verify_identity(prefix, Path(env[bridge]), env)
            manifest.write_text(json.dumps([fixture["data"]]))
            command = [sys.executable, str(package/f"{family}.py"), "--matrix", str(manifest), "--profile", profile]
            if family == "geometry":
                command.extend(["--test-source", str(package/"test_projective.py")])
            cpp_raw = subprocess.run([str(binary), case], cwd=work, env=env, text=True, capture_output=True)
            python_raw = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True)
            execution = {
                "cpp": {"status": cpp_raw.returncode, "stdout": cpp_raw.stdout, "stderr": cpp_raw.stderr},
                "python": {"status": python_raw.returncode, "stdout": python_raw.stdout,
                           "stderr": python_raw.stderr},
            }
            try:
                if cpp_raw.returncode or python_raw.returncode:
                    raise ValueError(f"C++ exit {cpp_raw.returncode}, Python exit {python_raw.returncode}")
                cpp = cpp_samples(family, case, cpp_raw.stdout)
                python = python_samples(case, python_raw.stdout, profile)
            except (ValueError, KeyError, TypeError) as error:
                failed.append(case)
                result["rows"].append(dict(case=case, family=family, installed=cpp_identity,
                                           status="FAIL", diagnostic=str(error), execution=execution))
                continue
            result["rows"].append(dict(case=case, family=family, installed=cpp_identity,
                                       cpp=cpp, python=python, execution=execution))
            print(case, "C++", cpp["repeat"]["classification"], "Python", python["classification"], flush=True)
        result["compiled_metadata"] = [
            json.loads(path.read_text()) for path in sorted((Path(work)/"triton-cache").rglob("*.json"))
            if not path.name.startswith("__grp__")
        ]
    result["status"] = "FAIL" if failed else "PASS"
    if summary:
        summary.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise ValueError("live Imgproc cases failed: " + ", ".join(failed))
