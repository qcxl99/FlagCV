"""Execute installed Imgproc C++ and Triton adapters for selected cases."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import math
import re
import shlex
from statistics import mean, median
import subprocess
import sys
from tempfile import TemporaryDirectory

from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES


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
                executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), opencv=opencv,
                hip_runtime=next((str(path) for name, path in paths.items()
                                  if name.startswith("libamdhip64.so")), None))


def verify_reference(cpp_libraries, python_libraries, hip):
    if hip:
        def is_cpu_oracle(name):
            return name.startswith(("libopencv_core.so", "libopencv_imgproc.so"))
        cpp_libraries = {name:info for name,info in cpp_libraries.items() if is_cpu_oracle(name)}
        if not all(any(name.startswith(f"libopencv_{part}.so") for name in cpp_libraries)
                   for part in ("core", "imgproc")):
            raise ValueError("C++ CPU OpenCV core/imgproc identity is incomplete")
    for module, details in cpp_libraries.items():
        if python_libraries.get(module, {}).get("sha256") != details["sha256"]:
            raise ValueError(f"C++ and Python resolve different {module}")


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
        samples = [row for row in rows if (row.get("sample") or "").isdigit()]
    if not samples or any(row.get("case", row.get("id")) != case for row in samples):
        raise ValueError(f"{case} missing public C++ samples")
    groups = {}
    for row in samples:
        groups.setdefault(row.get("background", "repeat"), []).append(row)
    result = {}
    for name, rows in groups.items():
        state = rows[0].get("classification", rows[0].get("status"))
        if state not in ("COMPARABLE", "SEMANTICS_DIFFER", "NATIVE_REFUSED", "NATIVE_LAYOUT_CHANGED", "CANDIDATE_ONLY"):
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


def python_samples(case, raw, profile, verify_only):
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [row for row in records if row.get("id") == case]
    if len(rows) != 1:
        raise ValueError(f"{case} missing or duplicate installed Python result")
    row = rows[0]
    state = row.get("classification")
    admitted = ("VERIFIED",) if verify_only else (
        "COMPARABLE", "SEMANTICS_DIFFER", "NATIVE_REFUSED", "NATIVE_LAYOUT_CHANGED", "CANDIDATE_ONLY")
    if state not in admitted or not row.get("launch"):
        raise ValueError(f"{case} failed independent Python CPU verification: {row.get('error', state)}")
    if verify_only:
        fields = {"id", "classification", "launch", "candidate_verified",
                  "candidate_cutoff", "cpu_cutoff", "flagcv_launches"}
        if row.get("candidate_verified") is not True or row.keys() - fields:
            raise ValueError(f"{case} invalid verify-only Python result")
        return row
    expected_count = PROFILES[profile].samples
    if len(row.get("pairs") or row.get("samples") or []) != expected_count:
        raise ValueError(f"{case} missing installed Python timing samples")
    return row


def hip_cpp_samples(case, raw, profile):
    groups = {}
    for line in raw.splitlines():
        if not line.startswith("HIPPERF,"):
            continue
        _, case_id, background, sample, warm, window, wall, event = line.split(",")
        if case_id != case:
            raise ValueError("public C++ timing was attributed to the wrong case")
        warm, window, wall = float(warm), float(window), float(wall)
        event = float(event) if event else None
        if (not all(math.isfinite(v) for v in (warm, window, wall)) or
                warm < PROFILES[profile].warmup_ms or window < MEASUREMENT_MS or wall <= 0 or
                (event is not None and (not math.isfinite(event) or event <= 0))):
            raise ValueError("invalid HIP timing sample or measurement window")
        groups.setdefault(background, {}).setdefault(int(sample), []).append(
            dict(warm_ms=warm, measure_ms=window, flagcv_sync_wall_us=wall, flagcv_event_us=event))
    if not groups:
        raise ValueError("no HIP public C++ timings")
    result = {}
    aggregate = mean if profile == "default" else median
    for name, rounds in groups.items():
        if sorted(rounds) != list(range(PROFILES[profile].samples)):
            raise ValueError("incomplete HIP public C++ timing rounds")
        timing = {}
        for metric in ("flagcv_sync_wall_us", "flagcv_event_us"):
            if all(row[metric] is not None for rows in rounds.values() for row in rows):
                timing[metric] = aggregate(median(row[metric] for row in rows) for rows in rounds.values())
        result[name] = dict(classification="CANDIDATE_ONLY", formal_ratios=None, rounds=rounds,
                            timing=timing, diagnostic="no C++ comparator at the same boundary")
    return result


def cpp_verified(case, raw):
    lines = raw.splitlines()
    evidence = [line for line in lines if line.startswith("VERIFY,")]
    has_csv_samples = any(len(row) >= 3 and row[2].isdigit()
                          for row in csv.reader(line for line in lines if not line.startswith("VERIFY,")))
    if evidence != [f"VERIFY,{case},PASS"] or has_csv_samples or any(
        line.startswith(("HIPPERF,", "PERF,")) for line in lines
    ):
        raise ValueError("missing verify-only public C++ receipt or unexpected timings")
    return {"repeat": dict(classification="VERIFIED", candidate_verified=True)}


def first_args(row):
    keys = ("op", "sw", "sh", "dw", "dh", "depth", "channels", "mode", "code", "geometry", "border", "fx", "fy")
    values = {**row, "depth": {"8U": 0, "16U": 2, "32F": 5}[row["depth"]]}
    return [str(values[key]) for key in keys]


def first_fixtures(document):
    """Use existing frozen parameters and IDs. No archived timing is imported."""
    if "cases" in document:
        return document["cases"]
    if document.get("schema") == "flagcv-imgproc-archive/2":
        first = next(group for group in document["groups"] if group["id"] == "first-batch")
        return [dict(id=row["cpp"], family="first", data={**row["parameters"], "id": row["cpp"]},
                     archive_python_id=row["python"]) for row in first["cases"]]
    if "legacy_csv_sha256" in document and "groups" in document:
        fixtures = []
        for row in document["groups"]:
            op = {"resize": "resize", "warpAffine": "warp", "cvtColor": "color"}[row["operator"]]
            name = row["group"]
            depth = {0: "8U", 2: "16U", 5: "32F"}[row["cv_depth"]]
            data = dict(id=name, op=op, depth=depth, sw=row["src_w"], sh=row["src_h"],
                        dw=row["dst_w"], dh=row["dst_h"], channels=row["src_cn"],
                        code=row["color_code"], mode=row["interp"], border=row["border"],
                        geometry=row["geometry"], fx=0, fy=0, seed=row["seed"],
                        background=row.get("background"), matrix=row.get("M"), inverse=row.get("inverse", False))
            fixtures.append(dict(id=name, family="first", data=data, legacy_ids=row["case_ids"],
                                 legacy_csv_sha256=document["legacy_csv_sha256"]))
        return fixtures
    raise ValueError("unrecognized frozen first-family case manifest")


def legacy_input(row, path):
    import numpy as np

    dtype = np.dtype({"8U":"uint8", "16U":"uint16", "32F":"float32"}[row["depth"]])
    rng = np.random.default_rng(row["seed"])
    shape = (row["sh"], row["sw"], row["channels"])
    pixels = (rng.uniform(-2, 2, shape).astype(dtype) if dtype.kind == "f"
              else rng.integers(0, np.iinfo(dtype).max + 1, shape, dtype=dtype))
    payload = pixels.tobytes()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def select_cases(requested, aliases, registry):
    mapping = {case: aliases.get(case, case) for case in requested}
    if not mapping or len(mapping) != len(requested) or not set(mapping.values()) <= registry.keys():
        raise ValueError("empty, duplicate or unknown live Imgproc case selection")
    return list(dict.fromkeys(mapping.values())), mapping


def run_live(prefix, cases, profile, summary, *, backend=None, verify_only=False, expected_uuid=None,
             case_manifest=None, baseline_adapter=None):
    prefix = prefix.resolve()
    packages = list(prefix.glob("**/flagcv/benchmark/imgproc/live-install.json"))
    if len(packages) != 1:
        raise ValueError("expected one installed Imgproc benchmark package")
    package = packages[0].parent
    data = package.parents[1]
    install = json.loads(packages[0].read_text())
    installed_backend = install.get("backend", "nvidia")
    if backend is not None and backend != installed_backend:
        raise ValueError("requested backend is not the installed runner backend")
    backend = installed_backend
    hip = backend in ("hip", "hygon")
    data_cases = json.loads((package / "cases.json").read_text())
    registry = {row["id"]: row for row in data_cases["cases"]}
    seen, aliases = set(), {}
    if case_manifest:
        extra_cases = first_fixtures(json.loads(case_manifest.read_text()))
        for fixture in extra_cases:
            name = fixture["id"]
            if (name in seen or fixture["family"] != "first" or fixture["data"]["id"] != name):
                raise ValueError(f"invalid frozen first-family case {name}")
            if name in registry and (registry[name]["family"] != "first" or
                                     first_args(registry[name]["data"]) != first_args(fixture["data"])):
                raise ValueError(f"frozen case conflicts with installed case {name}")
            registry[name] = fixture
            seen.add(name)
            for alias in fixture.get("legacy_ids", ()):
                if alias in aliases:
                    raise ValueError(f"duplicate legacy alias {alias}")
                aliases[alias] = name
    requested = cases or ([row["id"] for row in extra_cases] if case_manifest else data_cases["default"])
    selected, requested_groups = select_cases(requested, aliases, registry)
    if not re.fullmatch(r"[0-9a-f]{40}", install["revision"]):
        raise ValueError("installed Imgproc revision is missing")
    nv_identity = None
    if not hip:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", visible):
            raise ValueError("CUDA_VISIBLE_DEVICES must select a single physical NVIDIA UUID")
        nv_identity = output(["nvidia-smi", "--id", visible,
                              "--query-gpu=uuid,name,pci.bus_id,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"]).strip().split(", ")
        if len(nv_identity) != 5 or nv_identity[0] != visible or nv_identity[-1] != "9.0":
            raise ValueError("the installed NVIDIA Imgproc adapter expects one SM90 GPU")
    if hip and not expected_uuid:
        raise ValueError("HIP live runs need --device-uuid for the selected physical card")
    env = dict(os.environ)
    for key in ("FLAGCV_PLUGIN_DIR", "FLAGCV_IMGP_VERIFY_ONLY", "FLAGCV_IMGP_LONG",
                "FLAGCV_IMGP_BASELINE_ADAPTER", "FLAGCV_IMGP_INPUT", "FLAGCV_IMGP_BACKGROUND",
                "FLAGCV_IMGP_HIP_RUNTIME"):
        env.pop(key, None)
    env.update(FLAGCV_DATA_DIR=str(data), FLAGCV_PROJECTIVE_DATA=str(data),
               FLAGCV_PROJECTIVE_KERNELS=str(data/"kernels"), FLAGCV_IMGP_BACKEND=backend,
               PYTHONDONTWRITEBYTECODE="1")
    if verify_only:
        env["FLAGCV_IMGP_VERIFY_ONLY"] = "1"
    elif profile == "long":
        env["FLAGCV_IMGP_LONG"] = "1"
    if baseline_adapter and not verify_only:
        env["FLAGCV_IMGP_BASELINE_ADAPTER"] = str(baseline_adapter.resolve(strict=True))
    env["PYTHONPATH"] = os.pathsep.join([str(data), str(data/"compiler"), env.get("PYTHONPATH", "")])
    runners = prefix / install["runners"]
    env["FLAGCV_IMGP_FIRST_BRIDGE"] = str(runners/"libflagcv_imgproc_first_bridge.so")
    env["FLAGCV_IMGP_SPATIAL_BRIDGE"] = str(runners/"libflagcv_imgproc_spatial_bridge.so")
    for bridge in ("FLAGCV_IMGP_FIRST_BRIDGE", "FLAGCV_IMGP_SPATIAL_BRIDGE"):
        verify_identity(prefix, Path(env[bridge]), env)
    support = package / "live_support.py"
    device = json.loads(output([sys.executable, str(support)], env=env))
    if expected_uuid and device["uuid"].casefold() != expected_uuid.casefold():
        raise ValueError(f"wrong physical card, expected {expected_uuid}, observed {device['uuid']}")
    plugin = Path(device["plugin"]).resolve()
    if not plugin.is_relative_to(prefix):
        raise ValueError("backend plugin did not come from the relocated prefix")
    if hip:
        plugin_runtime = next((str(path) for name, path in library_paths(plugin, env).items()
                               if name.startswith("libamdhip64.so")), None)
        if not plugin_runtime or plugin_runtime != device.get("hip_runtime"):
            raise ValueError("HIP plugin and installed Python resolve different runtime libraries")
        env["FLAGCV_IMGP_HIP_RUNTIME"] = plugin_runtime
    env["FLAGCV_IMGP_EXPECTED_UUID"] = device["uuid"]
    device["plugin"] = str(plugin)
    device["plugin_sha256"] = hashlib.sha256(plugin.read_bytes()).hexdigest()
    import cv2
    modules = list(Path(cv2.__file__).parent.glob("**/cv2*.so"))
    if len(modules) != 1:
        raise ValueError("unable to identify the OpenCV Python extension")
    extension = modules[0].resolve()
    python_opencv = library_paths(extension, env)
    def path_identity(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    python_reference = dict(version=cv2.__version__, extension=path_identity(extension),
                            opencv={name:path_identity(path) for name,path in python_opencv.items()
                                    if name.startswith("libopencv_")})
    provider = data/"backends"/backend/"compiler.py"
    result = dict(schema="flagcv-imgproc-live/1", scope="current_install", backend=backend,
                  revision=install["revision"], prefix=str(prefix), physical_device=nv_identity or device,
                  device=device, verify_only=verify_only, python_reference=python_reference,
                  provider={"path": str(provider), "sha256": hashlib.sha256(provider.read_bytes()).hexdigest()},
                  case_manifest_sha256=hashlib.sha256((package/"cases.json").read_bytes()).hexdigest(),
                  extra_manifest_sha256=hashlib.sha256(case_manifest.read_bytes()).hexdigest() if case_manifest else None,
                  requested_case_groups=requested_groups,
                  python_profile=None if verify_only else profile,
                  cache_policy=None if verify_only else "warm buffers without L2 flush" if hip else "installed Triton do_bench",
                  rows=[])
    failed = []
    with TemporaryDirectory(prefix="flagcv-imgproc-") as work:
        work = Path(work)
        env["TRITON_CACHE_DIR"] = str(work/"triton-cache")
        trace = work/"compile.jsonl"
        compiler = work/"compiler"
        compiler.write_text("#!/bin/sh\nexec " + shlex.join([sys.executable, str(support), "--compile-trace"]) + '\n')
        compiler.chmod(0o700)
        env["FLAGCV_IMGP_COMPILE_TRACE"] = str(trace)
        env["FLAGCV_PYTHON"] = str(compiler)
        manifest = work/"selection.json"
        for index, case in enumerate(selected):
            fixture = registry[case]
            family = fixture["family"]
            binary = runners/f"flagcv_imgproc_{family}"
            cpp_identity = verify_identity(prefix, binary, env)
            if hip and cpp_identity["hip_runtime"] != env["FLAGCV_IMGP_HIP_RUNTIME"]:
                raise ValueError("C++ runner and Python resolve different HIP runtime libraries")
            verify_reference(cpp_identity["opencv"], python_reference["opencv"], hip)
            manifest.write_text(json.dumps([fixture["data"]]))
            command = [sys.executable, str(package/f"{family}.py"), "--matrix", str(manifest), "--profile", profile]
            if family == "geometry":
                command.extend(["--test-source", str(package/"test_projective.py")])
            cpp_command = [str(binary), case]
            if family == "first" and case_manifest and case in seen:
                cpp_command.extend(first_args(fixture["data"]))
            case_env = dict(env, FLAGCV_CACHE_DIR=str(work/f"flagcv-cache-{index}"))
            input_sha = None
            if "legacy_ids" in fixture:
                row = fixture["data"]
                matrices = ([1.,0.,0.,0.,1.,0.], [1.,0.,.5,0.,1.,.5], [.9,-.43589,40.,.43589,.9,-20.])
                if row["inverse"] or (row["matrix"] is not None and row["matrix"] != matrices[row["geometry"]]):
                    raise ValueError(f"unsupported frozen matrix for {case}")
                path = work/"input.raw"
                input_sha = legacy_input(row, path)
                case_env["FLAGCV_IMGP_INPUT"] = str(path)
                if row["background"]:
                    case_env["FLAGCV_IMGP_BACKGROUND"] = row["background"]
            trace.write_text("")
            cpp_raw = subprocess.run(cpp_command, cwd=work, env=case_env, text=True, capture_output=True)
            python_raw = subprocess.run(command, cwd=work, env=case_env, text=True, capture_output=True)
            compilation = []
            execution = {
                "cpp": {"status": cpp_raw.returncode, "stdout": cpp_raw.stdout, "stderr": cpp_raw.stderr},
                "python": {"status": python_raw.returncode, "stdout": python_raw.stdout,
                           "stderr": python_raw.stderr},
            }
            try:
                for line in trace.read_text().splitlines():
                    compilation.append(json.loads(line))
                if cpp_raw.returncode or python_raw.returncode:
                    raise ValueError(f"C++ exit {cpp_raw.returncode}, Python exit {python_raw.returncode}")
                if verify_only:
                    cpp = cpp_verified(case, cpp_raw.stdout)
                elif hip:
                    cpp = hip_cpp_samples(case, cpp_raw.stdout, profile)
                else:
                    cpp = cpp_samples(family, case, cpp_raw.stdout)
                python = python_samples(case, python_raw.stdout, profile, verify_only)
                if not any(item["request"]["op"] == "compile" for item in compilation):
                    raise ValueError("missing C++ code-object receipt")
            except (ValueError, KeyError, TypeError) as error:
                failed.append(case)
                result["rows"].append(dict(case=case, family=family, installed=cpp_identity,
                                           status="FAIL", diagnostic=str(error), execution=execution,
                                           cpp_compilation=compilation))
                continue
            result["rows"].append(dict(case=case, family=family, installed=cpp_identity, fixture=fixture, cpp=cpp,
                                       python=python, shared_input_sha256=input_sha, cpp_compilation=compilation, execution=execution))
            print(case, "C++", cpp["repeat"]["classification"], "Python", python["classification"], flush=True)
        result["compiled_metadata"] = [
            json.loads(path.read_text()) for path in sorted((work/"triton-cache").rglob("*.json"))
            if not path.name.startswith("__grp__")
        ]
    result["status"] = "FAIL" if failed else "PASS"
    if summary:
        summary.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise ValueError("live Imgproc cases failed: " + ", ".join(failed))
