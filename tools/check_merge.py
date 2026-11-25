"""Run FlagCV's merge checks and retain their raw evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from helpers.benchmark_protocol import PROFILES

NVIDIA_TESTS = {"flagcv.nvidia.public_api", "flagcv.nvidia.platform_failure",
                "flagcv.nvidia.warpaffine", "flagcv.nvidia.resize", "flagcv.nvidia.cvtcolor",
                "flagcv.nvidia.border_threshold", "flagcv.nvidia.spatial_filters",
                "flagcv.nvidia.projective"}


def check_junit(path, expected=None):
    cases = ET.parse(path).findall(".//testcase")
    names = {case.get("name") for case in cases}
    if not cases or (expected and not expected <= names):
        raise ValueError(f"{path.name}: unexpected test inventory {sorted(names)}")
    bad = [case.get("name") for case in cases if (
        case.get("status") in {"notrun", "disabled", "skipped"}
        or any(case.find(kind) is not None for kind in ("skipped", "failure", "error"))
    )]
    if bad:
        raise ValueError(f"{path.name}: unsuccessful tests {bad}")


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def source_id():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=SOURCE)
    files = sorted(set(os.fsdecode(tracked).split("\0")) - {""})
    entries = [(name, sha256(SOURCE / name) if (SOURCE / name).is_file() else None)
               for name in files]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


class Validation:
    def __init__(self, args):
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.profile = args.profile
        self.python_benchmark_profile = args.python_benchmark_profile
        self.opencv = args.opencv_dir.resolve() if args.opencv_dir else None
        self.imgproc_evidence = args.imgproc_evidence.resolve() if args.imgproc_evidence else None
        self.imgproc_scope = args.imgproc_scope
        self.build = self.output / "build"
        self.python = sys.executable
        self.env = {key: value for key, value in os.environ.items()
                    if key not in {"FLAGCV_DATA_DIR", "FLAGCV_PLUGIN_DIR",
                                   "FLAGCV_OPENCV_LIBRARY", "PYTEST_ADDOPTS", "LD_DEBUG_OUTPUT"}}
        self.env.update(FLAGCV_PYTHON=self.python, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                        FLAGCV_CACHE_DIR=str(self.output / "flagcv-cache"),
                        TRITON_CACHE_DIR=str(self.output / "triton-cache"))
        self.report = {"schema": "flagcv-merge-check/1", "profile": self.profile,
                       "source": str(SOURCE), "status": "FAIL", "steps": [], "artifacts": []}

    def run(self, name, *command, cwd=SOURCE, env=None):
        argv = list(map(str, command))
        log = self.output / f"{name}.log"
        self.report["artifacts"].append(log.name)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(argv, cwd=cwd, env=self.env if env is None else env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        self.report["steps"].append({"id": name, "argv": argv, "cwd": str(cwd),
                                     "exit_code": result.returncode, "log": log.name,
                                     "wall_seconds": round(time.monotonic() - started, 3)})
        if result.returncode:
            raise ValueError(f"{name} exited {result.returncode}; see {log}")
        return log

    def tests(self, name, *command, expected=None):
        report = self.output / f"{name}.xml"
        self.run(name, *command, report)
        self.report["artifacts"].append(report.name)
        check_junit(report, expected)

    def installed_consumer(self, gpu):
        staging = self.output / "staging"
        prefix = self.output / "relocated"
        self.run("install", "cmake", "--install", self.build, "--prefix", staging)
        staging.rename(prefix)
        packages = list(prefix.glob("**/cmake/FlagCV/FlagCVConfig.cmake"))
        if len(packages) != 1:
            raise ValueError("expected one installed FlagCV CMake package")
        package = packages[0].parent
        consumer = self.output / "consumer"
        self.run("consumer-configure", "cmake", "-S", SOURCE / "tests/consumer", "-B", consumer,
                 f"-DFlagCV_DIR={package}", f"-DFLAGCV_CONSUMER_NVIDIA={'ON' if gpu else 'OFF'}",
                 cwd=self.output)
        self.run("consumer-build", "cmake", "--build", consumer, "--parallel", "2",
                 cwd=self.output)
        env = {**self.env, "LD_DEBUG": "libs"}
        if gpu:
            # The compiler receives the Runtime's actual data path. Reject a
            # silent fallback to the checkout before delegating to real Python.
            data = prefix / "share/flagcv"
            guard = self.output / "installed-python"
            log_line = 'printf "FlagCV installed data: %s\\n" "$actual" >&2\n'
            guard.write_text("#!/bin/sh\n"
                             'actual=$(realpath "$FLAGCV_DATA_DIR") || exit 1\n'
                             f'[ "$actual" = {shlex.quote(str(data))} ] || exit 1\n'
                             + log_line + f'exec {shlex.quote(self.python)} "$@"\n',
                             encoding="utf-8")
            guard.chmod(0o700)
            env.update(FLAGCV_PYTHON=str(guard),
                       FLAGCV_CACHE_DIR=str(self.output / "installed-cache"),
                       TRITON_CACHE_DIR=str(self.output / "installed-triton-cache"))
        log = self.run("consumer-run", consumer / "flagcv_install_consumer",
                       cwd=self.output, env=env)
        output = log.read_text(encoding="utf-8", errors="replace")
        loaded = [line.split("calling init:", 1)[1].strip() for line in output.splitlines()
                  if "calling init:" in line]
        expected = ["libflagcv.so"] + (["libflagcv_nvidia.so"] if gpu else [])
        for library in expected:
            if not any(Path(path).name.startswith(library)
                       and Path(path).resolve().is_relative_to(prefix) for path in loaded):
                raise ValueError(f"consumer did not load installed {library} from {prefix}")
        if gpu and f"FlagCV installed data: {data}" not in output:
            raise ValueError("consumer did not start the compiler from its relocated data")

    def check(self):
        gpu = self.profile == "nvidia"
        self.run("revision", "git", "rev-parse", "HEAD")
        self.run("cmake-version", "cmake", "--version")
        self.run("cxx-version", "c++", "--version")
        version_info = (
            "import sys, cv2, torch; "
            "print(sys.version); print('OpenCV CPU:', cv2.__version__, cv2.__file__); "
            "print('PyTorch:', torch.__version__)"
        )
        flags = ["-DFLAGCV_BACKENDS=", "-DFLAGCV_WARNINGS_AS_ERRORS=ON"]
        if gpu:
            selection = self.env.get("CUDA_VISIBLE_DEVICES", "")
            if not selection.startswith("GPU-") or "," in selection:
                raise ValueError("select a single physical NVIDIA UUID in CUDA_VISIBLE_DEVICES")
            self.report["gpu_uuid"] = selection
            self.run("gpu", "nvidia-smi", "--id", selection,
                     "--query-gpu=uuid,pci.bus_id,name,driver_version", "--format=csv")
            version_info += (
                "; import triton; print('Triton:', triton.__version__); "
                "print('CUDA:', torch.version.cuda); "
                "assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_properties(0))"
            )
            flags = ["-DFLAGCV_BACKENDS=nvidia", "-DFLAGCV_WARNINGS_AS_ERRORS=ON",
                     "-DFLAGCV_TEST_NVIDIA=ON", "-DFLAGCV_TEST_NVIDIA_DEVICES=1",
                     "-DFLAGCV_BENCHMARK_NVIDIA=ON", f"-DOpenCV_DIR={self.opencv}"]
        self.run("python-versions", self.python, "-c", version_info)
        self.run("configure", "cmake", "-S", SOURCE, "-B", self.build, "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", *flags)
        self.run("build", "cmake", "--build", self.build, "--parallel", "2")
        self.report["artifacts"].append("build/CMakeCache.txt")
        if gpu:
            self.tests("cpp-gpu", "ctest", "--test-dir", self.build, "-L", "nvidia",
                       "--output-on-failure", "--no-tests=error", "--output-junit",
                       expected=NVIDIA_TESTS)
            tests = sorted((SOURCE / "tests").glob("test_*.py"))
            if not tests:
                raise ValueError("no Python operator tests found")
            self.tests("python-gpu", self.python, "-m", "pytest", "-q", *tests,
                       "--builtin", "--device", "cuda:0", "--junitxml")
        else:
            self.tests("cpp-host", "ctest", "--test-dir", self.build, "-L", "host",
                       "--output-on-failure", "--no-tests=error", "--output-junit",
                       expected={"flagcv.registry", "flagcv.runtime"})
            self.tests("python-host", self.python, "-m", "pytest", "-q", "tests/core",
                       "--junitxml")
        self.installed_consumer(gpu)
        if gpu:
            self.performance()
        if self.imgproc_evidence:
            summary = self.output / "imgproc-archive-summary.json"
            self.report["artifacts"].append(summary.name)
            scopes = [arg for scope in self.imgproc_scope for arg in ("--require-imgproc-scope", scope)]
            self.run("check-imgproc-archive", self.python, SOURCE / "benchmark/check_nvidia.py",
                     "--imgproc-evidence", self.imgproc_evidence, "--summary", summary, *scopes)
            self.report["imgproc_archive"] = {"scope": "historical", "groups": self.imgproc_scope}

    def opencv_libraries(self, name, binary, env):
        log = self.run(name, "ldd", binary, env=env)
        libraries = {}
        for line in log.read_text().splitlines():
            args = line.split()
            if args and args[0].startswith("libopencv_"):
                if len(args) < 3 or args[1] != "=>" or not Path(args[2]).is_file():
                    raise ValueError(f"failed to resolve {args[0]} for {binary.name}")
                libraries[args[0]] = sha256(Path(args[2]))
        if not any(name.startswith("libopencv_cudaarithm.") for name in libraries):
            raise ValueError(f"no live OpenCV CUDA dependency for {binary.name}")
        return libraries

    def performance_identity(self, cpp, python, cpp_libs, python_libs, source):
        if cpp_libs != python_libs:
            raise ValueError("C++ and Python resolve different OpenCV baselines")
        if source_id() != source:
            raise ValueError("FlagCV source changed during measurement")
        with (self.output / "gpu.log").open() as stream:
            device, = csv.DictReader(stream, skipinitialspace=True)
        protocol = PROFILES[self.python_benchmark_profile]
        record = {
            "schema": "flagcv-nvidia-evidence/2",
            "candidate": {"source_sha256": source,
                          "revision": (self.output / "revision.log").read_text().strip(),
                          "cpp_binary_sha256": sha256(self.build / "tests/flagcv_benchmark_nvidia"),
                          "runtime_sha256": sha256(self.build / "src/libflagcv.so")},
            "environment": {
                "device": {"uuid": device["uuid"], "pci_bus": device["pci.bus_id"],
                           "model": device["name"], "driver": device["driver_version"]},
                "opencv_libraries": cpp_libs,
                "python_versions": (self.output / "python-versions.log").read_text().strip(),
                "compiler": (self.output / "cxx-version.log").read_text().splitlines()[0],
                "workload_sha256": sha256(SOURCE / "benchmark/nvidia_cases.json"),
                "protocol": {"cpp": "flagcv-cpp-opencv-ticks-stream-wait/2",
                             "python": protocol.name,
                             "cpp_samples": 11, "python_samples": protocol.samples},
            },
            "artifacts": {"cpp": sha256(cpp), "python": sha256(python)},
        }
        if record["environment"]["device"]["uuid"] != self.report["gpu_uuid"]:
            raise ValueError("selected GPU disagrees with device evidence")
        identity = self.output / "performance-identity.json"
        identity.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        self.report["artifacts"].append(identity.name)
        return identity

    def performance(self):
        bridge = self.output / "opencv-bridge"
        self.run("opencv-bridge-configure", "cmake", "-S", SOURCE / "adapters/opencv_cuda",
                 "-B", bridge, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                 f"-DOpenCV_DIR={self.opencv}")
        self.run("opencv-bridge-build", "cmake", "--build", bridge, "--parallel", "2")
        cpp = self.output / "cpp.csv"
        python = self.output / "python.json"
        source = source_id()
        binary = self.build / "tests/flagcv_benchmark_nvidia"
        cpp_libs = self.opencv_libraries("cpp-baseline-libraries", binary, self.env)
        self.run("cpp-benchmark", binary, "0", cpp, "11")
        self.report["artifacts"].append(cpp.name)
        env = {**self.env, "FLAGCV_OPENCV_LIBRARY": str(bridge / "libflagcv_opencv_cuda.so")}
        python_libs = self.opencv_libraries("python-baseline-libraries",
                                            Path(env["FLAGCV_OPENCV_LIBRARY"]), env)
        self.run("python-benchmark", self.python, "-m", "pytest", "-q", "benchmark",
                 "--builtin", "--device", "cuda:0", "--benchmark-profile", self.python_benchmark_profile,
                 "--record", "json", "--output", python,
                 env=env)
        self.report["artifacts"].append(python.name)
        identity = self.performance_identity(cpp, python, cpp_libs, python_libs, source)
        summary = self.output / "performance-summary.json"
        self.run("check-performance", self.python, SOURCE / "benchmark/check_nvidia.py",
                 "--cpp", cpp, "--python", python, "--identity", identity, "--summary", summary,
                 "--python-benchmark-profile", self.python_benchmark_profile)
        self.report["artifacts"].append(summary.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("host", "nvidia"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="a new evidence directory")
    parser.add_argument("--imgproc-evidence", type=Path, help="root of the archived H20 runs")
    parser.add_argument("--imgproc-scope", action="append",
                        choices=("all", "first-batch", "geometry-small", "geometry-large", "arithmetic-filters", "resize-batch"))
    parser.add_argument("--opencv-dir", type=Path, help="OpenCV CMake package directory for NVIDIA")
    parser.add_argument("--python-benchmark-profile", choices=tuple(PROFILES), default="default")
    args = parser.parse_args()
    if bool(args.imgproc_evidence) != bool(args.imgproc_scope):
        parser.error("--imgproc-evidence and --imgproc-scope must be used together")
    if args.profile == "host" and args.python_benchmark_profile != "default":
        parser.error("--python-benchmark-profile applies to NVIDIA")
    if (args.profile == "nvidia") != bool(args.opencv_dir):
        parser.error("--opencv-dir is required for NVIDIA and unused for host")
    if args.opencv_dir and not (args.opencv_dir / "OpenCVConfig.cmake").is_file():
        parser.error("--opencv-dir must contain OpenCVConfig.cmake")
    validation = Validation(args)
    try:
        validation.check()
        validation.report["status"] = "PASS"
    except (OSError, ValueError, ET.ParseError) as error:
        validation.report["error"] = str(error)
        print(error, file=sys.stderr)
    finally:
        temporary = validation.output / ".acceptance.json.tmp"
        temporary.write_text(json.dumps(validation.report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(validation.output / "acceptance.json")
    return int(validation.report["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
