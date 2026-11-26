#!/usr/bin/env python3
"""Build, test and install FlagCV and optionally run its Core GPU benchmarks."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.report import read_cpp, read_python  # noqa: E402
from helpers.benchmark_protocol import PROFILES  # noqa: E402


def validate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build, package = output / "build", output / "installed"
    backend = "" if args.profile == "host" else args.profile
    vendor = backend.upper()
    env = dict(os.environ)
    for variable in ("FLAGCV_DATA_DIR", "FLAGCV_PLUGIN_DIR"):
        env.pop(variable, None)
    env.setdefault("FLAGCV_PYTHON", sys.executable)
    env["FLAGCV_CACHE_DIR"] = str(output / "cache")
    if backend:
        env["GEMS_VENDOR"] = backend

    def run(name, *command, extra_env=None, cwd=ROOT):
        command = list(map(str, command))
        print(shlex.join(command), flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=cwd, env={**env, **(extra_env or {})},
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((output / f"{name}.log").read_text(), file=sys.stderr)
            result.check_returncode()

    summary = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=ROOT, text=True).strip(),
               "profile": args.profile, "status": "failed"}
    try:
        flags = [f"-DFLAGCV_BACKENDS={backend}", "-DFLAGCV_BUILD_TESTS=ON",
                 "-DFLAGCV_WARNINGS_AS_ERRORS=ON", "-DCMAKE_BUILD_TYPE=Release"]
        if backend:
            flags += [f"-DFLAGCV_TEST_{vendor}=ON", f"-DFLAGCV_TEST_{vendor}_DEVICES={args.devices}",
                      f"-DOpenCV_DIR={args.opencv_dir.resolve()}", "-DFLAGCV_BUILD_BENCHMARKS=ON"]
        sdk_flags = [f"-Dhip_DIR={args.hip_dir.resolve()}"] if args.hip_dir else []
        run("configure", "cmake", "-S", ROOT, "-B", build, *flags, *sdk_flags)
        run("build", "cmake", "--build", build, "--parallel", args.jobs)
        run("cpp-host", "ctest", "--test-dir", build, "-L", "host", "--no-tests=error",
            "--output-on-failure", "--output-junit", output / "cpp-host.xml")
        pytest = [sys.executable, "-m", "pytest", "-q"]
        if backend:
            run("cpp-gpu", "ctest", "--test-dir", build, "-L", backend, "--no-tests=error",
                "--output-on-failure", "--output-junit", output / "cpp-gpu.xml")
            run("python-gpu", *pytest, *sorted((ROOT / "tests").glob("test_*.py")),
                "--builtin", "--device", "cuda:0", "--junitxml", output / "python-gpu.xml")
        else:
            run("python-host", *pytest, "tests/core", "--junitxml", output / "python-host.xml")

        prefix = output / "install-before-relocation"
        run("install", "cmake", "--install", build, "--prefix", prefix)
        prefix.rename(package)
        source = output / "consumer-source"
        shutil.copytree(ROOT / "tests" / "consumer", source)
        (output / "cpp").mkdir(exist_ok=True)
        shutil.copy(ROOT / "tests" / "cpp" / "hip_test_api.hpp", output / "cpp")
        consumer = output / "consumer-build"
        consumer_flags = [f"-DFLAGCV_CONSUMER_NVIDIA={'ON' if backend == 'nvidia' else 'OFF'}"]
        if backend in ("amd", "hygon"):
            consumer_flags.append(f"-DFLAGCV_CONSUMER_{vendor}=ON")
        run("consumer-configure", "cmake", "-S", source, "-B", consumer,
            f"-DCMAKE_PREFIX_PATH={package}", *consumer_flags, *sdk_flags)
        run("consumer-build", "cmake", "--build", consumer, "--parallel", args.jobs)
        application = consumer / "flagcv_install_consumer"
        ldd = subprocess.check_output(["ldd", application], text=True)
        runtime = next((line.split("=>", 1)[1].strip().split(" (", 1)[0]
                        for line in ldd.splitlines() if "libflagcv.so" in line and "=>" in line), "")
        if package not in Path(runtime).resolve().parents:
            raise ValueError("tree-out application did not link the relocated runtime")
        run("consumer", application, cwd=output)

        if backend:
            benchmark_env = {}
            run("compiler", env["FLAGCV_PYTHON"], "-c",
                "import sys, triton; print(sys.executable, triton.__version__, triton.__file__)")
            run("python-environment", sys.executable, "-c",
                "import sys, torch, cv2, triton; "
                "print(sys.executable, torch.__version__, cv2.__version__, cv2.__file__, "
                "triton.__version__, triton.__file__); print(torch.cuda.get_device_properties(0))")
            if backend == "nvidia":
                bridge = output / "bridge"
                run("bridge-configure", "cmake", "-S", ROOT / "adapters/opencv_cuda", "-B", bridge,
                    f"-DOpenCV_DIR={args.opencv_dir.resolve()}")
                run("bridge-build", "cmake", "--build", bridge, "--parallel", args.jobs)
                benchmark_env["FLAGCV_OPENCV_LIBRARY"] = str(bridge / "libflagcv_opencv_cuda.so")
            cpp, python = output / "cpp-core.csv", output / "python-core.json"
            exe = build / "backends" / backend / "validation" / f"flagcv_benchmark_{backend}"
            run("cpp-benchmark", exe, "0", cpp)
            options = ["--base-data", args.base_data.resolve()] if args.base_data else []
            run("python-benchmark", *pytest, "benchmark", "--builtin", "--device", "cuda:0",
                "--benchmark-profile", args.python_benchmark_profile, "--record", "json",
                "--output", python, *options, extra_env=benchmark_env)
            device, explicit, convenience = read_cpp(cpp)
            kernels = read_python(python)
            if len(convenience) != 3:
                raise ValueError("expected three steady-state setDevice comparison cases")
            summary.update(device=device, cpp_cases=len(explicit), convenience_cases=len(convenience),
                           python_cases=len(kernels))
        summary["status"] = "passed"
    finally:
        (output / "acceptance.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("host", "nvidia", "amd", "hygon"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="a new evidence directory")
    parser.add_argument("--opencv-dir", type=Path, help="OpenCV CMake package directory")
    parser.add_argument("--hip-dir", type=Path, help="HIP CMake package directory")
    parser.add_argument("--python-benchmark-profile", choices=tuple(PROFILES), default="default")
    parser.add_argument("--base-data", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.profile != "host" and args.opencv_dir is None:
        parser.error("GPU profiles require --opencv-dir")
    try:
        validate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
