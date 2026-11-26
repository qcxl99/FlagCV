"""Device binding, installed provider selection and common live measurements."""
import ctypes
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from uuid import UUID

from helpers.benchmark_protocol import MEASUREMENT_MS

DATA = Path(os.environ["FLAGCV_DATA_DIR"]).resolve()
BACKEND = os.environ.get("FLAGCV_IMGP_BACKEND", "nvidia")
HIP = BACKEND in ("hip", "hygon")
VERIFY_ONLY = "FLAGCV_IMGP_VERIFY_ONLY" in os.environ


def hip_runtime_path():
    sonames = ("libamdhip64.so", "libgalaxyhip.so")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = Path(fields[5])
            if path.is_absolute() and any(
                    path.name == soname or path.name.startswith(soname + ".") for soname in sonames):
                paths.add(path.resolve(strict=True))
    if len(paths) != 1:
        raise RuntimeError("the plugin and Torch need a single loaded HIP runtime")
    runtime_path = str(paths.pop())
    expected = os.environ.get("FLAGCV_IMGP_HIP_RUNTIME")
    if expected and expected != runtime_path:
        raise RuntimeError("the Python case resolved a different HIP runtime library")
    return runtime_path


@lru_cache
def identity():
    import torch
    import triton

    bridge = ctypes.CDLL(os.environ["FLAGCV_IMGP_FIRST_BRIDGE"])
    bridge.device_identity.restype = bridge.last_message.restype = ctypes.c_char_p
    response = bridge.device_identity()
    if not response:
        raise RuntimeError(bridge.last_message().decode())
    facts = json.loads(response)
    if facts["backend"] != BACKEND or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("expected the selected backend and exactly one visible logical device zero")
    target = triton.runtime.driver.active.get_current_target()
    if facts["target"] != f"{target.backend}:{target.arch}:{target.warp_size}":
        raise RuntimeError("plugin and Triton device targets disagree")
    props = torch.cuda.get_device_properties(0)
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if HIP:
        if torch.version.hip is None:
            raise RuntimeError("expected a HIP Torch build")
        runtime_path = hip_runtime_path()
        runtime = ctypes.CDLL(runtime_path)
        bus_id = ctypes.create_string_buffer(32)
        version = ctypes.c_int()
        raw_uuid = ctypes.create_string_buffer(16)
        if (runtime.hipDeviceGetPCIBusId(bus_id, len(bus_id), 0)
                or runtime.hipRuntimeGetVersion(ctypes.byref(version))
                or runtime.hipDeviceGetUuid(raw_uuid, 0)):
            raise RuntimeError("unable to query HIP runtime, PCI and physical UUID")
        uuid = str(UUID(bytes=raw_uuid.raw))
        pci = bus_id.value.decode()
        bus = int(pci.rsplit(":", 2)[-2], 16)
        if int(facts["plugin_device_id"].rsplit(":bus", 1)[1]) != bus:
            raise RuntimeError("plugin and HIP runtime physical PCI bus disagree")
        torch_uuid = getattr(props, "uuid", None)
        if torch_uuid and str(torch_uuid).casefold() != uuid:
            raise RuntimeError("Torch and HIP runtime physical UUID disagree")
        facts.update(pci=pci, hip_runtime=runtime_path, hip_runtime_version=version.value)
    wanted = os.environ.get("FLAGCV_IMGP_EXPECTED_UUID")
    if wanted and uuid.casefold() != wanted.casefold():
        raise RuntimeError("execution process is bound to the wrong physical card")
    facts.update(uuid=uuid, name=props.name, logical_device=0, torch_version=torch.__version__, torch_hip=torch.version.hip,
                 torch_cuda=torch.version.cuda, triton=triton.__version__,
                 vendor=("Hygon" if target.arch in ("gfx936", "gfx938") else "AMD") if HIP else "NVIDIA")
    return facts


@lru_cache
def plan_provider():
    path = DATA / "backends" / BACKEND / "compiler.py"
    spec = importlib.util.spec_from_file_location("imgproc_live_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compile_kernel = lambda request, tuning, warps, grid: (tuning, warps, grid)
    return module, sha256(path.read_bytes()).hexdigest()


def plan(request):
    observed = identity()
    if request["backend"] != observed["backend"] or request["target"] != observed["target"]:
        raise ValueError("compile request does not match installed device")
    return plan_provider()[0].compile(request)


def request(kernel, params, images):
    platform = DATA / "backends" / BACKEND / "kernels"
    common = DATA / "kernels"
    for directory in (platform, common):
        registry = directory / "registry.json"
        if not registry.is_file():
            continue
        for entry in json.loads(registry.read_text())["kernels"]:
            if entry["name"] == kernel:
                return dict(backend=BACKEND, target=identity()["target"], kernel=kernel,
                            source=str(directory / entry["source"]), function=entry["function"],
                            images=images, params=params, from_platform_registry=(directory == platform))
    raise ValueError(f"unregistered kernel {kernel}")


def artifact(compiled):
    binary = compiled.asm["hsaco" if HIP else "cubin"]
    return sha256(binary).hexdigest()


def first_output_roi(backing, shape):
    height, width = shape[:2]
    return backing[1:1 + height, 1:1 + width]


def output_layout(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), offset=tensor.storage_offset(),
                dtype=str(tensor.dtype), device=str(tensor.device))


def matching_output_roi(candidate, backing):
    other = first_output_roi(backing, candidate.shape)
    if output_layout(other) != output_layout(candidate) or other.data_ptr() == candidate.data_ptr():
        raise ValueError("candidate and comparator need distinct buffers with the same output layout")
    return other


def measure(calls, profile, profile_name):
    """The HIP event path reuses warm buffers. Existing NV uses Triton do_bench."""
    import torch
    import triton.testing

    if VERIFY_ONLY:
        raise RuntimeError("verify-only cannot sample")
    pairs = []
    for i in range(profile.samples):
        order = calls if i % 2 == 0 else calls[::-1]
        pair = {"order": [name for name, _ in order]}
        for name, fn in order:
            if not HIP:
                pair[name] = 1000 * triton.testing.do_bench(
                    fn, warmup=profile.warmup_ms, rep=MEASUREMENT_MS, return_mode="median")
                continue
            stream = torch.cuda.current_stream()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(16)]
            start = perf_counter()
            while (perf_counter() - start) * 1000 < profile.warmup_ms:
                for _ in events:
                    fn()
                stream.synchronize()
            warm = (perf_counter() - start) * 1000
            start = perf_counter()
            raw = []
            batches = []
            group = len(events)
            while (perf_counter() - start) * 1000 < MEASUREMENT_MS:
                batch_start = perf_counter()
                for begin, end in events[:group]:
                    begin.record(stream)
                    fn()
                    end.record(stream)
                stream.synchronize()
                raw.extend(1000 * a.elapsed_time(b) for a, b in events[:group])
                batches.append(group)
                remaining = MEASUREMENT_MS - (perf_counter() - start) * 1000
                cost = max((perf_counter() - batch_start) * 1000 / group, .001)
                group = max(1, min(len(events), int(remaining / cost)))
            pair[name] = median(raw)
            pair[f"{name}_window"] = dict(raw_us=raw, warm_ms=warm, measure_ms=(perf_counter()-start)*1000,
                                          measure_batches=batches)
        pairs.append(pair)
    aggregate = mean if profile_name == "default" else median
    return pairs, {name: aggregate(pair[name] for pair in pairs) for name, _ in calls}


def candidate_only(launch, candidate, profile, profile_name, reason="no equivalent comparator configured"):
    if VERIFY_ONLY:
        return dict(classification="VERIFIED", launch=launch, candidate_verified=True)
    pairs, event = measure([("flagcv", candidate)], profile, profile_name)
    return dict(classification="CANDIDATE_ONLY", launch=launch, candidate_verified=True, pairs=pairs,
                event_us=event, ratio=None, diagnostic=reason)


def compile_trace():
    """Record the installed C++ compiler's actual requests and responses."""
    from flagcv_codegen import main
    handler = main.handle
    log = Path(os.environ["FLAGCV_IMGP_COMPILE_TRACE"])

    def recorded(req):
        start = perf_counter()
        result = handler(req)
        record = dict(request=req, response=result, compiler_wall_s=perf_counter()-start)
        if req["op"] == "compile":
            record["sha256"] = sha256(Path(result["artifact"]).read_bytes()).hexdigest()
        with log.open("a") as out:
            out.write(json.dumps(record) + "\n")
        return result

    main.handle = recorded
    main.main()


if __name__ == "__main__":
    if "--compile-trace" in sys.argv:
        compile_trace()
    else:
        print(json.dumps(identity()))
