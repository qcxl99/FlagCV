"""Direct installed Triton calls with scoped native CUDA comparisons."""
import argparse
import ctypes as c
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean, median
import time

import cv2
import numpy as np
import torch
import triton.testing

from flagcv_codegen.triton_build import load_jit_function
from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES
import live_support as live

run = Path(__file__).resolve().parent
data = Path(os.environ["FLAGCV_DATA_DIR"]).resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--matrix", required=True, type=Path)
parser.add_argument("--profile", choices=PROFILES, default="default")
options = parser.parse_args()
protocol = PROFILES[options.profile]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

point = load_module("installed_pointwise_checker", data / "tools/check_border_threshold.py")
registry = {x["name"]: x for x in json.loads((data/"kernels/registry.json").read_text())["kernels"]}
depths = (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64, np.float16)
loader = lru_cache(maxsize=None)(load_jit_function)
cv2.setNumThreads(1)
stream = torch.cuda.Stream()
native_stream = None if live.HIP or live.VERIFY_ONLY else cv2.cuda.wrapStream(stream.cuda_stream)

lib = c.CDLL(os.environ["FLAGCV_IMGP_SPATIAL_BRIDGE"])
P, I, D, L = c.c_void_p, c.c_int, c.c_double, c.c_longlong
lib.ready_gaussian.argtypes = [I, I, I, I, I, D, D, I]
lib.ready_filter.argtypes = [I, I, I, I, I, I, P, I, I, D, I]
lib.ready_gaussian.restype = lib.ready_filter.restype = P
lib.table_bytes.argtypes = lib.table_data.argtypes = lib.release.argtypes = [P]
lib.table_bytes.restype = c.c_size_t
lib.table_data.restype = P
lib.describe.argtypes = [P, P, L, P, L, I, P, P, L]
lib.describe.restype = lib.last_message.restype = c.c_char_p

def row_bytes(array):
    return array.stride(0) * array.element_size()

def wrap(array, cvdepth):
    h, w, channels = array.shape
    return cv2.cuda.createGpuMatFromCudaMemory(
        h, w, cv2.CV_MAKETYPE(cvdepth, channels), array.data_ptr(), row_bytes(array))

def image(host, x=2, y=1):
    h, w, channels = host.shape
    view = torch.full((h + y + 2, w + x + 3, channels), 91,
                       dtype=torch.from_numpy(host).dtype, device="cuda")
    target = view[y:y+h, x:x+w]
    target.copy_(torch.from_numpy(host))
    return target, view

def evaluate_grid(plan, params):
    grid = []
    for item in plan:
        if isinstance(item, int):
            grid.append(item)
            continue
        size = 1
        for name, block in zip(item["product"], item.get("factor_blocks", [1] * len(item["product"]))):
            size *= triton.cdiv(params[name], block)
        grid.append(min(triton.cdiv(size, item["block"]), item.get("limit", 2**63)))
    return grid


def point_plan(call, large_u8_otsu):
    # The existing helpers own OpenCV-compatible scalar encoding and arguments.
    # Capture those arguments once, then let the installed provider plan each launch.
    planned = []
    names = ("copy_make_border", "threshold_clear", "threshold_histogram",
             "threshold_select", "threshold_apply")
    originals = {name: getattr(point, name) for name in names}

    class CaptureKernel:
        def __init__(self, name, fn):
            self.name, self.fn = name, fn
            if name == "threshold_histogram" and large_u8_otsu:
                self.name = "threshold_histogram_u8_large"
                spec = registry[self.name]
                self.fn = loader(str(data/"kernels"/spec["source"]), spec["function"])

        def __getitem__(self, _helper_grid):
            def capture(*args, **kwargs):
                params = dict(zip(self.fn.arg_names, args))
                params.update(kwargs)
                depths = {torch.uint8:"8U", torch.int8:"8S", torch.uint16:"16U", torch.int16:"16S",
                          torch.int32:"32S", torch.float16:"16F", torch.float32:"32F", torch.float64:"64F"}
                args_and_constants = {name:params[name] for name in self.fn.arg_names if name in params}
                values, images = [], []
                for index, (name, value) in enumerate(args_and_constants.items()):
                    if isinstance(value, torch.Tensor):
                        images.append(dict(name=name.removesuffix("_ptr"), depth=depths[value.dtype],
                                           channels=value.shape[2] if value.ndim == 3 else 1,
                                           aligned16=(value.data_ptr() % 16 == 0)))
                    elif isinstance(value, (int, np.integer)):
                        values.append(dict(name=name, value=int(value),
                                           type="constexpr" if index in self.fn.constexprs else "i32"))
                request = live.request(self.name, values, images)
                tune, warps, grid_spec = live.plan(request)
                grid = evaluate_grid(grid_spec, params)
                options = {**kwargs, **tune, "num_warps": warps, "num_stages": 1}
                if self.name == "threshold_select":
                    options["enable_fp_fusion"] = False
                planned.append((self.fn[tuple(grid)], args, options, dict(grid=grid, tile=tune, source=request["source"])))
            return capture

    try:
        for name, fn in originals.items():
            setattr(point, name, CaptureKernel(name, fn))
        call()
    finally:
        for name, fn in originals.items():
            setattr(point, name, fn)

    def launch():
        return [(run(*args, **opts), plan) for run,args,opts,plan in planned]
    return launch


def spatial_call(operation, depth, source, target, ksize, edge, two_pass, weights):
    h, w, channels = source.shape
    if operation == "gauss":
        state = lib.ready_gaussian(w, h, depth, ksize, ksize, 0, 0, edge)
    else:
        state = lib.ready_filter(w, h, depth, -1, ksize, ksize, weights.ctypes.data, -1, -1, 0, edge)
    if not state:
        raise RuntimeError(lib.last_message().decode())
    inner = torch.empty_like(source, dtype=torch.int32 if depth in (0, 2) else source.dtype) if two_pass else None
    try:
        blob = c.string_at(lib.table_data(state), lib.table_bytes(state))
        table = torch.from_numpy(np.frombuffer(blob, dtype=np.float64).copy()).cuda()
        built = lib.describe(state, source.data_ptr(), row_bytes(source),
                             target.data_ptr(), row_bytes(target), channels, table.data_ptr(),
                             inner.data_ptr() if inner is not None else 0,
                             row_bytes(inner) if inner is not None else 0)
        if not built:
            raise RuntimeError(lib.last_message().decode())
        requests = json.loads(built)
    finally:
        lib.release(state)
    pointers = {x.data_ptr():x for x in (source, target, table, inner) if x is not None}
    calls = []
    for request in requests:
        path = Path(request["source"])
        tile, warps, plan = live.plan(request)
        params = {x["name"]: x["value"] for x in request["params"]}
        grid = evaluate_grid(plan, params)
        launch = loader(str(path), request["function"])[tuple(grid)]
        arguments = {img["name"]+"_ptr": pointers[img["address"]] for img in request["images"]}
        arguments.update(params)
        arguments.update(tile, num_warps=warps, num_stages=1, enable_fp_fusion=False)
        calls.append((launch, arguments, dict(grid=grid, tile=tile, source=request["source"])))
    def run_spatial():
        return [(call(**arguments), plan) for call, arguments, plan in calls]
    return run_spatial

def check_pixels(actual, expected, tolerance):
    actual = actual.cpu().numpy().reshape(expected.shape)
    if not tolerance:
        return point.exact(actual, expected)
    error = np.abs(actual.astype(np.float64) - expected)
    return bool(np.isfinite(actual).all() and np.all(error <= tolerance * np.maximum(1., np.abs(expected))))

def samples(calls):
    paired = []
    for i in range(protocol.samples):
        order = calls if i % 2 == 0 else calls[::-1]
        step = {"order": [name for name, _ in order]}
        for name, call in order:
            step[name] = 1e3 * triton.testing.do_bench(
                call, warmup=protocol.warmup_ms, rep=MEASUREMENT_MS, return_mode="median")
        paired.append(step)
    aggregate = mean if options.profile == "default" else median
    return paired, {name: aggregate(x[name] for x in paired) for name, _ in calls}

def run_case(identifier):
    operation, *numbers = identifier.split("-")
    w, h, cvtype, *args = map(int, numbers)
    channels_minus_one, depth = divmod(cvtype, cv2.CV_8UC2)
    channels = channels_minus_one + 1
    dtype = depths[depth]
    rng = np.random.default_rng(872 if operation=="copy" else 876 if operation=="threshold" else 879)
    shape = (h, w, channels)
    if depth in (0, 1, 2, 3, 4):
        limits = np.iinfo(dtype)
        host = rng.integers(max(limits.min,-127), min(limits.max+1,256), shape, dtype=dtype)
    else:
        host = rng.uniform(-2, 2, shape).astype(dtype)
    source, _ = image(host)
    native_source = None if live.HIP or live.VERIFY_ONLY else wrap(source, depth)
    tolerance, expected_scalar, device_scalar, native_plan = 0, None, None, None
    automatic = 0
    if operation == "copy":
        mode, = args
        edge = point.BORDERS[mode]
        pad, value = (3, 5, 7, 9), (2.5, 3.5, 11., 15.)
        expected = cv2.copyMakeBorder(host, *pad, edge | cv2.BORDER_ISOLATED, value=value)
        out_shape = (h+8, w+16, channels)
    elif operation == "threshold":
        mode, automatic = args
        cutoff, maxval = (0. if automatic else 51.5), 201.5
        expected_scalar, expected = cv2.threshold(host, cutoff, maxval, mode | automatic)
        out_shape = shape
    else:
        ksize, mode, two_pass = args
        edge = point.BORDERS[mode]
        tolerance = 2e-5 if depth == cv2.CV_32F else 4e-12 if depth == cv2.CV_64F else 0
        weights = None
        if operation == "gauss":
            expected = cv2.GaussianBlur(host, (ksize, ksize), 0, borderType=edge | cv2.BORDER_ISOLATED)
        else:
            weights = rng.uniform(-.3, .3, (ksize, ksize))
            expected = cv2.filter2D(host, -1, weights, borderType=edge | cv2.BORDER_ISOLATED)
        out_shape = shape
    expected = expected.reshape(out_shape)
    initial = np.zeros(out_shape, dtype=dtype)
    target, backing = image(initial, 1, 3)
    native_target, _ = image(initial, 1, 3)
    native_output = None if live.HIP or live.VERIFY_ONLY else wrap(native_target, depth)
    if operation == "copy":
        candidate = point.border_call(source, target, pad, mode, value, dtype)
        def native():
            return None, cv2.cuda.copyMakeBorder(native_source, *pad, edge, native_output, value, native_stream)
    elif operation == "threshold":
        device_scalar = torch.empty((65537 if depth==2 else 257,), dtype=torch.int32, device="cuda") if automatic else None
        candidate, host_cutoff = point.threshold_call(source, target, cutoff, maxval, mode, dtype,
                                                       work=device_scalar, auto=automatic or None)
        def native():
            return cv2.cuda.threshold(native_source, cutoff, maxval, mode | automatic, native_output, native_stream)
    else:
        candidate = spatial_call(operation, depth, source, target, ksize, edge, bool(two_pass), weights)
        def native():
            nonlocal native_plan
            if native_plan is None:
                if operation == "gauss":
                    native_plan = cv2.cuda.createGaussianFilter(cvtype,cvtype,(ksize,ksize),0,0,edge,edge)
                else:
                    native_plan = cv2.cuda.createLinearFilter(cvtype,cvtype,weights,(-1,-1),edge)
            return None, native_plan.apply(native_source, native_output, native_stream)
    if operation in ("copy", "threshold"):
        candidate = point_plan(candidate, depth == 0 and automatic == cv2.THRESH_OTSU and h*w >= 1920*1080)
    compiled = candidate()
    launch = [dict(name=k.metadata.name, hash=k.hash, code_sha256=live.artifact(k), target=live.identity()["target"], stages=k.metadata.num_stages,
                   fusion=k.metadata.enable_fp_fusion, warps=k.metadata.num_warps, **plan)
              for k,plan in compiled]
    actual_scalar = device_scalar[0].item() if device_scalar is not None else host_cutoff if operation=="threshold" else None
    if not check_pixels(target, expected, tolerance) or actual_scalar != expected_scalar:
        return dict(id=identifier, classification="CANDIDATE_CPU_DIFFER", launch=launch,
                    actual_scalar=actual_scalar, expected_scalar=expected_scalar)
    guard = backing.cpu().numpy()
    guard[3:3+out_shape[0],1:1+out_shape[1]] = 91
    if not np.all(guard == 91) or not point.exact(source.cpu().numpy(), host):
        raise RuntimeError("arithmetic or spatial source/guard changed")
    if live.HIP or live.VERIFY_ONLY:
        return dict(id=identifier, candidate_cutoff=actual_scalar, cpu_cutoff=expected_scalar,
                    **live.candidate_only(launch, candidate, protocol, options.profile))
    category, available, error = "COMPARABLE", True, None
    try:
        scalar, native_result = native()
        stream.synchronize()
        if native_result.cudaPtr() != native_target.data_ptr():
            category, available = "NATIVE_LAYOUT_CHANGED", False
        elif scalar != expected_scalar or not check_pixels(native_target, expected, tolerance):
            category = "SEMANTICS_DIFFER"
    except cv2.error as exc:
        category, available, error = "NATIVE_REFUSED", False, str(exc)
    calls = [("flagcv", candidate)]
    if available and not automatic:
        calls.append(("opencv_cuda", native))
    pairs, event = samples(calls)
    ratio = event["opencv_cuda"]/event["flagcv"] if category=="COMPARABLE" and not automatic else None
    record = dict(id=identifier, classification=category, error=error, launch=launch,
                  provider_sha256=live.plan_provider()[1], samples=pairs, event_us=event, ratio=ratio)
    if available and automatic:
        wall = []
        for _ in range(11):
            start = time.perf_counter_ns()
            native()
            stream.synchronize()
            wall.append((time.perf_counter_ns()-start)/1000)
        record["native_sync_wall_us"] = wall
        record["return_contract"] = "FlagCV device scalar; CUDA host scalar"
    return record

rows = json.loads(options.matrix.read_text())
print(json.dumps(dict(cases=len(rows), opencv=cv2.__file__, kernel_data=str(data),
                      copy_file=str(Path(point.copy_make_border.fn.__code__.co_filename).resolve()),
                      threshold_file=str(Path(point.threshold_apply.fn.__code__.co_filename).resolve()),
                      profile=protocol._asdict(), provider=str(data / "backends" / live.BACKEND / "compiler.py"))), flush=True)
with torch.cuda.stream(stream):
    for case in rows:
        try:
            record = run_case(case)
        except (cv2.error, RuntimeError, AssertionError) as exc:
            record = dict(id=case, classification="CANDIDATE_OR_HARNESS_FAILED", error=str(exc))
        print(json.dumps(record), flush=True)
