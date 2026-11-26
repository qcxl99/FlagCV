"""Paired CUDA timing for the locked small-image geometry matrix."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import statistics

import cv2
import numpy as np
import torch
import triton.testing

from helpers.benchmark_protocol import MEASUREMENT_MS, PROFILES
import live_support as live

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=PROFILES, default="default")
parser.add_argument("--test-source", type=Path, required=True)
parser.add_argument("--matrix", type=Path, required=True)
options = parser.parse_args()
profile = PROFILES[options.profile]
spec = importlib.util.spec_from_file_location("geometry_test", options.test_source)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
# Use the relocated provider, including target-specific overrides.
@geometry.lru_cache(maxsize=None)
def provider_config(interpolation, border, channels, layout):
    args = {"INTER": 1 if interpolation == 3 else interpolation, "BORDER": border, "CN": channels}
    if layout >= 0:
        args["LAYOUT"] = layout
    request = live.request("warp_perspective" if layout < 0 else "remap",
                           [dict(name=name, value=value, type="constexpr") for name, value in args.items()],
                           [{"name":"src", "channels":channels}])
    tuning, warps, _ = live.plan(request)
    return tuning["BLOCK"], warps, tuning["COL_GRID_LIMIT"]

geometry.provider_config = provider_config
kernels = geometry.kernel_module()
kernel_path = Path(kernels.__file__).resolve()
kernel_sha = sha256(kernel_path.read_bytes()).hexdigest()
stream = torch.cuda.Stream()
native_stream = None if live.HIP or live.VERIFY_ONLY else cv2.cuda.wrapStream(stream.cuda_stream)
cv2.setNumThreads(1)

def wrap(array):
    h, w = array.shape[:2]
    channels = array.shape[2] if array.ndim == 3 else 1
    kind = {torch.uint8: cv2.CV_8U, torch.uint16: cv2.CV_16U,
            torch.int16: cv2.CV_16S, torch.float32: cv2.CV_32F}
    return cv2.cuda.createGpuMatFromCudaMemory(
        h, w, cv2.CV_MAKETYPE(kind[array.dtype], channels),
        array.data_ptr(), array.stride(0) * array.element_size())

def make_image(host, y, x):
    rows, cols, channels = host.shape
    view = torch.full((rows + 2 * y + 1, cols + 2 * x + 3, channels), 91,
                       dtype=torch.from_numpy(host).dtype, device="cuda")
    image = view[y:y+rows, x:x+cols]
    image.copy_(torch.from_numpy(host))
    return image, view

def validate(actual, expected):
    geometry.output_check(actual.cpu().numpy(), expected)

def measure(calls):
    pairs = []
    for cycle in range(profile.samples):
        order = calls if cycle % 2 == 0 else calls[::-1]
        pair = {"order": [name for name, _ in order]}
        for name, invoke in order:
            pair[name] = 1000 * triton.testing.do_bench(
                invoke, warmup=profile.warmup_ms, rep=MEASUREMENT_MS, return_mode="median")
        pairs.append(pair)
    aggregate = statistics.mean if options.profile == "default" else statistics.median
    return pairs, {name: aggregate(pair[name] for pair in pairs) for name, _ in calls}

def run(row):
    depth, channels, inter, border, flag, layout = row["configuration"]
    h, w = row["destination"]
    sh, sw = row["source"]
    host, initial = geometry.geometry_inputs((sh, sw, channels), (h, w, channels), depth, 294)
    source, _ = make_image(host, 1, 3)
    target, backing = make_image(initial, 2, 5)
    native_target, _ = make_image(initial, 2, 5)
    if layout < 0:
        matrix = tuple(map(float, row.get("matrix", geometry.GENERAL)))
        coefficients = matrix if flag else geometry.inverse_coefficients(matrix)
        maps, launch_layout = (), 0
    else:
        matrix, coefficients = None, None
        x_coord = y_coord = None
        if "locality" in row:
            y, x = np.mgrid[:h,:w]
            if row["locality"] == "scatter":
                x_coord = (937*x + 719*y + 17) % sw + .25
                y_coord = (433*x + 179*y + 19) % sh + .5
            else:
                x_coord = x + .25 - (11 if row["locality"] == "outside" else 0)
                y_coord = y + .5
        maps, launch_layout = geometry.map_inputs(h, w, flag, layout, x_coord, y_coord)
    reference = geometry.cv_result(host, initial, inter, border, flag, matrix, maps)
    gmaps = tuple(make_image(m[:, :, None] if m.ndim == 2 else m, 1, 3)[0] for m in maps)
    candidate = geometry.native_candidate(
        kernels, source, target, inter, border, flag, coefficients=coefficients, maps=gmaps,
        layout=launch_layout, values=geometry.constants(host.dtype, geometry.VALUES))
    compiled = candidate()
    validate(target, reference)
    guard = backing.cpu().numpy()
    guard[2:2+h,5:5+w] = 91
    if not np.all(guard == 91) or not np.array_equal(source.cpu().numpy(), host):
        raise RuntimeError("geometry source or destination guard changed")
    block, warps, limit = geometry.provider_config(inter, border, channels, -1 if layout < 0 else launch_layout)
    tiles = triton.cdiv(w, block)
    launch = dict(path=str(kernel_path), sha256=kernel_sha, name=compiled.metadata.name,
                  hash=compiled.hash, code_sha256=live.artifact(compiled), target=live.identity()["target"], block=block, num_warps=warps,
                  num_stages=compiled.metadata.num_stages,
                  enable_fp_fusion=compiled.metadata.enable_fp_fusion,
                  grid=[h, min(tiles, limit), triton.cdiv(tiles, limit)])
    if live.HIP or live.VERIFY_ONLY:
        return dict(id=row["id"], **live.candidate_only(launch, candidate, profile, options.profile))
    mode = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA)[inter]
    native_border = geometry.CV_BORDERS[border]
    src_cv, dst_cv = wrap(source), wrap(native_target)
    if matrix is not None:
        mat = np.asarray(matrix).reshape(3, 3)
        def native():
            return cv2.cuda.warpPerspective(
                src_cv, mat, (w, h), dst_cv, mode | (cv2.WARP_INVERSE_MAP if flag else 0),
                native_border, geometry.VALUES, native_stream)
    else:
        first = wrap(gmaps[0])
        second = wrap(gmaps[1]) if len(gmaps) == 2 else cv2.cuda.GpuMat()
        def native():
            return cv2.cuda.remap(
                src_cv, first, second, mode | (cv2.WARP_RELATIVE_MAP if flag else 0),
                dst_cv, native_border, geometry.VALUES, native_stream)
    classification, error = "COMPARABLE", None
    supported = True
    try:
        result = native()
        stream.synchronize()
        if result.cudaPtr() != native_target.data_ptr():
            classification, supported = "NATIVE_LAYOUT_CHANGED", False
        else:
            try:
                validate(native_target, reference)
            except AssertionError:
                classification = "SEMANTICS_DIFFER"
    except cv2.error as exc:
        classification, supported, error = "NATIVE_REFUSED", False, str(exc)
    calls = [("flagcv", candidate)]
    if supported:
        calls.append(("opencv_cuda", native))
    pairs, event = measure(calls)
    record = dict(id=row["id"], classification=classification, error=error, launch=launch,
                  pairs=pairs, event_us=event,
                  ratio=event["opencv_cuda"] / event["flagcv"] if classification == "COMPARABLE" else None)
    if border == 5 and not supported:
        background = torch.from_numpy(initial).to("cuda")
        def reset_and_submit():
            target.copy_(background)
            candidate()
        pair, event = measure([("flagcv", reset_and_submit)])
        record["reset_included"] = dict(pairs=pair, event_us=event, ratio=None)
    return record

matrix = json.loads(options.matrix.read_text())
print(json.dumps(dict(profile=profile.name, rows=len(matrix), kernel=str(kernel_path),
                      kernel_sha256=kernel_sha, opencv=cv2.__file__, triton=triton.__version__)), flush=True)
failed = 0
with torch.cuda.stream(stream):
    for row in matrix:
        try:
            result = run(row)
        except (RuntimeError, AssertionError, cv2.error) as exc:
            failed += 1
            result = dict(id=row["id"], classification="CANDIDATE_OR_HARNESS_FAILED", error=str(exc))
        print(json.dumps(result), flush=True)
raise SystemExit(bool(failed))
