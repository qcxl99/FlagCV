"""Installed or in-tree Triton geometry kernels against native OpenCV CPU."""

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys

import pytest


IDENTITY = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
GENERAL = (1.08, 0.043, -1.49, -0.02, 0.93, -0.51, 0.002, -0.003, 1.)
POLE = (0.3, 0., -1., 0.1, 0.3, 0., 1., 0., -4.)
VALUES = (2.5, -1.25, 123., 9.)
CV_BORDERS = (0, 1, 2, 3, 4, 5)


def kernel_module():
    root = Path(os.getenv("FLAGCV_PROJECTIVE_KERNELS", Path(__file__).parents[1] / "kernels"))
    spec = importlib.util.spec_from_file_location("flagcv_projective", root / "common/projective.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inverse_coefficients(matrix):
    import cv2
    import numpy as np

    _, inverse = cv2.invert(np.array(matrix, dtype=np.float64).reshape(3, 3), cv2.DECOMP_LU)
    return tuple(inverse.flat)


def constants(dtype, values):
    import numpy as np

    if dtype.kind == "f":
        return values
    return tuple(float(np.clip(round(v), 0, np.iinfo(dtype).max))
                 if round(v) <= 2147483647 else 0. for v in values)


def cv_result(src, initial, interpolation, border, flag, matrix=None, maps=(), values=VALUES):
    import cv2
    import numpy as np

    dst = initial.copy()
    mode = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA)[interpolation]
    if matrix is not None:
        transform = np.array(matrix, dtype=np.float64).reshape(3, 3)
        cv2.warpPerspective(src, transform, (dst.shape[1], dst.shape[0]), dst=dst,
                            flags=mode | (cv2.WARP_INVERSE_MAP if flag else 0),
                            borderMode=CV_BORDERS[border], borderValue=values,
                            hint=cv2.ALGO_HINT_ACCURATE)
    else:
        cv2.remap(src, maps[0], maps[1] if len(maps) == 2 else None,
                  mode | (cv2.WARP_RELATIVE_MAP if flag else 0), dst=dst,
                  borderMode=CV_BORDERS[border], borderValue=values,
                  hint=cv2.ALGO_HINT_ACCURATE)
    return dst


@lru_cache(maxsize=None)
def provider_config(interpolation, border, channels, layout):
    import torch

    data = os.getenv("FLAGCV_PROJECTIVE_DATA")
    if data:
        sys.path.insert(0, str(Path(data) / "compiler"))
    else:
        sys.path.insert(0, str(Path(__file__).parents[1] / "compiler"))
    from flagcv_codegen.providers import compile_with_plan

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if torch.version.hip:
        arch = props.gcnArchName.split(":")[0]
        backend = "hygon" if arch in ("gfx936", "gfx938") else "hip"
        target = f"hip:{arch}:{props.warp_size}"
    else:
        backend = "nvidia"
        target = f"cuda:{props.major}{props.minor}:32"
    args = dict(INTER=1 if interpolation == 3 else interpolation, BORDER=border, CN=channels)
    if layout >= 0:
        args["LAYOUT"] = layout
    request = dict(backend=backend, target=target,
                   kernel="warp_perspective" if layout < 0 else "remap",
                   params=[dict(name=name, value=value, type="constexpr") for name, value in args.items()])
    tuning, warps, _, _ = compile_with_plan(request)
    return tuning["BLOCK"], warps, tuning["COL_GRID_LIMIT"]


def native_candidate(module, src, dst, interpolation, border, flag, coefficients=None,
                     maps=(), layout=0, values=VALUES, block=None, warps=None, vector_c4=False):
    import triton

    height, width = dst.shape[:2]
    planned_block, planned_warps, limit = provider_config(
        interpolation, border, src.shape[2], -1 if coefficients is not None else layout)
    block = planned_block if block is None else block
    warps = planned_warps if warps is None else warps
    common = (*src.shape[:2], height, width, src.stride(0), dst.stride(0), *values)
    constexprs = dict(CN=src.shape[2], INTER=1 if interpolation == 3 else interpolation,
                      BORDER=border, BLOCK=block, num_warps=warps, num_stages=1,
                      enable_fp_fusion=False)
    columns = triton.cdiv(width, block)
    constexprs["COL_GRID_LIMIT"] = limit
    grid = ((height, min(columns, limit), triton.cdiv(columns, limit))
            if limit else (height, columns, 1))
    if coefficients is not None:
        if vector_c4:
            constexprs["VECTOR_C4"] = True
        return lambda: module.perspective_kernel[grid](src, dst, *coefficients, *common, **constexprs)
    first, second = maps if len(maps) == 2 else (maps[0], maps[0])
    return lambda: module.remap_kernel[grid](src, dst, first, second, first.stride(0),
                                            second.stride(0), *common, LAYOUT=layout,
                                            RELATIVE=flag, **constexprs)


def geometry_inputs(src_shape, dst_shape, depth, seed):
    import numpy as np

    rng = np.random.default_rng(seed)

    def image(shape):
        if depth == "float32":
            return rng.uniform(-2., 2., shape).astype(depth)
        return rng.integers(0, 256 if depth == "uint8" else 65536, shape, dtype=depth)

    return image(src_shape), image(dst_shape)


def map_inputs(height, width, relative, layout, x_values=None, y_values=None):
    import cv2
    import numpy as np

    y, x = np.mgrid[:height, :width]
    u = (x * 23 + y * 9) % 337 / 10 - 1.11 if x_values is None else x_values
    v = (x * 5 + y * 63) % 191 / 10 - 1.11 if y_values is None else y_values
    xy = np.stack((u - x if relative else u, v - y if relative else v), axis=-1).astype("float32")
    if layout == 0:
        return (xy,), 0
    if layout == 1:
        return (np.ascontiguousarray(xy[..., 0]), np.ascontiguousarray(xy[..., 1])), 1
    integer, frac = cv2.convertMaps(xy, None, cv2.CV_16SC2, nninterpolation=(layout == 4))
    if layout == 4:
        return (integer,), 3
    if layout == 3:
        frac = frac.view("int16")
    return (integer, frac), 2


def output_check(actual, expected):
    import numpy as np

    if expected.dtype.kind != "f":
        np.testing.assert_array_equal(actual, expected)
        return
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    error = np.abs(actual[finite].astype(np.float64) - expected[finite])
    bound = np.minimum(1e-4, 5e-5 * np.maximum(1.0, np.abs(expected[finite])))
    assert (error <= bound).all(), (error.max(), np.count_nonzero(error > bound))


@pytest.fixture(scope="session")
def device_geometry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    cv2 = pytest.importorskip("cv2")
    if not torch.cuda.is_available():
        pytest.skip("geometry tests require a CUDA-shaped GPU runtime")
    if not cv2.__version__.startswith("5."):
        pytest.skip("geometry reference requires OpenCV 5")
    return kernel_module()


def matrix_cases():
    for depth in ("uint8", "uint16", "float32"):
        for channels in (1, 3, 4):
            for inter in range(4):
                for border in range(6):
                    for flag in (False, True):
                        yield depth, channels, inter, border, flag, -1
                        for layout in range(5 if inter == 0 else 4):
                            yield depth, channels, inter, border, flag, layout


@pytest.mark.parametrize("depth,channels,inter,border,flag,layout", list(matrix_cases()))
def test_geometry(device_geometry, depth, channels, inter, border, flag, layout):
    import numpy as np
    import torch

    src_host, dst_host = geometry_inputs((21, 42, channels), (15, 145, channels), depth, 294)
    src = src_host[2:19, 3:34]
    dst = dst_host[2:13, 5:136]
    if layout < 0:
        matrix = GENERAL
        maps, fmt = (), 0
    else:
        matrix = None
        maps, fmt = map_inputs(*dst.shape[:2], flag, layout)
    expected = cv_result(src, dst, inter, border, flag, matrix, maps)
    s = torch.from_numpy(src_host).cuda()
    t = torch.from_numpy(dst_host).cuda()
    padded_maps = []
    for array in maps:
        h, w = array.shape[:2]
        staging = np.pad(array, ((2, 1), (3, 2)) + ((0, 0),) * (array.ndim - 2))
        padded_maps.append((staging, torch.from_numpy(staging).cuda(), h, w))
    with torch.cuda.stream(torch.cuda.Stream() if flag else torch.cuda.default_stream()):
        coefficients = None if matrix is None else matrix if flag else inverse_coefficients(matrix)
        call = native_candidate(device_geometry, s[2:19, 3:34], t[2:13, 5:136], inter,
                                border, flag, coefficients,
                                tuple(a[2:2 + h, 3:3 + w] for _, a, h, w in padded_maps),
                                fmt, constants(src.dtype, VALUES))
        torch.cuda.synchronize()
        call()
    torch.cuda.synchronize()
    got = t.cpu().numpy()
    output_check(got[2:13, 5:136], expected)
    got[2:13, 5:136] = dst
    np.testing.assert_array_equal(got.view("uint8"), dst_host.view("uint8"))
    np.testing.assert_array_equal(s.cpu().numpy().view("uint8"), src_host.view("uint8"))
    for reference, array, _, _ in padded_maps:
        np.testing.assert_array_equal(array.cpu().numpy().view("uint8"), reference.view("uint8"))


@pytest.mark.parametrize("mode", (0, 1, 2))
@pytest.mark.parametrize("border", (0, 1, 3, 5))
@pytest.mark.parametrize("constant", (-4., -4.000000476837158203125, -4.000000000001))
def test_projection_pole(device_geometry, mode, border, constant):
    import numpy as np
    import torch

    src, initial = geometry_inputs((17, 31, 1), (13, 17, 1), "uint16", 19456)
    matrix = (*POLE[:8], constant)
    expected = cv_result(src, initial, mode, border, True, matrix)
    s = torch.from_numpy(src).cuda()
    d = torch.from_numpy(initial).cuda()
    native_candidate(device_geometry, s, d, mode, border, True, matrix,
                     values=constants(np.dtype("uint16"), VALUES))()
    output_check(d.cpu().numpy(), expected)


@pytest.mark.parametrize("border", (0, 5))
@pytest.mark.parametrize("pixel", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize("width,coordinate", ((1, -2.5), (4, -3.5)))
@pytest.mark.parametrize("remap", (False, True))
def test_cubic_nonfinite(device_geometry, border, pixel, width, coordinate, remap):
    import numpy as np
    import torch

    src = np.full((1, 1, 1), pixel, np.float32)
    initial = np.full((1, width, 1), 13, np.float32)
    mapping = np.zeros((1, width, 2), np.float32)
    mapping[..., 0] = coordinate
    matrix = (0., 0., coordinate, 0., 0., 0., 0., 0., 1.)
    maps = (mapping,) if remap else ()
    reference = cv_result(src, initial, 2, border, not remap, None if remap else matrix,
                          maps, (7., 0., 0., 0.))
    s = torch.from_numpy(src).cuda()
    dst = torch.from_numpy(initial).cuda()
    native_candidate(device_geometry, s, dst, 2, border, not remap, None if remap else matrix,
                     tuple(torch.from_numpy(a).cuda() for a in maps), values=(7., 0., 0., 0.))()
    output_check(dst.cpu().numpy(), reference)


def test_cubic_transparent_group_inlier(device_geometry):
    import numpy as np
    import torch

    src = np.ones((8, 8, 1), np.float32)
    initial = np.array([float("inf"), 10., 11., 12.], np.float32).reshape(1, 4, 1)
    for x in (-.5, 2.5):
        coords = np.array([[[x, 2.5], [2.5, 2.5], [3.5, 2.5], [4.5, 2.5]]], np.float32)
        reference = cv_result(src, initial, 2, 5, False, maps=(coords,))
        s = torch.from_numpy(src).cuda()
        dst = torch.from_numpy(initial).cuda()
        native_candidate(device_geometry, s, dst, 2, 5, False, maps=(torch.from_numpy(coords).cuda(),))()
        output_check(dst.cpu().numpy(), reference)


@pytest.mark.parametrize("width", (2147450881, 2147483519))
def test_perspective_rectangular_grid_overflow(device_geometry, width):
    import numpy as np
    import torch
    import triton
    import triton.language as tl

    column = device_geometry._perspective_column

    @triton.jit
    def probe(indices, result, width: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.arange(0, BLOCK)
        candidate = tl.load(indices + lane)
        safe, valid = column(candidate, width, 128)
        tl.store(result + lane, tl.where(valid, safe * 128, -1))

    # The reproduced split-grid width and the largest accepted C1 extent.
    last = (width - 1) // 128
    indices = torch.tensor([0, last, last + 1, 16777216], dtype=torch.int32, device="cuda")
    result = torch.empty_like(indices)
    probe[(1,)](indices, result, width, 4)
    np.testing.assert_array_equal(result.cpu().numpy(), [0, last * 128, -1, -1])
