# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch plans shared by the platform providers.

A provider owns the platform-specific decisions. Most of them are the same
shape for every vendor — pick a tile, a warp count and a grid — and differ
only in the limits the platform imposes, which live in platform.py. Keeping
the shape here means a new operator is added once, not once per vendor.
"""

from flagcv_codegen.platform import platform

# Element sizes the kernels can be instantiated at.
DEPTH_SIZE = {
    "8U": 1,
    "8S": 1,
    "16U": 2,
    "16S": 2,
    "32S": 4,
    "16F": 2,
    "32F": 4,
    "64F": 8,
}


def _power_of_two_at_most(value, floor=64):
    block = floor
    while block * 2 <= value:
        block *= 2
    return block


def _grid_width(request):
    for param in request["params"]:
        if param["name"] == "VEC4" and param["value"]:
            return ["width"]
    return ["width", "CHANNELS"]


def plan_flip(request, facts=None):
    """Tile for flip: BLOCK_X channel values per program along a row.

    Each program moves BLOCK_Y x BLOCK_X values. Block sizes follow the
    vendor's element-block ceiling from FlagGems (`facts["tile"]`) divided by
    the element size, so wide elements do not blow past it, and are capped at
    a power of two for the vectorised axis.
    """
    facts = facts or platform(request["backend"], request["target"])
    size = DEPTH_SIZE[request["images"][0]["depth"]]
    block_x = _power_of_two_at_most(max(facts["tile"] // size, 64))
    block_y = 4
    if facts["warp"] == 1:  # Ascend has no SIMT warps: one block per program row
        block_y = 1
    grid = [
        {"product": _grid_width(request), "block": block_x},
        {"product": ["height"], "block": block_y},
        1,
    ]
    num_warps = min(4, facts["max_warps"])
    tuning = {"BLOCK_X": block_x, "BLOCK_Y": block_y, "ROW_GRID_LIMIT": 0}
    return tuning, num_warps, grid


def plan_pointwise(request, facts=None):
    tuning, warps, grid = plan_flip(request, facts)
    packed = any(
        param["name"] == "PACKED" and param["value"] for param in request["params"]
    )
    if packed:
        tuning["BLOCK_Y"] = 1
        grid = [
            {"product": [*_grid_width(request), "height"], "block": tuning["BLOCK_X"]},
            1,
            1,
        ]
    return tuning, warps, grid


def plan_warp_affine(request, facts):
    nearest = request["kernel"] == "warp_affine_nearest"
    block_h, block_w = (1, 128) if nearest else (1, 64)
    col_limit = 65535 if facts["triton_backend"] in ("cuda", "hip") else 0
    grid = [
        {"product": ["dst_h"], "block": block_h},
        {"product": ["dst_w"], "block": block_w},
        1,
    ]
    if col_limit:
        grid[1]["limit"] = col_limit
        grid[2] = {"product": ["dst_w"], "block": block_w * col_limit}
    tuning = {"BLOCK_H": block_h, "BLOCK_W": block_w, "COL_GRID_LIMIT": col_limit}
    return tuning, min(4, facts["max_warps"]), grid


def plan_projective(request, facts):
    # Row tiles keep output writes contiguous across the subgroup.
    block, warps = 64, min(4, facts["max_warps"])
    if request["backend"] == "hip" and request["target"].split(":")[1] == "gfx1100":
        modes = {p["name"]: p["value"] for p in request["params"] if p["type"] == "constexpr"}
        inter, border, channels = modes["INTER"], modes["BORDER"], modes["CN"]
        layout = modes.get("LAYOUT", -1)
        if inter == 2 and channels == 4 and border not in (0, 5):
            block, warps = 256, 4
        elif inter == 2 and channels != 1:
            block, warps = 128, 4
        elif inter == 1 and layout >= 2:
            block, warps = 64, 2
        elif inter == 1 and layout >= 0 and border == 5:
            block, warps = 256, 4
        elif inter == 0 and layout >= 0:
            block, warps = 128, 4
        else:
            block, warps = 128, 2
    col_limit = 65535 if facts["triton_backend"] in ("cuda", "hip") else 0
    grid = [{"product": ["dst_h"], "block": 1},
            {"product": ["dst_w"], "block": block}, 1]
    if col_limit:
        grid[1]["limit"] = col_limit
        grid[2] = {"product": ["dst_w"], "block": block * col_limit}
    return {"BLOCK": block, "COL_GRID_LIMIT": col_limit}, warps, grid


def plan_resize(request, facts):
    area = request["kernel"] in ("resize_area", "resize_area_2tap", "resize_area_fast",
                                 "resize_area_polyphase")
    block_h, block_w = (1, 64) if area else (2, 128)
    batch_limit = 65535 if facts["triton_backend"] in ("cuda", "hip") else 0
    batch = {"product": ["batch_count"], "block": 1}
    grid = [{"product": ["dst_h", "dst_w" if area else "W"],
             "factor_blocks": [block_h, block_w], "block": 1}, batch, 1]
    if batch_limit:
        batch["limit"] = batch_limit
        grid[2] = {"product": ["batch_count"], "block": batch_limit}
    tuning = {"BLOCK_H": block_h, "BLOCK_W": block_w, "BATCH_GRID_LIMIT": batch_limit}
    if request["kernel"] in ("resize_area", "resize_area_2tap"):
        tuning["UNROLL_TWO_TAPS"] = False
    if request["kernel"] == "resize_area_polyphase":
        tuning["POLYPHASE"] = False
    if request["kernel"] == "resize_area_fast":
        tuning["RY"] = 4
    return tuning, min(4, facts["max_warps"]), grid


def plan_cvtcolor(request, facts):
    block = 128
    col_limit = 65535 if facts["triton_backend"] in ("cuda", "hip") else 0
    grid = [{"product": ["rows"], "block": 1},
            {"product": ["cols"], "block": block}, 1]
    if col_limit:
        grid[1]["limit"] = col_limit
        grid[2] = {"product": ["cols"], "block": col_limit * block}
    tuning = {"BLOCK": block, "COL_GRID_LIMIT": col_limit}
    if request["kernel"] == "cvtcolor_swap_c4":
        tuning["VECTOR4"] = False
    return tuning, min(4, facts["max_warps"]), grid



def plan_flat(request, facts):
    block = 256
    rows = "dst_h" if request["kernel"] == "copy_make_border" else "rows"
    grid = [{"product": [rows, "row_values"], "block": block}, 1, 1]
    tuning = {"BLOCK": block}
    if request["kernel"] == "threshold_histogram_u8_large":
        tuning["PRIVATE_8U"] = False
    return tuning, min(4, facts["max_warps"]), grid


def plan_threshold_clear(request, facts):
    block = 256
    return {"BLOCK": block}, min(4, facts["max_warps"]), [{"product": ["bins"], "block": block}, 1, 1]


def plan_threshold_select(request, facts):
    params = {param["name"]: param.get("value") for param in request["params"]}
    gfx1100 = request["backend"] == "hip" and request["target"] == "hip:gfx1100:32"
    return {
        "OTSU_TILE": 32 if gfx1100 and params["BINS"] == 65536 and not params["TRIANGLE"] else 0,
        "TRIANGLE_VECTOR": gfx1100 and bool(params["TRIANGLE"]),
    }, 1, [1, 1, 1]


def plan_spatial_filter(request, facts):
    block = 128
    grid = [{"product": ["width", "height", "CHANNELS"], "block": block}, 1, 1]
    tuning = {"BLOCK": block}
    if request["kernel"] == "filter2d_dense":
        tuning["ROW_PAIR"] = False
    if request["kernel"] == "filter2d_interior":
        tuning["INTERIOR_TILE"] = False
    return tuning, min(4, facts["max_warps"]), grid


def plan_gaussian_f32_local(request, facts):
    if request["backend"] == "hip" and request["target"] == "hip:gfx1100:32":
        tuning = {"BLOCK": 64, "TILE_Y": 16}
        grid = [{"product": ["width", "CHANNELS"], "block": tuning["BLOCK"]},
                {"product": ["height"], "block": tuning["TILE_Y"]}, 1]
        return tuning, 4, grid
    tuning, warps, grid = plan_spatial_filter(request, facts)
    return {**tuning, "TILE_Y": 0}, warps, grid


PLANS = dict.fromkeys(
    (
        "add", "subtract", "multiply", "bitwise_and", "bitwise_not",
        "add_masked", "subtract_masked", "bitwise_and_masked", "bitwise_not_masked",
    ),
    plan_pointwise,
)
PLANS["flip"] = plan_flip
PLANS["cvtcolor"] = plan_cvtcolor
PLANS["cvtcolor_swap_c4"] = plan_cvtcolor
PLANS["copy_make_border"] = plan_flat
PLANS["threshold_apply"] = plan_flat
PLANS["threshold_histogram"] = plan_flat
PLANS["threshold_histogram_u8_large"] = plan_flat
PLANS["threshold_clear"] = plan_threshold_clear
PLANS["threshold_select"] = plan_threshold_select
for route in ("direct", "horizontal", "vertical"):
    PLANS[f"gaussian_{route}"] = plan_spatial_filter
PLANS["gaussian_f32_local"] = plan_gaussian_f32_local
PLANS["filter2d_direct"] = plan_spatial_filter
PLANS["filter2d_interior"] = plan_spatial_filter
PLANS["filter2d_dense"] = plan_spatial_filter
PLANS["warp_perspective"] = plan_projective
PLANS["remap"] = plan_projective
for mode in ("nearest", "linear", "cubic"):
    PLANS[f"warp_affine_{mode}"] = plan_warp_affine
for route in ("copy", "nearest", "separable", "area", "area_2tap", "area_fast", "area_polyphase"):
    PLANS[f"resize_{route}"] = plan_resize


def compile_with_plan(request):
    """Dispatch a compile request to the kernel's plan for this platform."""
    facts = platform(request["backend"], request["target"])
    plan = PLANS.get(request["kernel"])
    if plan is None:
        raise ValueError(
            f"{request['backend']} has no launch plan for kernel {request['kernel']!r}"
        )
    tuning, num_warps, grid = plan(request, facts)
    return tuning, num_warps, grid, facts
