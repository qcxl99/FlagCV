# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA compiler provider.

Kernel capability decisions live here, not in lowering: block sizes, shared
memory limits and grid extents are platform properties, so lowering stays
unchanged when a platform is added.

Resolution order is this platform's kernels/registry.json, then the common
kernels/registry.json; the runtime resolves it and names the chosen source in
the request. Platform registration is EXCLUSIVE: once an operation is
registered here, a constraint failure must be reported, not silently retried
against the common kernel. So every layout constraint a platform kernel relies
on gets checked here, before compiling, where the operand layout is fully
known.
"""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.triton_build import compile_kernel, digest

PROVIDER = "nvidia"


def identify():
    """Digest of everything that can change generated code.

    The Python interpreter, the Triton version, this file, the shared codegen
    helpers and every kernel source reachable through both registries. The
    runtime folds it into the cache key, so a kernel edit invalidates cached
    artifacts.
    """
    return digest(__file__, PROVIDER)


def compile(request):
    tuning, num_warps, grid, _ = compile_with_plan(request)
    if request["target"] == "cuda:90:32" and not request["from_platform_registry"]:
        source = request["images"][0]
        if (request["kernel"] == "warp_affine_linear" and
                source["depth"] in ("8U", "16U") and source["channels"] == 3):
            tuning["BLOCK_H"] = 2
            grid[0]["block"] = 2
        elif request["kernel"] == "resize_nearest" and source["channels"] in (1, 3):
            tuning.update(BLOCK_H=1, BLOCK_W=256)
            grid[0]["factor_blocks"] = [1, 256]
        elif request["kernel"] == "cvtcolor_swap_c4":
            tuning["VECTOR4"] = True
        elif request["kernel"] == "resize_area_polyphase":
            tuning.update(POLYPHASE=True, BLOCK_H=2, BLOCK_W=128)
            grid[0]["factor_blocks"] = [2, 256]
        elif request["kernel"] == "threshold_histogram_u8_large":
            tuning.update(PRIVATE_8U=True, BLOCK=2048)
            grid[0]["block"] = 2048
        elif request["kernel"] == "filter2d_interior":
            tuning.update(INTERIOR_TILE=True, BLOCK=256)
            grid = [{"product": ["height"], "block": 1},
                    {"product": ["width"], "block": 64}, 1]
        elif request["kernel"] == "gaussian_f32_local":
            tuning.update(BLOCK=64, TILE_Y=16)
            grid = [{"product": ["width", "CHANNELS"], "block": tuning["BLOCK"]},
                    {"product": ["height"], "block": tuning["TILE_Y"]}, 1]
        elif request["kernel"] == "resize_area_2tap":
            tuning["UNROLL_TWO_TAPS"] = True
    params = {param["name"]: param for param in request["params"]}
    if not request["from_platform_registry"]:
        if params.get("PACKED", {}).get("value"):
            # More CTAs for narrow/unaligned work while retaining wide,
            # coalesced accesses on aligned video workloads.
            bitwise = request["kernel"] in ("bitwise_and", "bitwise_not")
            if params["width"].get("div16"):
                tuning["BLOCK_X"] = 4096 if bitwise else 1024
            else:
                tuning["BLOCK_X"] = 256
            grid[0]["block"] = tuning["BLOCK_X"]
        elif (
            request["kernel"] in ("flip", "bitwise_and", "bitwise_not")
            and params["CHANNELS"]["value"] > 1
        ):
            # Pitched multichannel ROI rows are commonly unaligned.
            channels = params["CHANNELS"]["value"]
            if request["kernel"] == "flip" and channels == 3:
                tuning.update(BLOCK_X=512, BLOCK_Y=1)
                num_warps = 2
            else:
                tuning.update(BLOCK_X=256, BLOCK_Y=4)
            grid[0]["block"] = tuning["BLOCK_X"]
            grid[1]["block"] = tuning["BLOCK_Y"]
        if "ROW_GRID_LIMIT" in tuning and isinstance(grid[1], dict):
            # CUDA limits grid Y and Z to 65535 each. The pitched kernels
            # fold extra groups of rows into Z to preserve tall views.
            row_limit = 65535
            tuning["ROW_GRID_LIMIT"] = row_limit
            grid[1]["limit"] = row_limit
            grid[2] = {"product": ["height"], "block": tuning["BLOCK_Y"] * row_limit}
    return compile_kernel(request, tuning, num_warps, grid)
