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
