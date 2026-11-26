# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler provider for nvidia."""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.scratch_abi import raise_unqualified, validated_scratch_args

PROVIDER = "nvidia"
FACTS = {
    "triton_backend": "cuda",
    "binary": "cubin",
    "extension": "cubin",
    "warp": 32,
    "tile": 1024,
    "max_warps": 32,
}
FLAGTREE_RELEASE = "0.7.1.dev20261125"


def identify():
    from flagcv_codegen.triton_build import digest

    return digest(__file__, PROVIDER)


def kernel_config(request):
    tuning, num_warps, grid = compile_with_plan(request, FACTS)
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
        if isinstance(grid[1], dict):
            # CUDA limits grid Y and Z to 65535 each. The pitched kernels
            # fold extra groups of rows into Z to preserve tall views.
            row_limit = 65535
            tuning["ROW_GRID_LIMIT"] = row_limit
            grid[1]["limit"] = row_limit
            grid[2] = {"product": ["height"], "block": tuning["BLOCK_Y"] * row_limit}
    return tuning, num_warps, grid


def qualify(target, version, compiler_module):
    if (target.backend != "cuda" or target.warp_size != 32
            or version != "3.6.0" or compiler_module != "triton.backends.nvidia.compiler"):
        raise_unqualified(PROVIDER, target, version, compiler_module)

    from importlib import metadata
    from pathlib import Path

    try:
        distribution = metadata.distribution("flagtree-nvidia")
    except metadata.PackageNotFoundError:
        raise_unqualified(PROVIDER, target, version, compiler_module)
    if distribution.version != FLAGTREE_RELEASE:
        raise_unqualified(PROVIDER, target, version, compiler_module)

    import triton

    if (Path(triton.__file__).resolve()
            != distribution.locate_file("triton/__init__.py").resolve()):
        raise_unqualified(PROVIDER, target, version, compiler_module)


def validate_scratch(compiled, signature):
    return validated_scratch_args(compiled.metadata, compiled.asm.get("ptx"), signature)
