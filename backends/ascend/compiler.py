# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler provider for ascend."""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.scratch_abi import raise_unqualified

PROVIDER = "ascend"
FACTS = {
    "triton_backend": "npu",
    "binary": "npubin",
    "extension": "npubin",
    "warp": 1,
    "tile": 512,
    "max_warps": 32,
    "grid_limit": (48, 1, 1),
    "load_flags": 1,
}


def identify():
    from flagcv_codegen.triton_build import digest

    return digest(__file__, PROVIDER)


def kernel_config(request):
    tuning, num_warps, grid = compile_with_plan(request, FACTS)
    return tuning, num_warps, grid


def qualify(target, version, compiler_module):
    raise_unqualified(PROVIDER, target, version, compiler_module)
