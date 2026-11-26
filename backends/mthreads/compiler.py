# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler provider for mthreads."""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.scratch_abi import raise_unqualified

PROVIDER = "mthreads"
FACTS = {
    "triton_backend": "mtgpu",
    "binary": "mubin",
    "extension": "mubin",
    "warp": 32,
    "tile": 1024,
    "max_warps": 4,
}


def identify():
    from flagcv_codegen.triton_build import digest

    return digest(__file__, PROVIDER)


def kernel_config(request):
    tuning, num_warps, grid = compile_with_plan(request, FACTS)
    return tuning, num_warps, grid


def qualify(target, version, compiler_module):
    raise_unqualified(PROVIDER, target, version, compiler_module)
