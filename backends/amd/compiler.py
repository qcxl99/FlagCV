# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""AMD ROCm compiler provider."""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.triton_build import compile_kernel, digest

PROVIDER = "amd"


def identify():
    """Digest of the interpreter, Triton, this file and every registered kernel."""
    return digest(__file__, PROVIDER)


def compile(request):
    tuning, num_warps, grid, _ = compile_with_plan(request)
    return compile_kernel(request, tuning, num_warps, grid)
