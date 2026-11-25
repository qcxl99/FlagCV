# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HIP compiler provider shared with the legacy Hygon registration.

Kernel capability decisions live here, not in lowering: block sizes, shared
memory limits and grid extents are platform properties, so lowering stays
unchanged when a platform is added. The plan itself is shared
(flagcv_codegen.providers) and bounded by this platform's limits in
flagcv_codegen.platform; what is specific here is the identity digest and any
kernel this platform registers itself, which would be resolved through
the selected HIP backend registry first and never fall back to the common
kernel.
"""

from pathlib import Path

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.triton_build import compile_kernel, digest

PROVIDER = Path(__file__).parent.name


def identify():
    """Digest of the interpreter, Triton, this file and every registered kernel."""
    return digest(__file__, PROVIDER)


def compile(request):
    tuning, num_warps, grid, _ = compile_with_plan(request)
    # gfx936 has 64 lanes per wave. A 128-element F32 C3 correlation
    # block uses two waves; other targets and data routes keep their plan.
    if (request["target"] == "hip:gfx936:64"
            and request["kernel"] in ("filter2d_direct", "filter2d_dense")
            and request["images"][0]["depth"] == "32F"
            and request["images"][0]["channels"] == 3):
        num_warps = 2
        if request["kernel"] == "filter2d_dense":
            tuning["ROW_PAIR"] = True
            grid[0] = {"product": ["row_values", "height"],
                       "factor_blocks": [tuning["BLOCK"], 2], "block": 1}
    return compile_kernel(request, tuning, num_warps, grid)
