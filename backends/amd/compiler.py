# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler provider for amd."""

from flagcv_codegen.providers import compile_with_plan
from flagcv_codegen.scratch_abi import raise_unqualified, validated_hip_scratch_args

PROVIDER = "amd"
FACTS = {
    "triton_backend": "hip",
    "binary": "hsaco",
    "extension": "hsaco",
    "warp": 32,
    "tile": 2048,
    "max_warps": 16,
}
TARGETS = ('hip:gfx1100:32',)


def identify():
    from flagcv_codegen.triton_build import digest

    return digest(__file__, PROVIDER)


def kernel_config(request):
    if request["target"] not in TARGETS:
        raise ValueError(f"unsupported target {request['target']!r} for {PROVIDER}")
    tuning, num_warps, grid = compile_with_plan(request, FACTS)
    return tuning, num_warps, grid


def qualify(target, version, compiler_module):
    name = f"{target.backend}:{target.arch}:{target.warp_size}"
    if name not in TARGETS or version != "3.6.0" or compiler_module != "triton.backends.amd.compiler":
        raise_unqualified(PROVIDER, target, version, compiler_module)


def validate_scratch(compiled, signature):
    return validated_hip_scratch_args(compiled.metadata, compiled.asm.get("llir"), signature)
