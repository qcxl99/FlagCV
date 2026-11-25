# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the explicit kernel-argument ABI before publishing a launchable image."""

import re


def require_qualified_compiler(backend, target, version, arch=None, compiler_module=None):
    # Triton v3.6.0 NVIDIA's launcher appends global and profile scratch
    # pointers, in that order. Qualify vendor launchers separately.
    hip_compilers = {"gfx936": "triton.backends.hcu.compiler_hcu",
                     "gfx938": "triton.backends.hcu.compiler_hcu",
                     "gfx1100": "triton.backends.amd.compiler"}
    if (backend in ("hip", "hygon") and target == "hip" and version == "3.6.0"
            and compiler_module is not None and hip_compilers.get(arch) == compiler_module):
        return
    if (backend, target, version) != ("nvidia", "cuda", "3.6.0"):
        raise RuntimeError(
            f"scratch ABI is unqualified for {backend}/{target}:{arch} with {compiler_module} (Triton {version})"
        )


def validated_scratch_args(metadata, ptx, signature):
    for field in ("global_scratch_size", "profile_scratch_size"):
        size = getattr(metadata, field, None)
        if type(size) is not int or size < 0:
            raise RuntimeError(f"missing or invalid scratch metadata: {field}")
        if size:
            raise RuntimeError(f"{metadata.name} needs {field}={size}, which is unsupported")

    # The PTX and cubin come from the same compilation. Cross-check the
    # realized entry layout, including pointer attributes and scalar widths.
    entry = re.search(r"\.entry\s+" + re.escape(metadata.name) + r"\s*\(([^)]*)\)", ptx or "")
    expected = []
    for kind in signature.values():
        if kind.startswith("*"):
            expected.append(("u64", True))
        elif kind in ("i32", "i64"):
            expected.append(("u" + kind[1:], False))
        elif kind == "fp32":
            expected.append(("f32", False))
        elif kind != "constexpr":
            raise RuntimeError(f"unsupported CUDA kernel launch argument type: {kind}")
    expected.extend([("u64", True), ("u64", True)])
    params = entry.group(1).split(",") if entry else []
    actual = []
    for index, param in enumerate(params):
        match = re.fullmatch(
            r"\s*\.param\s+\.(u32|u64|f32)(\s+\.ptr\s+\.global\s+\.align\s+\d+)?\s+"
            + re.escape(f"{metadata.name}_param_{index}") + r"\s*", param
        )
        if match:
            actual.append((match[1], bool(match[2])))
    if len(actual) != len(params) or actual != expected:
        raise RuntimeError(f"{metadata.name} has an unexpected CUDA entry parameter ABI")
    return 2


def validated_hip_scratch_args(metadata, llir, signature):
    # These HIP launchers send two pointer slots and explicitly zero the
    # global one. The compiler may omit global_scratch_size. In that case,
    # accept it only with IR proof that the tail arguments are not used.
    profile = getattr(metadata, "profile_scratch_size", None)
    global_size = getattr(metadata, "global_scratch_size", None)
    explicit_global = hasattr(metadata, "global_scratch_size")
    if (type(profile) is not int or profile != 0
            or (explicit_global and (type(global_size) is not int or global_size != 0))):
        raise RuntimeError("HIP kernel needs scratch storage or has invalid scratch metadata")
    entry = re.search(r"^define.*?amdgpu_kernel void @" + re.escape(metadata.name)
                      + r"\((.*)\)[^{\n]*\{", llir or "", re.MULTILINE)
    if not entry:
        raise RuntimeError("missing compiled HIP kernel entry ABI")
    params = [arg.strip() for arg in entry[1].split(",")]
    expected = ["ptr addrspace(1)" if kind.startswith("*") else ("float" if kind == "fp32" else kind)
                for kind in signature.values() if kind != "constexpr"]
    expected.extend(["ptr addrspace(1)", "ptr addrspace(1)"])
    if (len(params) != len(expected) or any(not arg.startswith(kind + " ")
            for arg, kind in zip(params, expected))
            or any(not re.search(r"\breadnone\b", arg) for arg in params[-2:])):
        raise RuntimeError("unexpected HIP entry ABI or active scratch pointer")
    return 2
