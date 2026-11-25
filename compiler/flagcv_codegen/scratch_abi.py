# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the explicit kernel-argument ABI before publishing a launchable image."""

import re


def require_qualified_compiler(backend, target, version):
    # Triton v3.6.0 NVIDIA's launcher appends global and profile scratch
    # pointers, in that order. Qualify vendor launchers separately.
    if (backend, target, version) != ("nvidia", "cuda", "3.6.0"):
        raise RuntimeError(
            f"scratch ABI is unqualified for {backend}/{target} with Triton {version}"
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
        elif kind != "constexpr":
            raise RuntimeError(f"unsupported CUDA kernel launch argument type: {kind}")
    expected.extend([("u64", True), ("u64", True)])
    params = entry.group(1).split(",") if entry else []
    actual = []
    for index, param in enumerate(params):
        match = re.fullmatch(
            r"\s*\.param\s+\.(u32|u64)(\s+\.ptr\s+\.global\s+\.align\s+\d+)?\s+"
            + re.escape(f"{metadata.name}_param_{index}") + r"\s*", param
        )
        if match:
            actual.append((match[1], bool(match[2])))
    if len(actual) != len(params) or actual != expected:
        raise RuntimeError(f"{metadata.name} has an unexpected CUDA entry parameter ABI")
    return 2
