"""Test provider for the real Runtime compiler client. No GPU SDK is used."""
import os
from pathlib import Path


def identify():
    return "flagcv-test-compiler"


def compile(request):
    trace = Path(os.environ["FLAGCV_TEST_COMPILER_TRACE"])
    function = request["function"]
    selection = "platform" if request["from_platform_registry"] else "common"
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(f"{request['kernel']} {selection} {function}\n")
    width = "width"
    if request["kernel"] == "warp_affine_nearest":
        expected = ["M00", "M01", "M02", "M10", "M11", "M12", "bv0", "bv1", "bv2", "bv3"]
        actual = [param["name"] for param in request["params"] if param["type"] == "fp32"]
        if actual != expected:
            raise ValueError(f"unexpected fp32 arguments: {actual}")
        width = "dst_w"
    if request["kernel"] in ("copy_make_border", "threshold_apply", "threshold_histogram"):
        width = "row_values"
    elif request["kernel"] == "threshold_clear":
        width = "bins"
    elif request["kernel"] == "threshold_select":
        width = "pixels"
    artifact = trace.with_suffix(".bin")
    artifact.write_bytes(b"fixture")
    return {"artifact": str(artifact), "function": function, "num_warps": 1, "shared": 0, "scratch_args": 0,
            "grid": [{"product": [width], "block": 16}, 1, 1]}
