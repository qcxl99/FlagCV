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
    artifact = trace.with_suffix(".bin")
    artifact.write_bytes(b"fixture")
    return {"artifact": str(artifact), "function": function, "num_warps": 1, "shared": 0, "scratch_args": 0,
            "grid": [{"product": ["width"], "block": 16}, 1, 1]}
