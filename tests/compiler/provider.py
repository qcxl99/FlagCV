"""Driver-free provider used by the compiler-client contract test."""
import atexit
import os
from pathlib import Path
import time


def identify():
    return str(os.getpid())


def compile(request):
    if "kernel" in request:
        root = Path(os.environ["FLAGCV_DATA_DIR"])
        mode = (root / "scratch-mode").read_text()
        image = root / "test.bin"
        image.write_bytes(b"fixture")
        result = {"artifact": str(image), "function": "flip_kernel",
                  "num_warps": 1, "shared": 0, "grid": [1, 1, 1]}
        if mode != "missing":
            result["scratch_args"] = 2 if mode == "valid" else 1
        return result
    kind = request["kind"]
    if kind == "hang":
        Path(os.environ["FLAGCV_DATA_DIR"], "compiler.pid").write_text(str(os.getpid()))
        time.sleep(30)
    if kind == "shutdown_hang":
        atexit.register(time.sleep, 30)
    if kind == "compile_error":
        raise ValueError("deliberate compile error")
    return {"identity": str(os.getpid())}
