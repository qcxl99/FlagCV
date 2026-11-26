"""Internal Torch operand adapter for the production Triton compile path."""

import json
import operator
import os
import struct
import sys
from functools import lru_cache
from math import prod
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "compiler"))

from flagcv_codegen.main import load_provider  # noqa: E402
from flagcv_codegen.triton_build import compile_jit, digest  # noqa: E402

_MODES = {"add": 0, "subtract": 1, "multiply": 2, "bitwise_and": 3, "bitwise_not": 4}
_DEPTHS = {
    "uint8": "8U", "int8": "8S", "uint16": "16U", "int16": "16S", "int32": "32S",
    "float16": "16F", "float32": "32F", "float64": "64F",
}


@lru_cache(maxsize=None)
def _entry(root, vendor, kernel):
    for directory in (root / "backends" / vendor / "kernels", root / "kernels"):
        manifest = directory / "registry.json"
        if manifest.is_file():
            for entry in json.loads(manifest.read_text())["kernels"]:
                if entry["name"] == kernel:
                    return {
                        "source": str((directory / entry["source"]).resolve()),
                        "function": entry["function"],
                        "plan": entry["plan"],
                        "from_platform_registry": directory != root / "kernels",
                    }
    raise ValueError(f"no kernel registered for {kernel!r} on {vendor!r}")


@lru_cache(maxsize=None)
def _specialize(root, device, encoded):
    request = json.loads(encoded)
    provider = load_provider(request["backend"], root)
    compiled, scratch, grid = compile_jit(request, provider)
    identity = _identity(root, request["backend"])
    return compiled, grid, {
        "kernel": request["kernel"],
        "function": compiled.name,
        "plan": request["plan"],
        "target": request["target"],
        "compiler_identity": identity,
        "artifact_hash": compiled.hash,
        "signature": compiled.src.signature,
        "constexprs": {compiled.src.fn.arg_names[key[0]]: value
                       for key, value in compiled.src.constants.items()},
        "divisibility": [compiled.src.fn.arg_names[key[0]] for key in compiled.src.attrs],
        "num_warps": compiled.metadata.num_warps,
        "num_stages": compiled.metadata.num_stages,
        "scratch_args": scratch,
    }


@lru_cache(maxsize=None)
def _identity(root, vendor):
    return digest(load_provider(vendor, root).__file__, vendor, root)


def _params(op, args):
    # Mirror src/lowering's parameter order and specialization for these
    # internal Torch views. The compiler checks the selected entry's signature.
    unary = op in ("flip", "bitwise_not")
    dst = args[1 if unary else 2]
    src = args[0]
    a, b = (src, src if unary else args[1])
    extra = args[2 if unary else 3:]
    height, width = src.shape[:2]
    channels = src.shape[2] if src.ndim == 3 else 1
    runtime = {"width": width, "height": height}
    hinted = {"width"}

    if op == "flip":
        code = operator.index(extra[0])
        images = {"src": src, "dst": dst}
        constants = {
            "CHANNELS": channels,
            "FLIP_X": int(code != 0),
            "FLIP_Y": int(code <= 0),
            "IN_PLACE": int(src.data_ptr() == dst.data_ptr() and src.stride(0) == dst.stride(0)),
        }
        mask = None
    else:
        mask = extra[0] if extra and op != "multiply" else None
        images = {"src1": a, "src2": b, "dst": dst}
        constants = {"CHANNELS": channels}
        if op != "multiply":
            constants["MODE"] = _MODES[op]

    for name, tensor in images.items():
        param = f"{name}_row_stride"
        runtime[param] = tensor.stride(0)
        hinted.add(param)
    if mask is not None:
        images["mask"] = mask
        runtime["mask_row_stride"] = mask.stride(0)
    if op == "multiply":
        scale = extra[0] if extra else 1.0
        runtime["scale_bits"] = struct.unpack("=i", struct.pack("=f", scale))[0]
    if not unary or op == "bitwise_not":
        if mask is None:
            constants["PACKED"] = int(all(t.stride(0) == width * channels for t in images.values()))
    if mask is None and op in ("add", "subtract", "bitwise_and", "bitwise_not", "flip"):
        constants["VEC4"] = int(
            op in ("bitwise_and", "bitwise_not", "flip") and channels == 4
            and all(str(t.dtype) == "torch.uint8" and t.data_ptr() % 4 == 0
                    and t.stride(0) % 4 == 0 for t in images.values())
        )

    image_specs = [
        {"name": name, "depth": _DEPTHS[str(t.dtype).split(".")[-1]],
         "channels": 1 if name == "mask" else channels,
         "aligned16": t.data_ptr() % 16 == 0}
        for name, t in images.items()
    ]
    params = [
        {"name": name, "type": "i32" if -(1 << 31) <= value < (1 << 31) else "i64",
         "div16": name in hinted and value % 16 == 0}
        for name, value in runtime.items()
    ]
    params.extend({"name": name, "type": "constexpr", "value": value}
                  for name, value in constants.items())
    return dst, images, image_specs, runtime, params


def _prepare(op, vendor, args):
    from triton.runtime import driver

    dst, images, specs, runtime, params = _params(op, args)
    kernel = f"{op}_masked" if "mask" in images else op
    root = Path(os.environ.get("FLAGCV_DATA_DIR", _ROOT)).resolve()
    target = driver.active.get_current_target()
    target_name = f"{target.backend}:{target.arch}:{target.warp_size}"
    request = dict(backend=vendor, target=target_name, kernel=kernel, images=specs,
                   params=params, **_entry(root, vendor, kernel))
    compiled, grid_rule, metadata = _specialize(root, str(dst.device), json.dumps(request))
    values = {**runtime, **metadata["constexprs"]}
    grid = []
    for extent in grid_rule:
        if isinstance(extent, dict):
            count = prod(values[param] for param in extent["product"])
            block = extent["block"]
            grid.append(min((count + block - 1) // block, extent.get("limit", count)))
        else:
            grid.append(extent)
    launch = compiled[tuple(grid)]
    arguments = (*images.values(), *(values[name] for name in compiled.src.fn.arg_names[len(images):]))

    def run(*_inputs):
        launch(*arguments)
        return dst

    run.dispatch = {**metadata, "grid": grid}
    return run


def candidate_for(op, vendor):
    def prepare(*args):
        import torch

        device = args[0].device
        with torch.get_device_module(device).device(device):
            return _prepare(op, vendor, args)

    def run(*args):
        import torch

        device = args[0].device
        with torch.get_device_module(device).device(device):
            return _prepare(op, vendor, args)()

    run.prepare = prepare
    return run
