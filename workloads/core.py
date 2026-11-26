"""Core timing profiles and stable case identities for both evaluation paths."""

import argparse
import hashlib
import json
from collections import Counter
from importlib import import_module
from pathlib import Path

OPERATORS = ("add", "subtract", "multiply", "bitwise_and", "bitwise_not", "flip")
VGA = (480, 640)
FHD = (1080, 1920)
UHD = (2160, 3840)
ALIGNED_SMALL = (64, 64)
UNALIGNED_SMALL = (61, 67)
CPP_SIZES = (ALIGNED_SMALL, UNALIGNED_SMALL, VGA, FHD, UHD)
PYTHON_SIZES = (ALIGNED_SMALL, UNALIGNED_SMALL, (224, 224), (257, 509), VGA, (720, 1280), FHD)
SMALL_CORRECTNESS_SIZES = ((1, 1), (1, 257), (257, 1))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def case_id(config):
    digest = hashlib.sha256(canonical(config).encode()).hexdigest()[:24]
    return f"{config['op_name']}-{digest}"


def shape_key(config):
    # The collector adds dtype as a separate index. All other fields remain
    # in the key; equal dimensions can still have different layout or options.
    config = {name: value for name, value in config.items() if name != "dtype"}
    return case_id(config)


def python_case(op, dtype, shapes, params):
    return {
        "op_name": op, "variant": params.get("variant", f"{op}_masked" if params["mask"] else op),
        "dtype": dtype.removeprefix("torch."),
        "shapes": {name: list(shape) for name, shape in shapes.items()},
        "row_stride": params.get("row_stride", {
            name: shape[1] * params["channels"] for name, shape in shapes.items()
        }),
        "layout": params.get("layout", "packed" if params["contiguous"] else "pitched"),
        "params": params, "alias": params.get("alias"),
        "output": params.get("output", "preallocated"),
        "input_domain": params.get("input_domain", "default"),
    }


def python_cases(op):
    module = import_module(f"workloads.{op}")
    node = f"benchmark/test_{op}_benchmark.py::test_{op}_benchmark"
    for ordinal, binding in enumerate(module.TIMING_CASES):
        dtype, shapes, params = module.describe(binding)
        config = python_case(op, dtype, shapes, params)
        yield binding, {
            "case_id": case_id(config), "legacy_case_id": f"{node}::core::{dtype}::{ordinal}",
            "ordinal": ordinal, "dtype": f"torch.{dtype}", "shape": config["shapes"],
            "params": params, "config": config,
        }


def cpp_cases():
    for op in OPERATORS:
        depths = ("8U", "16U", "32F") if op in ("add", "subtract", "multiply", "flip") else ("8U",)
        configs = [(h, w, depth, 1, code, "packed")
                   for h, w in CPP_SIZES for depth in depths
                   for code in ((-1, 0, 1) if op == "flip" else (0,))]
        configs.extend((513, 1021, "8U", channels, -1 if op == "flip" else 0, "roi")
                       for channels in (3, 4))
        for height, width, depth, channels, code, layout in configs:
            roi = layout == "roi"
            # Match Image's allocation width (x offset plus trailing columns).
            margins = {"src1": 24, "src2": 15, "dst": 34}
            if op in ("flip", "bitwise_not"):
                margins = {"src1": 24, "dst": 34}
            input_domain = "u16_product_in_i32" if op == "multiply" and depth == "16U" else "default"
            params = {"channels": channels, "mask": None}
            if op == "multiply":
                params["scale"] = 1.0
            if op == "flip":
                params["flip_code"] = code
            config = {
                "op_name": op, "variant": op, "dtype": depth,
                "shapes": {name: [height, width] for name in margins},
                "row_stride": {name: (width + (margin if roi else 0)) * channels
                               for name, margin in margins.items()},
                "layout": layout, "params": params, "alias": None,
                "output": "preallocated", "input_domain": input_domain,
            }
            legacy = [op, str(height), str(width), depth, str(channels), layout, str(code), input_domain]
            yield legacy, config


def _unique(keys):
    counts = Counter(keys)
    if any(count != 1 for count in counts.values()):
        raise ValueError("duplicate Core workload")
    return set(counts)


def cpp_header():
    lines = []
    for legacy, config in cpp_cases():
        op, h, w, depth, channels, layout, code, domain = legacy
        lines.append(f'  {{Op::{op}, {h}, {w}, CV_{depth}, {channels}, {code}, '
                     f'{str(layout == "roi").lower()}, "{domain}", "{case_id(config)}"}},')
    _unique(case_id(config) for _, config in cpp_cases())
    return "constexpr Case core_cases[] = {\n" + "\n".join(lines) + "\n};\n"


def verify_legacy(path):
    old = json.loads(path.read_text())
    cpp = list(cpp_cases())
    python = [row for op in OPERATORS for _, row in python_cases(op)]
    assert _unique(canonical(case) for case in old["cpp"]) == _unique(
        canonical(legacy) for legacy, _ in cpp
    )
    assert _unique(canonical(case) for case in old["python"]) == _unique(
        canonical([row["config"]["op_name"], row["legacy_case_id"], row["dtype"], row["shape"],
                   row["params"]]) for row in python
    )
    _unique(row["case_id"] for row in python)
    return {"cpp": [{"legacy": old, "case_id": case_id(cfg), "config": cfg} for old, cfg in cpp],
            "python": python}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-header", type=Path)
    parser.add_argument("--verify-legacy", type=Path)
    args = parser.parse_args()
    if args.cpp_header:
        args.cpp_header.write_text(cpp_header())
    if args.verify_legacy:
        print(json.dumps(verify_legacy(args.verify_legacy), indent=2))


if __name__ == "__main__":
    main()
