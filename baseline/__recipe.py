"""Build the Optest native recipe from FlagCV's Core workloads."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.opencv_cuda import prepare  # noqa: E402
from helpers.tolerance import assert_bitwise_equal, assert_exact, assert_relative_close  # noqa: E402
from reference import get_reference  # noqa: E402
from workloads.core import python_cases, shape_key  # noqa: E402


def recipe(workload):
    op = workload.OP_NAME
    cases = {}
    dtypes = {}
    for binding, row in python_cases(op):
        key = shape_key(row["config"])
        dtype = getattr(torch, row["config"]["dtype"])
        cases.setdefault(key, {})[dtype] = (binding, row)
        dtypes.setdefault(dtype, None)

    def native():
        if not torch.cuda.is_available() or torch.version.hip or not torch.version.cuda:
            return None

        def run(prepared):
            return prepared()

        return run

    def grid():
        return list(cases)

    def build_inputs(binding, dtype, device):
        data, _ = cases[binding][dtype]
        args = workload.make_inputs(data, device)
        dst_index = 1 if op in ("flip", "bitwise_not") else 2
        dst = args[dst_index]
        expected = torch.empty_like(dst)
        got = get_reference(op, dst.device)(*args[:dst_index], expected, *args[dst_index + 1:])
        prepared = prepare(op, args)
        try:
            compare = (assert_bitwise_equal if op == "flip" else
                       assert_exact if dtype == torch.uint8 else assert_relative_close)
            if dtype == torch.uint8:
                dst.copy_(got ^ 0x80)
            else:
                dst.view(torch.int32).copy_(~got.view(torch.int32))
            compare(prepared(), got)
        except BaseException:
            prepared.close()
            raise
        return (prepared,), {}

    def config(binding, dtype):
        return cases[binding][dtype][1]["config"]

    return SimpleNamespace(OP_NAME=op, DTYPES=tuple(dtypes), IS_INPLACE=False,
                           native=native, grid=grid, build_inputs=build_inputs,
                           key_shape=lambda binding: binding, config=config)
