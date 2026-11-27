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
        if not torch.version.cuda or torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("FlagCV OpenCV CUDA recipe requires CUDA Torch and an NVIDIA device")

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
        snapshots = tuple(source.clone() for source in args[:dst_index])
        got = get_reference(op, dst.device)(*args[:dst_index], expected, *args[dst_index + 1:])
        prepared = prepare(op, args)
        try:
            compare = (assert_bitwise_equal if op == "flip" else
                       assert_exact if dtype == torch.uint8 else assert_relative_close)
            if dtype == torch.uint8:
                dst.copy_(got ^ 0x80)
            else:
                dst.view(torch.int32).copy_(~got.view(torch.int32))
            result = prepared()
            if result is not dst:
                raise RuntimeError("OpenCV CUDA bridge did not return the preallocated output")
            compare(result, got)
            for source, before in zip(args[:dst_index], snapshots):
                assert_bitwise_equal(source, before)
        except BaseException:
            prepared.close()
            raise
        return (prepared,), {}

    def config(binding, dtype):
        record = cases[binding].get(dtype)
        return record[1]["config"] if record else None

    def resource_descriptor(prepared):
        tensors = {value.data_ptr(): value for value in prepared.args if isinstance(value, torch.Tensor)}
        return (sum(value.numel() * value.element_size() for value in tensors.values()),
                {"baseline_source": "opencv_cuda_live", "baseline_version": prepared.version})

    return SimpleNamespace(OP_NAME=op, DTYPES=tuple(dtypes), IS_INPLACE=False,
                           native=native, grid=grid, build_inputs=build_inputs,
                           key_shape=lambda binding: binding, config=config,
                           resource_descriptor=resource_descriptor)
