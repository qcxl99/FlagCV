"""Qualified, preallocated ATen HIP comparators for Python timing."""

from runtime.identity import observe

_OUT = {"bitwise_and": ("uint8", 3), "bitwise_not": ("uint8", 2),
        "add": ("float32", 3), "subtract": ("float32", 3), "multiply": ("float32", 3)}


def _qualified_platform(vendor, torch, device):
    return (vendor == "hygon" and device.type == "cuda" and bool(torch.version.hip)
            and observe(vendor, torch, device)["chip"] == "BW1000")


def qualification(op, args, dtype, vendor):
    if op in ("add", "subtract", "multiply") and dtype == "torch.uint8":
        return "not_equivalent", "plain ATen uint8 arithmetic wraps; FlagCV saturates"
    if op == "flip":
        return "not_equivalent", "torch.flip allocates its result; no equivalent preallocated adapter"
    if op not in _OUT:
        return "unavailable", "no identified HIP comparator"
    element, arity = _OUT[op]
    if op == "multiply" and len(args) == arity + 1 and type(args[-1]) in (int, float) and args[-1] == 1:
        args = args[:-1]
    if dtype != f"torch.{element}" or len(args) != arity:
        return "pending", "ATen out= is not qualified for this dtype, scale, or operand list"

    import torch

    dst = args[-1]
    shape = tuple(dst.shape)
    if (len(shape) != 2 or min(shape) <= 0 or any(
            tensor.dtype != getattr(torch, element) or tensor.device != dst.device
            or tuple(tensor.shape) != shape or tensor.stride() != (shape[1], 1)
            or tensor.storage_offset() != 0 or tensor.data_ptr() % 16
            for tensor in args)
            or len({tensor.untyped_storage().data_ptr() for tensor in args}) != len(args)):
        return "pending", "ATen out= is not qualified for this shape, layout or aliasing"
    if not _qualified_platform(vendor, torch, dst.device):
        return "pending", "ATen out= execution boundary has not been verified on this device"
    if not callable(getattr(torch, op, None)):
        return "pending", "ATen out= operation is absent from this build"
    return "eligible", None


class _AtenOut:
    def __init__(self, function, version):
        self.function = function
        self.version = version

    def __call__(self, *args):
        return self.function(*args[:-1], out=args[-1])

    def close(self):
        pass


def prepare(op, args):
    import torch

    args = args[:_OUT[op][1]]
    function = getattr(torch, op)
    version = f"PyTorch {torch.__version__} ({torch.version.git_version}), HIP {torch.version.hip}"
    return _AtenOut(function, version), (*args[:-1], torch.empty_like(args[-1]))
