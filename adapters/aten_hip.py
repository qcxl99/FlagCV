"""Qualified, preallocated ATen HIP comparators for Python timing."""

_BITWISE_ARITY = {"bitwise_and": 3, "bitwise_not": 2}
_VERIFIED_SHAPES = {(64, 64), (61, 67), (224, 224), (257, 509), (480, 640),
                    (720, 1280), (1080, 1920)}


def _qualified_platform(vendor, torch, device):
    if (vendor != "hygon" or device.type != "cuda" or torch.__version__ != "2.9.0"
            or torch.version.hip != "6.3.26045"
            or torch.version.git_version != "275d08c233b2cf705e4b66d9f98f7590daae5cba"):
        return False
    props = torch.cuda.get_device_properties(device)
    return (props.name == "BW" and getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx936"
            and props.multi_processor_count == 80)


def qualification(op, args, dtype, vendor, candidate_source):
    if op in ("add", "subtract", "multiply"):
        if dtype == "torch.uint8":
            return "not_equivalent", "plain ATen uint8 arithmetic wraps; FlagCV saturates"
        return "pending", "ATen arithmetic out= discovered; semantics and execution scope not qualified"
    if op == "flip":
        return "not_equivalent", "torch.flip allocates its result; no equivalent preallocated adapter"
    if op not in _BITWISE_ARITY:
        return "unavailable", "no identified HIP comparator"
    if dtype != "torch.uint8" or len(args) != _BITWISE_ARITY[op]:
        return "pending", "ATen bitwise out= not qualified for this dtype or operand list"
    if candidate_source != "builtin":
        return "pending", "ATen HIP comparator qualified only against the builtin kernel scope"

    import torch

    dst = args[-1]
    shape = tuple(dst.shape)
    if (shape not in _VERIFIED_SHAPES or any(
            tensor.dtype != torch.uint8 or tensor.device != dst.device
            or tuple(tensor.shape) != shape or tensor.stride() != (shape[1], 1)
            or tensor.storage_offset() != 0 or tensor.data_ptr() % 16
            for tensor in args)
            or len({tensor.untyped_storage().data_ptr() for tensor in args}) != len(args)):
        return "pending", "ATen bitwise out= not qualified for this shape, layout or aliasing"
    if not _qualified_platform(vendor, torch, dst.device):
        return "pending", "ATen bitwise out= not qualified for this device and PyTorch/HIP build"
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

    function = {"bitwise_and": torch.bitwise_and, "bitwise_not": torch.bitwise_not}[op]
    version = f"PyTorch {torch.__version__} ({torch.version.git_version}), HIP {torch.version.hip}"
    return _AtenOut(function, version), (*args[:-1], torch.empty_like(args[-1]))
