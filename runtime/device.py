"""Resolve GEMS_VENDOR or detect the installed accelerator runtime."""

import importlib
import os
import shlex
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass

# Same vendor/device names used by FlagGems runtime backend metadata.
VENDORS = {
    "nvidia": ("cuda", "nvidia-smi"),
    "amd": ("cuda", "rocm-smi"),
    "ascend": ("npu", "npu-smi info"),
    "mthreads": ("musa", "mthreads-gmi"),
    "cambricon": ("mlu", "cnmon"),
    "enflame": ("gcu", "efsmi -L"),
    "hygon": ("cuda", "hy-smi"),
    "iluvatar": ("cuda", "ixsmi"),
    "kunlunxin": ("cuda", "xpu-smi"),
    "metax": ("cuda", "mx-smi"),
    "thead": ("cuda", "ppu-smi"),
}
TORCH_ATTRIBUTES = {
    "ascend": "npu",
    "cambricon": "mlu",
    "mthreads": "musa",
    "enflame": "gcu",
    "hygon": "__hcu_version__",
    "iluvatar": "corex",
}


@dataclass(frozen=True)
class RuntimeDevice:
    vendor: str
    device: str


def _torch_runtime():
    torch = importlib.import_module("torch")
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pass
    return torch


def _detect_vendor(torch):
    if "PPU_SDK" in os.environ:
        return "thead"
    for vendor, attribute in TORCH_ATTRIBUTES.items():
        if hasattr(torch, attribute):
            return vendor
    if getattr(torch.version, "hip", None):
        return "amd"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_properties(0).name.upper()
        if "NVIDIA" in name:
            return "nvidia"
    for vendor, (_, command) in VENDORS.items():
        try:
            result = subprocess.run(
                shlex.split(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return vendor
    raise RuntimeError(
        "No accelerator detected; set GEMS_VENDOR or --device cpu explicitly"
    )


def _hip_vendor(torch, device):
    if not (getattr(torch.version, "hip", None) or hasattr(torch, "__hcu_version__")):
        raise ValueError("amd and hygon require a HIP Torch runtime")
    index = torch.device(device or "cuda").index
    # The active Triton target is also checked for the selected device, not
    # necessarily the caller's current CUDA logical index.
    triton = importlib.import_module("triton")
    with torch.cuda.device(index) if index is not None else nullcontext():
        target = triton.runtime.driver.active.get_current_target()
    if target.backend == "hip":
        if target.arch == "gfx1100" and target.warp_size == 32:
            return "amd"
        if target.arch in ("gfx936", "gfx938") and target.warp_size == 64:
            return "hygon"
    raise ValueError(f"unsupported HIP device target {target!r}")


def resolve_device(device=None):
    """Return one vendor/device pair for tests, timing and profiler hooks.

    An explicit device preserves its index. GEMS_VENDOR must be a known
    vendor and must agree with an explicit accelerator device type.
    """
    vendor = os.environ.get("GEMS_VENDOR")
    if vendor is not None:
        vendor = vendor.lower()
        if vendor == "hip":
            raise ValueError("GEMS_VENDOR=hip was removed; select amd or hygon")
        if vendor not in VENDORS:
            raise ValueError(
                f"Unknown GEMS_VENDOR {vendor!r}; choose from {', '.join(VENDORS)}"
            )
    device_type = str(device).split(":", 1)[0] if device is not None else None
    if device_type == "cpu":
        return RuntimeDevice("cpu", str(device))
    if vendor is None and device_type in ("npu", "musa", "mlu", "gcu"):
        vendor = {
            "npu": "ascend",
            "musa": "mthreads",
            "mlu": "cambricon",
            "gcu": "enflame",
        }[device_type]
    if vendor is None:
        vendor = _detect_vendor(_torch_runtime())
    if vendor == "ascend":
        importlib.import_module("torch_npu")
    expected = VENDORS[vendor][0]
    if device_type is not None and device_type != expected:
        raise ValueError(
            f"GEMS_VENDOR {vendor!r} uses {expected!r}, incompatible with --device {device!r}"
        )
    if vendor == "nvidia" and os.environ.get("GEMS_VENDOR") is not None:
        torch = _torch_runtime()
        if getattr(getattr(torch, "version", None), "hip", None) or hasattr(torch, "__hcu_version__"):
            raise ValueError("GEMS_VENDOR='nvidia' does not match the installed HIP Torch runtime")
    if vendor in ("amd", "hygon"):
        actual = _hip_vendor(_torch_runtime(), device)
        if actual != vendor:
            if os.environ.get("GEMS_VENDOR") is not None:
                raise ValueError(f"GEMS_VENDOR {vendor!r} does not match the {actual!r} HIP device")
            vendor = actual
    return RuntimeDevice(vendor, str(device) if device is not None else expected)
