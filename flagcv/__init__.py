"""Source-checkout metadata for the Optest scheduler, not a C++ binding."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flagcv")
except PackageNotFoundError:
    __version__ = "0.1.0"


def __getattr__(name):
    if name not in ("vendor_name", "device"):
        raise AttributeError(name)
    try:
        from ._runtime import resolve_device
    except ModuleNotFoundError as error:
        if error.name != "flagcv._runtime":
            raise
        from runtime import resolve_device
    selected = resolve_device()
    return selected.vendor if name == "vendor_name" else selected.device
