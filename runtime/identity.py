"""Match a scheduler chip label to the observed device, not to a vendor family."""

import os


def is_bw1000(props):
    return (props.get("name") in ("BW", "BW1000") and props.get("arch") == "gfx936"
            and props.get("compute_units") == 80)


def cuda_chip(name):
    if isinstance(name, str) and name.startswith("NVIDIA "):
        return name.removeprefix("NVIDIA ")
    return None


def identify(vendor, props, configured):
    chip = None
    if vendor == "hygon" and is_bw1000(props):
        chip = "BW1000"
    elif vendor == "nvidia":
        chip = cuda_chip(props.get("name"))
    if not chip:
        basis = "observed device does not establish a supported chip identity"
    elif configured and configured != chip:
        basis = f"scheduled chip {configured} differs from observed {chip}"
        chip = None
    else:
        basis = "runtime properties and scheduler label" if configured else "runtime properties"
    return {"vendor": vendor, "chip": chip, "identity_basis": basis,
            "scheduled_chip": configured, **props}


def observe(vendor, torch, device):
    props = torch.cuda.get_device_properties(device)
    version = torch.version
    return identify(vendor, {
        "name": props.name,
        "arch": getattr(props, "gcnArchName", "").split(":", 1)[0] or None,
        "compute_units": getattr(props, "multi_processor_count", None),
        "memory_bytes": getattr(props, "total_memory", None),
        "l2_bytes": getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None),
        "uuid": str(props.uuid) if getattr(props, "uuid", None) else None,
        "pci_domain": getattr(props, "pci_domain_id", None),
        "pci_bus": getattr(props, "pci_bus_id", None),
        "pci_device": getattr(props, "pci_device_id", None),
        "torch": str(torch.__version__), "torch_git": getattr(version, "git_version", None),
        "hip": getattr(version, "hip", None),
    }, os.environ.get("OPTEST_TARGET_CHIP"))
