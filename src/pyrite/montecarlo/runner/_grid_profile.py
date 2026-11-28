"""Opt-in line-grid memory and stage profiling."""

import sys

from ..._backend import BACKEND


def _resident_mib():
    """Return current Linux RSS for opt-in line-grid profiling."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _device_mib():
    """Pool-used and driver-used device MiB for opt-in profiling, if a GPU."""
    stats = BACKEND.allocator_stats()
    out = {
        "device_used_mib": stats.get("used_mib"),
        "device_reserved_mib": stats.get("reserved_mib"),
    }
    runtime = getattr(getattr(getattr(BACKEND, "cp", None), "cuda", None), "runtime", None)
    if runtime is not None:
        free, total = runtime.memGetInfo()
        out["device_driver_used_mib"] = (total - free) / (1 << 20)
    return out


def _profile_stage(profile, **fields):
    """Record opt-in line-grid stage fields and flush them to stderr.

    Immediate output survives a scheduler time limit that discards the report.
    Device occupancy is sampled at every stage.
    """
    fields.update(_device_mib())
    profile.update(fields)
    items = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"line-grid profile: {items}", file=sys.stderr, flush=True)
