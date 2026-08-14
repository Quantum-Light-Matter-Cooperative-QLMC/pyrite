"""pyrite -- coherent X-ray radiation (PXR + coherent bremsstrahlung) from
table-top electron beams in crystals.

The package is intentionally light at import time: ``import pyrite`` pulls in
no heavy dependencies (matplotlib / cupy / the MC pipeline). Import submodules
explicitly, e.g. ``from pyrite import crystallography`` or
``from pyrite.montecarlo import run_cases``.

See the README for the scientific overview and CLAUDE.md for working conventions.
"""

import logging
from typing import Any

from ._compat import env_value
from .paths import data_dir

__version__ = "0.2.0"

# Packaged data (materials.toml, cifs/, mott_transport_cross_sections/,
# eaglexo_qe.csv, legacy atomic_scattering_factors/). Resolved relative to this
# file so it works installed (wheel) or from a source checkout.
DATA_DIR = data_dir()

# Package logger. Library convention: attach a NullHandler so a plain `import
# pyrite` (and every `cxr` CLI invocation, incl. each ProcessPoolExecutor
# worker in montecarlo.runner) stays silent -- submodules log routine
# noise (GPU/CPU backend probe, per-element Mott-table fallback) at DEBUG,
# which propagates nowhere by default. Set PYRITE_MC_DEBUG=1 to see it: this
# attaches our own StreamHandler at DEBUG on just the "pyrite" logger,
# without touching the caller's root logging config.
logger = logging.getLogger("pyrite")
logger.addHandler(logging.NullHandler())
if env_value("CXR_MC_DEBUG"):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

_PUBLIC_OBJECTS = {
    "Analysis": ("pyrite.campaign.model", "Analysis"),
    "Beam": ("pyrite.campaign.model", "Beam"),
    "BlazedGrooves": ("pyrite.campaign.geometry", "BlazedGrooves"),
    "Convergence": ("pyrite.campaign.model", "Convergence"),
    "Detector": ("pyrite.detectors", "Detector"),
    "EnergyBins": ("pyrite.detectors", "EnergyBins"),
    "Footprint": ("pyrite.campaign.geometry", "Footprint"),
    "Layer": ("pyrite.campaign.geometry", "Layer"),
    "Numerics": ("pyrite.campaign.model", "Numerics"),
    "Result": ("pyrite.results.model", "Result"),
    "Scene": ("pyrite.campaign.model", "Scene"),
    "Slab": ("pyrite.campaign.geometry", "Slab"),
    "Stack": ("pyrite.campaign.geometry", "Stack"),
    "Sweep": ("pyrite.campaign.model", "Sweep"),
    "simulate": ("pyrite.api", "simulate"),
}


def __getattr__(name: str) -> Any:
    """Resolve the public simulation surface lazily."""
    try:
        module_name, attribute = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = ["DATA_DIR", "__version__", *_PUBLIC_OBJECTS]
