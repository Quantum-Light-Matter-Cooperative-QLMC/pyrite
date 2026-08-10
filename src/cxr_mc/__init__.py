"""cxr_mc -- coherent X-ray radiation (PXR + coherent bremsstrahlung) from
table-top electron beams in crystals.

The package is intentionally light at import time: ``import cxr_mc`` pulls in
no heavy dependencies (matplotlib / cupy / the MC pipeline). Import submodules
explicitly, e.g. ``from cxr_mc import crystallography`` or
``from cxr_mc.montecarlo import run_cases``.

See the README for the scientific overview and CLAUDE.md for working conventions.
"""

import logging

from ._compat import env_value
from .paths import data_dir

__version__ = "0.2.0"

# Packaged data (materials.toml, cifs/, mott_transport_cross_sections/,
# eaglexo_qe.csv, legacy atomic_scattering_factors/). Resolved relative to this
# file so it works installed (wheel) or from a source checkout.
DATA_DIR = data_dir()

# Package logger. Library convention: attach a NullHandler so a plain `import
# cxr_mc` (and every `cxr` CLI invocation, incl. each ProcessPoolExecutor
# worker in montecarlo.runner) stays silent -- submodules log routine
# noise (GPU/CPU backend probe, per-element Mott-table fallback) at DEBUG,
# which propagates nowhere by default. Set PYRITE_MC_DEBUG=1 to see it: this
# attaches our own StreamHandler at DEBUG on just the "cxr_mc" logger,
# without touching the caller's root logging config.
logger = logging.getLogger("cxr_mc")
logger.addHandler(logging.NullHandler())
if env_value("CXR_MC_DEBUG"):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

__all__ = ["DATA_DIR", "__version__"]
