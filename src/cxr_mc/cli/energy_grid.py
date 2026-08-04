"""Compatibility alias for :mod:`cxr_mc.cli.commands.energy_grid`."""

import sys

from .commands import energy_grid as _implementation

sys.modules[__name__] = _implementation
