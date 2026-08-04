"""Compatibility alias for :mod:`cxr_mc.cli.commands.material`."""

import sys

from .commands import material as _implementation

sys.modules[__name__] = _implementation
