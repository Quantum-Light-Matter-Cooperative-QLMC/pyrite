"""Compatibility alias for :mod:`cxr_mc.cli.commands.performance`."""

import sys

from .commands import performance as _implementation

sys.modules[__name__] = _implementation
