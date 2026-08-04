"""Compatibility alias for :mod:`cxr_mc.cli.commands.sweep`."""

import sys

from .commands import sweep as _implementation

sys.modules[__name__] = _implementation
