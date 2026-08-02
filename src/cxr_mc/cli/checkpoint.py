"""Compatibility alias for :mod:`cxr_mc.cli.commands.checkpoint`."""

import sys

from .commands import checkpoint as _implementation

sys.modules[__name__] = _implementation
