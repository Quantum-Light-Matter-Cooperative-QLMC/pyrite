"""Compatibility alias for :mod:`cxr_mc.cli.commands.app`."""

import sys

from .commands import app as _implementation

sys.modules[__name__] = _implementation
