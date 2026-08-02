"""Compatibility alias for :mod:`cxr_mc.cli.commands.profile`."""

import sys

from .commands import profile as _implementation

sys.modules[__name__] = _implementation
