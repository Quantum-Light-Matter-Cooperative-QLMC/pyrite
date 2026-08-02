"""Compatibility alias for :mod:`cxr_mc.cli.commands.backend_setup`."""

import sys

from .commands import backend_setup as _implementation

sys.modules[__name__] = _implementation
