"""Compatibility alias for :mod:`cxr_mc.cli.commands.beam`."""

import sys

from .commands import beam as _implementation

sys.modules[__name__] = _implementation
