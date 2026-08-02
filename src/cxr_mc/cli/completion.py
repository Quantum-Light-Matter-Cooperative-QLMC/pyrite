"""Compatibility alias for :mod:`cxr_mc.cli.commands.completion`."""

import sys

from .commands import completion as _implementation

sys.modules[__name__] = _implementation
