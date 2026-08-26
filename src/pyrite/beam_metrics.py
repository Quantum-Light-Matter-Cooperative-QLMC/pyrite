"""Compatibility re-export for :mod:`pyrite.campaign.beam_metrics`."""

from ._module_deprecations import warn_module_deprecation
from .campaign.beam_metrics import *  # noqa: F401,F403

warn_module_deprecation(__name__)
