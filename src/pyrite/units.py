"""Unit construction through PyRITE's shared Pint registry.

Use the shared registry to construct quantities. Public configuration objects
normalize accepted quantities to their existing numeric field units before
case lowering; quantities are not stored in simulation kernels or checkpoints.
"""

from __future__ import annotations

import pint

ureg = pint.UnitRegistry(cache_folder=None)
"""The one project unit registry. Do not mutate its definitions after import."""

Q_ = ureg.Quantity
"""Convenience constructor bound to :data:`ureg`."""

__all__ = ["Q_", "ureg"]
