"""Unit construction for PyRITE's optional Pint integration.

Install the ``units`` extra before importing this module::

    pip install 'pyrite-mc[units]'

Use the shared registry to construct quantities. Public configuration objects
normalize accepted quantities to their existing numeric field units before
case lowering; quantities are not stored in simulation kernels or checkpoints.
"""

from __future__ import annotations

try:
    import pint
except ImportError as exc:  # pragma: no cover - exercised without the optional extra
    raise ImportError(
        "pyrite.units requires the optional Pint dependency; install it with "
        "`pip install 'pyrite-mc[units]'`"
    ) from exc

ureg = pint.UnitRegistry(cache_folder=None)
"""The one project unit registry. Do not mutate its definitions after import."""

Q_ = ureg.Quantity
"""Convenience constructor bound to :data:`ureg`."""

__all__ = ["Q_", "ureg"]
