"""Public ``Numerics.line_resolution`` vocabulary and validation (#1214).

Kept apart from ``_cxr_grid_policy`` so ``Numerics`` stays import-light; a test
pins :data:`LINE_RESOLUTIONS` to that module's uniform resolution policies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._energy_bins import EnergyBins

__all__ = ["LINE_RESOLUTIONS", "validate_line_resolution"]

#: Automatic line-axis steps ``Numerics.line_resolution`` may select.
LINE_RESOLUTIONS = ("sinc-oversampled", "sinc-nyquist")


def validate_line_resolution(value: object, energy_bins: EnergyBins | None) -> None:
    """Refuse an unknown step, or one paired with an explicit line grid."""
    if value is None:
        return
    if value not in LINE_RESOLUTIONS:
        raise ValueError(
            f"line_resolution must be one of {list(LINE_RESOLUTIONS)} or None, got {value!r}"
        )
    if energy_bins is not None and energy_bins.cxr is not None:
        raise ValueError(
            "line_resolution selects the automatic line axis; it cannot be combined "
            "with an explicit energy_bins.cxr grid"
        )
