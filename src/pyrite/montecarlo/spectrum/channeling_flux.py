"""Per-row channeling flux weights read by the incoherent scorers (#1148).

Planar and axial channeling transport attach ``channeling_flux`` ``(n_rows, n_keys)``
and ``channeling_flux_keys`` to its rows (:mod:`..channeling.flux_weights`):
the relative yield of a local incoherent process on each row, ``1`` on native
rows. Rows without the key (every non-channeling case) are unweighted, so their
scoring is bit-for-bit unchanged.

Validation: cr-zone-incoherent-flux-weight
"""

from __future__ import annotations

from typing import Any

from ..._backend import xp


def flux_columns(segments, keys) -> Any | None:
    """Weights of ``keys`` on every row, ``(n_rows, len(keys))``; ``None`` if unweighted.

    A key the transport did not weight gets a column of ones. Returns ``None``
    when the rows carry no weights or none of ``keys`` is weighted.

    Validation: cr-zone-incoherent-flux-weight
    """
    weights = segments.get("channeling_flux")
    names = segments.get("channeling_flux_keys")
    if weights is None or names is None:
        return None
    index = {name: column for column, name in enumerate(names)}
    if not any(key in index for key in keys):
        return None
    weights = xp.asarray(weights)
    ones = xp.ones(int(weights.shape[0]), dtype=weights.dtype)
    return xp.stack([weights[:, index[key]] if key in index else ones for key in keys], axis=1)


def flux_scaled_length(segments, seg_L, rows):
    """Scored path lengths ``seg_L`` times the rows' brem weight (#1148).

    ``rows`` maps each scored piece to its segment row. ``seg_L`` is returned
    unchanged (the same object) when the rows carry no brem weight.

    Validation: cr-zone-incoherent-flux-weight
    """
    flux = flux_columns(segments, ("brem",))
    if flux is None:
        return seg_L
    return seg_L * xp.asarray(flux[:, 0], dtype=seg_L.dtype)[rows]
