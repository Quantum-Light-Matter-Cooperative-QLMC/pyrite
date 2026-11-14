"""Native PXR/CBS ownership on planar channeling zone rows (#1055).

A planar zone row is the straight mean segment of a line electron from the
entrance face to its handoff (ADR-0018 D4-A), tagged ``channeling_zone``.
Inside the zone the electron is in the density-matrix solver's tracked
transverse states, and the plane's own reciprocal row (reflections ``g``
parallel to the plane normal) is exactly the potential that binds it. The
expected-value source ``spec_channeling`` owns that row's emission on the
zone: bound→bound, capture and the in-cut free→free block
(``final_states="tracked"``, ``cr-planar-free-free-block``). The native line
kernel treats the same row as a free plane wave at the beam angle and would
count it again (measured 5.4× the free→free block for Si(110) at 0.8 mrad,
17 MeV, #1055).

Reflections off the own row are modulated by the same tracked states and,
near a Bragg pole, carry diffracted channeling radiation (#1166). The zone
PXR/DCR source (:mod:`.planar_zone_pxr`) owns every such reflection that it
supports towards the scored direction (two-beam Laue geometry with a
physical root), so native line scoring of a planar channeling case is the
sum of two incoherent parts (reflections are always summed incoherently):

``spec = S[unowned; all rows] + S[own row + source-owned; rows not in zone]``.

Unowned reflections (Bragg-case or no root) keep every row. The per-incident
-electron normalization ``Ne`` is unchanged. Characteristic and incoherent
bremsstrahlung scoring never pass through here.

Validation: cr-planar-zone-row-ownership, cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np


class _LineCase(Mapping[str, Any]):
    """Read-only view of a case with replaced keys for one line-scoring part."""

    def __init__(self, base: Mapping[str, Any], **replaced: Any):
        self._base, self._replaced = base, replaced

    def __getitem__(self, key: str) -> Any:
        return self._replaced[key] if key in self._replaced else self._base[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._base
        yield from (key for key in self._replaced if key not in self._base)

    def __len__(self) -> int:
        return len(self._base) + sum(key not in self._base for key in self._replaced)


def is_own_row(hkl, plane_hkl) -> bool:
    """Whether reflection ``hkl`` lies on the reciprocal row of ``plane_hkl``.

    ``g(hkl)`` is linear in the Miller indices, so ``g`` is parallel to the
    plane normal exactly when the index vectors are parallel, in any lattice.

    Validation: cr-planar-zone-row-ownership
    """
    return bool(np.all(np.cross(np.asarray(hkl, dtype=int), np.asarray(plane_hkl, dtype=int)) == 0))


def _rows(segs, keep):
    """Rows ``keep`` (host indices) of every row array the line kernel reads."""
    from ..spectrum.cxr._kernels import _SEG_ARRAYS
    from .electron_blocks import _take

    out = dict(segs)
    for key in (*_SEG_ARRAYS, "channeling_zone"):
        if out.get(key) is not None:
            out[key] = _take(out[key], keep)
    return out


def planar_zone_line_parts(case, segs, n_hat=None):
    """Split native line scoring of a planar channeling case into its parts.

    Returns ``None`` for every case that is not planar channeling, or whose
    reflections hold no zone-owned member; scoring is then unchanged.
    Otherwise returns ``[(case_view, rows), ...]``: the unowned reflections on
    all rows and the zone-owned ones on the rows outside the zone. Zone-owned
    are the own-row reflections and, towards ``n_hat`` (when given), the
    reflections of :func:`.planar_zone_pxr.source_reflections`. Each view drops
    the ``channeling`` record, so the parts are scored as ordinary cases.
    A planar transport without the ``channeling_zone`` tag, or a multilayer
    radiator stack, refuses.

    Validation: cr-planar-zone-row-ownership, cr-planar-zone-pxr-modulation
    """
    record = case.get("channeling")
    if not isinstance(record, Mapping) or record.get("model") != "planar":
        return None
    plane = tuple(int(v) for v in record["plane_hkl"])
    hkls = [tuple(int(v) for v in hkl) for hkl in case["hkl_list"]]
    owned = {hkl for hkl in hkls if is_own_row(hkl, plane)}
    if n_hat is not None:
        from .planar_zone_pxr import source_reflections

        owned |= {row[0] for row in source_reflections(case, n_hat)}
    own = [hkl for hkl in hkls if hkl in owned]
    if not own:
        return None
    if case.get("layer_radiators") is not None:
        raise ValueError("planar channeling line scoring supports one crystal slab (#1055)")
    zone = segs.get("channeling_zone")
    if zone is None:
        raise ValueError("planar channeling transport carries no zone-row tag (#1055)")
    from ..._backend import _to_cpu

    outside = np.flatnonzero(~np.asarray(_to_cpu(zone), dtype=bool))
    other = [hkl for hkl in hkls if hkl not in owned]
    parts = []
    if other:
        parts.append((_LineCase(case, hkl_list=other, channeling=None), segs))
    parts.append((_LineCase(case, hkl_list=own, channeling=None), _rows(segs, outside)))
    return parts
