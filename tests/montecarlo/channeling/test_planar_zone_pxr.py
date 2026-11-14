"""Zone-row PXR/DCR source ownership on planar channeling cases (#1166).

The 17 MeV Si(1-10) fixture of ``test_planar_zone_lines`` with bin-mean
line quadrature, scored towards a Laue direction 40 deg from the beam. Every
source-owned root lies 3-60 keV from its Bragg pole, where the tracked-state
modulation is small, so the source must reproduce the native zone-row lines it
replaces.

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import channeling_emission, planar_zone_pxr
from pyrite.montecarlo.runner.planar_zone_lines import is_own_row
from pyrite.montecarlo.runner.planar_zone_pxr import source_reflections

from .test_planar_zone_lines import _case, _lines, _plain, _rows


@pytest.fixture(scope="module")
def laue():
    case = replace(_case(), line_quadrature="bin-mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # small basis: overflow warning
        transport = runner._transport_case(case, transport_core="per-electron")
    beam = planar_zone_pxr._beam(case)
    side = np.cross(beam, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    n = np.cos(np.radians(40.0)) * beam + np.sin(np.radians(40.0)) * side
    return case, {**transport, "n_hat": n}


def test_source_owned_reflections_leave_zone_rows_only(laue):
    case, transport = laue
    segs = transport["segs"]
    zone = np.asarray(segs["channeling_zone"], dtype=bool)
    hkls = [tuple(int(v) for v in hkl) for hkl in case["hkl_list"]]
    owned = {row[0] for row in source_reflections(case, transport["n_hat"])}
    in_grid = [row for row in source_reflections(case, transport["n_hat"]) if row[2] < 25e3]
    assert len(in_grid) >= 5
    for _hkl, g, E0, *_ in in_grid:  # far from every Bragg pole
        assert abs(-HBARC_EV_ANG * (g @ g) / (2 * transport["n_hat"] @ g) - E0) > 2500.0
    zone_owned = [h for h in hkls if h in owned or is_own_row(h, (1, -1, 0))]
    unowned = [h for h in hkls if h not in zone_owned]
    assert unowned
    expected = _lines(_plain(case, unowned), transport, segs) + _lines(
        _plain(case, zone_owned), transport, _rows(segs, ~zone)
    )
    np.testing.assert_allclose(_lines(case, transport, segs), expected, rtol=1e-12, atol=0.0)


@pytest.mark.slow
def test_zone_source_reproduces_native_zone_rows_far_from_poles(laue, monkeypatch):
    """Band within 2%: PXR+CBS interference, absorption and the zone flux all carry over."""
    case, transport = laue
    segs = transport["segs"]
    zone = np.asarray(segs["channeling_zone"], dtype=bool)
    owned = [row[0] for row in source_reflections(case, transport["n_hat"])]
    native = _lines(_plain(case, owned), transport, _rows(segs, zone))
    seen = {}
    original = planar_zone_pxr.case_zone_pxr_spectrum

    def spy(*args, **kwargs):
        seen["source"] = original(*args, **kwargs)
        return seen["source"]

    monkeypatch.setattr(planar_zone_pxr, "case_zone_pxr_spectrum", spy)
    channeling_emission.case_channeling_spectrum(case, transport)
    assert np.isfinite(seen["source"]).all() and native.sum() > 0
    assert seen["source"].sum() == pytest.approx(native.sum(), rel=0.02)
