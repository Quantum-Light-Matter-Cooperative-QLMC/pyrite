"""Planar zone-row line ownership (cr-planar-zone-row-ownership, #1055)."""

import warnings

import numpy as np
import pytest

import pyrite as pr
from pyrite.api import build_case
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins, PointDetector
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import channeling_emission
from pyrite.montecarlo.runner.planar_zone_lines import is_own_row, planar_zone_line_parts

# 3 deg from the (110) face normal; azimuth 46 deg puts the beam 0.913 mrad
# out of the Si (1-10) planes (psi_c = 1.55 mrad at 17 MeV): the own-row
# line lies near 14 keV.
TILT_AZIM_DEG = 46.0


def _case(tilt_azim_deg=TILT_AZIM_DEG, channeling=True, thickness_ang=20000.0):
    physics = pr.Physics(radiative_model="uncoupled")
    zone = (
        pr.Channeling(
            "planar",
            plane=(1, -1, 0),
            numerics=pr.ChannelingNumerics(basis_size=8, bloch_cells=4, step_ang=1000.0),
            admission="accept-inaccurate",
        )
        if channeling
        else None
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=17000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "silicon",
            thickness_ang=thickness_ang,
            tilt_deg=3.0,
            tilt_azim_deg=tilt_azim_deg,
            footprint=pr.geometry.Footprint(5.0, 5.0),
        ),
        PointDetector(),
        physics=physics,
        channeling=zone,
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=12, brem_histories=1),
        energy_bins=EnergyBins(cxr=np.linspace(5e3, 25e3, 81)),
    )
    return build_case(scene, numerics)


@pytest.fixture(scope="module")
def tilted():
    case = _case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # small basis: overflow warning
        transport = runner._transport_case(case, transport_core="per-electron")
    return case, transport


def _lines(case, transport, segs):
    return np.asarray(
        runner._cxr_for_segments(
            segs,
            np.asarray(transport["E_grid"], dtype=float),
            case,
            transport["n_hat"],
            case.get("abs_layers"),
            transport.get("groove"),
            coherent=False,
            Ne=transport["Ne_lines"],
            table_cache={},
        )
    )


def _plain(case, hkl_list):
    """The case as an ordinary (non-channeling) mapping with chosen reflections."""
    return {**case.to_dict(), "channeling": None, "hkl_list": hkl_list}


def _rows(segs, keep):
    from pyrite.montecarlo.spectrum.cxr._kernels import _SEG_ARRAYS

    out = dict(segs)
    for key in (*_SEG_ARRAYS, "channeling_zone"):
        if out.get(key) is not None:
            out[key] = np.asarray(out[key])[keep]
    return out


def test_own_row_is_the_plane_normal_row_in_index_space():
    assert is_own_row((2, -2, 0), (1, -1, 0)) and is_own_row((-4, 4, 0), (1, -1, 0))
    assert not is_own_row((2, 2, 0), (1, -1, 0)) and not is_own_row((1, 1, 1), (1, -1, 0))
    assert is_own_row((0, -2, 2), (0, 1, -1))


def test_other_cases_are_unchanged_and_untagged_planar_transport_refuses(tilted):
    case, transport = tilted
    segs = transport["segs"]
    assert planar_zone_line_parts({"channeling": None, "hkl_list": [(2, -2, 0)]}, segs) is None
    axial = {"channeling": {"model": "axial", "plane_hkl": (1, -1, 0)}, "hkl_list": [(2, -2, 0)]}
    assert planar_zone_line_parts(axial, segs) is None
    untagged = {key: value for key, value in segs.items() if key != "channeling_zone"}
    with pytest.raises(ValueError, match="zone-row tag"):
        planar_zone_line_parts(case, untagged)


def test_zone_rows_score_no_own_row_lines_and_keep_every_other_row(tilted):
    case, transport = tilted
    segs = transport["segs"]
    zone = np.asarray(segs["channeling_zone"], dtype=bool)
    assert zone.any() and (~zone).any()
    hkls = [tuple(int(v) for v in hkl) for hkl in case["hkl_list"]]
    own = [hkl for hkl in hkls if is_own_row(hkl, (1, -1, 0))]
    other = [hkl for hkl in hkls if not is_own_row(hkl, (1, -1, 0))]
    assert own and other

    actual = _lines(case, transport, segs)
    # Independently: other reflections on every row plus own row off the zone.
    expected = _lines(_plain(case, other), transport, segs) + _lines(
        _plain(case, own), transport, _rows(segs, ~zone)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    # The rule acts: the own row on zone rows alone is a bright in-band line.
    zone_own = _lines(_plain(case, own), transport, _rows(segs, zone))
    assert zone_own.max() > 0.0
    everything = _lines(_plain(case, hkls), transport, segs)
    np.testing.assert_allclose(everything - zone_own, actual, rtol=1e-10, atol=1e-30)


def test_spec_channeling_scores_the_in_cut_free_free_block(tilted, monkeypatch):
    case, transport = tilted
    seen = {}
    original = channeling_emission.channeling_spectrum

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(channeling_emission, "channeling_spectrum", spy)
    spectrum = channeling_emission.case_channeling_spectrum(case, transport)
    assert seen["final_states"] == "tracked"
    assert np.all(np.isfinite(spectrum)) and spectrum.max() > 0.0


def test_channeling_off_line_scoring_is_untouched():
    case = _case(channeling=False)
    assert case.get("channeling") is None
    assert planar_zone_line_parts(case, {"L_ang": np.zeros(0)}) is None


def test_spec_channeling_is_independent_of_call_order_and_process():
    """#1090: transport leaves the cached solver at the exit-face basis.

    Scoring after in-process transport must equal scoring twice in a row and
    equal a score on a freshly built solver (a worker process or a replay).
    5 um: the stopping drift passes the 1e-4 reprojection tolerance (2 um does not).
    """
    channeling_emission.zone_solver.cache_clear()
    case = _case(thickness_ang=50000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # small basis: overflow warning
        transport = runner._transport_case(case, transport_core="per-electron")
    record = case["channeling"]

    def solver():
        return channeling_emission.zone_solver(
            case["crystal"],
            tuple(record["plane_hkl"]),
            float(record["temperature_K"]),
            float(case["E0_keV"]) * 1e3,
            int(record["basis_size"]),
            int(record["bloch_cells"]),
            float(record["margin_eV"]),
        )[1]

    stale = solver()
    # Precondition: the shared solver really was left on the reprojected basis.
    assert stale.basis.gamma != stale.gamma0 and stale._scale != 1.0
    first = channeling_emission.case_channeling_spectrum(case, transport)
    second = channeling_emission.case_channeling_spectrum(case, transport)
    assert first.max() > 0.0
    np.testing.assert_array_equal(first, second)

    channeling_emission.zone_solver.cache_clear()
    fresh = channeling_emission.case_channeling_spectrum(case, transport)
    np.testing.assert_array_equal(first, fresh)
