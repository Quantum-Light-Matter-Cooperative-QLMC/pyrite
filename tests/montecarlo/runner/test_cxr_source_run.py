"""``cxr_source="none"`` runs no line work and keeps the continuum (#1327)."""

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.config import FixedSampling
from pyrite.montecarlo import run_case
from pyrite.montecarlo.runner import emission_adapters

HISTORIES = 8
LINE_GRID = np.linspace(1.0e3, 6.0e3, 120)
BREM_GRID = np.arange(1.0e3, 30.1e3, 500.0)


def scene():
    return pr.Scene(
        pr.Beam(30.0, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab("silicon", thickness_ang=2.0e4, tilt_deg=30.0, width=None, height=None),
    )


def numerics():
    return pr.Numerics(
        sampling=FixedSampling(cxr_histories=HISTORIES, brem_histories=HISTORIES),
        seed=7,
        energy_bins=pr.detectors.EnergyBins(cxr=LINE_GRID, brem=BREM_GRID),
    )


def test_cxr_off_scores_no_lines_and_runs_no_selection(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a cxr_source='none' run scored lines")

    monkeypatch.setattr(
        "pyrite.montecarlo.runner.reflection_cutoff.selected_batch_spectrum", refuse
    )
    monkeypatch.setattr("pyrite.montecarlo.runner._cxr_for_segments", refuse)
    monkeypatch.setattr(
        "pyrite.montecarlo.spectrum.diffracted_brem.diffracted_brem_spectrum", refuse
    )
    monkeypatch.setattr(
        "pyrite.montecarlo.spectrum.diffracted_characteristic.diffracted_characteristic_spectrum",
        refuse,
    )
    result = pr.simulate(scene(), numerics=numerics(), physics=pr.Physics(cxr_source="none"))
    assert not np.any(result.cxr.incoherent.density)
    assert result.cxr.incoherent.photon_energy_eV.shape == LINE_GRID.shape
    assert np.all(np.isfinite(result.brem.density)) and np.any(result.brem.density > 0)
    assert result.case["cxr_source"] == "none"
    assert "reflection_selection" not in result.case


def test_cxr_off_zero_adapter_matches_native_shapes():
    case = api.build_case(scene(), numerics(), physics=pr.Physics(cxr_source="none"))
    adapter = emission_adapters.select_emission(case).cxr
    batch = emission_adapters.EmissionBatch({"Ne": 3}, np.arange(5.0), case, None)
    assert adapter.evaluate(batch, "spec").values.shape == (5,)
    weighted = emission_adapters.EmissionBatch(
        {"Ne": 3}, np.arange(5.0), case, None, electron_band_weights=np.ones(2)
    )
    assert adapter.evaluate(weighted, "spec").values.shape == (3,)


def test_cxr_off_disables_lattice_share_correction():
    from pyrite.montecarlo.runner import lattice_share

    off = api.build_case(scene(), numerics(), physics=pr.Physics(cxr_source="none"))
    on = api.build_case(scene(), numerics())
    assert not lattice_share.lattice_share_applies(off)
    assert lattice_share.skip_reason(off, None) == "lines off (cxr_source='none')"
    assert lattice_share.lattice_share_applies(on)


def test_cxr_off_continuum_equals_lines_on_without_lattice_correction():
    """Same seed and transport: the continuum matches a lines-on run that corrects nothing."""
    off = api.build_case(scene(), numerics(), physics=pr.Physics(cxr_source="none"))
    on = dict(api.build_case(scene(), numerics()))
    on.pop("reflection_selection", None)  # four-family lines instead of the measured set
    on["_brem_lattice_share"] = False  # the uncorrected continuum a lines-off run scores
    off_out, on_out = run_case(off), run_case(on)
    np.testing.assert_array_equal(off_out["brem_wide"], on_out["brem_wide"])
    np.testing.assert_array_equal(off_out["spec_characteristic"], on_out["spec_characteristic"])
    assert np.any(on_out["spec"]) and not np.any(off_out["spec"])


@pytest.mark.parametrize("key", ["coherent_emission", "temporal_profile"])
def test_case_refuses_line_only_keys(key):
    case = dict(api.build_case(scene(), numerics(), physics=pr.Physics(cxr_source="none")))
    with pytest.raises(ValueError, match="cxr_source"):
        pr.montecarlo.Case(**{**case, key: True})
