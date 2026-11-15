"""Measured reflection selection inside the spectrum phase (#1173).

Validation: reflection-yield-tolerance-selector
"""

from functools import cache

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.detectors import EnergyBins
from pyrite.materials.crystal import dominant_reflections
from pyrite.montecarlo import runner

SELECTION = dict(tolerance=0.3, g_max_invang=16.0, shell_width_invang=2.0)


@cache
def _transported():
    sweep = Sweep(
        material="silicon",
        n_families=4,
        beam=BeamSpec(energy_keV=30.0),
        thickness_ang=20000.0,
        tilt_deg=30.0,
        energy_bins=EnergyBins(
            cxr=np.arange(500.0, 8000.0, 25.0), brem=np.array([5000.0, 10000.0, 15000.0])
        ),
    )
    case = build_cases(sweep, n_electrons=6, n_electrons_brem=3)[0]
    return case, runner._transport_case(case)


def test_selected_spectrum_is_the_spectrum_of_the_selected_set(monkeypatch):
    import pyrite.montecarlo.runner.diffracted_brem as dbr

    # Diffracted brem keeps the build-time reflections; isolate the lines.
    dbr_sets = []

    def no_dbr(case, segments, E_grid, n_hat, Ne):
        dbr_sets.append(list(case["hkl_list"]))
        return np.zeros(np.shape(E_grid))

    monkeypatch.setattr(dbr, "case_diffracted_brem", no_dbr)
    case, tp = _transported()
    out = runner._spectrum_case(dict(case, reflection_selection=SELECTION), tp)
    assert dbr_sets == [list(case["hkl_list"])]
    record = out["reflection_selection"]
    (radiator,) = record["radiators"]
    assert record["tolerance"] == 0.3 and record["quadrature"] == "node"
    assert radiator["converged"] and radiator["tail_fraction"] <= 0.3
    cutoff = radiator["cutoff_invang"]
    assert cutoff < SELECTION["g_max_invang"]
    yields = [shell["yield_"] for shell in radiator["shells"]]
    assert all(y > 0 for y in yields) and yields[-1] < yields[0]
    selected = dominant_reflections(
        "silicon", n_families=10**9, B_ang2=case["B_ang2"], g_max_invang=cutoff
    )
    assert radiator["reflections"] == len(selected)
    reference = runner._spectrum_case(dict(case, hkl_list=selected), tp)
    assert np.sum(reference["spec"]) > 0
    np.testing.assert_allclose(out["spec"], reference["spec"], rtol=1e-9, atol=0)
    grid = np.asarray(tp["E_grid"])
    assert sum(yields) == pytest.approx(np.trapezoid(reference["spec"], grid), rel=1e-9)
    assert "reflection_selection" not in reference


def test_selection_refuses_coherent_emission():
    case, tp = _transported()
    coherent = dict(case, reflection_selection=SELECTION, coherent_emission="both")
    with pytest.raises(ValueError, match="incoherent lines only"):
        runner._spectrum_case(coherent, tp)


def test_streamed_block_scoring_refuses_selection():
    from pyrite.montecarlo.runner.streamed_scoring import streaming_refusal

    case, _ = _transported()
    assert streaming_refusal(case) is None
    reason = streaming_refusal(dict(case, reflection_selection=SELECTION))
    assert reason is not None and "reflection_selection" in reason


def test_simulate_reports_the_selection_in_provenance():
    import pyrite as pr

    scene = pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "silicon",
            thickness_ang=20_000.0,
            tilt_deg=30.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
    )

    def numerics(**reflections):
        return pr.Numerics(
            sampling=pr.config.FixedSampling(cxr_histories=6, brem_histories=3),
            seed=7,
            energy_bins=EnergyBins(
                cxr=np.arange(500.0, 8000.0, 25.0), brem=np.arange(1000.0, 30001.0, 1000.0)
            ),
            cxr_reflections=pr.config.CxrReflections(**reflections),
        )

    result = pr.simulate(scene, numerics=numerics(line_yield_tolerance=0.3))
    record = result.provenance.reflection_selection
    assert record["tolerance"] == 0.3 and record["radiators"][0]["crystal"] == "silicon"
    assert record["radiators"][0]["converged"]
    plain = pr.simulate(scene, numerics=numerics(n_families=4)).provenance
    assert plain.reflection_selection is None and "reflection_selection" not in plain
