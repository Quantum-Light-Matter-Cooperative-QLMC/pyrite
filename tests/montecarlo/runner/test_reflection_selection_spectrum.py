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


def test_call_size_follows_tabulation_length_not_device():
    """Reflections per selection call (#1200): set by the table length alone, clamped."""
    from pyrite.montecarlo.spectrum.cxr._line_tables import (
        SELECTION_CALL_REFLECTIONS,
        SELECTION_CALL_TABLE_BYTES,
        selection_reflections_per_call,
    )
    from pyrite.montecarlo.spectrum.cxr._setup import line_table_points

    lo, hi = SELECTION_CALL_REFLECTIONS
    assert selection_reflections_per_call(1) == hi
    assert selection_reflections_per_call(10**9) == lo
    n_tab = 200_000
    count = selection_reflections_per_call(n_tab, itemsize=4)
    assert count * 16 * n_tab <= SELECTION_CALL_TABLE_BYTES < (count + 1) * 16 * n_tab
    assert selection_reflections_per_call(n_tab, itemsize=8) == count // 2
    case, _ = _transported()
    composition = case["composition"]
    band = line_table_points("silicon", composition, np.linspace(500.0, 10000.0, 9501))
    fine = line_table_points("silicon", composition, np.linspace(500.0, 10000.0, 95001))
    wide = line_table_points("silicon", composition, np.linspace(50.0, 300000.0, 9501))
    assert band == fine  # the span sets the tables, not the axis spacing
    # #1201: the edge-geometric mesh grows ~logarithmically with the span
    # (Si 50 eV-300 keV: 9,322 points, not the 1 eV mesh's 362,299).
    assert band < wide < 2 * band
    assert selection_reflections_per_call(wide) <= selection_reflections_per_call(band)


def test_selection_records_call_size_and_refuses_before_scoring(monkeypatch):
    from dataclasses import replace

    from pyrite._backend import BackendResourceError
    from pyrite.montecarlo.runner import reflection_cutoff

    case, tp = _transported()
    selected = dict(case, reflection_selection=SELECTION)
    grid = np.asarray(tp["E_grid"])
    sizes = reflection_cutoff.selection_call_sizes(selected, grid)
    out = runner._spectrum_case(selected, tp)
    (radiator,) = out["reflection_selection"]["radiators"]
    assert radiator["reflections_per_call"] == sizes[None]

    calls = []
    monkeypatch.setattr(
        runner,
        "_RESOURCE_POLICY",
        replace(runner._RESOURCE_POLICY, gpu=True, device_budget_bytes=1),
    )
    with pytest.raises(BackendResourceError, match="line tables"):
        reflection_cutoff.selected_line_spectrum(selected, calls.append, grid)
    assert calls == []
