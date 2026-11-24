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
from tests.helpers import scaled_rtol

SELECTION = dict(tolerance=0.3, g_max_invang=16.0, shell_width_invang=2.0)

#: The selected set is scored in reflection chunks (``selection_call_sizes``),
#: the reference in one call: the same positive per-bin contributions in other
#: batch shapes, so device FP32 cells round apart by a few eps per bin.
SELECTED_RTOL = scaled_rtol(1e-9, eps_multiple=4.0)


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
    # Deterministic soft loss keeps this six-history selection fixture stable.
    case = build_cases(sweep, n_electrons=6, n_electrons_brem=3, straggling=False)[0]
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
    # A corrected (crystalline BremsLib) continuum selects on the net yield (#1250).
    assert case["bremsstrahlung_model"] == "bremslib" and radiator["metric"] == "net"
    assert radiator["background_yield"] > 0
    shares = [shell["share"] for shell in radiator["shells"]]
    assert all(0 < s < y for s, y in zip(shares, yields, strict=True))
    selected = dominant_reflections(
        "silicon", n_families=10**9, B_ang2=case["B_ang2"], g_max_invang=cutoff
    )
    assert radiator["reflections"] == len(selected)
    reference = runner._spectrum_case(dict(case, hkl_list=selected), tp)
    assert np.sum(reference["spec"]) > 0
    np.testing.assert_allclose(out["spec"], reference["spec"], rtol=SELECTED_RTOL, atol=0)
    grid = np.asarray(tp["E_grid"])
    assert sum(yields) == pytest.approx(np.trapezoid(reference["spec"], grid), rel=SELECTED_RTOL)
    assert "reflection_selection" not in reference


def test_selection_refuses_coherent_emission():
    case, tp = _transported()
    coherent = dict(case, reflection_selection=SELECTION, coherent_emission="both")
    with pytest.raises(ValueError, match="incoherent lines only"):
        runner._spectrum_case(coherent, tp)


@pytest.mark.slow
@pytest.mark.parametrize("bandwidth", ["kinematic-ceiling", "resonance-population"])
def test_automatic_bandwidths_select_the_same_reflections(bandwidth, monkeypatch):
    """A shorter production axis must not change the net selector's band (#1243)."""
    from pyrite._cxr_grid_policy import resolve_cxr_grid_policy
    from pyrite.montecarlo.runner import case_driver, diffracted_brem

    # DBR has its own build-time reflection set, outside the selected native-line audit.
    monkeypatch.setattr(
        diffracted_brem, "case_diffracted_brem", lambda _case, _segs, grid, *_: np.zeros(grid.shape)
    )

    masses = []
    check = case_driver.check_cxr_truncation

    def capture_audit(case, audit):
        masses.append(float(audit["line_mass"]))
        return check(case, audit)

    monkeypatch.setattr(case_driver, "check_cxr_truncation", capture_audit)

    case, tp = _transported()
    policy = resolve_cxr_grid_policy(
        start_eV=500.0,
        stop_eV=30000.0,
        bandwidth_policy=bandwidth,
    ).payload()
    if bandwidth == "resonance-population":
        # This anchor tests selection on deliberately short axes, not tail acceptance.
        policy["bandwidth"]["truncation_limit"] = 0.99
    selected = dict(case, reflection_selection=SELECTION, line_grid_policy=policy)
    records = []
    for stop in (2000.0, 8000.0):
        grid = np.arange(500.0, stop + 1.0, 25.0)
        out = runner._spectrum_case(selected, {**tp, "E_grid": grid})
        records.append(out["reflection_selection"])
        assert out["spec"].shape == grid.shape
        if bandwidth == "resonance-population":
            # The reference pass must not add mass to the production audit.
            cutoff = records[-1]["radiators"][0]["cutoff_invang"]
            hkls = dominant_reflections(
                "silicon", n_families=10**9, B_ang2=case["B_ang2"], g_max_invang=cutoff
            )
            plain = runner._spectrum_case(
                dict(case, hkl_list=hkls, line_grid_policy=policy), {**tp, "E_grid": grid}
            )
            np.testing.assert_allclose(out["spec"], plain["spec"], rtol=1e-10)
            assert masses[-2] == pytest.approx(masses[-1], rel=1e-12)
    assert records[0]["radiators"] == records[1]["radiators"]
    assert records[0]["reference_band_eV"] == [500.0, 30000.0]


def _automatic_scene(bandwidth, stop_eV, monkeypatch):
    """The silicon scene with an automatic bandwidth on ``[500, stop_eV]``, DBR off."""
    from pyrite._cxr_grid_policy import resolve_cxr_grid_policy
    from pyrite.montecarlo.runner import diffracted_brem

    monkeypatch.setattr(
        diffracted_brem, "case_diffracted_brem", lambda _case, _segs, grid, *_: np.zeros(grid.shape)
    )
    case, tp = _transported()
    policy = resolve_cxr_grid_policy(
        start_eV=500.0, stop_eV=30000.0, bandwidth_policy=bandwidth
    ).payload()
    if bandwidth == "resonance-population":
        policy["bandwidth"]["truncation_limit"] = 0.99
    selected = dict(case, reflection_selection=SELECTION, line_grid_policy=policy)
    return selected, {**tp, "E_grid": np.arange(500.0, stop_eV + 1.0, 25.0)}


def _legacy_reference(monkeypatch):
    """Select through the bin-mean reference reduction, as before #1326."""
    from pyrite.montecarlo.runner import reflection_cutoff

    monkeypatch.setattr(reflection_cutoff, "_band_evaluator", lambda *_a, **_k: None)


def _same_selection(new, legacy):
    (a,), (b,) = (r["reflection_selection"]["radiators"] for r in (new, legacy))
    assert a["selected_set_sha256"] == b["selected_set_sha256"]
    assert a["cutoff_invang"] == b["cutoff_invang"] and a["converged"] == b["converged"]
    for x, y in zip(a["shells"], b["shells"], strict=True):
        assert x["yield_"] == pytest.approx(y["yield_"], rel=1e-9)
        assert x["share"] == pytest.approx(y["share"], rel=1e-9)
    np.testing.assert_allclose(new["spec"], legacy["spec"], rtol=SELECTED_RTOL, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("bandwidth", ["kinematic-ceiling", "resonance-population"])
def test_axis_spanning_the_band_selects_on_band_mass_then_rescores(bandwidth, monkeypatch):
    """A production axis spanning the band still decides on the mass-only
    reference pass and rescores the kept set: #1326 dropped the production-axis
    single pass, which reduced every rejected trailing shell and never paid."""
    from pyrite.montecarlo.runner import reflection_cutoff

    selected, tp = _automatic_scene(bandwidth, 30000.0, monkeypatch)
    rescored = []
    original = reflection_cutoff._selected_spectrum
    monkeypatch.setattr(
        reflection_cutoff,
        "_selected_spectrum",
        lambda *args: rescored.append(1) or original(*args),
    )
    new = runner._spectrum_case(selected, tp)
    assert rescored == [1]
    assert new["reflection_selection"]["reference_quadrature"] == "band-mass"
    assert new["reflection_selection"]["reference_band_eV"] == [500.0, 30000.0]
    _legacy_reference(monkeypatch)
    legacy = runner._spectrum_case(selected, tp)
    assert rescored == [1, 1]
    assert legacy["reflection_selection"]["reference_quadrature"] == "bin-mean"
    _same_selection(new, legacy)


@pytest.mark.slow
def test_short_axis_measures_band_mass_without_reducing_lines(monkeypatch):
    """An axis short of the band keeps the reference pass, now mass-only (#1326)."""
    from pyrite.montecarlo.runner import reflection_cutoff

    selected, tp = _automatic_scene("kinematic-ceiling", 8000.0, monkeypatch)
    rescored = []
    original = reflection_cutoff._selected_spectrum
    monkeypatch.setattr(
        reflection_cutoff,
        "_selected_spectrum",
        lambda *args: rescored.append(1) or original(*args),
    )
    new = runner._spectrum_case(selected, tp)
    assert rescored == [1]  # the production axis cannot supply the band's lines
    assert new["reflection_selection"]["reference_quadrature"] == "band-mass"
    _legacy_reference(monkeypatch)
    _same_selection(new, runner._spectrum_case(selected, tp))


@pytest.mark.parametrize(
    "unsupported", [{"max_dE_frac": 1e-3}, {"sinc_cutoff": 10.0}, {"node_support_cutoff": 10.0}]
)
def test_selection_keeps_node_only_routes_on_the_production_axis(unsupported):
    from pyrite.montecarlo.runner.reflection_cutoff import _selection_reference_grid

    case = {
        "line_grid_policy": {
            "bandwidth": {"policy": "kinematic-ceiling", "start_eV": 50.0, "stop_eV": 30000.0}
        },
        **unsupported,
    }
    assert _selection_reference_grid(case) is None


def test_selected_layered_union_is_scored_once_in_admitted_chunks(monkeypatch):
    from pyrite.montecarlo.runner import reflection_cutoff

    members = [[(1, 0, 0), (2, 0, 0), (3, 0, 0)], [(0, 1, 0), (0, 2, 0)]]
    case = {"layer_radiators": [{"hkl_list": h} for h in members] + [None]}
    monkeypatch.setattr(reflection_cutoff, "selection_call_sizes", lambda *_: {0: 2, 1: 1})
    seen = []

    def evaluate(pilot):
        active = [(i, r) for i, r in enumerate(pilot["layer_radiators"]) if r is not None]
        assert len(active) == 1
        layer, radiator = active[0]
        assert len(radiator["hkl_list"]) <= (2 if layer == 0 else 1)
        seen.extend((layer, hkl) for hkl in radiator["hkl_list"])
        return np.full(4, len(radiator["hkl_list"]))

    spectrum = reflection_cutoff._selected_spectrum(case, evaluate, np.arange(4.0))
    assert sorted(seen) == sorted((i, hkl) for i, hkls in enumerate(members) for hkl in hkls)
    np.testing.assert_array_equal(spectrum, np.full(4, 5.0))


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
    # The continuum correction reports itself per layer (#1250), selected or not.
    from pyrite.montecarlo.spectrum.brem_lattice_share import BREM_LATTICE_SHARE_MODEL

    for provenance in (result.provenance, plain):
        share = provenance.brem_lattice_share
        assert share["model"] == BREM_LATTICE_SHARE_MODEL
        (layer,) = share["layers"]
        assert layer["corrected"] and layer["line_cutoff_keV"] == 5.0
        (element,) = layer["elements"]
        assert element["element"] == "Si" and 0.0 < element["removed_fraction"] < 0.3
    assert (
        result.provenance.brem_lattice_share["layers"][0]["reflections"]
        == record["radiators"][0]["reflections"]
    )


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


def test_selection_scores_an_electron_prefix_then_every_history(monkeypatch):
    """The stop is decided on the first histories, the spectrum on all (#1281)."""
    import pyrite.montecarlo.runner.diffracted_brem as dbr
    from pyrite.montecarlo.runner import reflection_cutoff
    from pyrite.montecarlo.runner.electron_blocks import electron_prefix

    monkeypatch.setattr(
        dbr, "case_diffracted_brem", lambda _case, _segs, grid, *_: np.zeros(np.shape(grid))
    )
    case, tp = _transported()
    selected = dict(case, reflection_selection=SELECTION)
    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
    out = runner._spectrum_case(selected, tp)
    record = out["reflection_selection"]
    # The history cap binds (#1368): the row budget is far from reached.
    rows = int(np.sum(np.asarray(tp["segs"]["elec_id"]) < 2))
    assert rows < reflection_cutoff.SELECTION_ROWS
    assert record["selection_histories"] == {
        "lines": 2,
        "continuum": 2,
        "rows": rows,
        "row_budget": reflection_cutoff.SELECTION_ROWS,
    }
    # The selection equals a full selection on the two-history population alone.
    alone_tp = {**tp, "segs": electron_prefix(tp["segs"], 2), "Ne_lines": 2, "Ne_brem": 2}
    alone = runner._spectrum_case(dict(selected, Ne=2, Ne_brem=2), alone_tp)
    assert "selection_histories" not in alone["reflection_selection"]
    (sub,), (ref,) = record["radiators"], alone["reflection_selection"]["radiators"]
    assert sub["cutoff_invang"] == ref["cutoff_invang"]
    assert sub["reflections"] == ref["reflections"]
    for a, b in zip(sub["shells"], ref["shells"], strict=True):
        assert a["yield_"] == pytest.approx(b["yield_"], rel=1e-9)
        assert a["share"] == pytest.approx(b["share"], rel=1e-9)
    # The returned spectrum is the selected set on every history.
    hkls = dominant_reflections(
        "silicon", n_families=10**9, B_ang2=case["B_ang2"], g_max_invang=sub["cutoff_invang"]
    )
    full = runner._spectrum_case(dict(case, hkl_list=hkls), tp)
    np.testing.assert_allclose(out["spec"], full["spec"], rtol=SELECTED_RTOL, atol=0)
    assert np.sum(full["spec"]) > 0


def test_row_budget_binds_before_the_history_cap(monkeypatch):
    """A population that reaches ``SELECTION_ROWS`` within fewer electrons than
    the cap selects on that shorter prefix, exactly as the cap would (#1368)."""
    from pyrite.montecarlo.runner import reflection_cutoff

    case, tp = _transported()
    selected = dict(case, reflection_selection=SELECTION)
    counts = np.bincount(np.asarray(tp["segs"]["elec_id"]))
    assert counts[0] > 0 and counts[1] > 0
    budget = int(counts[:2].sum())  # electron 1's rows cross the budget: k = 2
    monkeypatch.setattr(reflection_cutoff, "SELECTION_ROWS", budget)
    out = runner._spectrum_case(selected, tp)
    record = out["reflection_selection"]
    assert record["selection_histories"] == {
        "lines": 2,
        "continuum": 2,
        "rows": budget,
        "row_budget": budget,
    }
    monkeypatch.setattr(reflection_cutoff, "SELECTION_ROWS", 1 << 62)
    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
    capped = runner._spectrum_case(selected, tp)
    assert record["radiators"] == capped["reflection_selection"]["radiators"]
    np.testing.assert_array_equal(out["spec"], capped["spec"])


def test_row_budget_prefix_counts_whole_electrons():
    from pyrite.montecarlo.runner.electron_blocks import row_budget_prefix

    # Rows per electron 0..4: 3, 0, 2, 5, 1, split over unordered blocks.
    blocks = [
        {"elec_id": np.array([3, 3, 0, 3, 2])},
        {"elec_id": np.array([0, 4, 3, 2, 0, 3])},
    ]
    assert row_budget_prefix(blocks, 5, 3) == (1, 3)
    assert row_budget_prefix(blocks, 5, 4) == (3, 5)  # electron 1 has no row
    assert row_budget_prefix(blocks, 5, 6) == (4, 10)
    assert row_budget_prefix(blocks, 5, 11) == (5, 11)
    assert row_budget_prefix(blocks, 5, 10**9) == (5, 11)  # budget unreached: the cap
    assert row_budget_prefix(blocks, 2, 10**9) == (2, 3)  # rows past the cap ignored
    assert row_budget_prefix([{"elec_id": np.zeros(0, dtype=np.int64)}], 4, 1) == (4, 0)


def test_electron_prefix_keeps_whole_histories():
    from pyrite.montecarlo.runner.electron_blocks import electron_prefix

    segs = {"elec_id": np.array([2, 0, 1, 0, 2]), "L_ang": np.arange(5.0), "Ne": 3}
    prefix = electron_prefix(segs, 2)
    assert prefix["Ne"] == 2
    np.testing.assert_array_equal(prefix["elec_id"], [0, 1, 0])
    np.testing.assert_array_equal(prefix["L_ang"], [1.0, 2.0, 3.0])
    ordered = electron_prefix({"elec_id": np.array([0, 0, 1, 2]), "L_ang": np.arange(4.0)}, 1)
    np.testing.assert_array_equal(ordered["L_ang"], [0.0, 1.0])


@pytest.mark.parametrize(("material", "automatic"), [("wte2", True), ("pdte2", False)])
def test_reference_only_pins_no_longer_block_automatic_selection(material, automatic):
    """#1302: an unpinned single crystal resolves automatic; B33-pinned stays fixed."""
    from pyrite.campaign.reflection_selection import resolve_reflection_sweep

    sweep = Sweep(material=material, beam=BeamSpec(energy_keV=30.0), thickness_ang=1000.0)
    resolved, is_automatic = resolve_reflection_sweep(sweep, False, None)
    assert is_automatic is automatic
    assert resolved.line_yield_tolerance == (1e-3 if automatic else None)


@pytest.mark.slow
def test_formerly_pinned_wte2_selects_beyond_its_old_basal_pin(monkeypatch):
    """#1302: WTe2 lost its reference-only (002) pin; pin its selected set.

    Frozen in-process transport: 4 histories, deterministic soft loss, fixed
    seed. Selection is shell-closed (all reflections up to the cutoff), so the
    digest follows the stop decision, whose tail (~0.08) sits far from 0.3.
    """
    import hashlib
    import json

    import pyrite.montecarlo.runner.diffracted_brem as dbr

    monkeypatch.setattr(
        dbr, "case_diffracted_brem", lambda case, segments, E_grid, n_hat, Ne: np.zeros(len(E_grid))
    )
    sweep = Sweep(
        material="wte2",
        n_families=4,
        beam=BeamSpec(energy_keV=30.0),
        thickness_ang=1000.0,
        tilt_deg=15.0,
        energy_bins=EnergyBins(
            cxr=np.arange(500.0, 8000.0, 25.0), brem=np.array([5000.0, 10000.0, 15000.0])
        ),
    )
    case = build_cases(
        sweep, n_electrons=4, n_electrons_brem=2, inelastic_model="continuous", straggling=False
    )[0]
    out = runner._spectrum_case(
        dict(case, reflection_selection=SELECTION), runner._transport_case(case)
    )
    (radiator,) = out["reflection_selection"]["radiators"]
    assert radiator["converged"] and radiator["metric"] == "net"
    assert radiator["cutoff_invang"] == 8.0
    selected = dominant_reflections(
        "wte2", n_families=10**9, B_ang2=case["B_ang2"], g_max_invang=radiator["cutoff_invang"]
    )
    assert radiator["reflections"] == len(selected) == 2554
    hkls = {tuple(int(i) for i in hkl) for hkl in selected}
    assert {(0, 0, 2), (0, 0, -2)} <= hkls
    assert any(h or k for h, k, _ in hkls)  # in-plane/oblique reflections now radiate
    digest = hashlib.sha256(json.dumps(sorted(hkls), separators=(",", ":")).encode()).hexdigest()
    assert digest[:16] == "1770ab10da0a3545"
