"""Measured ``resonance-population`` line bandwidth (issue #192).

Pins the tail bounds the edge is chosen from, the opt-in policy payload and its
refusals, case construction where the ceiling alone overflows the budget, and
the production-weight truncation audit.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from pyrite._backend import _to_cpu, xp
from pyrite._cxr_grid_policy import (
    CXR_GRID_POLICY_SCHEMA,
    DEFAULT_BANDWIDTH_TRUNCATION,
    KINEMATIC_BANDWIDTH_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    RESONANCE_CXR_GRID_POLICY_SCHEMA,
    CxrGridToleranceError,
    CxrGridTruncationWarning,
    CxrYieldStatisticsWarning,
    kinematic_fallback_payload,
    resolve_cxr_grid_policy,
    resolved_coordinates,
)
from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.energy_grid.bandwidth_check import reference_axis, shape_and_counts
from pyrite.montecarlo.runner.cxr_grid import (
    CXR_YIELD_RELATIVE_SE_LIMIT,
    _measured_cxr_grid,
    check_cxr_truncation,
    cxr_truncation_audit,
    cxr_yield_statistics,
)
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import sincsq_window_mass
from pyrite.montecarlo.spectrum.cxr._kernels import _accumulate_edge_truncation
from pyrite.montecarlo.spectrum.cxr_seeds import (
    ResonancePopulation,
    case_cxr_stop_eV,
    characteristic_stop_eV,
    resonance_population_stop_eV,
    sincsq_upper_tail_bound,
)

_RESONANCE = {"bandwidth": RESONANCE_BANDWIDTH_POLICY}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in (
        "PYRITE_ENERGY_GRID_RTOL",
        "PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE",
        "PYRITE_ENERGY_GRID_RTOL_DETECTED_COUNTS",
        "PYRITE_ENERGY_GRID_MAX_SPACING_EV",
        "PYRITE_ENERGY_GRID_ULPS",
        "PYRITE_ENERGY_GRID_MAX_POINTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sincsq_tail_bound_dominates_exact_tail():
    rng = np.random.default_rng(192)
    width = rng.uniform(0.1, 50.0, 400)
    distance = rng.uniform(0.01, 5000.0, 400)
    a_width = np.pi / width
    # Upper edge at E_res + distance: shift the resonance below a fixed edge.
    edges = np.array([-1e12, 0.0])
    _captured, truncated = sincsq_window_mass(a_width, -distance, edges)
    exact = truncated / width
    bound = sincsq_upper_tail_bound(width, distance)
    assert np.all(bound >= exact * (1.0 - 1e-12))
    # sin**2 averages 1/2, so the far-tail bound is twice the exact tail.
    far = distance > 100.0 * width
    assert np.allclose(bound[far] / exact[far], 2.0, rtol=2e-2)


def test_resonance_above_edge_counts_wholly():
    assert sincsq_upper_tail_bound(1.0, -3.0) == 1.0
    assert sincsq_upper_tail_bound(1.0, 0.0) == 1.0


def _population(energy, width, weight=None):
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    width = np.broadcast_to(np.asarray(width, dtype=float), energy.shape).copy()
    weight = np.ones_like(energy) if weight is None else np.asarray(weight, dtype=float)
    return ResonancePopulation("(0 0 2)", energy, weight, width)


def test_single_flight_stop_is_its_closed_form_limit():
    limit = 1e-4
    stop, summary = resonance_population_stop_eV(
        [_population(7000.0, 2.0)], ceiling_eV=1.0e6, truncation_limit=limit, round_to_eV=1.0
    )
    expected = 7000.0 + 2.0 / (np.pi**2 * limit)
    assert expected <= stop <= expected + 1.0
    assert summary["proxy_truncated_fraction"] <= limit
    assert not summary["capped"]


def test_stop_never_exceeds_ceiling_and_empty_population_has_none():
    stop, summary = resonance_population_stop_eV(
        [_population(7000.0, 2.0)], ceiling_eV=8000.0, truncation_limit=1e-4
    )
    assert stop == 8000.0 and summary["capped"]
    stop, summary = resonance_population_stop_eV(
        [_population([], 1.0)], ceiling_eV=8000.0, truncation_limit=1e-4
    )
    assert stop is None and summary["n_lines"] == 0


def test_weighted_stop_lets_light_wide_lines_through():
    heavy = _population(5000.0, 1.0, weight=[1.0])
    light = _population(5000.0, 1000.0, weight=[1e-12])
    alone, _ = resonance_population_stop_eV([heavy], ceiling_eV=1e7, truncation_limit=1e-4)
    mixed, _ = resonance_population_stop_eV([heavy, light], ceiling_eV=1e7, truncation_limit=1e-4)
    assert mixed == pytest.approx(alone, abs=100.0)


def test_characteristic_stop_bounds_each_lorentzian_tail():
    limit = 1e-4
    stop, summary = characteristic_stop_eV([[("Mo", 1.0), ("S", 2.0)]], truncation_limit=limit)
    assert summary["line"].startswith("Mo ")
    # Lorentzian: 1/2 - arctan(2 D / Gamma) / pi <= Gamma / (2 pi D) = limit.
    assert stop > 17_000.0


def test_kinematic_policy_payload_is_unchanged():
    policy = resolve_cxr_grid_policy(
        start_eV=50.0, stop_eV=1000.0, per_call={"bandwidth": KINEMATIC_BANDWIDTH_POLICY}
    )
    payload = policy.payload()
    assert payload["schema"] == CXR_GRID_POLICY_SCHEMA
    assert payload["bandwidth"] == {
        "policy": KINEMATIC_BANDWIDTH_POLICY,
        "start_eV": 50.0,
        "stop_eV": 1000.0,
    }
    assert "bandwidth" not in payload["sources"]


def test_resonance_policy_payload_carries_its_share():
    payload = resolve_cxr_grid_policy(start_eV=50.0, stop_eV=1000.0, per_call=_RESONANCE).payload()
    assert payload["schema"] == RESONANCE_CXR_GRID_POLICY_SCHEMA
    assert payload["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY
    assert payload["bandwidth"]["truncation_limit"] == DEFAULT_BANDWIDTH_TRUNCATION
    assert payload["sources"]["bandwidth"] == "per-call"


def test_resonance_policy_refuses_windows_and_unknown_names():
    with pytest.raises(ValueError, match="cannot be combined with windows"):
        resolve_cxr_grid_policy(
            start_eV=50.0, stop_eV=1000.0, per_call={**_RESONANCE, "windows": True}
        )
    with pytest.raises(ValueError, match="bandwidth must be one of"):
        resolve_cxr_grid_policy(start_eV=50.0, stop_eV=1000.0, per_call={"bandwidth": "log"})


def test_measured_stop_must_lie_under_the_cap():
    payload = resolve_cxr_grid_policy(start_eV=50.0, stop_eV=1000.0, per_call=_RESONANCE).payload()
    grid, record = resolved_coordinates(payload, 1.0, dtype=np.float64, stop_eV=500.0)
    assert grid[-1] == 500.0 and record["stop_eV"] == 500.0
    with pytest.raises(ValueError, match="outside"):
        resolved_coordinates(payload, 1.0, dtype=np.float64, stop_eV=2000.0)


def test_high_energy_hbn_builds_under_measured_bandwidth(monkeypatch):
    monkeypatch.setenv("PYRITE_ENERGY_GRID_MAX_POINTS", "200000")
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=5000.0), cxr_grid_policy=_RESONANCE)
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    bandwidth = case["line_grid_policy"]["bandwidth"]
    assert bandwidth["policy"] == RESONANCE_BANDWIDTH_POLICY
    # The ceiling is the clamped default stop (#1243): the 1720.8 keV kinematic
    # bound passes the atomic data's end, so both policies stop below it.
    assert bandwidth["stop_eV"] == 966_200.0
    assert case["line_grid_policy"]["sources"]["stop_eV"] == "atomic data ceiling"


def _audited_case(limit=1e-4):
    return {
        "name": "hbn test",
        "E0_keV": 5000.0,
        "line_grid_policy": {
            "bandwidth": {
                "policy": RESONANCE_BANDWIDTH_POLICY,
                "start_eV": 50.0,
                "stop_eV": 1e6,
                "truncation_limit": limit,
            }
        },
    }


def test_audit_is_only_built_for_the_measured_bandwidth():
    assert cxr_truncation_audit({"line_grid_policy": None}, np.arange(10.0)) is None
    default = {"line_grid_policy": {"bandwidth": {"policy": KINEMATIC_BANDWIDTH_POLICY}}}
    assert cxr_truncation_audit(default, np.arange(10.0)) is None
    audit = cxr_truncation_audit(_audited_case(), np.linspace(50.0, 1000.0, 5))
    assert audit == {"start_eV": 50.0, "stop_eV": 1000.0, "ceiling_eV": 1e6}


def test_audit_accumulates_and_gates_the_upper_edge():
    case = _audited_case()
    audit = {"start_eV": 50.0, "stop_eV": 1000.0}
    # One narrow line well inside, one wide line whose tail crosses the edge.
    E_r = xp.asarray([500.0, 900.0])
    a_width = xp.asarray([np.pi / 1.0, np.pi / 100.0])
    weight = xp.asarray([1.0, 1.0])
    _accumulate_edge_truncation(audit, E_r, a_width, weight)
    record_mass = float(audit["line_mass"])
    assert record_mass == pytest.approx(1.0 + 100.0)
    expected_above = (
        1.0 / (np.pi * np.pi / 1.0 * 500.0) * 1.0
        + min(1.0, 1.0 / (np.pi * np.pi / 100.0 * 100.0)) * 100.0
    )
    assert float(audit["mass_above"]) == pytest.approx(expected_above)
    with pytest.raises(CxrGridToleranceError, match="above the 0.0001 bandwidth share"):
        check_cxr_truncation(case, audit)
    narrow = {"start_eV": 50.0, "stop_eV": 1000.0}
    _accumulate_edge_truncation(narrow, E_r[:1], xp.asarray([np.pi / 0.01]), weight[:1])
    record = check_cxr_truncation(case, narrow)
    assert record["upper_fraction_bound"] <= 1e-4
    assert record["lower_fraction_bound"] > 0.0


def test_audit_at_the_kinematic_ceiling_warns_instead_of_refusing():
    """#192: at 100 keV the measured stop caps at the closed-form ceiling; tails
    above it are what the automatic bandwidth also drops, so this policy warns."""
    case = _audited_case()
    case["line_grid_policy"]["bandwidth"]["stop_eV"] = 1000.0
    audit = {"start_eV": 50.0, "stop_eV": 1000.0}
    _accumulate_edge_truncation(
        audit, xp.asarray([900.0]), xp.asarray([np.pi / 100.0]), xp.asarray([1.0])
    )
    with pytest.warns(CxrGridTruncationWarning, match="line-axis ceiling"):
        record = check_cxr_truncation(case, audit)
    assert record["capped_at_ceiling"]
    assert record["upper_fraction_bound"] > 1e-4


def test_audit_sums_line_mass_per_electron_when_counted():
    audit = cxr_truncation_audit(_audited_case(), np.linspace(50.0, 1000.0, 5), n_electrons=3)
    assert audit["n_electrons"] == 3
    E_r = xp.asarray([500.0, 600.0, 700.0])
    a_width = xp.asarray([np.pi, np.pi, np.pi / 2.0])
    _accumulate_edge_truncation(audit, E_r, a_width, xp.ones(3), xp.asarray([0, 2, 0]))
    np.testing.assert_allclose(_to_cpu(audit["electron_mass"]), [3.0, 0.0, 1.0])
    assert float(audit["line_mass"]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="outside the 3 line electrons"):
        _accumulate_edge_truncation(audit, E_r[:1], a_width[:1], xp.ones(1), xp.asarray([3]))


def test_line_yield_statistics_is_the_relative_standard_error_of_the_mean():
    mass = np.array([1.0, 2.0, 3.0, 0.0])
    stats = cxr_yield_statistics(mass)
    expected = mass.std(ddof=1) / np.sqrt(mass.size) / mass.mean()
    assert stats["relative_se"] == pytest.approx(expected)
    assert stats["max_electron_share"] == pytest.approx(0.5)
    assert cxr_yield_statistics(np.zeros(4))["relative_se"] is None


def test_audit_warns_when_a_rare_electron_carries_the_line_yield():
    """#201: a statistics-limited yield is flagged and warned about, not refused."""
    case = _audited_case()
    even = {"start_eV": 50.0, "stop_eV": 1000.0, "n_electrons": 400}
    rare = dict(even)
    E_r = xp.full(400, 500.0)
    a_width = xp.full(400, np.pi / 0.01)
    ids = xp.arange(400)
    _accumulate_edge_truncation(even, E_r, a_width, xp.ones(400), ids)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CxrYieldStatisticsWarning)
        record = check_cxr_truncation(case, even)
    stats = record["cxr_yield_statistics"]
    assert stats["relative_se"] < CXR_YIELD_RELATIVE_SE_LIMIT
    assert not stats["statistics_limited"]
    weights = xp.asarray(np.r_[1.0e3, np.ones(399)])
    _accumulate_edge_truncation(rare, E_r, a_width, weights, ids)
    with pytest.warns(CxrYieldStatisticsWarning, match="relative standard error"):
        record = check_cxr_truncation(case, rare)
    stats = record["cxr_yield_statistics"]
    assert stats["statistics_limited"]
    assert stats["max_electron_share"] == pytest.approx(1.0e3 / 1399.0)


def test_production_line_collection_retains_resonance_width_and_weight():
    audit = {"start_eV": 50.0, "stop_eV": 2000.0, "collect": []}
    _accumulate_edge_truncation(
        audit,
        xp.asarray([500.0, 900.0]),
        xp.asarray([np.pi, np.pi / 10]),
        xp.asarray([2.0, 0.5]),
    )
    energy, width, weight = audit["collect"][0]
    np.testing.assert_allclose(energy, [500.0, 900.0])
    np.testing.assert_allclose(width, [1.0, 10.0])
    np.testing.assert_allclose(weight, [2.0, 0.5])
    assert "line_mass" not in audit


def test_measured_grid_uses_collected_production_weights(monkeypatch):
    from pyrite.montecarlo import runner

    policy = resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=100_000.0,
        per_call={"bandwidth": RESONANCE_BANDWIDTH_POLICY},
    ).payload()
    case = {"composition": [("B", 0.5), ("N", 0.5)]}
    collected = (
        np.array([500.0, 900.0]),
        np.array([1.0, 100.0]),
        np.array([100.0, 1.0]),
    )

    def collect(_segments, axis, _case, _direction, _layers, _groove, **kwargs):
        np.testing.assert_array_equal(axis, [50.0, 100_000.0])
        kwargs["truncation_audit"]["collect"].append(collected)
        return np.zeros(axis.size)

    monkeypatch.setattr(runner, "_cxr_for_segments", collect)
    grid, _record, bandwidth = _measured_cxr_grid(
        policy, case, {}, (np.array([0.0, 0.0, 1.0]),), 2, 1.0, None, None
    )
    expected, _ = case_cxr_stop_eV(
        case,
        [ResonancePopulation("production lines", collected[0], collected[2], collected[1])],
        start_eV=50.0,
        ceiling_eV=100_000.0,
        truncation_limit=1e-4,
        proxy_safety=2.0,
    )
    assert grid[-1] == expected == bandwidth["stop_eV"]


def test_small_hbn_case_collects_production_lines_before_resolution():
    from pyrite.montecarlo.runner import _transport_case

    sweep = material_sweep("hbn")
    sweep = replace(
        sweep,
        beam=replace(sweep.beam, energy_keV=5000.0),
        cxr_grid_policy=_RESONANCE,
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    result = _transport_case(case, transport_core="lockstep")
    record = result["diagnostic_grid"]
    assert record["measured_bandwidth"]["kinematic"]["n_lines"] > 0
    assert result["E_grid"][-1] < case["line_grid_policy"]["bandwidth"]["stop_eV"]


def test_local_grid_reference_keeps_sinc_spacing_to_ceiling():
    local = np.array([50.0, 50.5, 51.0, 54.0, 57.0])
    reference = reference_axis(local, 100.0, spacing_eV=0.5)
    assert reference[0] == 50.0
    assert reference[-1] == 100.0
    np.testing.assert_allclose(np.diff(reference), 0.5)


def test_light_far_line_spends_the_share_instead_of_widening_the_axis():
    main = _population(5000.0, 1.0, weight=[1.0])
    stray = _population(60_000.0, 1.0, weight=[1e-9])
    alone, _ = resonance_population_stop_eV([main], ceiling_eV=1e6, truncation_limit=1e-5)
    both, summary = resonance_population_stop_eV(
        [main, stray], ceiling_eV=1e6, truncation_limit=1e-5
    )
    assert both < 60_000.0
    assert both == pytest.approx(alone, abs=200.0)
    assert summary["proxy_truncated_fraction"] <= 1e-5


def _lines(E, shift=0.0):
    centres = np.array([1500.0, 2200.0, 3100.0])
    return sum(np.exp(-0.5 * ((E - c - shift) / 30.0) ** 2) for c in centres)


def test_shape_and_counts_agree_for_one_spectrum_on_two_axes():
    """Same density on a fine uniform and a coarse piecewise axis: every shape
    and detected-count deviation is quadrature-small."""
    reference = np.arange(1000.0, 4000.0 + 0.25, 0.5)
    candidate = np.unique(np.r_[np.arange(1000.0, 2000.0, 2.0), np.arange(2000.0, 4000.0 + 1, 3.0)])
    out = shape_and_counts(
        reference, _lines(reference), candidate, _lines(candidate), detector_max_eV=4000.0
    )
    assert out["intrinsic_bins"]["l1_rel"] < 1e-3
    assert abs(out["fwhm_rel"]) < 1e-2
    assert abs(out["eaglexo_counts_rel"]) < 1e-4
    assert abs(out["timepix3"]["counts_rel"]) < 1e-4
    assert out["timepix3"]["l1_rel"] < 1e-3


def test_shape_and_counts_detect_a_shifted_line():
    reference = np.arange(1000.0, 4000.0 + 0.25, 0.5)
    out = shape_and_counts(
        reference,
        _lines(reference),
        reference,
        _lines(reference, shift=60.0),
        detector_max_eV=4000.0,
    )
    assert out["intrinsic_bins"]["l1_rel"] > 0.1
    assert out["timepix3"]["l1_rel"] > 0.05


def test_intrinsic_bins_are_exact_for_a_line_on_a_bin_edge():
    """One sinc**2 line straddling a 100 eV edge, bin-mean on a fine and a 3 eV
    axis: edges snapped to the candidate's cells leave no split artifact (a
    fixed 100 eV lattice misplaces ~17% of the line)."""
    from pyrite.energy_grid.bandwidth_check import _intrinsic_bins
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_axis, sincsq_bin_lineshape

    def density(axis):
        cells, inv_width = bin_axis(axis)
        return np.asarray(
            sincsq_bin_lineshape(
                np.pi / np.array([3.0]),
                np.array([2000.5]),
                np.asarray(cells),
                inv_width,
                exact_widths=None,
            )
        )[0]

    fine, coarse = np.arange(1000.0, 3000.1, 0.5), np.arange(1000.0, 3000.1, 3.0)
    out = _intrinsic_bins(
        fine, density(fine), coarse, density(coarse), bin_eV=100.0, quadrature="bin-mean"
    )
    assert out["l1_rel"] < 1e-6
    assert out["reference_split_bound"] > out["l1_rel"]


def test_dominant_fwhm_matches_a_gaussian():
    from pyrite.energy_grid.bandwidth_check import _dominant_fwhm_eV

    E = np.linspace(0.0, 100.0, 20_001)
    density = np.exp(-0.5 * ((E - 40.0) / 2.0) ** 2) + 0.3 * np.exp(-0.5 * ((E - 70.0) / 1.0) ** 2)
    assert _dominant_fwhm_eV(E, density) == pytest.approx(
        2.0 * np.sqrt(2.0 * np.log(2.0)) * 2.0, rel=1e-6
    )


@pytest.mark.slow
def test_selected_reflections_feed_the_measured_bandwidth_audit(monkeypatch):
    """Selected lines outside the measured population are still audited (#1243).

    Under a measured reflection selection, the resonance population that sets
    the stop is collected over the case's built ``hkl_list`` (4 for h-BN), not
    over the thousands of selected reflections. The spectrum-phase truncation
    audit must therefore see every selection pilot: its line mass then carries
    the whole scored spectrum, ``int S dE * N_e / mass = 1 - (edge shares)``.
    Were the pilots unaudited, the audit would hold no selected line mass and
    the ratio would sit far above one. Selected lines beyond the axis's own
    keep window are scored or charged up to the ceiling's window (#1243), so
    they enter both sides of the ratio.

    Validation: line-grid-resonance-bandwidth
    """
    import pyrite.montecarlo.runner.case_driver as case_driver

    sweep = material_sweep("hbn")
    sweep = replace(
        sweep,
        beam=replace(sweep.beam, energy_keV=100.0),
        cxr_grid_policy=_RESONANCE,
        line_yield_tolerance=1e-3,
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    seen = {}
    check = case_driver.check_cxr_truncation

    def spy(case, audit):
        seen.update(audit)
        return check(case, audit)

    monkeypatch.setattr(case_driver, "check_cxr_truncation", spy)
    result = case_driver.run_case(case, transport_core="lockstep")
    selected = result["reflection_selection"]["radiators"][0]["reflections"]
    assert selected > 100 * len(case["hkl_list"])
    record = result["line_grid_resolved"]["truncation_audit"]
    scored = float(np.trapezoid(result["spec"], result["E_grid"])) * 2  # n_electrons
    ratio = scored / float(seen["line_mass"])
    edges = record["upper_fraction_bound"] + record["lower_fraction_bound"]
    assert 1.0 - edges <= ratio <= 1.0 + 1e-3


def _hbn_case(energy_keV, cxr_grid_policy=None):
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=energy_keV))
    if cxr_grid_policy is not None:
        sweep = replace(sweep, cxr_grid_policy=cxr_grid_policy)
    return build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]


def test_defaulted_measured_bandwidth_falls_back_to_the_explicit_kinematic_payload():
    """The fallback axis is exactly what ``kinematic-ceiling`` would build (#1243)."""
    default = _hbn_case(1000.0)["line_grid_policy"]
    assert default["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY
    explicit = _hbn_case(1000.0, {"bandwidth": KINEMATIC_BANDWIDTH_POLICY})["line_grid_policy"]
    assert kinematic_fallback_payload(default) == explicit
    chosen = _hbn_case(1000.0, _RESONANCE)["line_grid_policy"]
    assert kinematic_fallback_payload(chosen) is None
    assert kinematic_fallback_payload(explicit) is None


def test_defaulted_refusal_rescores_on_the_ceiling_and_explicit_refusal_raises(monkeypatch):
    """A refused default stop keeps its transport and rescores to the ceiling (#1243).

    P3 on qlmc: automatic selection scored higher-|g| reflections above stops
    measured over the built ``hkl_list`` (Si 1 MeV: 3.8e-3 above the edge). A
    vanishing share at the spectrum-phase gate forces that refusal here.

    Validation: line-grid-resonance-bandwidth
    """
    import pyrite.montecarlo.runner.case_driver as case_driver

    check = case_driver.check_cxr_truncation

    def strict(case, audit):
        payload = case["line_grid_policy"]
        bandwidth = {**payload["bandwidth"], "truncation_limit": 1e-300}
        return check({**case, "line_grid_policy": {**payload, "bandwidth": bandwidth}}, audit)

    monkeypatch.setattr(case_driver, "check_cxr_truncation", strict)
    case = _hbn_case(1000.0)
    ceiling = case["line_grid_policy"]["bandwidth"]["stop_eV"]
    with pytest.warns(CxrGridTruncationWarning, match="rescoring on the kinematic-ceiling"):
        result = case_driver.run_case(case, transport_core="lockstep")
    assert result["E_grid"][-1] == pytest.approx(ceiling)
    fallback = result["line_grid_resolved"]["bandwidth_fallback"]
    assert fallback["refused"]["stop_eV"] < ceiling
    assert "truncation_audit" not in result["line_grid_resolved"]
    with pytest.raises(CxrGridToleranceError, match="bandwidth share"):
        case_driver.run_case(_hbn_case(1000.0, _RESONANCE), transport_core="lockstep")


def test_audited_keep_window_reaches_the_kinematic_window():
    """Lines up to the ceiling's window are scored or charged, never skipped (#1243).

    The fresh-context check found selected MoS2 lines between the measured
    axis's 20% pad and the ceiling counted by neither population nor audit.

    Validation: line-grid-resonance-bandwidth
    """
    from pyrite.montecarlo.spectrum.cxr._kernels import _line_keep_window

    grid = np.linspace(100.0, 1100.0, 11)
    assert _line_keep_window(grid) == pytest.approx((-100.0, 1300.0))
    assert _line_keep_window(grid, {"start_eV": 100.0}) == pytest.approx((-100.0, 1300.0))
    audit = {"start_eV": 100.0, "stop_eV": 1100.0, "ceiling_eV": 5100.0}
    # The kinematic axis [100, 5100] keeps resonances below 5100 + 0.2 * 5000.
    assert _line_keep_window(grid, audit) == pytest.approx((-100.0, 6100.0))
    audit["ceiling_eV"] = 1100.0  # at the ceiling: unchanged
    assert _line_keep_window(grid, audit) == pytest.approx((-100.0, 1300.0))
    beyond = {"start_eV": 100.0, "stop_eV": 1100.0, "ceiling_eV": 5100.0}
    _accumulate_edge_truncation(
        beyond, xp.asarray([5500.0]), xp.asarray([np.pi / 10.0]), xp.asarray([1.0])
    )
    assert float(beyond["mass_above"]) == pytest.approx(float(beyond["line_mass"]))


@pytest.mark.parametrize(
    "extra",
    [
        {"channeling": {"plane_hkl": (1, -1, 0)}},
        {"transition_radiation": {"coherence": "off"}},
        {"transport_mode": "mixed"},
    ],
)
def test_unmeasured_line_axis_routes_resolve_the_kinematic_payload(monkeypatch, extra):
    """Spectra the measured stop cannot see keep the kinematic axis (#1243).

    Planar channeling under the default stopped at 2.7 keV (Si (1-10), 17 MeV)
    with 0.99 of the channeling-radiation peak above the axis and a passing,
    empty audit.

    Validation: line-grid-resonance-bandwidth
    """
    from pyrite.montecarlo.runner import cxr_grid

    seen = []
    monkeypatch.setattr(
        cxr_grid, "_resolve_policy_cxr_grid", lambda payload, *a: seen.append(payload) or (None, {})
    )
    case = {**_hbn_case(1000.0), **extra}
    cxr_grid.resolve_cxr_grid(case, None, None, 2, None)
    assert seen[-1] == kinematic_fallback_payload(case["line_grid_policy"])
    chosen = {**_hbn_case(1000.0, _RESONANCE), **extra}
    with pytest.raises(ValueError, match="cannot measure"):
        cxr_grid.resolve_cxr_grid(chosen, None, None, 2, None)


_FORWARD = np.array([np.sin(0.6), 0.0, np.cos(0.6)])  # Laue DBR toward the exit face


def _silicon_case(energy_keV, cxr_grid_policy=None):
    sweep = material_sweep("silicon")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=energy_keV))
    if cxr_grid_policy is not None:
        sweep = replace(sweep, cxr_grid_policy=cxr_grid_policy)
    return build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]


def test_bragg_support_bounds_every_dbr_and_kossel_window_node():
    """The measured stop covers each finite DBR/Kossel deposition window (#1243).

    Both corrections are added after the native truncation audit and are
    exactly zero beyond their ``bragg_window``; a stop at or above the highest
    node therefore adds no unaudited upper loss of either.

    Validation: line-grid-resonance-bandwidth
    """
    from pyrite.montecarlo.runner.diffracted_brem import bragg_support_stop_eV
    from pyrite.montecarlo.spectrum.diffracted_brem import (
        _bragg_energy,
        bragg_reflections,
        bragg_window,
    )

    case = _silicon_case(1000.0)
    bandwidth = case["line_grid_policy"]["bandwidth"]
    start, ceiling = bandwidth["start_eV"], bandwidth["stop_eV"]
    support = bragg_support_stop_eV(case, (_FORWARD,), start, ceiling)
    reflections = bragg_reflections(case, _FORWARD, [start, ceiling])
    assert reflections and support is not None
    s_hat = np.array([0.0, 0.0, 1.0])
    thickness = float(case["thickness_ang"])
    top = 0.0
    for g, chi_g_abs, chi_0 in reflections:
        E_B = _bragg_energy(g, _FORWARD)
        for max_step in (np.inf, 0.05):  # DBR default; a narrow-line Kossel step
            window = bragg_window(thickness, _FORWARD, g, s_hat, E_B, chi_g_abs, chi_0, max_step)
            top = max(top, float(window[-1]))
    assert top <= support <= top + 3.1  # within one default step (<= 1.5 eV) and rounding
    # Outside the Laue geometry (side-on, toward the entrance face) nothing to cover.
    side = np.array([1.0, 0.0, -0.0076])
    assert bragg_support_stop_eV(case, (side,), start, ceiling) is None


def test_measured_stop_rises_to_the_bragg_support_under_the_ceiling():
    """Validation: line-grid-resonance-bandwidth"""
    from pyrite.montecarlo.runner.cxr_grid import _cover_bragg_support
    from pyrite.montecarlo.runner.diffracted_brem import bragg_support_stop_eV

    case = _silicon_case(1000.0)
    bandwidth = case["line_grid_policy"]["bandwidth"]
    start, ceiling = bandwidth["start_eV"], bandwidth["stop_eV"]
    support = bragg_support_stop_eV(case, (_FORWARD,), start, ceiling)
    stop, record = _cover_bragg_support(
        case, (_FORWARD,), start, ceiling, 2000.0, {"stop_eV": 2000.0}
    )
    assert stop == pytest.approx(min(support, ceiling))
    assert record["native_stop_eV"] == 2000.0 and record["stop_eV"] == stop
    unchanged = _cover_bragg_support(
        case, (_FORWARD,), start, ceiling, ceiling, {"stop_eV": ceiling}
    )
    assert unchanged == (ceiling, {"stop_eV": ceiling})
    capped, _ = _cover_bragg_support(case, (_FORWARD,), start, support - 1.0, 2000.0, {})
    assert capped == support - 1.0


def test_measured_grid_cache_key_carries_the_coverage_revision():
    """Old measured grids (no Bragg coverage) are never served from the cache."""
    from pyrite.montecarlo.runner import cxr_grid

    measured = _silicon_case(1000.0)
    key, _ = cxr_grid._policy_cache_key(
        measured["line_grid_policy"], measured, (), observation=False
    )
    kinematic = _silicon_case(1000.0, {"bandwidth": KINEMATIC_BANDWIDTH_POLICY})
    reference, _ = cxr_grid._policy_cache_key(
        kinematic["line_grid_policy"], kinematic, (), observation=False
    )
    assert cxr_grid.MEASURED_COVERAGE_REVISION >= 1 and key != reference


@pytest.mark.parametrize(
    "extra",
    [
        {"channeling": {"plane_hkl": (1, -1, 0)}},
        {"transition_radiation": {"coherence": "off"}},
        {"transport_mode": "mixed"},
    ],
)
def test_observation_resolution_shares_the_unmeasured_axis_fallback(monkeypatch, extra):
    """Multi-direction scoring keeps the kinematic axis too (#1243).

    The fresh-context check found ``resolve_observation_cxr_grid`` bypassing
    the scalar guard and re-measuring channeling/TR on every direction.

    Validation: line-grid-resonance-bandwidth
    """
    from pyrite.montecarlo.runner import cxr_grid

    seen = []
    monkeypatch.setattr(
        cxr_grid,
        "_resolve_policy_grid",
        lambda payload, *a, **k: seen.append(payload) or (None, {}),
    )
    case = {**_hbn_case(1000.0), **extra}
    cxr_grid.resolve_observation_cxr_grid(case, None, (_FORWARD,), 2, None)
    assert seen[-1] == kinematic_fallback_payload(case["line_grid_policy"])
    chosen = {**_hbn_case(1000.0, _RESONANCE), **extra}
    with pytest.raises(ValueError, match="cannot measure"):
        cxr_grid.resolve_observation_cxr_grid(chosen, None, (_FORWARD,), 2, None)


def test_replay_refuses_a_recorded_axis_short_of_todays_requirement():
    """Artifact replay validates its stored line axis before scoring (#1243).

    Validation: line-grid-resonance-bandwidth
    """
    from pyrite.montecarlo.runner.cxr_grid import check_recorded_line_axis
    from pyrite.montecarlo.runner.diffracted_brem import bragg_support_stop_eV

    channeling = {**_hbn_case(1000.0), "channeling": {"plane_hkl": (1, -1, 0)}}
    bandwidth = channeling["line_grid_policy"]["bandwidth"]
    start, ceiling = bandwidth["start_eV"], bandwidth["stop_eV"]
    with pytest.raises(ValueError, match="recorded line axis"):
        check_recorded_line_axis(channeling, np.linspace(start, 0.5 * ceiling, 5), _FORWARD)
    check_recorded_line_axis(channeling, np.linspace(start, ceiling, 5), _FORWARD)

    case = _silicon_case(1000.0)
    bandwidth = case["line_grid_policy"]["bandwidth"]
    start, ceiling = bandwidth["start_eV"], bandwidth["stop_eV"]
    support = bragg_support_stop_eV(case, (_FORWARD,), start, ceiling)
    assert support < ceiling
    with pytest.raises(ValueError, match="recorded line axis"):
        check_recorded_line_axis(case, np.linspace(start, support - 10.0, 5), _FORWARD)
    check_recorded_line_axis(case, np.linspace(start, support, 5), _FORWARD)
    side = np.array([1.0, 0.0, -0.0076])  # no Laue DBR: a short measured axis replays
    check_recorded_line_axis(case, np.linspace(start, 2000.0, 5), side)
    check_recorded_line_axis({"name": "fixed"}, np.linspace(1.0, 2.0, 3), side)


def test_measured_cache_key_separates_mosaic_from_perfect_crystals():
    """A mosaic case's uncovered stop is never served to a perfect crystal (#1243).

    Fresh re-verification 3: ``mosaic_fwhm_rad`` gates the DBR claim, so the
    Bragg-support stop depends on it; Si 100 keV hit a mosaic-first entry at
    84800 eV instead of the 96860 eV support.

    Validation: line-grid-resonance-bandwidth
    """
    from pyrite.montecarlo.runner import cxr_grid

    perfect = _silicon_case(100.0)
    mosaic = {**perfect, "mosaic_fwhm_rad": 1e-3}
    keys = [
        cxr_grid._policy_cache_key(case["line_grid_policy"], case, (), observation=False)[0]
        for case in (perfect, mosaic)
    ]
    assert keys[0] != keys[1]
