"""Deterministic line-window seeds (issue #101).

Kinematic seeds are checked against the closed-form resonance on synthetic
flights, edge seeds against the Chantler table the attenuation model reads, and
the provider registry against a synthetic non-PXR component standing in for
#103 transition-radiation fringes.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyrite._grid_semantics import validate_backend_coordinates
from pyrite._line_windows import FeatureSeed, build_window_plan
from pyrite.materials.atomic import load_henke
from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    reciprocal_g_vector,
    refractive_index,
)
from pyrite.montecarlo.spectrum import line_seeds
from pyrite.montecarlo.spectrum.line_seeds import (
    CHARACTERISTIC_SOURCE,
    EDGE_MIN_F2_RATIO,
    EDGE_SOURCE,
    KINEMATIC_SOURCE,
    SeedContext,
    absorption_edge_seeds,
    characteristic_line_seeds,
    collect_feature_seeds,
    kinematic_line_seeds,
    register_seed_provider,
)
from pyrite.montecarlo.transport import beta_from_keV

_CRYSTAL = next(key for key, info in CRYSTALS.items() if {e for e, _ in info["basis"]} == {"C"})
_N_HAT = np.array([np.sin(np.deg2rad(60.0)), 0.0, np.cos(np.deg2rad(60.0))])


def _flights(energies_keV, lengths_ang, directions):
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return {
        "E_keV": np.asarray(energies_keV, dtype=float),
        "L_ang": np.asarray(lengths_ang, dtype=float),
        "v_hat": directions,
        "elec_id": np.zeros(len(energies_keV), dtype=int),
    }


def _resonance_eV(energy_keV, direction, hkl):
    g_vector, _ = reciprocal_g_vector(hkl, CRYSTALS[_CRYSTAL]["lattice"])
    beta = float(beta_from_keV(energy_keV))
    v = beta * np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return HBARC_EV_ANG * float(v @ g_vector) / (1.0 - float(v @ _N_HAT))


def _in_medium_denominator(energy_keV, direction, hkl):
    """``1 - Re n(E) v.n`` at the root of ``E = hbar c v.g / (1 - Re n(E) v.n)``.

    Solved with pointwise ``refractive_index`` rather than the kernels' table.
    """
    g_vector, _ = reciprocal_g_vector(hkl, CRYSTALS[_CRYSTAL]["lattice"])
    beta = float(beta_from_keV(energy_keV))
    v = beta * np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    v_dot_g, v_dot_n = float(v @ g_vector), float(v @ _N_HAT)

    def n_re(energy_eV):
        return float(np.real(refractive_index(_CRYSTAL, np.array([energy_eV]))[0]))

    vacuum = HBARC_EV_ANG * v_dot_g / (1.0 - v_dot_n)
    root = brentq(
        lambda e: e * (1.0 - n_re(e) * v_dot_n) - HBARC_EV_ANG * v_dot_g,
        0.9 * vacuum,
        1.1 * vacuum,
        xtol=1e-9,
    )
    return root, 1.0 - n_re(root) * v_dot_n


def test_band_seeds_centre_on_the_in_medium_root_the_kernels_use():
    hkl = (0, 0, 2)
    g_vector, _ = reciprocal_g_vector(hkl, CRYSTALS[_CRYSTAL]["lattice"])
    segments = _flights([30.0], [3000.0], [g_vector])
    (seed,), _ = kinematic_line_seeds(
        segments,
        _N_HAT,
        crystal=_CRYSTAL,
        hkl_list=[hkl],
        feature_width_eV=0.8,
        composition=[("C", 0.11)],
        band_eV=(10.0, 2600.0),
    )
    root, _ = _in_medium_denominator(30.0, g_vector, hkl)
    vacuum = _resonance_eV(30.0, g_vector, hkl)
    # Tabulated Re n puts the kernels' root within a few percent of the shift.
    assert abs(seed.centre_eV - root) < 0.05 * abs(root - vacuum)
    assert abs(root - vacuum) > 0.02  # eV; the in-medium shift at this geometry


def test_single_flight_window_is_centred_on_the_closed_form_resonance():
    hkl = (0, 0, 2)
    g_vector, _ = reciprocal_g_vector(hkl, CRYSTALS[_CRYSTAL]["lattice"])
    segments = _flights([100.0], [2000.0], [g_vector])
    seeds, summary = kinematic_line_seeds(
        segments, _N_HAT, crystal=_CRYSTAL, hkl_list=[hkl], feature_width_eV=0.8
    )
    (seed,) = seeds
    assert seed.source == KINEMATIC_SOURCE and seed.label == "(0 0 2)"
    assert seed.centre_eV == pytest.approx(_resonance_eV(100.0, g_vector, hkl), rel=1e-12)
    assert seed.below_eV == pytest.approx(2.0 * 0.8)
    assert seed.above_eV == pytest.approx(2.0 * 0.8)
    assert seed.spacing_eV == pytest.approx(0.1)
    assert summary["dropped_reflections"] == []


def test_window_covers_the_weighted_band_and_ignores_a_negligible_tail():
    hkl = (0, 0, 2)
    g_vector, _ = reciprocal_g_vector(hkl, CRYSTALS[_CRYSTAL]["lattice"])
    tilted = g_vector / np.linalg.norm(g_vector) + np.array([0.3, 0.0, 0.0])
    heavy = [g_vector] * 50 + [tilted] * 50
    rare = [g_vector + np.array([0.0, 0.9 * np.linalg.norm(g_vector), 0.0])]
    segments = _flights([100.0] * 101, [1000.0] * 100 + [1.0], heavy + rare)
    (seed,), _ = kinematic_line_seeds(
        segments,
        _N_HAT,
        crystal=_CRYSTAL,
        hkl_list=[hkl],
        feature_width_eV=0.5,
        aliased_weight_limit=1.0e-3,
        tail_widths=0.0,
    )
    band = sorted([_resonance_eV(100.0, g_vector, hkl), _resonance_eV(100.0, tilted, hkl)])
    assert seed.centre_eV - seed.below_eV == pytest.approx(band[0], rel=1e-12)
    assert seed.centre_eV + seed.above_eV == pytest.approx(band[1], rel=1e-12)
    rare_resonance = _resonance_eV(100.0, rare[0], hkl)
    assert not seed.centre_eV - seed.below_eV <= rare_resonance <= seed.centre_eV + seed.above_eV


def test_a_reflection_that_never_radiates_is_reported_not_seeded():
    g_vector, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS[_CRYSTAL]["lattice"])
    segments = _flights([100.0], [2000.0], [g_vector])
    seeds, summary = kinematic_line_seeds(
        segments, _N_HAT, crystal=_CRYSTAL, hkl_list=[(0, 0, 2), (0, 0, -2)], feature_width_eV=1.0
    )
    assert [seed.label for seed in seeds] == ["(0 0 2)"]
    assert summary["dropped_reflections"] == ["(0 0 -2)"]
    assert summary["dropped_weight_fraction"] == 0.0


def test_mosaic_spread_widens_the_window():
    hkl = (0, 0, 2)
    g_vector, _ = reciprocal_g_vector(hkl, CRYSTALS[_CRYSTAL]["lattice"])
    segments = _flights([100.0], [2000.0], [g_vector + np.array([0.2, 0.0, 0.0])])
    (perfect,), _ = kinematic_line_seeds(
        segments, _N_HAT, crystal=_CRYSTAL, hkl_list=[hkl], feature_width_eV=0.5, tail_widths=1.0
    )
    (mosaic,), _ = kinematic_line_seeds(
        segments,
        _N_HAT,
        crystal=_CRYSTAL,
        hkl_list=[hkl],
        feature_width_eV=0.5,
        tail_widths=1.0,
        mosaic_fwhm_rad=np.deg2rad(3.0),
        mosaic_nodes=3,
    )
    assert perfect.below_eV + perfect.above_eV == pytest.approx(1.0)
    assert mosaic.below_eV + mosaic.above_eV > 2.0


def test_edge_anchors_bracket_the_chantler_jump_not_the_nominal_edge():
    seeds, summary = absorption_edge_seeds(["C"], 100.0, 1000.0, tables=("chantler",))
    native, _f1, f2 = load_henke("C")
    anchors = sorted(seed.centre_eV for seed in seeds if seed.anchor)
    assert [seed.label for seed in seeds] == ["C K", "C K above"]
    assert all(seed.source == EDGE_SOURCE for seed in seeds)
    below, above = np.searchsorted(native, anchors)
    assert above == below + 1
    near = np.flatnonzero((native > 270.0) & (native < 300.0))
    assert f2[above] / f2[below] == pytest.approx(np.max(f2[near[1:]] / f2[near[:-1]]))
    assert 284.2 not in anchors
    assert summary["skipped"] == []
    assert absorption_edge_seeds(["C"], 1000.0, 2000.0)[0] == []


def test_a_secondary_shell_beside_a_stronger_jump_gets_its_own_anchors():
    # Se L2 (1474 eV) sits 2.8% above L3; Chantler smears the L3 jump over
    # several brackets, and L2 must anchor on its own step, not L3's tail.
    seeds, _ = absorption_edge_seeds(["Se"], 1000.0, 2000.0, tables=("chantler",))
    native, _f1, f2 = load_henke("Se")
    centres = {seed.label: seed.centre_eV for seed in seeds}
    assert set(centres) == {"Se L3", "Se L3 above", "Se L2", "Se L2 above"}
    assert centres["Se L3"] == pytest.approx(1433.9, rel=5e-3)
    assert centres["Se L2"] == pytest.approx(1474.3, rel=5e-3)
    below = int(np.searchsorted(native, centres["Se L2"]))
    assert f2[below + 1] / f2[below] >= EDGE_MIN_F2_RATIO


@pytest.mark.parametrize("table", ["chantler", "epdl"])
def test_edge_bracket_is_a_single_pair_of_nodes_in_the_plan(table):
    seeds, _ = absorption_edge_seeds(["C"], 100.0, 1000.0, tables=(table,))
    grid = build_window_plan(100.0, 1000.0, 3.0, seeds).coordinates()
    below, above = sorted(seed.centre_eV for seed in seeds)
    i = int(np.flatnonzero(grid == below)[0])
    assert grid[i + 1] == above


def test_epdl_edge_anchors_straddle_the_exact_discontinuity():
    """The escape mu jumps exactly at the EPDL edge (C K 288 eV), a few eV from
    Chantler's; its anchor pair is centred on it, so the bin boundary is the edge."""
    seeds, _ = absorption_edge_seeds(["C"], 100.0, 1000.0)
    centres = {seed.label: seed.centre_eV for seed in seeds}
    assert set(centres) == {"C K", "C K above", "C K (EPDL)", "C K (EPDL) above"}
    assert 0.5 * (centres["C K (EPDL)"] + centres["C K (EPDL) above"]) == pytest.approx(288.0)
    assert centres["C K (EPDL)"] < 288.0 < centres["C K (EPDL) above"]
    with pytest.raises(ValueError, match="subset"):
        absorption_edge_seeds(["C"], 100.0, 1000.0, tables=("henke",))


def test_characteristic_windows_follow_the_emitted_lines():
    seeds, _ = characteristic_line_seeds([[("C", 0.1)]], samples_per_feature=8)
    by_label = {seed.label: seed for seed in seeds}
    assert all(seed.source == CHARACTERISTIC_SOURCE for seed in seeds)
    ka1 = by_label["C Ka1"]
    assert ka1.centre_eV == pytest.approx(277.0, abs=0.5)
    fwhm = ka1.spacing_eV * 8
    assert ka1.below_eV == pytest.approx(10.0 * fwhm)
    assert characteristic_line_seeds([[("C", 0.1)]], relaxation_cutoff_eV=1000.0)[0] == []


def _context():
    g_vector, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS[_CRYSTAL]["lattice"])
    return SeedContext(
        case={"crystal": _CRYSTAL, "hkl_list": [(0, 0, 2)], "composition": [("C", 0.11)]},
        segments=_flights([30.0], [3000.0], [g_vector]),
        n_hat=_N_HAT,
        electron_limit=None,
        start_eV=10.0,
        stop_eV=2600.0,
        feature_width_eV=1.0,
    )


def test_default_providers_build_a_valid_float32_plan_for_a_carbon_case():
    seeds, summaries = collect_feature_seeds(_context(), line_seeds.DEFAULT_SEED_PROVIDERS)
    assert {seed.source for seed in seeds} == set(line_seeds.DEFAULT_SEED_PROVIDERS)
    assert summaries[KINEMATIC_SOURCE]["seeds"] == 1
    plan = build_window_plan(10.0, 2600.0, 3.0, seeds)
    grid = plan.coordinates()
    assert plan.windowed
    assert validate_backend_coordinates(grid, dtype=np.float32) > 0.0


def test_a_registered_non_pxr_component_contributes_windows(monkeypatch):
    monkeypatch.setattr(line_seeds, "_PROVIDERS", dict(line_seeds._PROVIDERS))

    def fringes(context):
        centre = 0.5 * (context.start_eV + context.stop_eV)
        return [
            FeatureSeed("transition-radiation", f"fringe {k}", centre + 40.0 * k, 5.0, 5.0, 0.05)
            for k in range(3)
        ], {"fringes": 3}

    register_seed_provider("transition-radiation", fringes)
    with pytest.raises(ValueError, match="already registered"):
        register_seed_provider("transition-radiation", fringes)
    seeds, summaries = collect_feature_seeds(_context(), ["transition-radiation"])
    assert summaries == {"transition-radiation": {"fringes": 3, "seeds": 3}}
    plan = build_window_plan(10.0, 2600.0, 3.0, seeds)
    assert sum(spacing <= 0.05 for _, _, spacing in plan.pieces) == 3


def test_unknown_or_misattributed_providers_are_refused(monkeypatch):
    monkeypatch.setattr(line_seeds, "_PROVIDERS", dict(line_seeds._PROVIDERS))
    with pytest.raises(ValueError, match="unknown seed provider"):
        collect_feature_seeds(_context(), ["no-such-provider"])
    register_seed_provider(
        "liar", lambda context: ([FeatureSeed("other", "x", 100.0, 1.0, 1.0, 0.1)], {})
    )
    with pytest.raises(ValueError, match="source is 'liar'"):
        collect_feature_seeds(_context(), ["liar"])


def test_kinematic_summary_reports_the_narrowest_feature_not_only_the_quantile():
    """The spacing resolves the eps-quantile width; the summary exposes the floor.

    Issue #101. Nothing guarantees the quantile width tracks the narrowest
    feature, so a run has to be able to see how many nodes actually land across
    the narrowest one. On these cases the two coincide, which is the evidence
    that the ``samples_per_feature`` heuristic means what it says.
    """
    _seeds, summary = collect_feature_seeds(_context(), [line_seeds.KINEMATIC_SOURCE])
    kinematic = summary[line_seeds.KINEMATIC_SOURCE]

    g_vector, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS[_CRYSTAL]["lattice"])
    t_L = 3000.0 / float(beta_from_keV(30.0))
    # The kernels' a_width carries the in-medium denominator.
    _root, denominator = _in_medium_denominator(30.0, g_vector, (0, 0, 2))
    expected = 2.0 * np.pi * HBARC_EV_ANG / (denominator * t_L)

    assert kinematic["narrowest_feature_width_eV"] == pytest.approx(expected, rel=1e-6)
    assert kinematic["samples_at_narrowest"] == pytest.approx(expected / kinematic["spacing_eV"])
