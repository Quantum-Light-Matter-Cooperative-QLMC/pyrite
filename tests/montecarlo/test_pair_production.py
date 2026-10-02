"""Pair conversion of coupled hard photons (``pair_production_model``, #275).

Validation: pair-production-sampling, photon-pair-first-interaction
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
from pyrite.montecarlo.transport.pair_production import (
    PAIR_THRESHOLD_EV,
    convert_hard_photons,
    pair_photon_stream_key,
    pair_reduced_energy_density,
    photon_first_interactions,
    sample_pair,
    sample_pair_polar_cosine,
    sample_pair_reduced_energy,
)

MC2 = 0.5 * PAIR_THRESHOLD_EV
ALPHA = 1.0 / 137.035999084
RE2_BARN = (2.8179403262e-13) ** 2 * 1e24
SYMBOL = {6: "C", 14: "Si", 42: "Mo", 82: "Pb"}


def _density_integral(E_eV, Z, upper=None):
    kappa = E_eV / MC2
    lo, hi = 1.0 / kappa, 1.0 - 1.0 / kappa
    return integrate.quad(
        lambda e: float(pair_reduced_energy_density(e, E_eV, Z)),
        lo,
        hi if upper is None else upper,
        limit=200,
        epsabs=0.0,
        epsrel=1e-10,
    )[0]


@pytest.mark.parametrize("Z", sorted(SYMBOL))
def test_bethe_heitler_total_tracks_epdl_nuclear_pair(Z):
    """Eq. 2.85 (eta = 0) integrates to the nuclear-field total it was fitted to.

    PENELOPE-2024 states the residual is appreciable near threshold, 4 % at
    3 MeV and 2 % above 6 MeV; EPDL2025 is an independent evaluation of the
    same Hubbell et al. totals.
    """
    for E_MeV, tolerance in ((1.5, 0.13), (2.0, 0.07), (3.0, 0.045), (5.0, 0.04), (10.0, 0.02)):
        E = E_MeV * 1e6
        sigma_barn = RE2_BARN * ALPHA * Z * Z * 1.0093 * (2.0 / 3.0) * _density_integral(E, Z)
        epdl_barn = photon_cross_sections_ang2(SYMBOL[Z], E)["pair_nuclear"] * 1e8
        assert sigma_barn / epdl_barn == pytest.approx(1.0, abs=tolerance), (E_MeV, Z)


def test_reduced_energy_density_is_symmetric_and_bounded():
    E, Z = 3.0e6, 82
    kappa = E / MC2
    eps = np.linspace(1.0 / kappa + 1e-6, 1.0 - 1.0 / kappa - 1e-6, 101)
    density = pair_reduced_energy_density(eps, E, Z)
    np.testing.assert_allclose(density, density[::-1], rtol=1e-12)
    assert np.all(density >= 0.0) and np.argmax(density) == 50
    outside = [0.5 / kappa, 1.0 - 0.5 / kappa, 0.0, 1.0]
    np.testing.assert_array_equal(pair_reduced_energy_density(outside, E, Z), 0.0)
    with pytest.raises(ValueError, match="above 2 m_e c"):
        sample_pair_reduced_energy(PAIR_THRESHOLD_EV, Z, np.random.default_rng(0))
    with pytest.raises(ValueError, match="Z=1..99"):
        pair_reduced_energy_density(0.5, E, 100)


@pytest.mark.parametrize(("E_MeV", "Z"), [(1.1, 82), (1.5, 6), (3.0, 82), (10.0, 6), (10.0, 82)])
def test_sampled_reduced_energy_matches_eq_2_90(E_MeV, Z):
    """Chi-square of 20 000 composition/rejection draws in 20 equal-mass bins."""
    E = E_MeV * 1e6
    rng = np.random.default_rng(275 + Z)
    samples = np.array([sample_pair_reduced_energy(E, Z, rng) for _ in range(20_000)])
    kappa = E / MC2
    assert samples.min() > 1.0 / kappa and samples.max() < 1.0 - 1.0 / kappa
    grid = np.linspace(1.0 / kappa, 1.0 - 1.0 / kappa, 20_001)
    cdf = integrate.cumulative_trapezoid(pair_reduced_energy_density(grid, E, Z), grid, initial=0)
    cdf /= cdf[-1]
    edges = np.interp(np.linspace(0.0, 1.0, 21), cdf, grid)
    observed, _ = np.histogram(samples, edges)
    assert stats.chisquare(observed).pvalue > 1e-3


def test_polar_cosine_matches_leading_term():
    """Eq. 2.99 inverts ``p(c) ~ (1 - beta c)^-2``; beta -> 0 is isotropic."""
    rng = np.random.default_rng(5)
    for kinetic in (2.0e4, 5.0e5, 4.0e6):
        beta = np.sqrt(kinetic * (kinetic + 2 * MC2)) / (kinetic + MC2)
        cosines = sample_pair_polar_cosine(np.full(20_000, kinetic), rng.random(20_000))
        assert np.all(np.abs(cosines) <= 1.0)

        def cdf(c, beta=beta):
            return (1 / (1 - beta * c) - 1 / (1 + beta)) / (1 / (1 - beta) - 1 / (1 + beta))

        assert stats.kstest(cosines, cdf).pvalue > 1e-3
    np.testing.assert_allclose(sample_pair_polar_cosine(0.0, [0.0, 0.25, 1.0]), [-1.0, -0.5, 1.0])


def test_pair_conserves_energy_and_returns_unit_directions():
    rng = np.random.default_rng(9)
    photon = np.array([0.0, 0.6, 0.8])
    for k in (1.03e6, 2.0e6, 4.5e6):
        for Z in (6, 42, 82):
            pair = sample_pair(k, Z, photon, rng)
            assert pair.electron_eV >= 0.0 and pair.positron_eV >= 0.0
            assert pair.electron_eV + pair.positron_eV + PAIR_THRESHOLD_EV == pytest.approx(
                k, rel=1e-15
            )
            for direction in (pair.electron_direction, pair.positron_direction):
                assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-12)


def _beta(kinetic_eV):
    return np.sqrt(kinetic_eV * (kinetic_eV + 2 * MC2)) / (kinetic_eV + MC2)


def _leading_term_cdf(c, beta):
    return (1 / (1 - beta * c) - 1 / (1 + beta)) / (1 / (1 - beta) - 1 / (1 + beta))


@pytest.mark.parametrize("Z", [6, 82])
def test_sampled_pair_angles_match_leading_term_per_particle(Z):
    """``sample_pair`` end to end: each cosine about the photon follows Eq. 2.99
    at its own particle's beta (probability-integral transform, then KS)."""
    rng = np.random.default_rng(2750 + Z)
    photon = np.array([0.36, -0.48, 0.8])
    e1 = np.array([1.0, 0.0, 0.0]) - 0.36 * photon
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(photon, e1)
    for k in (1.5e6, 4.0e6, 10.0e6):
        pit, azimuth = [], []
        for _ in range(3_000):
            pair = sample_pair(k, Z, photon, rng)
            total = pair.electron_eV + pair.positron_eV + PAIR_THRESHOLD_EV
            assert abs(total - k) <= 2 * np.spacing(k)  # exact up to summation rounding
            for kinetic, direction in (
                (pair.electron_eV, pair.electron_direction),
                (pair.positron_eV, pair.positron_direction),
            ):
                pit.append(_leading_term_cdf(float(direction @ photon), _beta(kinetic)))
            transverse = pair.electron_direction - (pair.electron_direction @ photon) * photon
            azimuth.append(np.arctan2(transverse @ e2, transverse @ e1))
        assert stats.kstest(pit, "uniform").pvalue > 1e-3, k
        assert stats.kstest(np.asarray(azimuth), stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue > (
            1e-3
        )


GEANT4_REFERENCE = (
    Path(__file__).resolve().parents[2] / "checks" / "pair_production_geant4" / "reference"
)


def _merged_two_sample_chi2(a, b):
    """Two-sample chi-square on bins merged left to right until both hold 10."""
    merged, acc = [], np.zeros(2)
    for pair in zip(a, b, strict=True):
        acc += pair
        if acc.min() >= 10:
            merged.append(acc)
            acc = np.zeros(2)
    merged[-1] = merged[-1] + acc
    return stats.chi2_contingency(np.array(merged).T, correction=False).pvalue


@pytest.mark.parametrize("case", ["c_2mev", "c_5mev", "pb_2mev", "pb_5mev"])
def test_sample_pair_matches_geant4_penelope_reference(case):
    """Regression anchor: binned ``x = E_-/(k - 2mc^2)`` and both polar cosines
    against Geant4 11.4.2 ``G4PenelopeGammaConversionModel`` (TestEm5, see
    ``checks/pair_production_geant4``), the same PENELOPE section 2.4 model."""
    import json

    ref = json.loads((GEANT4_REFERENCE / f"{case}_empenelope.json").read_text())
    Z, k = ref["Z"], ref["photon_energy_eV"]
    rng = np.random.default_rng(27_500 + Z + int(k // 1e6))
    photon = np.array([1.0, 0.0, 0.0])
    n = 8_000
    x, cos_minus, cos_plus = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        pair = sample_pair(k, Z, photon, rng)
        x[i] = pair.electron_eV / (k - PAIR_THRESHOLD_EV)
        cos_minus[i] = pair.electron_direction[0]
        cos_plus[i] = pair.positron_direction[0]
    for name, values, edges in (
        ("x", x, ref["x_edges"]),
        ("cos_minus", cos_minus, ref["cos_edges"]),
        ("cos_plus", cos_plus, ref["cos_edges"]),
    ):
        ours, _ = np.histogram(values, edges)
        assert _merged_two_sample_chi2(ours, ref[name]) > 1e-3, name


def _slab(composition, thickness):
    return [(0.0, thickness, composition)]


def _mu(composition, E):
    pair = other = 0.0
    for element, density in composition:
        sigma = photon_cross_sections_ang2(element, E)
        p = sigma["pair_nuclear"] + sigma["pair_electron"]
        pair += density * p
        other += density * (sum(sigma.values()) - p)
    return float(pair), float(other)


def test_first_interaction_depth_and_channel_follow_epdl():
    """Escape is exp(-mu L); depth is truncated exponential; pair share mu_pair/mu."""
    composition = [("Pb", 0.033)]
    E = 4.0e6
    mu_pair, mu_other = _mu(composition, E)
    mu = mu_pair + mu_other
    thickness = 1.0 / mu
    n = 40_000
    rng = np.random.default_rng(1)
    origins = np.zeros((n, 3))
    directions = np.tile([0.0, 0.0, 1.0], (n, 1))
    hit = photon_first_interactions(
        origins, directions, np.full(n, E), rng.random((n, 2)), _slab(composition, thickness)
    )
    interacts = np.isfinite(hit["distance_ang"])
    p_escape = np.exp(-mu * thickness)
    assert stats.binomtest(int(np.count_nonzero(~interacts)), n, p_escape).pvalue > 1e-3
    depth = hit["distance_ang"][interacts]
    assert stats.kstest(depth, lambda s: (1 - np.exp(-mu * s)) / (1 - p_escape)).pvalue > 1e-3
    n_pair = int(np.count_nonzero(hit["pair"]))
    assert stats.binomtest(n_pair, int(interacts.sum()), mu_pair / mu).pvalue > 1e-3
    np.testing.assert_array_equal(hit["layer"][~interacts], -1)
    np.testing.assert_array_equal(hit["element"][~hit["pair"]], -1)


def test_first_interaction_crosses_layers_backwards_and_picks_elements():
    """A photon heading to the entrance face crosses the upper layer last."""
    top, bottom = [("C", 0.11)], [("Pb", 0.02), ("O", 0.06)]
    E = 6.0e6
    mu_top = sum(_mu(top, E))
    pair_b = [
        density
        * (
            photon_cross_sections_ang2(el, E)["pair_nuclear"]
            + photon_cross_sections_ang2(el, E)["pair_electron"]
        )
        for el, density in bottom
    ]
    mu_bottom = sum(_mu(bottom, E))
    L_top, L_bottom = 0.7 / mu_top, 0.5 / mu_bottom
    layers = [(0.0, L_top, top), (L_top, L_top + L_bottom, bottom)]
    n = 40_000
    rng = np.random.default_rng(2)
    origins = np.tile([0.0, 0.0, L_top + L_bottom - 1.0], (n, 1))
    directions = np.tile([0.0, 0.0, -1.0], (n, 1))
    hit = photon_first_interactions(origins, directions, np.full(n, E), rng.random((n, 2)), layers)
    p_bottom = 1 - np.exp(-mu_bottom * (L_bottom - 1.0))
    p_top = np.exp(-mu_bottom * (L_bottom - 1.0)) * (1 - np.exp(-mu_top * L_top))
    observed = [np.sum(hit["layer"] == 1), np.sum(hit["layer"] == 0), np.sum(hit["layer"] == -1)]
    expected = n * np.array([p_bottom, p_top, 1 - p_bottom - p_top])
    assert stats.chisquare(observed, expected).pvalue > 1e-3
    in_bottom = hit["pair"] & (hit["layer"] == 1)
    lead = int(np.count_nonzero(hit["element"][in_bottom] == 0))
    assert stats.binomtest(lead, int(in_bottom.sum()), pair_b[0] / sum(pair_b)).pvalue > 1e-3


def test_first_interaction_depth_follows_layered_optical_depth():
    """An oblique ray through three layers: escape, depth CDF and pair share per layer.

    The depth CDF is ``1 - exp(-tau(s))`` with ``tau`` piecewise linear in the
    path length, conditioned on interacting before the exit face.
    """
    stack = [[("C", 0.11)], [("Pb", 0.02), ("O", 0.06)], [("Si", 0.05)]]
    E = 5.0e6
    mus = [_mu(comp, E) for comp in stack]
    mu = np.array([pair + other for pair, other in mus])
    cos_z = 0.6
    path = np.array([0.6, 0.4, 0.5]) / mu  # path length per layer
    bounds = np.concatenate([[0.0], np.cumsum(path * cos_z)])
    layers = [(bounds[i], bounds[i + 1], comp) for i, comp in enumerate(stack)]
    n = 60_000
    rng = np.random.default_rng(6)
    direction = [np.sqrt(1 - cos_z**2), 0.0, cos_z]
    hit = photon_first_interactions(
        np.zeros((n, 3)), np.tile(direction, (n, 1)), np.full(n, E), rng.random((n, 2)), layers
    )
    s_edges = np.concatenate([[0.0], np.cumsum(path)])
    tau_edges = np.concatenate([[0.0], np.cumsum(mu * path)])

    def tau(s):
        return np.interp(s, s_edges, tau_edges)

    p_escape = np.exp(-tau_edges[-1])
    interacts = np.isfinite(hit["distance_ang"])
    assert stats.binomtest(int(np.count_nonzero(~interacts)), n, p_escape).pvalue > 1e-3
    depth = hit["distance_ang"][interacts]
    assert stats.kstest(depth, lambda s: (1 - np.exp(-tau(s))) / (1 - p_escape)).pvalue > 1e-3
    np.testing.assert_array_equal(
        hit["layer"][interacts], np.searchsorted(s_edges, depth, side="right") - 1
    )
    for index, (mu_pair, mu_other) in enumerate(mus):
        here = hit["layer"] == index
        share = mu_pair / (mu_pair + mu_other)
        assert stats.binomtest(int(hit["pair"][here].sum()), int(here.sum()), share).pvalue > 1e-3


def test_first_interaction_respects_finite_footprint():
    """A lateral photon exits the side face at the half-width."""
    composition = [("W", 0.063)]
    E = 3.0e6
    mu = sum(_mu(composition, E))
    half_width = 0.4 / mu
    n = 20_000
    rng = np.random.default_rng(3)
    hit = photon_first_interactions(
        np.tile([0.0, 0.0, 5.0], (n, 1)),
        np.tile([1.0, 0.0, 0.0], (n, 1)),
        np.full(n, E),
        rng.random((n, 2)),
        _slab(composition, 10.0),
        width_ang=2 * half_width,
        height_ang=2 * half_width,
    )
    escaped = int(np.count_nonzero(~np.isfinite(hit["distance_ang"])))
    assert stats.binomtest(escaped, n, np.exp(-mu * half_width)).pvalue > 1e-3
    assert np.nanmax(np.where(np.isfinite(hit["distance_ang"]), hit["distance_ang"], np.nan)) < (
        half_width
    )


def _photon_rows(n, k_eV, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return {
        "electron_id": rng.integers(0, 4, n),
        "flight_id": np.arange(n),
        "substep_id": np.zeros(n, dtype=np.int64),
        "r_mid": np.column_stack([np.zeros(n), np.zeros(n), rng.uniform(1.0, 9.0, n)]),
        "v_hat": np.tile([0.0, 0.0, 1.0], (n, 1)),
        "L_ang": np.full(n, 0.5),
        "t_end_ang": rng.uniform(0.0, 100.0, n),
        "hard_radiative_k_eV": np.asarray(k_eV, dtype=float),
        "hard_radiative_direction": direction,
    }


def test_photon_streams_are_keyed_and_order_independent():
    """Reordering rows keeps each photon's outcome; threshold photons are skipped."""
    n = 400
    k = np.where(np.arange(n) % 5 == 0, 5.0e5, 3.0e6)
    rows = _photon_rows(n, k)
    # A dense heavy layer so most photons convert inside 10 Angstrom.
    layers = [(0.0, 10.0, [("Pb", 1.0e10)])]
    events, counts = convert_hard_photons(rows, seed=4, parent_track_offset=7, layers=layers)
    assert counts["photons"] == int(np.count_nonzero(k > PAIR_THRESHOLD_EV))
    assert counts["pair"] == events["row"].size > 0
    assert counts["pair"] + counts["other"] + counts["escaped"] == counts["photons"]
    np.testing.assert_allclose(
        events["electron_keV"] + events["positron_keV"] + PAIR_THRESHOLD_EV * 1e-3,
        events["k_eV"] * 1e-3,
        rtol=1e-14,
    )
    assert np.all(events["t_ang"] >= rows["t_end_ang"][events["row"]])
    permutation = np.random.default_rng(1).permutation(n)
    shuffled = {name: value[permutation] for name, value in rows.items()}
    again, _ = convert_hard_photons(shuffled, seed=4, parent_track_offset=7, layers=layers)
    original = np.lexsort((events["photon_ordinal"], events["parent"]))
    moved = np.lexsort((again["photon_ordinal"], again["parent"]))
    for name in ("parent", "photon_ordinal", "electron_keV", "positron_keV", "r_ang"):
        np.testing.assert_array_equal(events[name][original], again[name][moved])
    np.testing.assert_array_equal(permutation[again["row"][moved]], events["row"][original])
    other, _ = convert_hard_photons(rows, seed=5, parent_track_offset=7, layers=layers)
    assert not np.array_equal(other["electron_keV"], events["electron_keV"])
    assert pair_photon_stream_key(4, 7, 0) != pair_photon_stream_key(4, 8, 0)


def test_no_photon_above_threshold_converts_nothing():
    rows = _photon_rows(10, np.full(10, 9.0e5))
    events, counts = convert_hard_photons(
        rows, seed=0, parent_track_offset=0, layers=[(0.0, 10.0, [("Pb", 1.0e10)])]
    )
    assert counts == {"photons": 0, "pair": 0, "other": 0, "escaped": 0}
    assert events["row"].size == 0 and events["r_ang"].shape == (0, 3)


def _cascade(monkeypatch, *, pair_scale, model="penelope-2024", seed=11):
    """Thin silicon, 3 MeV, synthetic BremsLib boosted to emit many MeV photons.

    ``pair_scale`` multiplies EPDL's nuclear pair cross section seen by the
    photon step only, so conversions happen inside a 10 um slab.
    """
    from dataclasses import replace

    from pyrite.materials import CATALOG
    from pyrite.montecarlo import shell_configuration as config
    from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
    from pyrite.montecarlo.transport import pair_production, simulate_trajectories
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")

    def scaled(element, energies):
        sigma = dict(photon_cross_sections_ang2(element, energies))
        sigma["pair_nuclear"] = sigma["pair_nuclear"] * pair_scale
        sigma["pair_electron"] = sigma["pair_electron"] * pair_scale
        return sigma

    monkeypatch.setattr(pair_production, "photon_cross_sections_ang2", scaled)
    arrays = synthetic_bremslib_arrays(t1_MeV=np.array([1e-2, 1e-1, 1.0, 2.0, 5.0]))
    table = prepare_bremslib_table(arrays, atomic_number=14)
    table = replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 3e3,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 3e3,
    )
    return simulate_trajectories(
        3000.0,
        6,
        1.0e5,
        composition=CATALOG.crystal("silicon").composition,
        E_cut_keV=100.0,
        seed=seed,
        energy_model="midpoint",
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=["silicon"],
        secondary_threshold_eV=100_000.0,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"Si": table},
        transport_core="per-electron",
        pair_production_model=model,
    )


def test_cascade_launches_pair_electrons_and_closes_energy(monkeypatch):
    from pyrite.montecarlo.transport.events import check_segment_event_contract
    from pyrite.montecarlo.transport.secondaries import LAUNCH_PAIR, secondary_energy_balance

    with pytest.warns(UserWarning, match="not transported"):
        result = _cascade(monkeypatch, pair_scale=1e9)
    pair = result["pair_production"]
    events, tracks = pair["events"], result["secondary_tracks"]
    assert pair["positrons_transported"] is False
    assert pair["photon_counts"]["pair"] == events["row"].size > 0
    np.testing.assert_allclose(
        events["electron_keV"] + events["positron_keV"] + PAIR_THRESHOLD_EV * 1e-3,
        events["k_eV"] * 1e-3,
        rtol=1e-14,
    )
    # Converted rows really carry the photon, from the recorded parent history.
    np.testing.assert_array_equal(result["hard_radiative_k_eV"][events["row"]], events["k_eV"])
    np.testing.assert_array_equal(result["electron_id"][events["row"]], events["electron_id"])
    launched = events["electron_launched"]
    np.testing.assert_array_equal(launched, events["electron_keV"] > 100.0)
    assert launched.any()
    track = events["electron_track"][launched]
    np.testing.assert_array_equal(events["electron_track"][~launched], -1)
    np.testing.assert_array_equal(tracks["launch_kind"][track], LAUNCH_PAIR)
    np.testing.assert_array_equal(tracks["launch_E_keV"][track], events["electron_keV"][launched])
    np.testing.assert_array_equal(tracks["parent_id"][track], events["parent_track"][launched])
    np.testing.assert_array_equal(tracks["electron_id"][track], events["electron_id"][launched])
    assert np.count_nonzero(tracks["launch_kind"] == LAUNCH_PAIR) == launched.sum()
    check_segment_event_contract(dict(result, electron_id=result["track_id"]))
    terms = secondary_energy_balance(result)
    assert terms["pair_rest_mass_keV"] == pytest.approx(events["row"].size * 1021.99790)
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    per_history = secondary_energy_balance(result, per_history=True)
    np.testing.assert_allclose(per_history["residual_keV"], 0.0, atol=1e-9 * 3000.0)


def test_cascade_pair_mode_is_reproducible_and_inert_without_pairs(monkeypatch):
    from pyrite.montecarlo.transport.secondaries import secondary_energy_balance

    with pytest.warns(UserWarning, match="not transported"):
        first = _cascade(monkeypatch, pair_scale=1e9)
    with pytest.warns(UserWarning, match="not transported"):
        again = _cascade(monkeypatch, pair_scale=1e9)
    for name in ("E_end_keV", "r_mid", "track_id"):
        np.testing.assert_array_equal(first[name], again[name])
    for name, value in first["pair_production"]["events"].items():
        np.testing.assert_array_equal(value, again["pair_production"]["events"][name])
    # With no pair cross section the mode adds nothing: rows equal the mode off.
    inert = _cascade(monkeypatch, pair_scale=0.0)
    off = _cascade(monkeypatch, pair_scale=0.0, model=None)
    assert "pair_production" not in off
    assert inert["pair_production"]["events"]["row"].size == 0
    for name in ("E_end_keV", "r_mid", "v_hat", "track_id", "parent_id", "hard_radiative_k_eV"):
        np.testing.assert_array_equal(inert[name], off[name])
    for name, value in off["secondary_tracks"].items():
        np.testing.assert_array_equal(inert["secondary_tracks"][name], value)
    assert secondary_energy_balance(off).keys() <= secondary_energy_balance(inert).keys()


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"pair_production_model": "penelope-2024"}, "requires secondary_threshold_eV"),
        (
            {"pair_production_model": "bethe", "secondary_threshold_eV": 1e4},
            "pair_production_model must be one of",
        ),
        (
            {"pair_production_model": "penelope-2024", "secondary_threshold_eV": 1e4},
            "requires coupled BremsLib",
        ),
    ],
)
def test_pair_mode_rejects_unsupported_configurations(kw, message):
    from pyrite.materials import CATALOG
    from pyrite.montecarlo import shell_configuration as config
    from pyrite.montecarlo.transport import simulate_trajectories
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    with pytest.raises(ValueError, match=message):
        simulate_trajectories(
            50.0,
            2,
            1.0e4,
            composition=CATALOG.crystal("silicon").composition,
            E_cut_keV=10.0,
            energy_model="midpoint",
            stopping_tables=[resolve_catalog_table("silicon").arrays()],
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=50.0,
            inelastic_materials=["silicon"],
            transport_core="per-electron",
            **kw,
        )


def test_photon_outside_epdl_domain_is_refused():
    with pytest.raises(ValueError, match="outside the EPDL2025"):
        photon_first_interactions(
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0]],
            [2.0e11],
            [[0.5, 0.5]],
            [(0.0, 10.0, [("Pb", 0.03)])],
        )
