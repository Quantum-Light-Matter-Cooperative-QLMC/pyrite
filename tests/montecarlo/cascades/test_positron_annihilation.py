"""Positron annihilation in flight and at rest (#295).

Validation: heitler-annihilation, positron-annihilation-at-rest
"""

import numpy as np
import pytest
from scipy import integrate, stats

from pyrite.montecarlo.transport import annihilation as ann
from pyrite.montecarlo.transport.annihilation import (
    ELECTRON_REST_KEV,
    heitler_cross_section_ang2,
    heitler_zeta_density,
    row_optical_depth,
    sample_at_rest_directions,
    sample_heitler,
    sample_heitler_zeta,
    truncate_in_flight,
)
from tests.helpers.positron_tables import require_positron_tables

_PI_RE2 = np.pi * 2.8179403262e-5**2
_ENERGIES_KEV = (10.0, 1_000.0, 10_000.0)


def _zeta_min(kinetic_keV):
    g = 1.0 + kinetic_keV / ELECTRON_REST_KEV
    return 1.0 / (g + 1.0 + np.sqrt(g * g - 1.0))


@pytest.mark.parametrize("kinetic_keV", (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 1e5))
def test_cross_section_is_the_integral_of_the_dcs(kinetic_keV):
    """Eq. 3.189 equals the Eq. 3.187 DCS integrated over [zeta_min, 1/2]."""
    total, _ = integrate.quad(
        heitler_zeta_density, _zeta_min(kinetic_keV), 0.5, args=(kinetic_keV,), epsabs=0.0
    )
    assert total == pytest.approx(1.0, rel=1e-9)


def test_cross_section_limits():
    # Slow positrons: sigma ~ pi r_e^2 / beta (the 1/v law).
    for kinetic in (1e-3, 1e-2):
        g = 1.0 + kinetic / ELECTRON_REST_KEV
        beta = np.sqrt(1.0 - 1.0 / g**2)
        assert heitler_cross_section_ang2(kinetic) * beta == pytest.approx(_PI_RE2, rel=1e-4)
    # Fast positrons: sigma -> pi r_e^2 (ln 2g - 1) / g (Dirac).
    g = 1.0 + 1e9 / ELECTRON_REST_KEV
    dirac = _PI_RE2 * (np.log(2.0 * g) - 1.0) / g
    assert heitler_cross_section_ang2(1e9) == pytest.approx(dirac, rel=1e-4)
    # PENELOPE-2024 Fig. 3.20: about 0.1 b per electron at 1 MeV.
    assert 0.05e-8 < heitler_cross_section_ang2(1_000.0) < 0.2e-8
    assert np.all(np.diff(heitler_cross_section_ang2(np.geomspace(1.0, 1e6, 50))) < 0.0)


def test_rejection_envelope_bounds_g():
    """max g = g^2 + 2g - 1 at 1/(g + 1); PENELOPE's g(zeta_min) is below it."""
    for kinetic in (1.0, 100.0, 1_000.0, 1e5):
        g = 1.0 + kinetic / ELECTRON_REST_KEV
        zmin = _zeta_min(kinetic)
        u = np.linspace(zmin, 1.0 - zmin, 200_001)
        values = -((g + 1.0) ** 2) * u + (g * g + 4.0 * g + 1.0) - 1.0 / u
        assert values.min() >= -1e-9
        assert values.max() == pytest.approx(g * g + 2.0 * g - 1.0, rel=1e-8)
        assert values.max() >= values[0]


@pytest.mark.parametrize("kinetic_keV", _ENERGIES_KEV)
def test_sampled_zeta_follows_the_heitler_dcs(kinetic_keV):
    rng = np.random.Generator(np.random.Philox(key=int(kinetic_keV) + 7))
    zeta = np.array([sample_heitler_zeta(kinetic_keV, rng) for _ in range(20_000)])
    zmin = _zeta_min(kinetic_keV)
    assert zeta.min() >= zmin and zeta.max() <= 0.5
    grid = np.linspace(zmin, 0.5, 4001)
    pdf = heitler_zeta_density(grid, kinetic_keV)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    result = stats.kstest(zeta, lambda z: np.interp(z, grid, cdf))
    assert result.pvalue > 1e-3, result


@pytest.mark.parametrize("kinetic_keV", (10.0, 300.0, 1_000.0, 10_000.0))
def test_in_flight_photons_conserve_energy_and_momentum(kinetic_keV):
    rng = np.random.Generator(np.random.Philox(key=3))
    direction = np.array([0.3, -0.4, 0.866])
    direction /= np.linalg.norm(direction)
    g = 1.0 + kinetic_keV / ELECTRON_REST_KEV
    momentum = ELECTRON_REST_KEV * np.sqrt(g * g - 1.0) * direction
    for _ in range(200):
        energies, dirs = sample_heitler(kinetic_keV, direction, rng)
        assert energies.sum() == pytest.approx(kinetic_keV + 2.0 * ELECTRON_REST_KEV, rel=1e-15)
        assert energies[0] <= energies[1]
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(
            energies @ dirs, momentum, atol=1e-9 * (kinetic_keV + ELECTRON_REST_KEV)
        )


def test_at_rest_photons_are_back_to_back_and_isotropic():
    rng = np.random.Generator(np.random.Philox(key=11))
    dirs = np.array([sample_at_rest_directions(rng) for _ in range(20_000)])
    np.testing.assert_array_equal(dirs[:, 0], -dirs[:, 1])
    np.testing.assert_allclose(np.linalg.norm(dirs[:, 0], axis=1), 1.0, rtol=1e-12)
    cos_theta = dirs[:, 0, 2]
    phi = np.arctan2(dirs[:, 0, 1], dirs[:, 0, 0])
    assert stats.kstest(cos_theta, stats.uniform(-1.0, 2.0).cdf).pvalue > 1e-3
    assert stats.kstest(phi, stats.uniform(-np.pi, 2.0 * np.pi).cdf).pvalue > 1e-3


def _straight_tracks(n_tracks, n_rows, E_start, E_end, L_ang):
    """Synthetic rows: each track goes straight down +z, energy linear in depth."""
    energies = np.linspace(E_start, E_end, n_rows + 1)
    track = np.repeat(np.arange(n_tracks), n_rows)
    substep = np.tile(np.arange(n_rows), n_tracks)
    z_mid = (substep + 0.5) * L_ang
    zeros = np.zeros(track.size)
    v = np.tile([0.0, 0.0, 1.0], (track.size, 1))
    return {
        "electron_id": track,
        "flight_id": np.zeros(track.size, dtype=np.int64),
        "substep_id": substep,
        "layer": np.zeros(track.size, dtype=np.int64),
        "E_start_keV": energies[substep],
        "E_end_keV": energies[substep + 1],
        "L_ang": np.full(track.size, L_ang),
        "r_mid": np.column_stack([zeros, zeros, z_mid]),
        "v_hat": v,
        "t_start_ang": substep * L_ang * 1.2,
        "t_end_ang": (substep + 1) * L_ang * 1.2,
        "E_repr_keV": 0.5 * (energies[substep] + energies[substep + 1]),
        "event_kind": np.where(substep == n_rows - 1, 7, 1).astype(np.int8),
        "hard_channel": np.full(track.size, -1, dtype=np.int64),
        "hard_W_keV": zeros.copy(),
    }


def test_in_flight_fraction_matches_the_optical_depth():
    """P(annihilate) = 1 - exp(-int N Z sigma ds) over a slowing-down track."""
    n_tracks, density = 40_000, 0.7
    rows = _straight_tracks(n_tracks, 10, 400.0, 20.0, 1.0e7)
    depth = row_optical_depth(
        rows["E_start_keV"][:10], rows["E_end_keV"][:10], rows["L_ang"][:10], density
    ).sum()
    # Independent quadrature of the same integral over the track.
    exact, _ = integrate.quad(lambda s: heitler_cross_section_ang2(400.0 - 380.0 * s), 0.0, 1.0)
    assert depth == pytest.approx(density * 1.0e8 * exact, rel=1e-8)
    p = 1.0 - np.exp(-depth)
    tau = -np.log1p(-np.random.Generator(np.random.Philox(key=5)).random(n_tracks))
    _, hit = truncate_in_flight(rows, tau, [density], tuple(rows))
    cut = hit["row"] >= 0
    sigma = np.sqrt(n_tracks * p * (1.0 - p))
    assert abs(cut.sum() - n_tracks * p) < 5.0 * sigma
    # Depth of the cut follows the survival law: the optical depth at the cut is tau.
    z = hit["r_ang"][cut, 2]
    E_ann = hit["E_keV"][cut]
    np.testing.assert_allclose(E_ann, 400.0 - 380.0 * z / 1.0e8, rtol=1e-9)
    reached = np.array(
        [
            density
            * 1.0e8
            * integrate.quad(
                lambda s: heitler_cross_section_ang2(400.0 - 380.0 * s), 0.0, zz / 1.0e8
            )[0]
            for zz in z[:200]
        ]
    )
    np.testing.assert_allclose(reached, tau[cut][:200], rtol=1e-7)


def test_truncation_cuts_the_crossing_row_and_drops_the_rest():
    from pyrite.montecarlo.transport.events import EVENT_ANNIHILATION

    rows = _straight_tracks(3, 4, 300.0, 100.0, 1.0e6)
    rows["hard_channel"][1] = 2
    rows["hard_W_keV"][1] = 5.0
    density = 1.0
    depth = row_optical_depth(rows["E_start_keV"], rows["E_end_keV"], rows["L_ang"], density)
    # Track 0 survives; track 1 crosses mid row 1; track 2 crosses in row 0.
    tau = np.array([1e9, depth[4] + 0.5 * depth[5], 0.25 * depth[8]])
    out, hit = truncate_in_flight(rows, tau, [density], tuple(rows))
    np.testing.assert_array_equal(hit["row"], [-1, 5, 8])
    np.testing.assert_array_equal(out["electron_id"], [0, 0, 0, 0, 1, 1, 2])
    assert (
        out["event_kind"][-1] == EVENT_ANNIHILATION and out["event_kind"][5] == EVENT_ANNIHILATION
    )
    assert out["hard_channel"][1] == 2  # untouched rows keep their payload
    f = out["L_ang"][5] / 1.0e6
    assert 0.45 < f < 0.55
    assert out["E_end_keV"][5] == pytest.approx(hit["E_keV"][1])
    assert out["r_mid"][5, 2] + 0.5 * out["L_ang"][5] == pytest.approx(hit["r_ang"][1, 2])
    assert out["t_end_ang"][5] == pytest.approx(rows["t_start_ang"][5] + f * 1.2e6)


def test_stream_keys_separate_budget_and_emission():
    a = ann.annihilation_stream_keys(7, [3, 3, 4], [0, 1, 0], in_flight=True)
    b = ann.annihilation_stream_keys(7, [3, 3, 4], [0, 1, 0], in_flight=False)
    assert len(set(a.tolist()) | set(b.tolist())) == 6
    np.testing.assert_array_equal(
        a, ann.annihilation_stream_keys(7, [3, 3, 4], [0, 1, 0], in_flight=True)
    )


def test_scored_photons_escape_or_interact():
    layers = [(0.0, 1.0e8, [("Si", 0.05)])]
    keys = ann.annihilation_stream_keys(1, np.arange(400), np.zeros(400), in_flight=False)
    kinds = np.tile([ann.ANNIHILATION_AT_REST, ann.ANNIHILATION_IN_FLIGHT], 200)
    origins = np.tile([0.0, 0.0, 5.0e7], (400, 1))
    v = np.tile([0.0, 0.0, 1.0], (400, 1))
    photons = ann.score_annihilation_photons(kinds, np.full(400, 2000.0), v, origins, keys, layers)
    assert photons["k_keV"].size == 800
    rest = np.repeat(kinds == ann.ANNIHILATION_AT_REST, 2)
    np.testing.assert_array_equal(photons["k_keV"][rest], ELECTRON_REST_KEV)
    totals = np.bincount(photons["event"], weights=photons["k_keV"])
    np.testing.assert_allclose(totals[kinds == 1], 2000.0 + 2.0 * ELECTRON_REST_KEV, rtol=1e-14)
    absorbed = photons["absorbed"]
    assert absorbed.any() and (~absorbed).any()
    assert np.all(np.isfinite(photons["distance_ang"][absorbed]))
    assert np.all(photons["layer"][~absorbed] == -1)


# --- the cascade -----------------------------------------------------------


@pytest.fixture
def _positron_tables():
    require_positron_tables()


def _thick(monkeypatch, *, sigma_scale=1.0, **kw):
    from .test_pair_production import _cascade

    monkeypatch.setattr(
        ann, "heitler_cross_section_ang2", lambda E: sigma_scale * heitler_cross_section_ang2(E)
    )
    return _cascade(monkeypatch, pair_scale=1e9, positron_transport=True, thickness_ang=1.0e6, **kw)


@pytest.mark.usefixtures("_positron_tables")
def test_cascade_annihilates_every_stopped_positron_and_closes_energy(monkeypatch):
    from pyrite.montecarlo.transport.events import (
        EVENT_ANNIHILATION,
        check_segment_event_contract,
    )
    from pyrite.montecarlo.transport.secondaries import (
        FATE_AT_REST,
        FATE_ESCAPED,
        FATE_IN_FLIGHT,
        FATE_SUBTHRESHOLD,
        secondary_energy_balance,
    )

    result = _thick(monkeypatch, sigma_scale=300.0)
    pair = result["pair_production"]
    events, photons = pair["events"], pair["annihilation_photons"]
    fate = events["positron_fate"]
    assert set(np.unique(fate)) <= {FATE_SUBTHRESHOLD, FATE_AT_REST, FATE_IN_FLIGHT, FATE_ESCAPED}
    assert np.any(fate == FATE_IN_FLIGHT) and np.any(fate == FATE_AT_REST)
    check_segment_event_contract(dict(result, electron_id=result["track_id"]))

    # One ANNIHILATION row ends each in-flight track, at the recorded point.
    flight = np.flatnonzero(fate == FATE_IN_FLIGHT)
    for index in flight:
        rows = np.flatnonzero(result["track_id"] == events["positron_track"][index])
        last = rows[np.lexsort((result["substep_id"][rows], result["flight_id"][rows]))][-1]
        assert result["event_kind"][last] == EVENT_ANNIHILATION
        assert np.count_nonzero(result["event_kind"][rows] == EVENT_ANNIHILATION) == 1
        assert result["E_end_keV"][last] == events["annihilation_keV"][index]
        end = result["r_mid"][last] + 0.5 * result["L_ang"][last] * result["v_hat"][last]
        np.testing.assert_allclose(end, events["annihilation_r_ang"][index], atol=1e-6)

    # Two photons per non-escaping positron, 511 keV each at rest.
    annihilating = np.flatnonzero(fate != FATE_ESCAPED)
    np.testing.assert_array_equal(np.unique(photons["event"]), annihilating)
    rest = np.isin(photons["event"], np.flatnonzero(fate != FATE_IN_FLIGHT))
    np.testing.assert_array_equal(photons["k_keV"][rest], ELECTRON_REST_KEV)
    totals = np.bincount(photons["event"], weights=photons["k_keV"], minlength=fate.size)
    np.testing.assert_allclose(
        totals[flight], events["annihilation_keV"][flight] + 2.0 * ELECTRON_REST_KEV, rtol=1e-14
    )

    terms = secondary_energy_balance(result)
    assert "positron_rest_pending_keV" not in terms
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    per_history = secondary_energy_balance(result, per_history=True)
    np.testing.assert_allclose(per_history["residual_keV"], 0.0, atol=1e-9 * 3000.0)
    counts = result["secondaries"]["counts_per_generation"]
    assert sum(c.get("n_annihilated", 0) for c in counts) == flight.size


@pytest.mark.usefixtures("_positron_tables")
def test_cascade_annihilation_is_reproducible(monkeypatch):
    first = _thick(monkeypatch, sigma_scale=300.0)
    again = _thick(monkeypatch, sigma_scale=300.0)
    for name in ("E_end_keV", "r_mid", "track_id", "event_kind"):
        np.testing.assert_array_equal(first[name], again[name])
    a, b = (r["pair_production"]["annihilation_photons"] for r in (first, again))
    for name in ("k_keV", "direction", "distance_ang"):
        np.testing.assert_array_equal(a[name], b[name])
