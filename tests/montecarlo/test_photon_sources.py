"""Unattenuated weighted continuous photon source and ordinary-hit scoring (#341)."""

import numpy as np
import pytest

from pyrite.montecarlo.transport.photon_sources import (
    sample_bremslib_continuum,
    sample_continuous_photons,
)


def _segments():
    return {
        "r_mid": np.array([[0.0, 0.0, 10.0], [20.0, 0.0, 30.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "L_ang": np.array([10.0, 20.0]),
        "elec_id": np.array([0, 1]),
        "E_keV": np.array([20.0, 30.0]),
        "E_repr_keV": np.array([15.0, 25.0]),
    }


def _sample(segs, rate, n=1000, **kw):
    return sample_continuous_photons(
        segs,
        rate,
        n_samples=n,
        n_primaries=2,
        seed=341,
        cutoff_eV=1000.0,
        ceiling_eV=40_000.0,
        **kw,
    )


def test_constant_emissivity_normalization_and_full_sphere():
    # The first row has 14 keV available; the second has 24 keV. The exact
    # integral uses both *path lengths* and energy ranges, no detector cone.
    intensity = 2.0e-8
    batch = _sample(_segments(), lambda i, e, d: np.full(i.size, intensity), 10000)
    integrated = intensity * 4 * np.pi * (10 * 14_000 + 20 * 24_000)
    assert batch.weight.sum() == pytest.approx(integrated, rel=2.0e-15)
    assert np.any(batch.direction[:, 2] < 0) and np.any(batch.direction[:, 2] > 0)
    np.testing.assert_allclose(np.linalg.norm(batch.direction, axis=1), 1, atol=1e-15)
    assert batch.segment_index.min() == 0 and batch.segment_index.max() == 1
    assert len(np.unique(batch.event_id)) == 10_000
    assert np.all((batch.energy_eV > 1000) & (batch.energy_eV <= 25_000))
    for row in [0, 1]:
        selected = batch.segment_index == row
        r = batch.position_ang[selected] - _segments()["r_mid"][row]
        segment_ray = _segments()["v_hat"][row]
        np.testing.assert_allclose(np.cross(r, segment_ray), 0, atol=1e-12)
        assert np.max(np.abs(r @ segment_ray)) <= _segments()["L_ang"][row] / 2


def test_energy_angle_correlations_are_evaluated_jointly():
    # Integral of c*(1 + a*(k-kmin)/dE * mu) over isotropic angles is just c.
    # A proposal that only used one observation direction would not be unbiased.
    def intensity(index, energy, direction):
        upper = np.array([15_000.0, 25_000.0])[index]
        return 1.0e-8 * (1.0 + 0.8 * (energy - 1000.0) / (upper - 1000.0) * direction[:, 2])

    out = _sample(_segments(), intensity, 70_000)
    expected = 1.0e-8 * 4 * np.pi * (10 * 14_000 + 20 * 24_000)
    assert out.weight.sum() == pytest.approx(expected, rel=0.007)
    assert (out.weight > 0).all()


def _mev_segments():
    # Representative energies 5 and 0.2 MeV; directions +z and +x.
    return dict(_segments(), E_keV=np.array([5000.0, 200.0]), E_repr_keV=np.array([5000.0, 200.0]))


def _sample_mev(rate, n, proposal, seed=341):
    return sample_continuous_photons(
        _mev_segments(),
        rate,
        n_samples=n,
        n_primaries=2,
        seed=seed,
        cutoff_eV=1000.0,
        ceiling_eV=5.0e6,
        proposal=proposal,
    )


def _forward_inverse_k(index, energy, direction):
    # j = c / k * (1 - beta^2) / (1 - beta mu)^2: integrates to c * 4pi * ln(upper/cutoff).
    beta = np.array([0.9957, 0.6953])[index]
    axis = _segments()["v_hat"][index]
    mu = np.sum(axis * direction, axis=1)
    return 1.0e-6 / energy * (1 - beta**2) / (1 - beta * mu) ** 2


def test_bremsstrahlung_proposal_is_unbiased_with_full_support():
    upper = np.array([5.0e6, 2.0e5])
    log_mass = (10.0 * np.log(upper[0] / 1e3) + 20.0 * np.log(upper[1] / 1e3)) * 4 * np.pi
    flat = _sample_mev(lambda i, e, d: np.full(i.size, 1e-9), 200_000, "bremsstrahlung")
    exact_flat = 1e-9 * 4 * np.pi * (10 * (upper[0] - 1e3) + 20 * (upper[1] - 1e3))
    assert flat.weight.sum() == pytest.approx(exact_flat, rel=0.02)
    assert np.any(flat.direction[:, 2] < -0.99) and np.any(flat.energy_eV > 4.0e6)
    np.testing.assert_allclose(np.linalg.norm(flat.direction, axis=1), 1, atol=1e-12)

    peaked = _sample_mev(_forward_inverse_k, 20_000, "bremsstrahlung")
    assert peaked.weight.sum() == pytest.approx(1.0e-6 * log_mass, rel=0.01)


def test_bremsstrahlung_proposal_cures_heavy_tailed_weights():
    # Same integrand, same trial count: the uniform/isotropic proposal leaves
    # most of the estimate on a few trials; the mixture spreads it.
    def top_share(batch):
        w = np.sort(batch.weight)[::-1]
        return w[: w.size // 1000].sum() / w.sum()

    uniform = _sample_mev(_forward_inverse_k, 20_000, "uniform")
    mixture = _sample_mev(_forward_inverse_k, 20_000, "bremsstrahlung")
    assert top_share(uniform) > 0.2
    assert top_share(mixture) < 0.01


def test_bremsstrahlung_proposal_replays_and_rejects_unknown():
    one = _sample_mev(_forward_inverse_k, 500, "bremsstrahlung")
    two = _sample_mev(_forward_inverse_k, 500, "bremsstrahlung")
    for x, y in zip(one.track_args, two.track_args, strict=True):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="proposal must be one of"):
        _sample_mev(_forward_inverse_k, 1, "isotropic")


def test_rng_replay_empty_and_zero_rate():
    one = _sample(_segments(), lambda i, e, d: np.full(i.size, 1e-8))
    two = _sample(_segments(), lambda i, e, d: np.full(i.size, 1e-8))
    for x, y in zip(one.track_args, two.track_args, strict=True):
        np.testing.assert_array_equal(x, y)
    empty = _sample(_segments(), lambda i, e, d: np.zeros(i.size))
    assert empty.energy_eV.size == empty.weight.size == 0
    none = _sample(_segments(), lambda i, e, d: np.zeros(i.size), n=0)
    assert none.position_ang.shape == (0, 3)
    assert len(none.track_args) == 6
    no_support = dict(_segments(), E_repr_keV=np.array([0.5, 0.9]))
    assert _sample(no_support, lambda i, e, d: np.ones(i.size)).weight.size == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"n_primaries": 0}, "positive"),
        ({"n_samples": -1}, "nonnegative"),
        ({"cutoff_eV": 0.0}, "positive"),
        ({"ceiling_eV": 1000.0}, "exceed"),
        ({"source_kind": -1}, "nonnegative"),
    ],
)
def test_invalid_options(change, match):
    opts = dict(n_samples=20, n_primaries=2, seed=1, cutoff_eV=1000.0, ceiling_eV=40_000.0)
    opts.update(change)
    with pytest.raises(ValueError, match=match):
        sample_continuous_photons(_segments(), lambda i, e, d: np.ones(i.size), **opts)


def test_invalid_segment_geometry_or_rate():
    with pytest.raises(ValueError, match="unit vectors"):
        _sample(dict(_segments(), v_hat=np.ones((2, 3))), lambda i, e, d: np.ones(i.size))
    with pytest.raises(ValueError, match="nonnegative"):
        _sample(dict(_segments(), L_ang=np.array([10.0, -20.0])), lambda i, e, d: np.ones(i.size))
    with pytest.raises(ValueError, match="nonnegative rates"):
        _sample(_segments(), lambda i, e, d: -np.ones(i.size))
    with pytest.raises(ValueError, match="shape"):
        _sample(_segments(), lambda i, e, d: np.ones(1))


def test_tracker_normalizes_weighted_hits_once():
    from pyrite.montecarlo.transport.photon_histories import (
        PhotonAperture,
        detected_spectrum,
        ledger_residual_eV,
        track_photon_histories,
    )

    # Zero material: only geometric aperture acceptance. Weight is for the
    # *whole electron batch*; detected_spectrum divides by Ne exactly once.
    batch = _sample(_segments(), lambda i, e, d: np.full(i.size, 1e-6), 10_000)
    aperture = PhotonAperture((0.0, 0.0, 1e6), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1e9, 1e9)
    histories = track_photon_histories(
        *batch.track_args,
        [],
        seed=11,
        photon_cutoff_eV=1000.0,
        electron_cutoff_eV=1000.0,
        source_kind=batch.source_kind,
        aperture=aperture,
    )
    np.testing.assert_allclose(ledger_residual_eV(histories.ledger), 0.0, atol=1e-7)
    hits = histories.exits["detected"].astype(bool)
    weighted_hits = histories.exits["weight"][hits].sum()
    spectrum = detected_spectrum(histories, [1000.0, 40_000.0], 2)
    assert (spectrum * 39_000.0 * 2).sum() == pytest.approx(weighted_hits)
    assert weighted_hits > 0


def test_vacuum_aperture_score_matches_analytic_solid_angle():
    from pyrite.montecarlo.runner.photon_emission import _score_exits
    from pyrite.montecarlo.transport.photon_histories import (
        PhotonAperture,
        track_photon_histories,
    )

    # A square with half-width equal to its axial distance subtends
    # 4 atan(1/sqrt(3)) = 2*pi/3 sr: exactly 1/6 of isotropic emission.
    # Source extent / distance < 1e-6, negligible next to binomial noise.
    count, acceptance, rate = 30_000, 1 / 6, 1e-8
    segments = dict(_segments(), r_mid=np.zeros((2, 3)))
    batch = _sample(segments, lambda i, e, d: np.full(i.size, rate), count)
    aperture = PhotonAperture((0, 0, 1e8), (1, 0, 0), (0, 1, 0), 1e8, 1e8)
    histories = track_photon_histories(
        *batch.track_args,
        [],
        seed=342,
        photon_cutoff_eV=1000,
        electron_cutoff_eV=1000,
        source_kind=batch.source_kind,
        aperture=aperture,
    )
    total, direct, redistributed, _, outside = _score_exits(
        histories,
        np.array([1000, 40_000]),
        2,
    )
    expected = rate * (10 * 14_000 + 20 * 24_000) * (2 * np.pi / 3) / (2 * 39_000)
    relative_six_sigma = 6 * np.sqrt((1 - acceptance) / (count * acceptance))
    assert total[0] == pytest.approx(expected, rel=relative_six_sigma)
    np.testing.assert_array_equal(direct, total)
    np.testing.assert_array_equal(redistributed, [0])
    assert outside == 0


def test_bremslib_source_matches_sdcs_energy_integral():
    from pyrite._backend import xp

    if xp is not np:
        pytest.skip("CPU-only BremsLib source reference")
    from pyrite.montecarlo.spectrum.brem_bremslib import (
        bremslib_segment_state,
        evaluate_bremslib,
        prepare_bremslib_table,
        stage_bremslib_table,
    )
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    rows = _segments()
    rows["E_repr_keV"] = np.array([50.0, 50.0])
    sample = sample_bremslib_continuum(
        rows,
        [[("C", 0.1)]],
        {"C": table},
        n_samples=70_000,
        n_primaries=2,
        seed=45,
        cutoff_eV=5_000.0,
        ceiling_eV=30_000.0,
    )
    staged = stage_bremslib_table(table)
    state = bremslib_segment_state(staged, np.array([50.0]))
    k = np.linspace(5_000.0, 30_000.0, 501)
    dsigma = np.asarray(evaluate_bremslib(staged, state, k))[0]
    # 1D energy integral of angle-integrated SDCS, with same source path.
    expected = np.trapezoid(dsigma, k) * (10.0 + 20.0) * 0.1 * 1e16
    assert sample.weight.sum() == pytest.approx(expected, rel=0.045)
    assert len(np.unique(sample.event_id)) == sample.energy_eV.size
    assert set(sample.history_id) == {0, 1}


def test_bremslib_missing_table_and_material_assignment_fail_closed():
    from pyrite._backend import xp

    if xp is not np:
        pytest.skip("CPU-only BremsLib source reference")
    rows = _segments()
    with pytest.raises(ValueError, match="segment_layer is required"):
        sample_bremslib_continuum(
            rows,
            [[("C", 0.1)], [("Si", 0.05)]],
            {},
            n_samples=1,
            n_primaries=2,
            seed=1,
            cutoff_eV=1000.0,
            ceiling_eV=10_000.0,
        )
    with pytest.raises(ValueError, match="table missing"):
        sample_bremslib_continuum(
            rows,
            [[("C", 0.1)]],
            {},
            n_samples=1,
            n_primaries=2,
            seed=1,
            cutoff_eV=1000.0,
            ceiling_eV=10_000.0,
        )
