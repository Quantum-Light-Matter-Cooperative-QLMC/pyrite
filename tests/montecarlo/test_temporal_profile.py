"""Limiting-case gates for the opt-in temporal intensity profile I(t) (#292).

Validation: temporal-intensity-profile
"""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG, beta_from_Ee
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.spectrum.lines import temporal_profile_for
from pyrite.montecarlo.spectrum.lines._temporal import (
    TemporalProfile,
    add_boxes,
    segment_arrival_times,
)
from pyrite.montecarlo.transport import C_ANG_PER_FS

E_GRID = np.arange(700.0, 1500.0, 0.25)
N_HAT = np.array([1.0, 0.0, 0.01])
KWARGS = {"crystal": "hopg", "hkl_list": [(0, 0, 2)], "B_ang2": 0.8, "n_hat": N_HAT}
BETA = float(beta_from_Ee(30e3))


def _segments(starts, length=2000.0, *, elec_id=None, t0=None, Ne=None):
    """Collinear straight flights along +z, one row each, in a thick slab."""
    starts = np.asarray(starts, dtype=float)
    count = starts.size
    elec_id = np.zeros(count, dtype=int) if elec_id is None else np.asarray(elec_id)
    return {
        "r_mid": np.column_stack([np.zeros(count), np.zeros(count), 10.0 + starts + 0.5 * length]),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, length),
        "E_keV": np.full(count, 30.0),
        "t_ang": starts / BETA,
        "t0_ang": np.zeros(count) if t0 is None else np.asarray(t0, dtype=float),
        "elec_id": elec_id,
        "layer": np.zeros(count, dtype=int),
        "Ne": int(elec_id.max()) + 1 if Ne is None else Ne,
        "thickness_ang": 1.0e6,
        "crystal_width_ang": 100.0,
        "crystal_height_ang": 100.0,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _run(segments, *, coherent, grid=E_GRID, **kwargs):
    profile = temporal_profile_for(segments, grid, KWARGS["n_hat"] / np.linalg.norm(N_HAT))
    spec = mc_spectrum(segments, grid, coherent=coherent, temporal=profile, **KWARGS, **kwargs)
    out = profile.result()
    return spec, out["t_fs"], out["intensity"]


def _moments(t, intensity):
    mass = intensity.sum()
    mean = (t * intensity).sum() / mass
    return mass, mean, np.sqrt(((t - mean) ** 2 * intensity).sum() / mass)


def test_add_boxes_conserves_mass_and_support():
    profile = TemporalProfile(t_start_ang=0.0, dt_ang=1.0, E_start_eV=0.0, dE_eV=1.0, n=50)
    buf = profile.buffer()
    add_boxes(
        profile,
        buf,
        np.array([10.3, 20.0, 30.0]),
        np.array([2.6, 0.2, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert buf.sum() * profile.dt_ang == pytest.approx(6.0, rel=1e-12)
    # The wide box spans [7.7, 12.9]: interior bins at height 1 / 5.2.
    np.testing.assert_allclose(buf[9:12], 1.0 / 5.2, rtol=1e-12)
    assert buf[7] == 0.0 and buf[14] == 0.0


@pytest.mark.parametrize("coherent", [False, True])
def test_parseval_matches_spectrum(coherent):
    spec, t, intensity = _run(_segments([0.0]), coherent=coherent)
    dt = t[1] - t[0]
    yield_t = intensity.sum() * dt
    yield_E = spec.sum() * (E_GRID[1] - E_GRID[0])
    assert yield_E > 0.0
    # Coherent is band-limited to the same axis (quadrature only); the
    # incoherent box carries the whole line, including tails off the axis.
    assert yield_t == pytest.approx(yield_E, rel=2e-3 if coherent else 2e-2)


def test_single_flight_is_doppler_compressed_box_at_arrival_time():
    segments = _segments([0.0])
    n_hat = N_HAT / np.linalg.norm(N_HAT)
    d, t_L = segment_arrival_times(segments, n_hat)
    D = 1.0 - BETA * n_hat[2]
    duration_fs = D * t_L[0] / C_ANG_PER_FS
    _, t, box = _run(segments, coherent=False)
    _, _, coh = _run(segments, coherent=True)
    for intensity in (box, coh):
        _, mean, rms = _moments(t, intensity)
        assert mean == pytest.approx(d[0] / C_ANG_PER_FS, abs=2 * (t[1] - t[0]))
        assert rms == pytest.approx(duration_fs / np.sqrt(12.0), rel=0.05)
    # Same pulse from the two routes up to the coherent band limit.
    assert np.abs(box - coh).sum() / box.sum() < 0.1


def test_two_flights_give_two_separated_pulses():
    gap = 20000.0
    segments = _segments([0.0, gap])
    _, t, coh = _run(segments, coherent=True)
    _, _, box = _run(segments, coherent=False)
    n_hat = N_HAT / np.linalg.norm(N_HAT)
    d, _ = segment_arrival_times(segments, n_hat)
    for intensity in (coh, box):
        early = intensity[t < d.mean() / C_ANG_PER_FS].sum()
        late = intensity[t >= d.mean() / C_ANG_PER_FS].sum()
        assert early == pytest.approx(late, rel=0.05)
        assert intensity[np.argmin(np.abs(t - d.mean() / C_ANG_PER_FS))] < 1e-2 * intensity.max()


def test_bunch_offsets_spread_the_envelope():
    rng = np.random.default_rng(3)
    n_e = 40
    sigma_ang = 3.0 * C_ANG_PER_FS
    t0 = rng.normal(0.0, sigma_ang, n_e)
    segments = _segments(np.zeros(n_e), elec_id=np.arange(n_e), t0=t0)
    segments["initial_t0_ang"] = t0
    segments["initial_r_ang"] = np.zeros((n_e, 3))
    _, t, coh = _run(segments, coherent=True, longitudinal_rms_fs=3.0)
    _, _, box = _run(segments, coherent=False)
    _, _, single_rms = _moments(*_run(_segments([0.0]), coherent=False)[1:])
    expected = np.sqrt(single_rms**2 + np.var(t0 / C_ANG_PER_FS))
    for intensity in (coh, box):
        mass, _, rms = _moments(t, intensity)
        assert rms == pytest.approx(expected, rel=0.05)
    # Long bunch: the cross-electron term vanishes, so both policies agree.
    assert _moments(t, coh)[0] == pytest.approx(_moments(t, box)[0], rel=0.05)


def test_temporal_does_not_change_the_spectrum():
    segments = _segments([0.0, 500.0])
    for coherent in (False, True):
        plain = mc_spectrum(segments, E_GRID, coherent=coherent, **KWARGS)
        with_t, _, _ = _run(segments, coherent=coherent)
        np.testing.assert_allclose(with_t, plain, rtol=1e-6, atol=1e-12 * plain.max())


def test_window_cap_refuses_oversized_grid():
    with pytest.raises(ValueError, match="time samples"):
        temporal_profile_for(_segments([0.0]), E_GRID, N_HAT, max_samples=8)


def test_energy_grid_conjugate_to_time_grid():
    profile = temporal_profile_for(_segments([0.0]), E_GRID, N_HAT)
    assert profile.n * profile.dt_ang * profile.dE_eV == pytest.approx(
        2.0 * np.pi * HBARC_EV_ANG, rel=1e-12
    )
