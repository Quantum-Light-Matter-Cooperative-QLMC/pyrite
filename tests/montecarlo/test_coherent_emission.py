"""Numerical limiting-case gates for opt-in coherent segment summation."""

import numpy as np
import pytest

from pyrite._backend import REAL
from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    beta_from_Ee,
    reciprocal_g_vector,
)
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.transport import C_ANG_PER_FS
from tests.helpers import scaled_rtol

E_GRID = np.arange(700.0, 1500.0)

KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": np.array([1.0, 0.0, 0.1]),
}

RTOL = max(1e-12, 100.0 * float(np.finfo(REAL).eps))
# The batched (n_seg, N_g) path intentionally reassociates a handful of float
# operations relative to the legacy per-hkl reference.  Give that cross-path
# comparison the same order-of-eps latitude as the chunk-invariance gate while
# retaining a peak-scaled absolute floor for bins near an exact zero.
BATCH_RTOL = max(1e-10, 500.0 * float(np.finfo(REAL).eps))
ATOL = 1e-8


def _blend_rtol(sensitivity):
    """rtol for a decoherence-blend reconstruction at the backend's REAL.

    A blend ``(1-F) G + F C`` is a convex combination of two spectra the kernel
    also produces, so the reconstruction error is not the combination's own
    rounding -- it is the error in the WEIGHT. F is a smooth function of a
    phase the kernel evaluates in the working precision, and a relative
    perturbation ``eps`` of omega moves it by ``|dF| <= sensitivity * eps``;
    since G and C are both O(peak), that absolute weight error is also the
    fractional error of the blended result. *sensitivity* is therefore
    ``|dF / (domega/omega)|`` for the specific F under test, evaluated at the
    grid's high end where it is largest. fp64 keeps the historical 1e-9.
    """
    return scaled_rtol(1e-9, eps_multiple=sensitivity)


def _assert_batch_close(actual, reference):
    peak = float(max(np.max(np.abs(actual)), np.max(np.abs(reference))))
    assert peak > 0.0
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=BATCH_RTOL,
        atol=BATCH_RTOL * 1e-2 * peak,
    )


def _segments(count=1):
    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 10.0,
        "crystal_height_ang": 10.0,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _straight_flight_segments(*, split: bool, reference: bool = False, length: float = 40.0):
    """One constant-velocity flight, optionally split and paired with a reference."""
    beta = float(beta_from_Ee(30e3))
    starts = np.array([0.0, 0.5 * length]) if split else np.array([0.0])
    lengths = np.full(starts.size, length / starts.size)
    elec_id = np.zeros(starts.size, dtype=int)
    if reference:
        starts = np.append(starts, 0.0)
        lengths = np.append(lengths, length)
        elec_id = np.append(elec_id, 1)
    segments = _segments(starts.size)
    segments.update(
        r_mid=np.column_stack(
            [
                np.zeros(starts.size),
                np.zeros(starts.size),
                10.0 + starts + 0.5 * lengths,
            ]
        ),
        L_ang=lengths,
        t_ang=starts / beta,
        elec_id=elec_id,
        Ne=2 if reference else 1,
        thickness_ang=100.0,
        crystal_width_ang=10.0,
        crystal_height_ang=10.0,
    )
    return segments


def _constant_velocity_field(segments, energy_eV):
    """Independent centered-segment integral, up to one common amplitude."""
    n_hat = np.asarray(KWARGS["n_hat"], dtype=float)
    n_hat /= np.linalg.norm(n_hat)
    _, g_norm = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    beta = float(beta_from_Ee(30e3))
    omega = np.asarray(energy_eV) / HBARC_EV_ANG
    omega_res = beta * g_norm / (1.0 - beta * n_hat[2])
    fields = np.zeros(omega.shape, dtype=complex)
    for r_mid, length, t_start in zip(
        segments["r_mid"], segments["L_ang"], segments["t_ang"], strict=True
    ):
        duration = length / beta
        t_mid = t_start + 0.5 * duration
        detuning = 0.5 * (1.0 - beta * n_hat[2]) * (omega - omega_res)
        finite_time = duration * np.sinc(detuning * duration / np.pi)
        phase = omega * (t_mid - n_hat @ r_mid) - g_norm * r_mid[2]
        fields += finite_time * np.exp(1j * phase)
    return fields


def test_single_segment_coherent_equals_incoherent_self_term(monkeypatch):
    from pyrite.montecarlo.spectrum.lines import _setup

    # Constant-index limit: full delta(E) has a derivative Jacobian, separately
    # pinned by test_dispersive_formation_integral_has_the_derivative_jacobian.
    monkeypatch.setattr(
        _setup,
        "refractive_index",
        lambda crystal, energy, *args: np.ones_like(energy, dtype=complex),
    )
    segments = _segments()

    incoherent = mc_spectrum(segments, E_GRID, coherent=False, **KWARGS)
    coherent = mc_spectrum(segments, E_GRID, coherent=True, **KWARGS)
    # Parseval gives the same energy yield in this constant-index limit.
    # Full delta(E) instead carries the derivative Jacobian documented and
    # anchored under xray-in-medium-resonance.
    # Validation: coherent-formation-absorption
    assert np.max(incoherent) > 0.0
    np.testing.assert_allclose(coherent.sum(), incoherent.sum(), rtol=1e-6)
    peak = max(np.max(np.abs(coherent)), np.max(np.abs(incoherent)))
    np.testing.assert_allclose(coherent, incoherent, rtol=RTOL, atol=peak * 2e-5)


def test_coherent_decoherence_blend_matches_reference_formula():
    """Two electrons with distinct longitudinal offsets: the coherent output
    must equal the closed-form blend (1-F)*sum_e|S_e|^2 + F*|sum_e S_e|^2.
    Both reference terms come from mc_spectrum itself on ALREADY-validated
    (offset-free, decoherence-inactive) sub-cases: the fully-coherent flat
    term is mc_spectrum on the same two electrons with t0_ang=0 (the
    degenerate limit locked in by test_identical_in_phase_electrons_reach_
    n_squared_limit), and the intra-electron floor is the SUM of mc_spectrum
    on each electron alone (Ne=1, always trivially self-coherent, per
    test_single_segment_coherent_equals_incoherent_self_term). F is the
    empirical characteristic function of the actual t0 offsets, computed
    independently here.

    Validation: coherent-inter-electron-decoherence
    """
    energy_grid = np.arange(700.0, 1500.0, 2.0)
    t0_values = np.array([137.0, -412.0])  # Ang, c=1 -- O(1/omega)-scale, arbitrary

    def _no_footprint(segs):
        # decoherence_active rejects the finite-footprint branch (out of
        # scope, see the rejection test below); an apples-to-apples
        # comparison also needs the SAME footprint setting on every one of
        # this test's three mc_spectrum calls, active or not.
        segs.update(crystal_width_ang=None, crystal_height_ang=None)
        return segs

    active = _no_footprint(_segments(2))
    active.update(t0_ang=t0_values, initial_t0_ang=t0_values, initial_r_ang=np.zeros((2, 3)))
    actual = mc_spectrum(active, energy_grid, coherent=True, **KWARGS)

    # mc_spectrum divides its row sum by Ne exactly once, at the very end --
    # AFTER the (1-F)*grouped + F*flat blend, not per term. So the two
    # reference terms must be un-normalized back to that same RAW (pre-/Ne)
    # scale before blending: flat_ref's own Ne=2 call already divided by 2
    # (undo it); grouped_ref's two Ne=1 calls each divided by 1, a no-op, so
    # their sum is already the raw sum_e|S_e|^2.
    ne_total = active["Ne"]
    flat_raw = (
        mc_spectrum(_no_footprint(_segments(2)), energy_grid, coherent=True, **KWARGS) * ne_total
    )
    grouped_raw = sum(
        mc_spectrum(_no_footprint(_segments(1)), energy_grid, coherent=True, **KWARGS)
        for _ in range(2)
    )

    omega = energy_grid / HBARC_EV_ANG
    chi = np.mean(np.exp(1j * omega[:, None] * t0_values[None, :]), axis=1)
    F = np.abs(chi) ** 2

    expected = ((1.0 - F) * grouped_raw + F * flat_raw) / ne_total
    peak = float(np.max(np.abs(expected)))
    assert peak > 0.0
    # F = |<exp(i omega t0)>|^2, so dF = 2|chi||dchi| <= 2 |omega t0| eps: the
    # sensitivity is twice the largest coherent phase in radians (313 here).
    sensitivity = 2.0 * float(np.max(omega)) * float(np.max(np.abs(t0_values)))
    np.testing.assert_allclose(actual, expected, rtol=_blend_rtol(sensitivity), atol=peak * 1e-12)


@pytest.mark.parametrize("sinc_cutoff", [None, 4.0], ids=["exact", "windowed"])
def test_coherent_decoherence_blend_holds_on_the_per_hkl_route(sinc_cutoff):
    """Same blend on the per-(reflection, orientation) ``_accumulate`` loop,
    which owns its own reduction separate from the batched path the test above
    exercises. Reached here through the grooved-escape branch, which keeps the
    test on the compatibility route with and without sinc windowing.

    LONGITUDINAL offsets only: the groove's escape distance depends on the
    lateral emission point, so a transverse offset would move S_e's amplitude
    as well as its phase -- the same amplitude/phase coupling the
    finite-footprint branch is excluded for (see the validation doc's limits).

    Validation: coherent-inter-electron-decoherence
    """
    from pyrite.montecarlo.geometry import tilted_geometry
    from pyrite.montecarlo.groove import blazed_groove_spec

    tilt = np.deg2rad(45.0)
    _, n_hat = tilted_geometry(np.pi / 2, tilt, np.pi)
    kwargs = {
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.8,
        "n_hat": n_hat,
        "theta_obs_rad": np.pi / 2,
        "groove": blazed_groove_spec(
            spacing_ang=2.0e4,
            theta_obs_rad=np.pi / 2,
            tilt_polar_rad=tilt,
            tilt_azim_rad=np.pi,
        ),
    }
    energy_grid = np.arange(700.0, 1500.0, 2.0)
    t0_values = np.array([137.0, -412.0])

    def _flat(count):
        segs = _segments(count)
        segs.update(crystal_width_ang=None, crystal_height_ang=None)
        return segs

    active = _flat(2)
    active.update(t0_ang=t0_values, initial_t0_ang=t0_values, initial_r_ang=np.zeros((2, 3)))
    actual = mc_spectrum(active, energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **kwargs)

    ne_total = active["Ne"]
    flat_raw = (
        mc_spectrum(_flat(2), energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **kwargs)
        * ne_total
    )
    grouped_raw = sum(
        mc_spectrum(_flat(1), energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **kwargs)
        for _ in range(2)
    )
    omega = energy_grid / HBARC_EV_ANG
    F = np.abs(np.mean(np.exp(1j * omega[:, None] * t0_values[None, :]), axis=1)) ** 2

    expected = ((1.0 - F) * grouped_raw + F * flat_raw) / ne_total
    peak = float(np.max(np.abs(expected)))
    assert peak > 0.0
    # F = |<exp(i omega t0)>|^2, so dF = 2|chi||dchi| <= 2 |omega t0| eps: the
    # sensitivity is twice the largest coherent phase in radians (313 here).
    sensitivity = 2.0 * float(np.max(omega)) * float(np.max(np.abs(t0_values)))
    np.testing.assert_allclose(actual, expected, rtol=_blend_rtol(sensitivity), atol=peak * 1e-12)


def test_coherent_decoherence_inactive_by_default():
    """No initial_t0_ang/initial_r_ang keys (every pre-existing fixture) ->
    decoherence_active is False and the result is bit-for-bit the old,
    already-validated fully-coherent path -- a direct regression guard on
    the coherent-inter-electron-decoherence change itself."""
    segments = _segments(2)
    assert "initial_t0_ang" not in segments and "initial_r_ang" not in segments
    coherent = mc_spectrum(segments, E_GRID, coherent=True, **KWARGS)
    np.testing.assert_array_equal(
        coherent, mc_spectrum(_segments(2), E_GRID, coherent=True, **KWARGS)
    )


def test_long_gaussian_bunch_supports_finite_footprint():
    """A regular 200 fs bunch fully removes inter-electron terms on this
    X-ray grid. The remaining per-electron floor must retain each sampled
    transverse position's finite-prism escape attenuation.

    Validation: coherent-inter-electron-decoherence
    """
    segments = _segments(2)
    positions = np.array([[4.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
    segments.update(
        r_mid=positions,
        t0_ang=np.array([137.0, -412.0]),
        initial_t0_ang=np.array([137.0, -412.0]),
        initial_r_ang=np.array([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
    )
    assert segments["crystal_width_ang"] is not None
    assert segments["crystal_height_ang"] is not None
    actual = mc_spectrum(
        segments,
        E_GRID,
        coherent=True,
        longitudinal_rms_fs=200.0,
        **KWARGS,
    )
    grouped_raw = sum(
        mc_spectrum(
            {**_segments(1), "r_mid": position[None, :]},
            E_GRID,
            coherent=True,
            **KWARGS,
        )
        for position in positions
    )
    np.testing.assert_allclose(actual, grouped_raw / 2.0, rtol=1e-12, atol=0.0)


def test_finite_footprint_partially_coherent_longitudinal_blend():
    """Arrival-time averaging remains exact when transverse position affects
    amplitude: retain each sampled finite-footprint field and blend only its
    independent longitudinal cross terms.

    Validation: finite-footprint-longitudinal-decoherence
    """
    energy_grid = np.arange(700.0, 1500.0, 2.0)
    longitudinal_rms_fs = 1.0e-3
    segments = _segments(2)
    positions = np.array([[4.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
    segments.update(
        r_mid=positions,
        t0_ang=np.array([137.0, -412.0]),
        initial_t0_ang=np.array([137.0, -412.0]),
        initial_r_ang=np.array([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
    )
    actual = mc_spectrum(
        segments,
        energy_grid,
        coherent=True,
        longitudinal_rms_fs=longitudinal_rms_fs,
        **KWARGS,
    )

    flat_raw = (
        mc_spectrum(
            {**_segments(2), "r_mid": positions},
            energy_grid,
            coherent=True,
            **KWARGS,
        )
        * 2.0
    )
    grouped_raw = sum(
        mc_spectrum(
            {**_segments(1), "r_mid": position[None, :]},
            energy_grid,
            coherent=True,
            **KWARGS,
        )
        for position in positions
    )
    sigma_z_ang = longitudinal_rms_fs * C_ANG_PER_FS
    F_z = np.exp(-((energy_grid / HBARC_EV_ANG * sigma_z_ang) ** 2))
    expected = ((1.0 - F_z) * grouped_raw + F_z * flat_raw) / 2.0

    peak = float(np.max(np.abs(expected)))
    assert peak > 0.0
    # F_z = exp(-(omega sigma_z)^2), so dF_z = -2 (omega sigma_z)^2 F_z
    # domega/omega and the sensitivity is twice that squared phase (5.2 here),
    # F_z <= 1 bounding the rest.
    sensitivity = 2.0 * (float(np.max(energy_grid)) / HBARC_EV_ANG * sigma_z_ang) ** 2
    np.testing.assert_allclose(actual, expected, rtol=_blend_rtol(sensitivity), atol=peak * 1e-12)


def test_identical_in_phase_electrons_reach_n_squared_limit():
    single = mc_spectrum(_segments(), E_GRID, coherent=True, **KWARGS)
    pair = mc_spectrum(_segments(2), E_GRID, coherent=True, **KWARGS)

    # Field doubles, intensity quadruples, then per-electron /Ne normalization
    # leaves twice the one-electron yield.
    np.testing.assert_allclose(pair, 2.0 * single, rtol=RTOL)


# Subdivision invariance is exact: each piece carries the complex formation
# integral of the same propagation phase that separates pieces -- vacuum
# retardation, reciprocal harmonic, and the in-medium escape leg
# ``-delta(E) omega(E) L_esc,j`` with its within-piece slope (issue #181) --
# so a flight's halves sum to it to rounding. Before issue #181 the sinc
# dropped the escape-leg slope and the residual was 1e-6..1e-5 of peak.
SPLIT_RESIDUAL_TOL = max(1e-12, 1000.0 * float(np.finfo(REAL).eps))


def _split_residual(length, sinc_cutoff):
    energy_grid = np.arange(1050.0, 1400.0, 0.5)
    spectra = [
        mc_spectrum(
            _straight_flight_segments(split=split, reference=True, length=length),
            energy_grid,
            coherent=True,
            sinc_cutoff=sinc_cutoff,
            **KWARGS,
        )
        for split in (False, True)
    ]
    peak = float(max(np.max(np.abs(s)) for s in spectra))
    assert peak > 0.0
    return float(np.max(np.abs(spectra[1] - spectra[0]))) / peak


@pytest.mark.parametrize("sinc_cutoff", [None, 1.0e6])
def test_straight_flight_is_invariant_to_two_half_segments(sinc_cutoff):
    """Validation: coherent-segment-midpoint-time.

    A constant-velocity segment integral is independent of numerical
    subdivision when each stored midpoint position is paired with midpoint
    transport age -- exactly, including the in-medium escape-path phase
    (Validation: coherent-formation-absorption).  ``None`` selects the batched route; a very
    large finite cutoff selects the per-reflection route without removing this
    grid's tails.
    """
    single = _straight_flight_segments(split=False)
    halves = _straight_flight_segments(split=True)
    energy_grid = np.arange(1050.0, 1400.0, 0.5)

    # The independent centered-segment integral carries no in-medium leg, so it
    # keeps the exact identity and still pins the midpoint pairing itself.
    single_field = _constant_velocity_field(single, energy_grid)
    halves_field = _constant_velocity_field(halves, energy_grid)
    np.testing.assert_allclose(halves_field, single_field, rtol=1e-11, atol=1e-12)

    residuals = [_split_residual(length, sinc_cutoff) for length in (40.0, 20.0, 10.0, 5.0)]
    assert max(residuals) < SPLIT_RESIDUAL_TOL


@pytest.mark.parametrize("sinc_cutoff", [None, 4.0])
def test_straight_trajectory_segments_are_in_phase_at_resonance(sinc_cutoff):
    """The reciprocal-harmonic phase cancels propagation phase on resonance."""
    n_hat = np.asarray(KWARGS["n_hat"], dtype=float)
    n_hat /= np.linalg.norm(n_hat)
    _, g_norm = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    beta = float(beta_from_Ee(30e3))
    k_gamma = beta * g_norm / (1.0 - beta * n_hat[2])
    target_energy = HBARC_EV_ANG * k_gamma
    dz = np.pi / g_norm
    segments = _segments(2)
    segments.update(
        r_mid=np.array([[4.0, 0.0, 5.0], [4.0, 0.0, 5.0 + dz]]),
        L_ang=np.full(2, dz),
        t_ang=np.array([0.0, dz / beta]),
        elec_id=np.zeros(2, dtype=int),
        Ne=1,
        thickness_ang=100.0,
        crystal_width_ang=100.0,
        crystal_height_ang=100.0,
    )
    single = {
        key: value[:1] if isinstance(value, np.ndarray) else value
        for key, value in segments.items()
    }
    single["Ne"] = 1

    energy_grid = target_energy + np.array([-1.0, 0.0, 1.0])
    one_segment = mc_spectrum(single, energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **KWARGS)
    two_segments = mc_spectrum(
        segments, energy_grid, coherent=True, sinc_cutoff=sinc_cutoff, **KWARGS
    )

    assert one_segment[1] > 0.0
    np.testing.assert_allclose(two_segments[1], 4.0 * one_segment[1], rtol=RTOL)


def test_batched_coherent_matches_legacy_per_hkl_reference():
    """The new batched coherent setup must preserve the legacy per-hkl result.

    ``sinc_cutoff=None`` selects the batched coherent route.  A deliberately
    enormous finite cutoff selects the legacy per-hkl route, but its energy
    window is far wider than this test grid, so no sinc tail is actually
    removed.  The two calls therefore differ only in implementation/reduction
    ordering, not in the modeled physics.
    """
    segments = _segments(2)

    batched = mc_spectrum(segments, E_GRID, coherent=True, sinc_cutoff=None, **KWARGS)
    legacy = mc_spectrum(segments, E_GRID, coherent=True, sinc_cutoff=1.0e6, **KWARGS)

    _assert_batch_close(batched, legacy)


def test_coherent_reflections_remain_incoherent_under_batching():
    """Batching across g must not create cross-reflection field interference."""
    segments = _segments(2)
    energy_grid = np.arange(500.0, 3200.0, 2.0)
    hkls = ((0, 0, 2), (0, 0, 4))

    singles = []
    for hkl in hkls:
        spec = mc_spectrum(
            segments,
            energy_grid,
            coherent=True,
            **{**KWARGS, "hkl_list": [hkl]},
        )
        assert np.max(spec) > 0.0
        singles.append(spec)

    together = mc_spectrum(
        segments,
        energy_grid,
        coherent=True,
        **{**KWARGS, "hkl_list": list(hkls)},
    )

    _assert_batch_close(together, singles[0] + singles[1])


def test_coherent_components_are_rejected():
    with pytest.raises(ValueError, match="incompatible with components"):
        mc_spectrum(_segments(), E_GRID, coherent=True, components=True, **KWARGS)


def _runner_case(*, coherent=False):
    return {
        "crystal": "hopg",
        "hkl_list": KWARGS["hkl_list"],
        "B_ang2": KWARGS["B_ang2"],
        "composition": None,
        "E0_keV": 30.0,
        "coherent_emission": coherent,
    }


def _runner_tp(segs, *, ne_lines=None, ne_brem=None):
    ne = int(segs["Ne"])

    return {
        "E_grid": E_GRID,
        "E_brem": E_GRID,
        "n_hat": KWARGS["n_hat"],
        "segs": segs,
        "Ne_lines": ne if ne_lines is None else ne_lines,
        "Ne_brem": ne if ne_brem is None else ne_brem,
        "groove": None,
    }


def test_runner_always_stores_incoherent_spec_and_omits_spec_coherent(monkeypatch):
    """Default (incoherent) emission: `spec` is the incoherent line sum and no
    `spec_coherent` is attached -- so no downstream `record["spec"]` consumer can
    KeyError and no coherent grid is paid for."""
    import pyrite.montecarlo.runner as runner

    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(E_GRID))
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *a, **k: np.zeros_like(E_GRID),
    )

    segs = _segments(2)
    tp = _runner_tp(segs, ne_lines=2, ne_brem=2)
    out = runner._spectrum_case_impl(_runner_case(), tp)

    assert "spec_coherent" not in out
    direct_incoherent = mc_spectrum(segs, E_GRID, coherent=False, **KWARGS)
    np.testing.assert_allclose(out["spec"], direct_incoherent, rtol=RTOL)


def test_runner_dual_spectra_from_one_transport(monkeypatch):
    """emission includes coherent: ONE transport (`tp["segs"]`) yields both the
    incoherent `spec` and a `spec_coherent`, each matching a direct
    `mc_spectrum` on the SAME segments -- the single-transport / dual-kernel
    invariant."""
    # Both accumulation routes tabulate reflections through `_kernels`, so that
    # is the one place a tabulation counter sees every evaluation.
    import pyrite.montecarlo.runner as runner
    from pyrite.montecarlo.spectrum.lines import _kernels as line_kernels

    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(E_GRID))
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *a, **k: np.zeros_like(E_GRID),
    )
    real_tables = line_kernels.reflection_coupling_tables
    tabulated = []

    def counted_tables(crystal, hkl_list, *args, **kwargs):
        tabulated.append(len(hkl_list))
        return real_tables(crystal, hkl_list, *args, **kwargs)

    monkeypatch.setattr(line_kernels, "reflection_coupling_tables", counted_tables)
    segs = _segments(2)
    tp = _runner_tp(segs, ne_lines=2, ne_brem=2)

    case = _runner_case(coherent=True)
    out = runner._spectrum_case_impl(case, tp)
    # One tabulation covers every reflection and serves both kernels.
    assert tabulated == [len(KWARGS["hkl_list"])]

    direct_incoherent = mc_spectrum(segs, E_GRID, coherent=False, **KWARGS)
    direct_coherent = mc_spectrum(segs, E_GRID, coherent=True, **KWARGS)

    np.testing.assert_allclose(out["spec"], direct_incoherent, rtol=RTOL)
    np.testing.assert_allclose(out["spec_coherent"], direct_coherent, rtol=RTOL)
    # The two kernels genuinely differ for this in-phase pair (n^2 build-up),
    # so `spec` is NOT silently the coherent array.
    assert not np.allclose(out["spec"], out["spec_coherent"], rtol=RTOL, atol=0.0)


# ---- store_result: the transport payload must reach the checkpoint record ----
def _store_case():
    return {
        "name": "coh",
        "E0_keV": 30.0,
        "theta_obs_rad": np.pi / 2,
        "dtheta_obs_rad": 0.0,
        "domega_sr": 1.0,
    }


def _store_out(*, coherent=False):
    E = np.arange(50.0, 151.0, 1.0)
    out = {
        "E_grid": E,
        "spec": np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2),
        "brem": np.full_like(E, 0.1),
        "eta": 1.0,
    }
    if coherent:
        out["spec_coherent"] = 2.0 * out["spec"]
    return out


def test_store_result_keeps_spec_coherent_from_transport():
    """Regression: store_result built its record from a hardcoded field list that
    never mentioned `spec_coherent`, so every emission coherent/both run stored
    incoherent-only data no matter what the transport computed -- the analysis UI
    then found Coherent/Both permanently disabled."""
    from pyrite.results import store_result

    out = _store_out(coherent=True)
    results = {}
    store_result(results, _store_case(), out)
    record = results["coh"][30.0]

    np.testing.assert_array_equal(record["spec_coherent"], out["spec_coherent"])
    # and the incoherent array is untouched by the companion
    np.testing.assert_array_equal(record["spec"], out["spec"])


def test_store_result_omits_spec_coherent_for_incoherent_run():
    """An incoherent transport grows no key at all, so presence stays the honest
    gate every reader (emission_menu, the altair overlay, reline) tests on."""
    from pyrite.results import store_result

    results = {}
    store_result(results, _store_case(), _store_out())

    assert "spec_coherent" not in results["coh"][30.0]
