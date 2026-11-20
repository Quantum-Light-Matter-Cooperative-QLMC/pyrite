"""Impulse-approximation Doppler broadening of Compton energies (#1231).

Expectations are built here from PENELOPE-2024 §2.3 (Eqs. 2.30, 2.38, 2.40,
2.43, 2.54) by numerical integration, not from the sampler's closed-form CDF,
inverse or root solver.
"""

import numpy as np
import pytest

from pyrite.materials.crystal import M_E_EV
from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation
from pyrite.montecarlo.transport import compton_profiles
from pyrite.montecarlo.transport import photon_scattering as scattering
from pyrite.montecarlo.transport.photon_scattering import sample_compton

ALPHA = 1 / 137.035999  # independent of the package constant's spelling


def _assert_uniform_cdf(values):
    ordered = np.sort(values)
    n = len(ordered)
    error = max(np.max(np.arange(1, n + 1) / n - ordered), np.max(ordered - np.arange(n) / n))
    # DKW bound: false rejection probability <= 2 exp(-2*2.2²).
    assert error < 2.2 / np.sqrt(n)


def _pz_au(energy, photon, delta):
    """PENELOPE-2024 Eq. 2.30 in atomic units, from energies alone."""
    mu = 1 - delta
    cq = np.sqrt(energy**2 + photon**2 - 2 * energy * photon * mu)
    return (energy * photon * delta - M_E_EV * (energy - photon)) / (M_E_EV * cq) / ALPHA


def _profile(j0, p):
    """PENELOPE-2024 Eq. 2.54 (n = 2), one-electron normalization."""
    a = np.sqrt(0.5) + np.sqrt(2) * j0 * np.abs(p)
    return j0 * np.sqrt(2) * a * np.exp(0.5 - a * a)


def _conditional_cdf(energy, delta, momentum, shells):
    """CDF of p_z given theta for F(p_z) sum_i f_i J_i Theta(E-E'-U_i) (Eq. 2.40)."""
    tau_c = 1 / (1 + energy / M_E_EV * delta)
    e_c = energy * tau_c
    cq_c = np.sqrt(energy**2 + e_c**2 - 2 * energy * e_c * (1 - delta))
    slope = cq_c / energy * (1 + e_c * (e_c - energy * (1 - delta)) / cq_c**2)
    lowest = -1 / ALPHA
    highest = np.array(
        [_pz_au(energy, energy - u, delta) if u < energy else lowest for u in shells[1]]
    )
    top = max(highest.max(), momentum)
    s = np.sinh(np.linspace(np.arcsinh(lowest / 0.02), np.arcsinh(top / 0.02), 20_001))
    grid = np.unique(np.r_[s * 0.02, highest[highest > lowest]])
    factor = np.maximum(1 + slope * np.clip(grid * ALPHA, -0.2, 0.2), 0)
    density = sum(
        f * _profile(j0, grid) * (grid <= h)
        for f, h, j0 in zip(shells[0], highest, shells[2], strict=True)
    )
    density = factor * density
    cdf = np.r_[0, np.cumsum(np.diff(grid) * (density[1:] + density[:-1]) / 2)]
    return np.interp(momentum, grid, cdf) / cdf[-1]


@pytest.mark.parametrize("element,energy", [("W", 511e3), ("Cu", 60e3), ("C", 2e3), ("Pb", 3e6)])
def test_doppler_momentum_matches_independent_impulse_approximation(element, energy):
    rng = np.random.default_rng(1231_01)
    ray = np.array([2, -3, 6]) / 7
    events = [sample_compton(element, energy, ray, iter(rng.random, None)) for _ in range(800)]
    eadl = load_eadl_relaxation(element)
    j0 = dict(
        zip(
            compton_profiles.compton_shells(element).designators,
            compton_profiles.compton_shells(element).profile_j0_au,
            strict=True,
        )
    )
    shells = (
        [s.electrons for s in eadl.subshells],
        [s.binding_energy_eV for s in eadl.subshells],
        [j0[s.shell_designator] for s in eadl.subshells],
    )
    levels = []
    for event in events:
        assert event.photon_energy_eV <= energy - event.binding_energy_eV * (1 - 1e-12)
        momentum = _pz_au(energy, event.photon_energy_eV, event.one_minus_cosine)
        levels.append(_conditional_cdf(energy, event.one_minus_cosine, momentum, shells))
    _assert_uniform_cdf(np.array(levels))


def test_w_backscatter_is_broadened_around_the_compton_line():
    # The #341 gap: free kinematics put every 511 keV backscatter in W at
    # 170-178 keV; Geant4 Penelope/Livermore spread it over ~150-220 keV.
    rng = np.random.default_rng(1231_02)
    events = [sample_compton("W", 511e3, [0, 0, 1], iter(rng.random, None)) for _ in range(4000)]
    back = [e for e in events if e.one_minus_cosine > 1.9]
    shift = np.array(
        [e.photon_energy_eV * (1 + 511e3 / M_E_EV * e.one_minus_cosine) / 511e3 for e in back]
    )
    assert len(back) > 100
    assert np.median(np.abs(shift - 1)) > 0.01  # free kinematics: exactly 0
    assert np.percentile(shift, 5) < 0.93 and np.percentile(shift, 95) > 1.07


def test_delta_profile_without_binding_recovers_free_compton(monkeypatch):
    shells = compton_profiles.ComptonShells(
        np.array([1]), np.array([6.0]), np.array([1e-9]), np.array([1e9])
    )
    monkeypatch.setattr(scattering, "compton_shells", lambda element: shells)
    rng = np.random.default_rng(1231_03)
    for _ in range(200):
        event = sample_compton("C", 2e6, [0, 0, 1], iter(rng.random, None))
        line = 2e6 / (1 + 2e6 / M_E_EV * event.one_minus_cosine)
        assert event.photon_energy_eV == pytest.approx(line, rel=1e-6)
        assert event.electron_energy_eV == pytest.approx(2e6 - line, rel=1e-6, abs=1e-3)


def test_draw_prefix_reproduces_event_and_consumes_three_doppler_draws():
    tape = np.random.default_rng(1231_04).random(4096)
    event = sample_compton("Cu", 3e4, [0, 1, 0], tape)
    exact = sample_compton("Cu", 3e4, [0, 1, 0], tape[: event.draws_used])
    for name in ("photon_energy_eV", "electron_energy_eV", "draws_used", "subshell_designator"):
        assert getattr(exact, name) == getattr(event, name)
    np.testing.assert_array_equal(exact.photon_direction, event.photon_direction)
    with pytest.raises(ValueError, match="exhausted"):
        sample_compton("Cu", 3e4, [0, 1, 0], tape[: event.draws_used - 1])
    # Angle trials use 3 draws, the azimuth 1, every Doppler trial 3.
    assert (event.draws_used - 1) % 3 == 0


def test_packaged_profiles_are_pinned_and_cover_eadl_shells():
    for element in ("H", "C", "Si", "Cu", "Ga", "As", "W", "Pb", "Bk", "Fm"):
        shells = compton_profiles.compton_shells(element)
        eadl = load_eadl_relaxation(element)
        np.testing.assert_array_equal(shells.designators, eadl.shell_designators)
        assert np.all(shells.profile_j0_au > 0)
        assert not shells.profile_j0_au.flags.writeable


def test_profiles_reproduce_xraylib_biggs_values():
    xraylib = pytest.importorskip("xraylib")
    from pyrite.montecarlo.eedl_ionization import EEDL_SUBSHELL_LABELS

    for z, element in ((6, "C"), (29, "Cu"), (74, "W"), (82, "Pb")):
        shells = compton_profiles.compton_shells(element)
        for designator, value in zip(shells.designators, shells.profile_j0_au, strict=True):
            label = EEDL_SUBSHELL_LABELS[int(designator)]
            try:
                expected = xraylib.ComptonProfile_Partial(z, getattr(xraylib, f"{label}_SHELL"), 0)
            except ValueError:
                continue  # spin-orbit partner substitution, recorded in the table
            assert value == pytest.approx(expected, rel=5e-6)


def test_photon_identity_carries_the_compton_model_but_mode_off_identity_does_not(monkeypatch):
    from pyrite import _photon_support
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.profiles import build_cases, case_content_key
    from pyrite.montecarlo.transport.photon_histories import PhotonAperture

    aperture = PhotonAperture((0, 0, 1e6), (1, 0, 0), (0, 1, 0), 1e6, 1e6)
    kwargs = dict(
        n_samples=8,
        seed=1231,
        photon_cutoff_eV=1e3,
        electron_cutoff_eV=1e3,
        source_ceiling_eV=1e5,
    )
    digest, payload = _photon_support._identity("0" * 64, aperture, (), [1e3, 1e5], **kwargs)
    assert payload["model"] == "bremslib-continuum-photon-only-cpu.v3"
    assert payload["data_sha256"]["compton_profiles"] == compton_profiles.COMPTON_PROFILE_SHA256
    case = build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]
    assert case.to_dict().get("photon_transport") is None
    off = case_content_key(case)
    # A different Compton table changes the enabled observation identity...
    monkeypatch.setattr(compton_profiles, "COMPTON_PROFILE_SHA256", "f" * 64)
    for name in ("MODEL", "EVENT_MODEL", "CUDA_MODEL", "CUDA_EVENT_MODEL"):
        monkeypatch.setattr(_photon_support, name, "changed")
    assert _photon_support._identity("0" * 64, aperture, (), [1e3, 1e5], **kwargs)[0] != digest
    # ...but never the photon-off case identity (pinned in identity-pins.json).
    assert case_content_key(case) == off
