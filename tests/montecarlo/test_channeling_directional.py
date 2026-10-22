"""Direction-resolved valence loss (#536): uniaxial mixing, sum rules and the kernel hook."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling.directional import directional_channels
from pyrite.montecarlo.channeling.operators import electronic_channels, electronic_kernel
from pyrite.montecarlo.transport.directional_loss import (
    ProvisionalModelWarning,
    UniaxialDielectric,
    catalog_directional_oscillators,
    directional_valence,
    load_directional_loss,
)
from pyrite.montecarlo.transport.shell_oscillators import ShellOscillator
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators

ENERGY_EV = 16.92e6
SMALL = {"n_cells": 4, "max_order": 8}
ELECTRONS = 18.0


@pytest.fixture(scope="module")
def omega_ev():
    """Plasma energy of the 18 MoS2 valence electrons, ``sqrt(18/N_total) Omega_p``."""
    shells = catalog_shell_oscillators("mos2")
    return shells.plasma_energy_eV * np.sqrt(ELECTRONS / shells.electrons_per_formula)


def _lorentz(energy, omega, poles):
    """Synthetic ``eps = 1 + sum A_i/(w_i^2 - w^2 - i g_i w)``, ``sum A_i = Omega^2``."""
    eps = np.ones_like(energy, dtype=complex)
    for w0, share, gamma in poles:
        eps += share * omega**2 / (w0**2 - energy**2 - 1j * gamma * energy)
    return eps


PERP = ((5.0, 0.12, 1.5), (9.5, 0.18, 1.5), (18.0, 0.30, 6.0), (26.0, 0.40, 8.0))
PAR = ((6.0, 0.12, 1.5), (14.0, 0.28, 2.0), (20.0, 0.25, 6.0), (27.0, 0.35, 8.0))


def _dielectric(omega, perp=PERP, par=PAR, emax=600.0):
    energy = np.geomspace(0.3, emax, 6000)
    static = lambda poles: 1.0 + sum(s * omega**2 / w0**2 for w0, s, _ in poles)  # noqa: E731
    return UniaxialDielectric(
        energy,
        _lorentz(energy, omega, perp),
        _lorentz(energy, omega, par),
        "synthetic Lorentz test model",
        static(perp),
        static(par),
    )


@pytest.fixture(scope="module")
def valence(omega_ev):
    return directional_valence(_dielectric(omega_ev), ELECTRONS, omega_ev)


@pytest.fixture(scope="module")
def isotropic(omega_ev):
    return directional_valence(_dielectric(omega_ev, PERP, PERP), ELECTRONS, omega_ev)


def test_sum_rules_close_per_direction(valence):
    # Lorentz model: the f-sum and static sum hold analytically, so the binned data must too.
    assert np.max(np.abs(valence.closure - 1.0)) < 5e-3
    assert np.max(np.abs(valence.static_sum / valence.static_target - 1.0)) < 2e-2
    assert np.allclose(valence.strengths.sum(axis=1), ELECTRONS, rtol=1e-12)
    assert np.all(valence.strengths >= 0.0)
    assert np.max(valence.tail_fraction) < 1e-2


def test_directions_differ_and_mix_through_epsilon_not_loss(valence, omega_ev):
    # The mixed loss function is -Im 1/(eps_perp sin^2 + eps_par cos^2), not a loss mixture.
    perp, par = valence.strengths[0], valence.strengths[-1]
    assert np.sum(np.abs(perp - par)) > 0.1 * ELECTRONS
    mid = valence.strengths[valence.cos2_nodes.size // 2]
    assert np.sum(np.abs(mid - 0.5 * (perp + par))) > 0.02 * ELECTRONS
    dielectric = _dielectric(omega_ev)
    energy = np.array([9.0])
    mixed = dielectric.elf(0.5, energy)[0]
    linear = 0.5 * (dielectric.elf(0.0, energy)[0] + dielectric.elf(1.0, energy)[0])
    assert abs(mixed - linear) > 0.05 * linear


def test_isotropic_limit_is_the_radial_channels(isotropic, omega_ev):
    assert np.allclose(isotropic.ratio(np.full((3, isotropic.energies_eV.size), 0.37)), 1.0)
    base = catalog_directional_oscillators("mos2", isotropic)
    for axis in "xyz":
        channels = directional_channels(base, isotropic, ENERGY_EV, axis)
        shells = tuple(
            ShellOscillator(0, f"cb{k}", float(f), 0.0, float(w))
            for k, (f, w) in enumerate(
                zip(isotropic.mean_strengths, isotropic.energies_eV, strict=True)
            )
        )
        reference = electronic_channels(
            replace(
                base,
                oscillators=shells + tuple(o for o in base.oscillators if o.atomic_number != 0),
            ),
            ENERGY_EV,
        )
        q = np.geomspace(1e-3, 50.0, 40)
        phi = np.linspace(0.0, np.pi / 2, 7)[:, None]
        got = channels.weight_pq(q * np.cos(phi), q * np.sin(phi))
        assert np.allclose(got, reference.weight(q), rtol=1e-12)
        assert np.allclose(channels.distant(q), reference.distant(q), rtol=1e-12)


def test_sphere_average_strengths_close_the_sum_rule(valence):
    assert valence.mean_strengths.sum() == pytest.approx(ELECTRONS, rel=1e-4)
    # the Bethe-sum W_cb lies between the directional extremes
    logs = valence.log_sums / ELECTRONS
    assert np.min(logs) <= np.log(valence.mean_log_energy_eV) <= np.max(logs)


def test_interpolation_converges_in_alpha_nodes(omega_ev):
    dielectric = _dielectric(omega_ev)
    coarse = directional_valence(dielectric, ELECTRONS, omega_ev, nodes=33)
    fine = directional_valence(dielectric, ELECTRONS, omega_ev, nodes=129)
    probe = np.array([0.03, 0.21, 0.5, 0.77, 0.96])
    k = np.arange(coarse.energies_eV.size)
    got = coarse.ratio(np.broadcast_to(probe[:, None], (5, k.size)))
    ref = fine.ratio(np.broadcast_to(probe[:, None], (5, k.size)))
    weight = coarse.mean_strengths / coarse.mean_strengths.sum()
    assert np.max(np.sum(weight * np.abs(got - ref), axis=1)) < 2e-2


def test_nonclosing_dataset_is_rejected(omega_ev):
    with pytest.raises(ValueError, match="closure"):
        directional_valence(_dielectric(omega_ev), ELECTRONS, 1.5 * omega_ev)


def test_negative_loss_is_rejected(omega_ev):
    d = _dielectric(omega_ev)
    with pytest.raises(ValueError, match="Im eps"):
        UniaxialDielectric(d.energy_eV, d.eps_perp.conj(), d.eps_par, "bad")


@pytest.fixture(scope="module")
def basal():
    return planar_profile("mos2", (0, 0, 1), 295.0)


def _blocks(kernel):
    return kernel.momentum_kernel


def test_directional_kernel_is_psd_and_warns(basal, valence):
    with pytest.warns(ProvisionalModelWarning, match="not an anchored"):
        kernel = electronic_kernel(basal, ENERGY_EV, valence=valence, c_axis="x", **SMALL)
    herm = _blocks(kernel)
    assert np.allclose(herm, np.conj(np.swapaxes(herm, 1, 2)))
    assert np.min(np.linalg.eigvalsh(herm)) > -1e-9 * np.max(np.abs(herm))
    assert np.all(kernel.diagonal_coefficients[:1].real > 0.0)


def test_basal_kernel_depends_on_which_axis_carries_c(basal, valence):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProvisionalModelWarning)
        along_x = electronic_kernel(basal, ENERGY_EV, valence=valence, c_axis="x", **SMALL)
        along_y = electronic_kernel(basal, ENERGY_EV, valence=valence, c_axis="y", **SMALL)
        along_z = electronic_kernel(basal, ENERGY_EV, valence=valence, c_axis="z", **SMALL)
    # In-plane c (x or y) shares one circle-averaged total loss but not the jump kernel.
    scale = np.max(np.abs(along_x.momentum_kernel))
    assert np.max(np.abs(along_x.momentum_kernel - along_y.momentum_kernel)) > 1e-3 * scale
    assert np.allclose(along_x.diagonal_coefficients, along_y.diagonal_coefficients, rtol=1e-9)
    # c along the beam leaves only q_min along c, so the in-plane response dominates.
    assert not np.allclose(along_z.diagonal_coefficients, along_x.diagonal_coefficients, rtol=1e-3)


def test_isotropic_valence_reproduces_the_radial_kernel(basal, isotropic):
    base = catalog_directional_oscillators("mos2", isotropic)
    shells = tuple(
        ShellOscillator(0, f"cb{k}", float(f), 0.0, float(w))
        for k, (f, w) in enumerate(
            zip(isotropic.mean_strengths, isotropic.energies_eV, strict=True)
        )
    )
    split = replace(
        base, oscillators=shells + tuple(o for o in base.oscillators if o.atomic_number != 0)
    )
    radial = electronic_kernel(basal, ENERGY_EV, oscillators=split, **SMALL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProvisionalModelWarning)
        directional = electronic_kernel(basal, ENERGY_EV, valence=isotropic, c_axis="x", **SMALL)
    scale = np.max(np.abs(radial.momentum_kernel))
    assert np.max(np.abs(directional.momentum_kernel - radial.momentum_kernel)) < 1e-9 * scale
    assert np.allclose(directional.diagonal_coefficients, radial.diagonal_coefficients, rtol=1e-9)


def test_mean_excitation_closure_is_preserved(valence):
    base = catalog_directional_oscillators("mos2", valence)
    cb = next(o for o in base.oscillators if o.atomic_number == 0)
    assert cb.resonance_energy_eV == pytest.approx(valence.mean_log_energy_eV)
    total = sum(o.strength * np.log(o.resonance_energy_eV) for o in base.oscillators)
    assert total / base.electrons_per_formula == pytest.approx(
        np.log(base.mean_excitation_eV), rel=1e-9
    )


# ---- registered MoS2 dataset (skipped until the digitized curves are committed) ----


def _mos2_or_skip():
    try:
        return load_directional_loss("mos2")
    except (FileNotFoundError, ValueError, OSError) as exc:
        pytest.skip(f"MoS2 digitized curves not available: {exc}")


def test_mos2_dataset_reproduces_stated_peaks(omega_ev):
    dielectric, entry = _mos2_or_skip()
    energy = np.linspace(2.0, dielectric.energy_eV[-1], 4000)
    perp = dielectric.elf(0.0, energy)
    par = dielectric.elf(1.0, energy)
    peak = lambda y, lo, hi: energy[(energy > lo) & (energy < hi)][  # noqa: E731
        np.argmax(y[(energy > lo) & (energy < hi)])
    ]
    # Zeppenfeld 1970 Sec. 2: 8.75 eV and ~23 eV for q perp c; 23 eV unaffected and no pi plasmon for q || c.
    assert peak(perp, 7.5, 10.5) == pytest.approx(8.75, abs=0.4)
    assert peak(perp, 20.0, 26.0) == pytest.approx(23.0, abs=0.7)
    assert peak(par, 20.0, 26.0) == pytest.approx(23.0, abs=0.7)
    assert np.max(par[(energy > 7.5) & (energy < 10.5)]) < 0.4 * np.max(
        perp[(energy > 7.5) & (energy < 10.5)]
    )


def test_mos2_dataset_closes_both_sum_rules(omega_ev):
    dielectric, entry = _mos2_or_skip()
    valence = directional_valence(
        dielectric,
        ELECTRONS,
        omega_ev,
        status=entry["status"],
        tolerance=entry["closure_tolerance"],
    )
    assert np.all(np.abs(valence.closure - 1.0) <= entry["closure_tolerance"])
    # Digitized Fig. 2 ends at 31.5 eV: the Im-eps-matched tail carries 16-21% of the
    # f-sum, and the static sum falls ~8% short of 1 - 1/eps1(0) (#541 diagnostics).
    assert np.all(valence.tail_fraction < 0.25)
    assert np.all(np.abs(valence.static_sum / valence.static_target - 1.0) < 0.1)


def test_mos2_dielectric_matches_the_digitized_loss_function():
    # Fig. 1 (-Im 1/eps, elf_perp.csv / elf_par.csv) is the measured loss; Fig. 2 eps is its
    # KK transform. Each must reproduce the other within 2 sigma of the digitization error
    # stated in the CSV headers, propagated through ELF = eps2/|eps|^2 (at the 23 eV plasmon
    # eps1 ~ 0, so a 0.03 error on eps2 ~ 0.3 is already ~10% of the peak).
    from pyrite import DATA_DIR
    from pyrite.montecarlo.transport.directional_loss import _read_curve

    dielectric, entry = _mos2_or_skip()
    folder = DATA_DIR / entry["directory"]
    # (cos^2 alpha, Fig. 2 x10-branch start, Fig. 1 x3-branch end) in eV.
    for name, cos2, eps_split, elf_split in (("perp", 0.0, 15.0, 14.0), ("par", 1.0, 14.0, 17.0)):
        energy, elf = _read_curve(folder / f"elf_{name}.csv")
        keep = (energy > 3.0) & (energy < min(30.0, dielectric.energy_eV[-1]))
        energy, elf = energy[keep], elf[keep]
        eps = dielectric.eps_perp if cos2 == 0.0 else dielectric.eps_par
        e1 = np.interp(energy, dielectric.energy_eV, eps.real)
        e2 = np.interp(energy, dielectric.energy_eV, eps.imag)
        mod2 = e1**2 + e2**2
        d_eps = np.where(energy < eps_split, 0.3, 0.03)
        sigma = np.hypot(
            np.hypot(2.0 * e1 * e2, e1**2 - e2**2) / mod2**2 * d_eps,
            np.where(energy < elf_split, 0.02, 0.05),
        )
        assert np.all(np.abs(dielectric.elf(cos2, energy) - elf) < 2.0 * sigma)
