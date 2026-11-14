"""Axial zone flux weights against the Trofymenko & Kyryllin shared physics (#1175).

Trofymenko & Kyryllin, Phys. Rev. A 113, 032808 (2026) simulate Si K-shell
CXR at 1--1000 GeV through 40 um crystals; Andreyashkin et al. (2001) measure
W <111> at 600--1000 MeV over 1.2 mm. The admitted axial cases are
si-fibre/A and si-fibre/B (Si <001>, 3.5 MeV, 0.3 um), so no overlap point
exists. These tests compare the shared physics on the same transverse density instead:

* their distant-collision impact-parameter profile, Eq. (6),
  ``w(r) ∝ (1 - exp(-r^2/b^2)) / r^2`` with ``r_min = (hbar / 2 m omega_K)^1/2``
  [Eq. (5)] and ``b = eta r_min e^{C/2}`` [Eq. (7), ``eta = 0.995``], summed
  over strings inside ``rho_0 = 5 A`` [Eq. (9)] and uniform beyond, in place of
  the production uniform distant-transverse term, on the si-fibre/B retained
  density (their shape on the PENELOPE transverse magnitude only: an
  over-localization bound, not their model);
* the orientation ordering for electrons, axial <100> > (110) > (100) > 1
  (their Figs. 4--5), at a common 3.5 MeV and 0.3 um, with the planar well
  depths they quote (11.8 and 21.4 eV).

The quantitative ratios are not compared: theirs fold dechanneling over a
12 um Kalpha escape layer at GeV energies.

Validation: cr-zone-incoherent-flux-weight
"""

import numpy as np
import pytest
from scipy.special import j0

from pyrite.materials.attenuation import linear_attenuation_inv_mm
from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling import flux_weights as fw
from pyrite.montecarlo.channeling.axial_fiber_case import axial_fiber_inputs
from pyrite.montecarlo.channeling.axial_fiber_regression import fiber_on_axis_source
from pyrite.montecarlo.channeling.axial_flux_weights import (
    AxialFluxKernels,
    _string_densities,
    axial_flux_kernels,
)
from pyrite.montecarlo.channeling.handoff import _cell_grid, _solve
from pyrite.montecarlo.transport.shell_gos import shell_gos_moments
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES

pytestmark = pytest.mark.slow

ENTRY = ADMITTED_AXIAL_CASES["si-fibre/B"]
LINES = 4
RHO0_ANG = 5.0
EULER = 0.5772156649015329
HBARC_EV_ANG = 1973.269804
MC2_EV = 510998.95
GRID = np.linspace(800.0, 10000.0, 47)


def _impact_profile(g, b, r_cut):
    """``int_0^r_cut (1 - e^{-r^2/b^2}) / r J0(g r) dr`` on a log grid."""
    t = np.linspace(np.log(1e-5 * b), np.log(r_cut), 20001)
    r = np.exp(t)
    return np.trapezoid((1.0 - np.exp(-((r / b) ** 2)))[None, :] * j0(np.outer(g, r)), t, axis=1)


def _trofymenko_kernel(profile, m_max):
    """Si K kernel with the Eq. (6) distant-transverse profile, on the axial layout."""
    e = ENTRY.kinetic_energy_eV
    m, n = np.meshgrid(
        np.arange(-2 * m_max, 2 * m_max + 1), np.arange(-2 * m_max, 2 * m_max + 1), indexing="ij"
    )
    gx, gy = profile.g_inv_ang
    order = np.argsort((m**2 + n**2).ravel(), kind="stable")
    vectors = np.stack([m.ravel() * gx, n.ravel() * gy], axis=-1)[order]
    g = np.hypot(vectors[:, 0], vectors[:, 1])
    densities = _string_densities(profile, vectors)
    oscillators = catalog_shell_oscillators("silicon")
    index, osc = next(
        (i, o)
        for i, o in enumerate(oscillators.oscillators)
        if o.label == "K" and int(o.atomic_number) == 14
    )
    transverse = float(shell_gos_moments(oscillators, e).distant_transverse[index, 0])
    transverse *= 1e16 / oscillators.electrons_per_formula
    kernel = fw._shell_harmonics(None, oscillators, e, densities, g)["14:K"]
    kernel[0] -= densities["Si"][0] * transverse  # production: uniform transverse
    gamma = 1.0 + e / MC2_EV
    beta = np.sqrt(1.0 - gamma**-2)
    r_min = HBARC_EV_ANG / np.sqrt(2.0 * MC2_EV * osc.ionization_energy_eV)
    b = 0.995 * r_min * np.exp(0.5 * EULER)
    adiabatic = beta * gamma * HBARC_EV_ANG / osc.ionization_energy_eV
    shape = _impact_profile(g, b, min(RHO0_ANG, adiabatic))
    shape /= _impact_profile(np.zeros(1), b, adiabatic)[0]
    shape[0] = 1.0  # rho_0 .. adiabatic cutoff stays uniform
    kernel = kernel + densities["Si"] * transverse * shape
    flat = np.empty(g.size, dtype=complex)
    flat[order] = kernel / kernel[0].real
    return flat.reshape(m.shape)


@pytest.fixture(scope="module")
def axial_weights():
    order = ENTRY.numerics.max_order
    profile = axial_profile(ENTRY.crystal, ENTRY.axis_uvw, temperature_K=ENTRY.temperature_K)
    production = axial_flux_kernels(profile, ENTRY.kinetic_energy_eV, (order, order))
    harmonics = np.concatenate([production.harmonics, _trofymenko_kernel(profile, order)[None]])
    kernels = AxialFluxKernels((*production.keys, "trofymenko:K"), harmonics, (order, order))
    inputs = axial_fiber_inputs(ENTRY)
    z = inputs.depths_ang
    source = fiber_on_axis_source(
        inputs.channels,
        inputs.entry_orders,
        kinetic_energy_eV=ENTRY.kinetic_energy_eV,
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
        depths_ang=z,
        photon_energy_eV=GRID,
        incident_ids=range(LINES),
        seed=0,
        carrier_inv_ang=inputs.carrier_inv_ang,
        boundary_eV=inputs.boundary_eV,
        absorption_inv_ang=linear_attenuation_inv_mm("silicon", GRID) * 1e-7,
        exit_path_ang=z[-1] - z,
        search_step_ang=inputs.search_step_ang,
        flux_kernels=kernels,
    )
    weights = fw.row_flux_weights(z, source.flux_factors, np.array([z[-1]]))[0]
    return dict(zip(kernels.keys, weights, strict=True))


def _planar_weights(hkl):
    result, _, profile, _, depth = _solve(
        "silicon", hkl, ENTRY.temperature_K, ENTRY.kinetic_energy_eV, 12, 16, 300.0,
        ENTRY.thickness_ang, 15.0, 0.0, 0.0, 0.038,
    )  # fmt: skip
    kernels = fw.flux_kernels(profile, ENTRY.kinetic_energy_eV, 24)
    z = np.asarray(result.z_ang)
    factors = fw.flux_factors(kernels, _cell_grid(profile, 12), result.spatial_density_per_ang)
    weights = fw.row_flux_weights(z, factors, np.array([z[-1]]))[0]
    return dict(zip(kernels.keys, weights, strict=True)), depth


def test_impact_parameter_k_profile_stays_within_the_localization_tolerance(axial_weights):
    # Uniform transverse is the least-localized limit: Eq. (6) only raises W_K.
    ratio = axial_weights["trofymenko:K"] / axial_weights["14:K"]
    assert 1.0 < ratio < 1.10
    assert ratio == pytest.approx(1.058, abs=0.005)  # 1.062 / 1.066 at 32 / 128 lines


def test_orientation_ordering_and_planar_well_depths_match_trofymenko(axial_weights):
    p100, depth100 = _planar_weights((1, 0, 0))
    p110, depth110 = _planar_weights((1, 1, 0))
    assert depth100 == pytest.approx(11.8, abs=0.5)  # Trofymenko Sec. III
    assert depth110 == pytest.approx(21.4, abs=0.5)
    for key in (fw.BREM_KEY, "14:K"):
        assert axial_weights[key] > p110[key] > p100[key] > 1.0
    assert p100["14:K"] == pytest.approx(1.4129, rel=1e-3)
    assert p110["14:K"] == pytest.approx(1.4733, rel=1e-3)
