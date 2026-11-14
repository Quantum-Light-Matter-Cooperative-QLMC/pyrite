"""Localization-model spread of zone-row flux weights at the demo case (#1148).

The production kernels (``flux_weights``) make two modelling choices that set
how strongly zone rows weight incoherent emission:

* distant-transverse (photon-exchange) ionization is uniform over the cell;
* brem localizes on the screened Bethe--Heitler transfer of the nuclear and
  electron fields with stipulated ``Z^2 : Z`` strengths.

Independent alternatives on the same retained density of the admitted
Si(1-10) 17 MeV ``channeling_demo`` zone (2.0027 um path):

* impact-parameter (equivalent-photon) transverse excitation ``~ db/b`` from
  the shell size to the adiabatic/plasma cutoff, Trofymenko & Kyryllin,
  Phys. Rev. A 113, 032808 (2026), Eq. (1); in the zone LDA convention
  ``db/b <-> dq/q`` with ``exp(-G^2/2q^2)``;
* the most-localized brem limit, a thermally smeared point nuclear source
  (no screening smoothing, no electron field).

Both must stay within the first-pass ≤10% band-yield tolerance of the issue;
measured spreads are +1.6..+3.0% (K) and +3.5% (brem).

Validation: cr-zone-incoherent-flux-weight
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.atomic import Z_TABLE
from pyrite.montecarlo.channeling import flux_weights as fw
from pyrite.montecarlo.channeling.handoff import _cell_grid, _solve
from pyrite.montecarlo.channeling.operators import _electronic_transform, electronic_channels
from pyrite.montecarlo.transport.shell_gos import shell_gos_moments
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators

E_EV = 17e6
H = 16
PATH_ANG = 20027.44691995842
#: ``hbar omega_p`` of silicon [eV]; sets the c/omega_p transverse cutoff.
SI_PLASMON_EV = 31.05
HBARC_EV_ANG = 1973.269804
TOLERANCE = 0.10


@pytest.fixture(scope="module")
def demo_zone():
    # Arguments of the admitted ``channeling_demo`` zone solve (aligned entry).
    result, _, profile, _, _ = _solve(
        "silicon", (1, -1, 0), 293.15, E_EV, H, 16, 300.0, PATH_ANG, 250.0, 0.0, 0.0, 0.0409277
    )
    x = _cell_grid(profile, H)
    return profile, x, np.asarray(result.spatial_density_per_ang), np.asarray(result.z_ang)


def _row_weight(profile, x, density, z, harmonics):
    kernel = fw.FluxKernels(("c",), (harmonics / harmonics[0].real)[None, :], profile.g_inv_ang)
    factors = fw.flux_factors(kernel, x, density)
    return float(fw.row_flux_weights(z, factors, np.array([z[-1]]))[0, 0])


def _log_transverse(g, q_min, q_max, n=4000):
    q = np.geomspace(q_min, q_max, n)
    smooth = np.exp(-0.5 * (g[None, :] / q[:, None]) ** 2)
    return np.trapezoid(smooth, np.log(q), axis=0) / np.log(q_max / q_min)


def _k_shell_kernels(profile):
    """``(production, impact-parameter alternative)`` Si K-shell kernel harmonics."""
    oscillators = catalog_shell_oscillators("silicon")
    densities = fw._element_densities(profile, 2 * H)
    g = np.arange(2 * H + 1) * profile.g_inv_ang
    channels = electronic_channels(oscillators, E_EV)
    moments = shell_gos_moments(oscillators, E_EV)
    index, osc = next(
        (i, o)
        for i, o in enumerate(oscillators.oscillators)
        if o.label == "K" and int(o.atomic_number) == int(Z_TABLE["Si"])
    )
    transverse = float(moments.distant_transverse[index, 0]) * 1e16
    transverse /= oscillators.electrons_per_formula
    names = (
        "distant_ang2",
        "distant_qmin_inv_ang",
        "distant_qmax_inv_ang",
        "close_ang2_eV",
        "close_lower_eV",
        "close_upper_eV",
        "close_prime_eV",
    )
    one = replace(channels, **{n: getattr(channels, n)[index : index + 1] for n in names})
    form = fw.slater_form_factor(g, osc.ionization_energy_eV, 1)
    base = densities["Si"] * form * _electronic_transform(one, g)
    production = base.copy()
    production[0] += densities["Si"][0] * transverse
    gamma = 1.0 + E_EV / 510998.95
    beta = np.sqrt(1.0 - gamma**-2)
    zeta = np.sqrt(2.0 * osc.ionization_energy_eV / 27.211386245988) / 0.529177210903
    q_min = max(
        osc.ionization_energy_eV / (HBARC_EV_ANG * beta * gamma), SI_PLASMON_EV / HBARC_EV_ANG
    )
    alternatives = [
        base + densities["Si"] * form * transverse * _log_transverse(g, q_min, scale * zeta)
        for scale in (0.5, 1.0, 2.0)
    ]
    return production, alternatives


def test_production_kernels_reproduce_the_recorded_row_weights(demo_zone):
    profile, x, density, z = demo_zone
    kernels = fw.flux_kernels(profile, E_EV, 2 * H)
    factors = fw.flux_factors(kernels, x, density)
    weights = dict(
        zip(kernels.keys, fw.row_flux_weights(z, factors, np.array([z[-1]]))[0], strict=True)
    )
    # Production values on the demo full path (recorded 2026-11-12, #1148).
    assert weights[fw.BREM_KEY] == pytest.approx(1.34065, rel=1e-4)
    assert weights["14:K"] == pytest.approx(1.17556, rel=1e-4)


def test_k_shell_weight_is_insensitive_to_transverse_localization(demo_zone):
    profile, x, density, z = demo_zone
    production, alternatives = _k_shell_kernels(profile)
    w_prod = _row_weight(profile, x, density, z, production)
    w_alt = [_row_weight(profile, x, density, z, k) for k in alternatives]
    # Uniform transverse is the least-localized limit: alternatives only raise W.
    assert all(w >= w_prod for w in w_alt)
    spread = max(w_alt) / w_prod - 1.0
    assert 0.01 < spread < TOLERANCE
    assert spread == pytest.approx(0.0297, abs=0.003)


def test_brem_weight_is_within_tolerance_of_the_point_nuclear_limit(demo_zone):
    profile, x, density, z = demo_zone
    densities = fw._element_densities(profile, 2 * H)
    g = np.arange(2 * H + 1) * profile.g_inv_ang
    w_prod = _row_weight(profile, x, density, z, fw._brem_harmonics(profile, densities, g))
    point = float(Z_TABLE["Si"]) ** 2 * densities["Si"]
    w_point = _row_weight(profile, x, density, z, point.astype(complex))
    assert w_point > w_prod > 1.0
    assert w_point / w_prod - 1.0 < TOLERANCE
    assert w_point / w_prod - 1.0 == pytest.approx(0.0348, abs=0.003)
