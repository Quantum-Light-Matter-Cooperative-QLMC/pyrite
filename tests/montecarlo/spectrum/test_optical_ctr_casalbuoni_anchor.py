"""Published absolute CTR energy and form factors: Casalbuoni, Schmidt and Schmüser.

"Far-Infrared Transition and Diffraction Radiation, Part I", TESLA Report
2005-15 (DESY, 2005). Eq. (37) is the far-field coherent TR of ``N`` electrons,

    d^2U/(dw dOmega) = d^2U_GF/(dw dOmega) N^2 |F_long(w)|^2
                       [2c/(w r_b sin th) J_1(w r_b sin th / c) - (...) T]^2,

with the Ginzburg-Frank single-electron energy of Eq. (8), a uniform-disc
transverse charge density of radius ``r_b`` and the screen term ``T`` that
vanishes for an infinitely large screen. Fig. 14 plots its integral over the
acceptance cone ``th < th_det`` as an absolute energy in µJ per GHz for a 1 nC
bunch at ``gamma = 1000`` with zero length; the dashed curves are the infinite
screen (``T = 0``) for ``r_b`` = 50, 150, 350 and 1000 µm and ``th_det`` = 100
and 10 mrad. Fig. 15 (left) is ``|F_long|^2`` of a Gaussian bunch with
``sigma = 50 fs``. Every parameter is stated; nothing is fitted.

Here the bunch runs through the production scorer as a conductor-to-vacuum
exit crossing (a perfect conductor emits the same Ginzburg-Frank pattern
forward as the report's backward geometry) with the population ``N = Q/e``
taken from the 1 nC charge:

- ``empirical`` law: the disc as a resolved transverse density profile;
- ``sampled`` law: the disc carried by the histories' own crossing positions,
  so the form factor comes from the distinct-pair estimator alone;
- ``gaussian`` law: the 50 fs RMS crossing-time law against Fig. 15.

``photons/eV/sr`` per bunch times ``E`` and the 1 GHz width ``h * 1 GHz`` is
the plotted energy. The equal-area lattice points stand in for iid draws.

Data: ``tests/data/optical/casalbuoni-2005-fig14-15.json``, curve vertices
from the PDF's vector paths. The published curves lie about 0.5 % (0.2-0.7 %)
above Eq. (37) evaluated with CODATA constants (cause not identified; inside
the tolerance). The ``r_b`` = 1000 µm, 100 mrad curve falls below Eq. (37) under
1.2 THz, by up to 1.9 %: an independent quadrature of the equation departs
from the figure identically, so that corner is a figure/equation difference
of the source and gets a wider tolerance (a smaller shortfall of the same kind
shows at 350 µm, 100 mrad; it grows with ``r_b``). The laws here are even, so
phase signs are not tested. Not covered: finite screens
(solid curves), the near-field curve of Fig. 15 (right), any measurement.

Validation: optical-ctr-population
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j1

from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.spectrum.coherent_population import CoherentSamplingWarning
from pyrite.montecarlo.spectrum.optical_transition import (
    OpticalBunchSource,
    optical_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import BoundaryCrossings

_DATA = json.loads(
    (Path(__file__).parents[2] / "data" / "optical" / "casalbuoni-2005-fig14-15.json").read_text()
)
_GAMMA = _DATA["parameters"]["gamma"]
_CHARGE_PC = _DATA["parameters"]["charge_nC"] * 1e3
_SIGMA_FS = _DATA["parameters"]["gaussian_rms_fs"]
_H_EV_S = 4.135667696e-15
_E_C = 1.602176634e-19
_C_ANG_PER_S = 2.99792458e18
_CONDUCTOR = [np.array(-1e12 + 1e9j)]
_F_THZ = np.array([0.6, 1.0, 2.0, 4.0, 8.0, 15.0, 28.0])
_CURVES = {
    (curve["theta_det_rad"], curve["bunch_radius_um"]): np.array(curve["f_THz_uJ_per_GHz"])
    for curve in _DATA["fig14"]["curves"]
}


def _published(theta_det, radius_um, f_thz):
    curve = _CURVES[theta_det, radius_um]
    return np.interp(np.log(f_thz), np.log(curve[:, 0]), curve[:, 1])


def _exit_crossings(xy_ang):
    """Normal conductor-to-vacuum exit crossings at the given transverse points."""
    m = len(xy_ang)
    return BoundaryCrossings(
        electron_id=np.arange(m), track_id=np.arange(m), charge=-np.ones(m, np.int8),
        position_ang=np.column_stack([xy_ang, np.full(m, 1e3)]),
        direction=np.tile([0.0, 0.0, 1.0], (m, 1)),
        kinetic_energy_keV=np.full(m, M_E_EV * (_GAMMA - 1) / 1e3), time_ang=np.zeros(m),
        interface=np.ones(m, int), normal=np.tile([0.0, 0.0, 1.0], (m, 1)),
        layer_before=np.zeros(m, int), layer_after=-np.ones(m, int),
        z_interfaces_ang=np.array([0.0, 1e3]), n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


def _disc(radius_um, count):
    """Equal-area (sunflower) points of a uniform disc [Å]."""
    i = np.arange(count) + 0.5
    r, angle = radius_um * 1e4 * np.sqrt(i / count), i * np.pi * (3 - np.sqrt(5))
    return np.column_stack([r * np.cos(angle), r * np.sin(angle)])


def _cone(theta_det, n_theta, n_phi):
    """Directions and solid-angle weights [sr]: Gauss-Legendre in ``asinh(gamma theta)``."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    t_max = np.arcsinh(_GAMMA * theta_det)
    t = 0.5 * t_max * (x + 1)
    theta = np.sinh(t) / _GAMMA
    w_theta = 0.5 * t_max * w * np.cosh(t) / _GAMMA * np.sin(theta)
    phi = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    return n.reshape(-1, 3), np.repeat(w_theta, n_phi) * 2 * np.pi / n_phi


def _detected_uJ_per_GHz(crossings, source, f_thz, theta_det, *, n_theta, n_phi):
    """Bunch energy inside the cone per GHz, from the production scorer."""
    n, weights = _cone(theta_det, n_theta, n_phi)
    energy = _H_EV_S * f_thz * 1e12
    result = optical_transition_radiation(
        crossings, energy[:, None], n[None], _CONDUCTOR,
        source=source, n_histories=int(crossings.electron_id.max()) + 1,
    )  # fmt: skip
    photons_per_eV = result.bunch_yield @ weights
    return photons_per_eV * energy * _E_C * (_H_EV_S * 1e9) * 1e6, result


def _profile_source(radius_um, theta_det, *, rms_t_fs=None):
    """Uniform disc as a resolved density profile, pitch below ``0.8/k_par``."""
    k_par = _F_THZ.max() * 1e12 * _H_EV_S / HBARC_EV_ANG * theta_det
    count = max(400, int(np.ceil(np.pi * (radius_um * 1e4 * k_par / 0.8) ** 2)))
    return OpticalBunchSource(
        "empirical", _CHARGE_PC, "TESLA 2005-15 uniform disc of radius r_b",
        profile_ang=np.column_stack([np.zeros(count), _disc(radius_um, count)]),
        profile_kind="profile", time_resolution_ang=1.0,
        transverse_resolution_ang=radius_um * 1e4 * np.sqrt(np.pi / count),
    )  # fmt: skip


def _eq37_uJ_per_GHz(f_thz, radius_um, theta_det):
    """Eq. (37) with ``T = 0`` and ``F_long = 1``, integrated over the cone (SI)."""
    beta = np.sqrt(1 - _GAMMA**-2)
    k_rb = 2 * np.pi * f_thz * 1e12 / _C_ANG_PER_S * radius_um * 1e4

    def integrand(theta):
        x = k_rb * np.sin(theta)
        disc = 1.0 if x < 1e-8 else (2 * j1(x) / x) ** 2
        ginzburg_frank = beta**2 * np.sin(theta) ** 2 / (1 - beta**2 * np.cos(theta) ** 2) ** 2
        return ginzburg_frank * disc * 2 * np.pi * np.sin(theta)

    cone = quad(integrand, 0, theta_det, limit=2000, points=[1e-3, 3e-3, 1e-2])[0]
    population = _CHARGE_PC * 1e-12 / _E_C
    hbar_J_s = _H_EV_S * _E_C / (2 * np.pi)
    # e^2/(4 pi^3 eps0 c) = alpha hbar / pi^2 [J s / sr]; d(omega) = 2 pi GHz.
    return population**2 * ALPHA_FS * hbar_J_s / np.pi**2 * cone * 2 * np.pi * 1e9 * 1e6


@pytest.mark.parametrize("theta_det", [0.1, 0.01])
@pytest.mark.parametrize("radius_um", [50, 150, 350, 1000])
def test_resolved_disc_profile_reproduces_fig14_absolute_energy(theta_det, radius_um):
    source = _profile_source(radius_um, theta_det)
    energy, result = _detected_uJ_per_GHz(
        _exit_crossings(np.zeros((2, 2))), source, _F_THZ, theta_det, n_theta=128, n_phi=2
    )
    equation = np.array([_eq37_uJ_per_GHz(f, radius_um, theta_det) for f in _F_THZ])
    np.testing.assert_allclose(energy, equation, rtol=2e-3)  # lattice + cone quadrature
    relative = energy / _published(theta_det, radius_um, _F_THZ) - 1
    source_corner = (theta_det == 0.1) & (radius_um == 1000) & (_F_THZ < 1.2)
    assert np.max(np.abs(relative[~source_corner])) < 0.01
    if np.any(source_corner):  # the figure lies below its own Eq. (37) here (docstring)
        assert np.all((relative[source_corner] > 0) & (relative[source_corner] < 0.025))
    # 1 nC: the N(N - 1) pair term is the whole signal; the self term is 1/N of it.
    population = source.physical_electrons
    np.testing.assert_allclose(population, 6.2415e9, rtol=1e-5)
    np.testing.assert_allclose(
        result.yield_per_electron,
        result.self_yield + (population - 1) * result.form_factor * result.pair_yield,
        rtol=1e-12,
    )
    assert np.all(result.self_yield <= 1e-9 * np.max(result.yield_per_electron))


@pytest.mark.parametrize(
    ("theta_det", "radius_um", "f_thz"),
    [
        (0.1, 350, np.array([0.6, 1.0, 2.0, 4.0, 8.0])),
        (0.01, 1000, np.array([0.6, 2.0, 5.0, 10.0, 18.0, 28.0])),
    ],
)
def test_sampled_disc_pair_estimator_reproduces_fig14(theta_det, radius_um, f_thz):
    """The form factor from crossing positions alone: ``|F|^2 = 1`` is applied."""
    histories = 1000
    source = OpticalBunchSource.sampled(_CHARGE_PC, "TESLA 2005-15 uniform disc, lattice points")
    with warnings.catch_warnings():
        # Lattice points give -O(1/M) pair estimates at the disc form-factor zeros.
        warnings.simplefilter("ignore", CoherentSamplingWarning)
        energy, result = _detected_uJ_per_GHz(
            _exit_crossings(_disc(radius_um, histories)), source, f_thz, theta_det,
            n_theta=48, n_phi=4,
        )  # fmt: skip
    assert np.all(result.form_factor == 1.0)
    published = _published(theta_det, radius_um, f_thz)
    assert np.max(np.abs(energy / published - 1)) < 0.01
    # The curve is not a plateau: the disc form factor removes a third or more.
    assert published[-1] < 0.67 * published[0]
    equation = np.array([_eq37_uJ_per_GHz(f, radius_um, theta_det) for f in f_thz])
    np.testing.assert_allclose(energy, equation, rtol=4e-3)


def test_gaussian_law_reproduces_fig15_form_factor():
    curve = np.array(_DATA["fig15_left"]["f_THz_form_factor_squared"])
    f_thz, published = curve[::3, 0], curve[::3, 1]
    point = OpticalBunchSource.gaussian(_CHARGE_PC, "point bunch", rms_t_fs=1e-9)
    bunch = OpticalBunchSource.gaussian(_CHARGE_PC, "TESLA 2005-15 Fig. 15", rms_t_fs=_SIGMA_FS)
    crossings = _exit_crossings(np.zeros((2, 2)))
    short, _ = _detected_uJ_per_GHz(crossings, point, f_thz, 0.1, n_theta=32, n_phi=1)
    long, result = _detected_uJ_per_GHz(crossings, bunch, f_thz, 0.1, n_theta=32, n_phi=1)
    np.testing.assert_allclose(long / short, published, atol=3e-3)
    np.testing.assert_allclose(result.form_factor[:, 0], published, atol=3e-3)
    assert published[0] > 0.97 and published[-1] < 0.01  # spans the whole turnover
