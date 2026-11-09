"""Published optical TR comparison: Coenen, Vesseur and Polman, 30 keV electrons on gold.

Reference: T. Coenen, E. J. R. Vesseur and A. Polman, "Angle-resolved
cathodoluminescence spectroscopy", arXiv:1107.3632v1 (2011; the arXiv record
was later withdrawn, the letter is Appl. Phys. Lett. 99, 143103,
doi:10.1063/1.3644985). Fig. 3 is their calculated TR emission probability per
incoming 30 keV electron per nm, hemisphere-integrated, for a gold substrate
with Johnson–Christy optical constants (their ref. 12). Values below were
digitized from the 200 dpi rendering of v1 page 11 (blue curve centroid per
column, linear axes from the frame and tick marks: 400–800 nm ticks, 1.0–2.2
frame); digitization uncertainty about 0.005 x 1e-7 /nm.

Fig. 4c (APL version) is the *measured* angular pattern at 400 and 900 nm
(40 nm band-pass filters), integrated over azimuth on an equally spaced
(theta, phi) grid of photon flux per solid angle, phi = 150-210 deg excluded,
and normalized to its peak. Values below were digitized from the embedded
752x1485 px raster (pdfimages, page 4) by colour (red 400 nm, green 900 nm),
polar centre (376, 470) px and unit radius 330 px (0.6/0.8 grid circles agree
to 0.3 %); median radius per 1-deg bin, about 0.01 uncertainty, worse where
the curves overlap below 0.3.

Validation: optical-planar-tr
"""

import numpy as np

from pyrite.materials.crystal import ALPHA_FS, HC_EV_ANG
from pyrite.materials.optical import gold_johnson_christy_response
from pyrite.montecarlo.spectrum.transition_radiation import (
    beta_from_kinetic_eV,
    planar_transition_radiation,
)

#: Coenen et al. Fig. 3 [1e-7 photons / nm / electron] at vacuum wavelength [nm].
COENEN_FIG3 = {
    375: 1.5909, 400: 1.5238, 425: 1.4358, 450: 1.3413, 475: 1.2350, 485: 1.2154,
    500: 1.2830, 525: 1.5945, 550: 1.7963, 600: 2.0012, 650: 2.0564, 700: 2.0314,
    750: 1.9801, 800: 1.9236, 850: 1.8692, 900: 1.8151,
}  # fmt: skip


def _hemisphere_per_nm(wavelength_nm, kinetic_eV=30e3):
    """Backward TR of a normally incident electron, integrated over 2 pi sr, per nm."""
    response = gold_johnson_christy_response()
    energy = HC_EV_ANG / (wavelength_nm * 10.0)
    mu, weight = np.polynomial.legendre.leggauss(64)
    mu, weight = 0.5 * (mu + 1.0), 0.5 * weight
    direction = np.stack([np.sqrt(1.0 - mu**2), 0.0 * mu, mu], axis=-1)
    beta = beta_from_kinetic_eV(kinetic_eV)
    yield_ = planar_transition_radiation(
        energy[:, None], direction[None], np.array([0.0, 0.0, -beta]), [],
        substrate_eps=response.epsilon(energy)[:, None],
    )  # fmt: skip
    return 2.0 * np.pi * (yield_ @ weight) * energy / wavelength_nm  # dE/dlambda = E/lambda


def test_coenen_gold_spectral_shape_with_johnson_christy_optics():
    wavelength = np.array(sorted(COENEN_FIG3), dtype=float)
    published = np.array([COENEN_FIG3[int(w)] for w in wavelength]) * 1e-7
    ours = _hemisphere_per_nm(wavelength)
    peak = wavelength == 650.0
    shape, reference = ours / ours[peak], published / published[peak]
    # Interband region (375-550 nm): J&C node spacing (~0.1 eV) and the unknown
    # interpolation of the published calculation dominate; red band is tighter.
    blue = wavelength <= 550.0
    np.testing.assert_allclose(shape[blue], reference[blue], rtol=0.0, atol=0.04)
    np.testing.assert_allclose(shape[~blue], reference[~blue], rtol=0.0, atol=0.006)
    assert wavelength[np.argmin(ours[wavelength < 600])] == 485.0  # plasmon-region dip
    # Absolute scale: the published curve is a constant 0.696 of this
    # hemisphere integral over 600-900 nm. The APL text confirms Fig. 3 is the
    # full upper hemisphere (not the 1.46 pi sr mirror acceptance); a lower
    # effective beam energy is excluded (shape spread grows below 30 keV). The
    # paper's absolute check (3.9e6/s detected vs ~4e6/s expected) rests on an
    # unitemized ~50 % collection/QE efficiency, which our scale would need to
    # be ~34 %: not discriminating. The cited theory (RMP 82, 209) reproduces our
    # scale to 1e-3 (test below), so 0.696 belongs to the published curve.
    # Pinned so a change in either is noticed.
    ratio = published[wavelength >= 600.0] / ours[wavelength >= 600.0]
    np.testing.assert_allclose(ratio, 0.696, rtol=0.005)


#: Coenen et al. APL Fig. 4c measured, peak-normalized intensity at polar angle [deg].
COENEN_FIG4C = {
    400: {
        24: 0.163, 28: 0.229, 32: 0.351, 36: 0.528, 40: 0.662, 44: 0.748, 48: 0.831,
        52: 0.936, 56: 0.961, 60: 0.984, 64: 0.942, 68: 0.904, 72: 0.773, 76: 0.584,
        80: 0.354, 84: 0.204,
    },
    900: {
        24: 0.093, 28: 0.110, 32: 0.156, 36: 0.276, 40: 0.446, 44: 0.552, 48: 0.672,
        52: 0.763, 56: 0.848, 60: 0.941, 64: 0.967, 68: 0.970, 72: 0.934, 76: 0.800,
        80: 0.630, 84: 0.411,
    },
}  # fmt: skip


def _band_pattern(center_nm, theta, bandwidth_nm=40.0):
    """Per-sr backward TR per nm, averaged over a flat band-pass filter."""
    response = gold_johnson_christy_response()
    wavelength = np.linspace(center_nm - bandwidth_nm / 2, center_nm + bandwidth_nm / 2, 9)
    energy = HC_EV_ANG / (wavelength * 10.0)
    direction = np.stack([np.sin(theta), 0.0 * theta, np.cos(theta)], axis=-1)
    beta = beta_from_kinetic_eV(30e3)
    yield_ = planar_transition_radiation(
        energy[:, None], direction[None], np.array([0.0, 0.0, -beta]), [],
        substrate_eps=response.epsilon(energy)[:, None],
    )  # fmt: skip
    return np.mean(yield_ * (energy / wavelength)[:, None], axis=0)


def test_coenen_measured_angular_pattern_is_the_azimuth_integrated_flux():
    for center, rms_limit in ((400, 0.035), (900, 0.065)):
        theta_deg = np.array(sorted(COENEN_FIG4C[center]), dtype=float)
        measured = np.array([COENEN_FIG4C[center][int(t)] for t in theta_deg])
        fine = np.radians(np.arange(1.0, 90.0, 0.25))
        per_sr = _band_pattern(center, fine)
        azimuth_integrated = per_sr * np.sin(fine)  # dN/dtheta: phi-integral over dOmega
        model = np.interp(
            theta_deg, np.degrees(fine), azimuth_integrated / azimuth_integrated.max()
        )
        wrong = np.interp(theta_deg, np.degrees(fine), per_sr / per_sr.max())
        assert np.sqrt(np.mean((model - measured) ** 2)) < rms_limit
        # The per-steradian convention is excluded by the measurement.
        assert np.sqrt(np.mean((wrong - measured) ** 2)) > 0.1


def _garcia_de_abajo_hemisphere_per_nm(wavelength_nm, kinetic_eV=30e3, n=400):
    """Coenen's cited theory: Garcia de Abajo, RMP 82, 209 (arXiv:0903.1669), Sec. IV.C.

    ``mu_1`` (Eq. 52), ``D`` (Eq. 54), ``f_H = i k cos(theta) D mu_1`` at
    ``q_par = k sin(theta)``, and the per-solid-angle probability
    ``Gamma_CL = |f|^2 / (4 pi^2 hbar k)`` (Sec. IV.B), integrated over the
    vacuum hemisphere with ``dOmega = 2 pi sin(theta) dtheta`` (the printed
    Eq. 56 omits ``sin(theta)``). Gaussian units, ``hbar = c = 1``, energies in eV.
    """
    response = gold_johnson_christy_response()
    beta = beta_from_kinetic_eV(kinetic_eV)
    t, w = np.polynomial.legendre.leggauss(n)
    theta, w = 0.25 * np.pi * (t + 1.0), 0.25 * np.pi * w
    out = []
    for lam in np.atleast_1d(wavelength_nm):
        k = HC_EV_ANG / (lam * 10.0)
        eps = complex(response.epsilon(np.array([k]))[0])
        q_par = k * np.sin(theta)
        qz1 = np.sqrt(k**2 - q_par**2 + 0j)
        qz2 = np.sqrt(k**2 * eps - q_par**2 + 0j)
        qz2 = np.where(qz2.imag < 0, -qz2, qz2)
        q2 = q_par**2 + (k / beta) ** 2
        mu1 = (-(k / beta) * eps + qz2) / (q2 - k**2) - (-(k / beta) + qz2) / (q2 - k**2 * eps)
        d = 2j * np.sqrt(ALPHA_FS) * q_par / (qz1 * eps + qz2)
        f = 1j * k * np.cos(theta) * d * mu1
        per_sr = np.abs(f) ** 2 / (4 * np.pi**2 * k)
        out.append(2 * np.pi * np.sum(w * np.sin(theta) * per_sr) * k / lam)
    return np.array(out)


def test_cited_theory_reproduces_our_absolute_scale_not_the_published_curve():
    wavelength = np.array([375.0, 485.0, 650.0, 900.0])
    ours = _hemisphere_per_nm(wavelength)
    np.testing.assert_allclose(_garcia_de_abajo_hemisphere_per_nm(wavelength), ours, rtol=1e-3)
