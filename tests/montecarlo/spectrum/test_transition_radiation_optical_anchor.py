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

Validation: optical-planar-tr
"""

import numpy as np

from pyrite.materials.crystal import HC_EV_ANG
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
    # hemisphere integral over 600-900 nm. Unresolved normalization (evidence
    # gate in the ledger); pinned so a change in either is noticed.
    ratio = published[wavelength >= 600.0] / ours[wavelength >= 600.0]
    np.testing.assert_allclose(ratio, 0.696, rtol=0.005)
