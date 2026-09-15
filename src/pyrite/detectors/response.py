"""Legacy EDS response operators owned by the detector domain."""

import numpy as np

from .._grid_semantics import is_uniform_grid, node_bin_edges_and_widths
from ..materials.attenuation import _mu_total_inv_ang


def detector_efficiency(E_eV, polymer_nm=300.0, al_nm=40.0, grid_open=0.78):
    """
    Soft X-ray collection efficiency of a polymer-window SDD (the dominant
    loss below ~2 keV; the Si diode itself is ~fully absorbing there):

        QE(E) = grid_open * T_polymer(E) * T_Al(E)

    computed from Henke f2 attenuation for a polyimide film (C22 H10 N2 O5,
    rho = 1.42 g/cm^3) of thickness polymer_nm, an aluminum light-blocking
    coat of al_nm, and the etched-Si support grid (open-area fraction
    grid_open; the grid bars are opaque at these energies). Defaults model a
    Moxtek AP3.3-class window -- the actual Oxford UltimMax window is
    proprietary, so treat the parameters as tunable. Carries the C, N, O
    edge structure (e.g. the deep notch just above the O-K edge at 532 eV).

    Parameters
    ----------
    E_eV
        Photon energy in eV.
    polymer_nm
        Polyimide window thickness in nm.
    al_nm
        Aluminium light-blocking coating thickness in nm.
    grid_open
        Support-grid open-area fraction.

    Returns
    -------
    numpy.ndarray
        Dimensionless efficiency shaped like ``E_eV``.
    """
    E = np.asarray(E_eV, dtype=float)
    # polyimide C22 H10 N2 O5: formula units per Ang^3 at rho = 1.42 g/cm^3
    n_f = 1.42 / 382.31 * 0.602214076
    mu_poly = sum(
        count * _mu_total_inv_ang([(el, n_f)], E)
        for el, count in (("C", 22), ("H", 10), ("N", 2), ("O", 5))
    )
    n_al = 2.70 / 26.982 * 0.602214076
    mu_al = _mu_total_inv_ang([("Al", n_al)], E)
    # Above the Henke ceiling (~30 keV) the window attenuation is unavailable
    # (NaN, and inf at E=0); the thin polymer/Al window is transparent to hard
    # X-rays, so treat an unavailable mu as zero -> QE -> grid_open rather than
    # NaN (which would clip the detected spectrum on a log plot). NB this
    # window-transmission model assumes the Si diode is fully absorbing, so it
    # OVERestimates QE above ~20 keV where the Si itself turns transparent -- use
    # the Timepix forward model (plot_timepix_detected) for a faithful hard-X-ray
    # detector response.
    mu_poly = np.nan_to_num(mu_poly, nan=0.0, posinf=0.0, neginf=0.0)
    mu_al = np.nan_to_num(mu_al, nan=0.0, posinf=0.0, neginf=0.0)
    return grid_open * np.exp(-mu_poly * polymer_nm * 10.0 - mu_al * al_nm * 10.0)


def convolve_detector(E_grid_eV, spec, fwhm_eV):
    """
    Convolve a spectrum with a unit-area Gaussian of the given FWHM [eV].
    Output always matches the input length for ANY fwhm (np.convolve with
    mode="same" returned an OVERSIZED array whenever the kernel outgrew the
    spectrum -- large FWHM on a short grid; truncating the kernel instead
    distorts the lineshape). Edges are zero-padded: counts blurred past the
    grid ends are lost, consistent with a detector band edge.

    On a uniform grid this is exactly the previous sample-space
    ``gaussian_filter1d`` path (bit-for-bit; one sigma in bins is a fixed
    energy width there). On a graded grid, sample-space bins are not a fixed
    energy width, so the kernel is evaluated directly in physical energy and
    each source node is weighted by its own local width
    (:func:`node_bin_edges_and_widths`) rather than one shared ``dE``.

    Parameters
    ----------
    E_grid_eV
        Photon-energy coordinate in eV; need not be uniform.
    spec
        Spectral samples on ``E_grid_eV``.
    fwhm_eV
        Positive Gaussian full width at half maximum in eV.

    Returns
    -------
    numpy.ndarray
        Convolved spectrum with the same shape and units as ``spec``.
    """
    E = np.asarray(E_grid_eV, dtype=float)
    values = np.asarray(spec, dtype=float)
    sigma_eV = fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if is_uniform_grid(E):
        from scipy.ndimage import gaussian_filter1d

        dE = E[1] - E[0]
        sigma_bins = sigma_eV / dE
        return gaussian_filter1d(values, sigma_bins, mode="constant", cval=0.0)
    # Direct quadrature in physical energy: kernel[i, j] is the Gaussian
    # response at E[i] to a unit-density source at E[j], weighted by E[j]'s
    # own local width so density * width is the per-node mass being blurred.
    _, widths = node_bin_edges_and_widths(E)
    kernel = np.exp(-0.5 * ((E[:, None] - E[None, :]) / sigma_eV) ** 2)
    kernel *= widths[None, :] / (sigma_eV * np.sqrt(2.0 * np.pi))
    flattened = values.reshape(-1, E.size)
    return (flattened @ kernel.T).reshape(values.shape)
