"""Position density of a translation-averaged planar quantum ensemble."""

import numpy as np


def spatial_density(
    x_ang: np.ndarray,
    density: np.ndarray,
    coefficients: np.ndarray,
    orders: np.ndarray,
    kappa_inv_ang: np.ndarray,
    period_ang: float,
) -> np.ndarray:
    """Reconstruct the folded cell density [angstrom**-1], without conditioning.

    Source: Born rule ``p(x) = <x|rho|x>`` (MIT OCW 5.74, Spring 2009,
    Lecture 12, The Density Matrix) and the Bloch basis of Andersen et al.
    (1983), Phys. Scr. 28, 308. ``density[r,n,m]`` contains retained weight
    per incident electron; ``coefficients[r,h,n]`` contains orthonormal
    eigenvectors in integer plane-wave ``orders`` at ``kappa_inv_ang[r]``.
    All inputs must refer to the same instantaneous basis.

    Assumes translation averaging (no inter-block coherences). With period d,
    ``psi[r,n](x) = sum_h C[r,h,n] exp(i(kappa[r]+2pi h/d)x) / sqrt(d)``
    and ``p(x) = sum_rnm psi[r,n](x) rho[r,n,m] psi[r,m](x)*``.
    Folding the normalized supercell into one cell removes its factor
    ``1/n_cells``. The cell integral is ``sum_r trace(rho[r])``, so the
    result includes survival weight. No clipping or renormalization is applied.
    In the pure-state limit it is the squared coherent wavefunction; mixtures
    add linearly. This does not reconstruct a unique ensemble wavefunction.

    Validation: cr-planar-spatial-density-reconstruction
    """
    x = np.asarray(x_ang, dtype=float)
    rho = np.asarray(density, dtype=complex)
    vectors = np.asarray(coefficients, dtype=complex)
    h = np.asarray(orders, dtype=float)
    kappa = np.asarray(kappa_inv_ang, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or x.size == 0:
        raise ValueError("x_ang must be a nonempty finite one-dimensional grid")
    if not np.isfinite(period_ang) or period_ang <= 0:
        raise ValueError("period_ang must be positive and finite")
    if h.ndim != 1 or kappa.ndim != 1 or not np.all(np.isfinite(kappa)):
        raise ValueError("orders and kappa_inv_ang must be one-dimensional and finite")
    if not np.all(np.isfinite(h)) or not np.all(h == np.round(h)) or np.unique(h).size != h.size:
        raise ValueError("orders must contain distinct finite integers")
    if rho.ndim != 3 or rho.shape[1] != rho.shape[2]:
        raise ValueError("density must have shape (n_blocks, n_states, n_states)")
    if vectors.shape != (kappa.size, h.size, rho.shape[1]) or rho.shape[0] != kappa.size:
        raise ValueError("coefficients, density and Bloch grid shapes do not match")
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(vectors)):
        raise ValueError("density and coefficients must be finite")
    # Process blocks separately: temporary storage scales with one cell basis.
    result = np.zeros(x.size)
    for block, k in enumerate(kappa):
        waves = np.exp(1j * np.multiply.outer(x, k + 2 * np.pi * h / period_ang))
        psi = waves @ vectors[block] / np.sqrt(period_ang)
        result += np.einsum("xn,nm,xm->x", psi, rho[block], psi.conj(), optimize=True).real
    return result
