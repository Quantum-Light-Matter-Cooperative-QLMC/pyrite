"""Small numerical diagnostics for the planar channeling demo."""

import numpy as np

from pyrite.materials.channeling import PlanarProfile, longitudinal_kinematics
from pyrite.materials.crystal import M_E_EV


def critical_angle_rad(profile: PlanarProfile, kinetic_energy_eV: float) -> float:
    """Return the existing Lindhard well-depth diagnostic in radians.

    Source: the transverse Hamiltonian of Andersen et al., Phys. Scr. 28,
    308 (1983), and Azadegan (2013), Eqs. 2–6. In the small-angle limit,
    equating ``pv theta**2 / 2`` at the well minimum to the planar well
    depth gives ``sqrt(2 (max U - min U) / pv)``. A flat potential gives
    zero. This is a display diagnostic, not an entry selection rule.

    Validation: cr-transverse-eigenproblem
    """
    minimum, maximum = profile.extrema()
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    return float(np.sqrt(2.0 * (maximum - minimum) / (gamma * M_E_EV * beta**2)))


def far_field_maps(gamma: float, coherency: np.ndarray, span: float = 3.0, n: int = 121):
    """Fixed-basis circular far-field maps of a dipole line on a ``gamma theta`` grid.

    ``coherency`` is the transverse ``C_ab = sum d_a d_b^*`` (``a, b = x, y``).
    Returns ``(axis, intensity, phase, purity)``: ``intensity[s]`` is the
    photon density per ``(1/gamma)^2`` per line photon of component
    ``s = +, -``, summed incoherently over the eigenvectors of ``C``;
    ``phase[s]`` is that of the leading eigenvector, up to its arbitrary global
    phase (exact when ``purity``, its share of ``tr C``, is one). Display assembly of
    :func:`~pyrite.montecarlo.channeling.dipole.dipole_far_field`.

    Validation: cr-dipole-far-field-amplitude
    """
    from pyrite.montecarlo.channeling.dipole import dipole_far_field

    values, vectors = np.linalg.eigh(np.asarray(coherency, dtype=complex))
    total = float(values.sum())
    if not total > 0.0:
        raise ValueError("coherency has no emission")
    axis = np.linspace(-span, span, n)
    tx, ty = np.meshgrid(axis / gamma, axis / gamma, indexing="ij")
    intensity = np.zeros((2, n, n))
    for value, vector in zip(values, vectors.T, strict=True):
        if value > 1e-12 * total:
            field = dipole_far_field(tx, ty, gamma, vector)[:2]
            intensity += value / total * np.abs(field) ** 2 / gamma**2
    phase = np.angle(dipole_far_field(tx, ty, gamma, vectors[:, -1])[:2])
    return axis, intensity, phase, float(values[-1] / total)


def far_field_figure(gamma: float, coherency: np.ndarray, title: str):
    """Draw intensity and phase of the ``E_+`` and ``E_-`` far-field components."""
    import matplotlib.pyplot as plt

    axis, intensity, phase, purity = far_field_maps(gamma, coherency)
    beta = np.sqrt(1.0 - gamma**-2)
    grid = np.hypot(*np.meshgrid(axis, axis, indexing="ij")) / gamma
    ratio = (1.0 - beta) / (1.0 - beta * np.cos(grid))
    extent = (axis[0], axis[-1], axis[0], axis[-1])
    peak = intensity.max()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    for k, label in enumerate(("E₊ (σ = +1)", "E₋ (σ = −1)")):
        mesh = axes[2 * k].imshow(
            intensity[k].T, origin="lower", extent=extent, vmin=0, vmax=peak, cmap="magma"
        )
        contours = axes[2 * k].contour(
            axis, axis, ratio.T, levels=[0.25, 0.5, 0.75], colors="w", linewidths=0.6
        )
        axes[2 * k].clabel(contours, fmt="E/E₀=%.2g", fontsize=6)
        axes[2 * k].set(title=f"{label} intensity", xlabel="γθx", ylabel="γθy")
        fig.colorbar(mesh, ax=axes[2 * k], label="photons per (1/γ)² per line photon")
        shown = np.where(intensity[k] > 1e-3 * peak, phase[k], np.nan)
        mesh = axes[2 * k + 1].imshow(
            shown.T, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        axes[2 * k + 1].set(title=f"{label} phase", xlabel="γθx", ylabel="γθy")
        fig.colorbar(mesh, ax=axes[2 * k + 1], label="rad")
    fig.suptitle(f"{title} · leading-mode purity {purity:.3f}")
    return fig
