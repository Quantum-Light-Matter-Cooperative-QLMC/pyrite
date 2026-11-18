"""Lattice-coherent share of the amorphous bremsstrahlung continuum (#1250).

BremsLib (and EEDL) tabulate isolated-atom bremsstrahlung. In a crystal, the
Debye--Waller share ``e^{-2W}`` of that emission is lattice-coherent, and the
CXR line route already scores it as the CBS lines with ``|g| <= G``. Adding both
counts it twice. This module supplies ``S_j(T, theta, k; G)``, the
orientation-averaged Born--Debye--Waller CBS-only line yield of those
reflections per unit path, so the continuum can be scored with it removed.

The orientation average of one reciprocal-lattice shell ``|g|`` depends on no
crystal: per unit ``|U_g|^2`` it is a universal kernel ``H(|g|; T, theta, x)``.
It is binned in the resonance variable

    x = hbar omega (1 - beta cos theta) / (hbar c |g| beta),

which is ``g_hat . v_hat`` for the classical root, so the kernel's shape is
nearly invariant across ``|g|``, ``T`` and ``theta`` and interpolates well
(the line energy itself moves with ``|g|``; interpolating at fixed photon
energy fails per bin). The released table holds ``H`` on a node grid; the
crystal enters only at evaluation. Because the orientation average of one
reflection depends on ``|g|`` alone, the share of any scored set is the exact
sum ``S = sum_g |U_g|^2 H(|g|)`` over the reflections the line route scores:
the selector's ``|g| <= G`` set, an explicit list, or an incomplete star.
Each reflection's share is split among the elements by their incoherent
weights ``n_j |u_j(g)|^2``.

Interpolation (measured on #1250, P5a): linear in ``ln |g|`` and in
``theta`` at fixed ``x``; between ``T`` nodes the bin total is interpolated
in ``ln T`` and the normalized ``x`` shape linearly.

Validation: brem-lattice-coherent-fraction
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from .cxr._recoil import (
    _line_recoil_terms,
    _quantum_line_factors,
    _recoil_line_allowed,
    _recoil_slope_factor,
)

__all__ = [
    "KERNEL_SCHEMA",
    "LatticeShare",
    "LatticeShareKernel",
    "kernel_x",
    "lattice_share_density",
    "lattice_share_table",
    "reflection_shell_weights",
    "shell_kernel_x",
]

#: Schema of the released kernel table; bump on any change to its contents.
KERNEL_SCHEMA = "pyrite.brem-lattice-share-kernel.v1"

#: Identity marker of crystalline BremsLib continua scored with the share
#: removed (#1250); bump with the kernel release tag or the subtraction rule.
BREM_LATTICE_SHARE_MODEL = "born-dw-cbs-share-v1/tables-brem-lattice-share-1/scored-set"

_ALPHA = 1.0 / 137.035999084
_H_ANG = HBARC_EV_ANG / M_E_EV  # hbar c / m c^2 [Angstrom]

#: Production line kernels drop resonances below this photon energy [eV].
_MIN_LINE_EV = 10.0


def _kinematics(T_keV: float) -> tuple[float, float]:
    gamma = 1.0 + T_keV * 1e3 / M_E_EV
    return gamma, float(np.sqrt(1.0 - 1.0 / gamma**2))


def _frame(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Electron direction ``v = z``, transverse basis, and the photon direction at ``theta``."""
    v = np.array([0.0, 0.0, 1.0])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    return v, e1, e2, np.cos(theta) * v + np.sin(theta) * e1


def _line_weights(gvec, v_hat, n_hat, T_keV):
    """CBS-only line yield per unit path and unit ``|U_g|^2`` [photons/sr/Angstrom/eV^2], and ``hbar omega`` [eV].

    The production CBS amplitude with exact recoil (``cbs-amplitude``,
    ``line-recoil-in-medium-root``) in vacuum kinematics, summed over the two
    polarizations; the line is integrated over its energy profile, so only
    its Jacobian (the recoil-exact slope) enters.
    """
    gamma, beta = _kinematics(T_keV)
    v = beta * v_hat
    g2 = (gvec**2).sum(-1)
    n_dot_g = gvec @ n_hat
    r_num, r_den, inv_E = _line_recoil_terms(gamma, g2, n_dot_g)
    D = 1.0 - v @ n_hat + r_den
    vdg = gvec @ v
    om = (vdg - r_num) / D
    k_dot_v = om * (1.0 - D + r_den)
    k_dot_g = om * n_dot_g
    es = np.cross(n_hat, gvec)
    norm = np.linalg.norm(es, axis=-1, keepdims=True)
    es = np.divide(es, norm, out=np.zeros_like(es), where=norm > 0)
    ep = np.cross(es, n_hat)
    f2 = 0.0
    for e in (es, ep):
        _, f = _quantum_line_factors(
            _H_ANG, gamma, vdg, k_dot_v, k_dot_g, g2, om**2, e @ v, (gvec * e).sum(-1), 0.0
        )
        f2 = f2 + np.abs(f) ** 2
    slope = D * _recoil_slope_factor(om, inv_E)
    ok = (
        _recoil_line_allowed(D, om, inv_E, gamma)
        & (om * HBARC_EV_ANG > _MIN_LINE_EV)
        & (norm[:, 0] > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        w = _ALPHA * om * f2 / M_E_EV**2 / (2.0 * np.pi * slope) / beta  # per Angstrom path
    return np.where(ok, w, 0.0), om * HBARC_EV_ANG


@lru_cache(maxsize=8)
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    # leggauss is an O(n^3) eigen-solve: at n = 4000 it costs ~100x the kernel itself.
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def shell_kernel_x(
    g_invang: float,
    T_keV: float,
    theta_rad: float,
    x_edges: np.ndarray,
    *,
    n_mu: int = 4000,
    n_phi: int = 48,
) -> np.ndarray:
    """Orientation-averaged CBS line yield of one ``|g|`` shell, binned in ``x``.

    Mass per ``x`` bin, per unit path and unit ``|U_g|^2`` [photons/sr/Angstrom/eV^2],
    averaged over the direction of ``g`` relative to the electron velocity and
    photon direction (the sum rule's orientation average). ``mu = g_hat.v_hat``
    uses Gauss--Legendre nodes on ``(0, 1]`` (a resonance needs ``g.v > 0``)
    and ``phi`` a uniform rule; ``n_mu = 4000`` keeps the per-bin aliasing of
    the ``x`` histogram below the interpolation error of the released grid.

    Validation: brem-lattice-coherent-fraction
    """
    v, e1, e2, n = _frame(theta_rad)
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    nodes, weights = _gauss_legendre(int(n_mu))
    mu = 0.5 * (nodes + 1.0)
    M, P = np.meshgrid(mu, phi, indexing="ij")
    s = np.sqrt(np.clip(1.0 - M**2, 0.0, None))
    dirs = (
        M[..., None] * v + s[..., None] * (np.cos(P)[..., None] * e1 + np.sin(P)[..., None] * e2)
    ).reshape(-1, 3)
    wq = (0.5 * weights[:, None] * np.full(n_phi, 2.0 * np.pi / n_phi)).reshape(-1) / (4.0 * np.pi)
    w, E = _line_weights(g_invang * dirs, v, n, T_keV)
    x = kernel_x(E, g_invang, T_keV, theta_rad)
    mass, _ = np.histogram(x, bins=x_edges, weights=w * wq)
    return mass


def kernel_x(E_eV, g_invang, T_keV, theta_rad):
    """Resonance variable ``x = E (1 - beta cos theta) / (hbar c g beta)``.

    Validation: brem-lattice-coherent-fraction
    """
    _, beta = _kinematics(T_keV)
    return (
        np.asarray(E_eV)
        * (1.0 - beta * np.cos(theta_rad))
        / (HBARC_EV_ANG * np.asarray(g_invang) * beta)
    )


@dataclass(frozen=True)
class LatticeShareKernel:
    """Universal kernel ``H(|g|; T, theta, x)`` on a node grid.

    ``mass[i_g, i_T, i_theta, i_x]`` is the :func:`shell_kernel_x` mass of bin
    ``i_x`` [photons/sr/Angstrom/eV^2]. Nodes are logarithmic in ``|g|`` and
    ``T``; ``theta`` is in degrees.
    """

    g_invang: np.ndarray
    T_keV: np.ndarray
    theta_deg: np.ndarray
    x_edges: np.ndarray
    mass: np.ndarray
    n_mu: int
    n_phi: int

    def arrays(self) -> dict[str, np.ndarray]:
        return dict(
            schema=np.array(KERNEL_SCHEMA),
            g_invang=self.g_invang,
            T_keV=self.T_keV,
            theta_deg=self.theta_deg,
            x_edges=self.x_edges,
            mass=self.mass.astype(np.float32),
            n_mu=np.array(self.n_mu),
            n_phi=np.array(self.n_phi),
        )

    @classmethod
    def load(cls, path) -> LatticeShareKernel:
        with np.load(path, allow_pickle=False) as z:
            if str(z["schema"]) != KERNEL_SCHEMA:
                raise ValueError(f"{path}: kernel schema {z['schema']!s}, expected {KERNEL_SCHEMA}")
            return cls(
                g_invang=z["g_invang"],
                T_keV=z["T_keV"],
                theta_deg=z["theta_deg"],
                x_edges=z["x_edges"],
                mass=z["mass"].astype(np.float64),
                n_mu=int(z["n_mu"]),
                n_phi=int(z["n_phi"]),
            )


#: 1 Angstrom^2 in millibarn.
_MB_PER_ANG2 = 1.0e11


def _incoherent_u2(element: str, g: np.ndarray) -> np.ndarray:
    """``|u_j(g)|^2`` of the static Mott--Bethe atomic potential [eV^2 Angstrom^6].

    ``u = 4 pi e^2 (Z - f0(g)) / g^2`` with Cromer--Mann ``f0`` within its
    validity ``|g| / 4 pi <= 6`` and the bare nucleus beyond.
    """
    from ...materials.crystal import E2_EV_ANG, Z_TABLE, cromer_mann_f0

    g = np.atleast_1d(np.asarray(g, dtype=float))
    f0 = np.array([cromer_mann_f0(element, x) if x / (4 * np.pi) <= 6.0 else 0.0 for x in g])
    return (4.0 * np.pi * E2_EV_ANG * (Z_TABLE[element] - f0) / g**2) ** 2


def reflection_shell_weights(crystal, hkls, B_ang2, composition):
    """Distinct ``|g|`` of the scored reflections and each element's ``|U_g|^2`` share.

    Returns ``(g [1/Angstrom], weights)`` where ``weights[element]`` is
    ``sum_{g in shell} |U_g|^2 n_j |u_j|^2 / sum_i n_i |u_i|^2`` [eV^2].
    ``U_g`` is the production coupling (bonding charge included); it carries
    no photon-energy dependence.

    Validation: brem-lattice-coherent-fraction
    """
    from ...materials.crystal import CRYSTALS, U_g, reciprocal_g_vector

    lattice = CRYSTALS[crystal]["lattice"]
    shells: dict[float, float] = {}
    for hkl in hkls:
        g = round(float(reciprocal_g_vector(tuple(hkl), lattice)[1]), 9)
        u = complex(np.asarray(U_g(crystal, tuple(hkl), 5000.0, B_ang2=B_ang2)).ravel()[0])
        shells[g] = shells.get(g, 0.0) + abs(u) ** 2
    g = np.array(sorted(shells))
    total = np.array([shells[x] for x in g])
    incoherent = {el: float(n) * _incoherent_u2(el, g) for el, n in composition}
    norm = sum(incoherent.values())
    return g, {el: total * w / norm for el, w in incoherent.items()}


def _kernel_at_T(kernel: LatticeShareKernel, T_keV: float) -> np.ndarray | None:
    """``H(g_nodes, theta_nodes, x)`` at ``T``; ``None`` outside the kernel's ``T`` range."""
    lt = np.log(kernel.T_keV)
    t = float(np.log(T_keV))
    if not lt[0] <= t <= lt[-1]:
        return None
    i = int(np.clip(np.searchsorted(lt, t) - 1, 0, lt.size - 2))
    w = (t - lt[i]) / (lt[i + 1] - lt[i])
    a, b = kernel.mass[:, i], kernel.mass[:, i + 1]
    ta, tb = a.sum(-1, keepdims=True), b.sum(-1, keepdims=True)
    both = (ta > 0) & (tb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.where(both, np.exp((1 - w) * np.log(ta) + w * np.log(tb)), (1 - w) * ta + w * tb)
        shape_a = np.where(ta > 0, a / ta, 0.0)
        shape_b = np.where(tb > 0, b / tb, 0.0)
    shape = np.where(both, (1 - w) * shape_a + w * shape_b, np.where(ta > 0, shape_a, shape_b))
    return total * shape


def lattice_share_density(kernel, g, weights, T_keV, k_eV):
    """Share density ``S(theta_node, k)`` [photons/(eV sr Angstrom)] per unit path.

    ``g`` and ``weights`` from :func:`reflection_shell_weights` (one element's
    weights, or their sum); rows are the kernel's ``theta`` nodes. Zero
    outside the kernel's ``T`` range and for ``|g|`` above its last node.

    Validation: brem-lattice-coherent-fraction
    """
    k = np.atleast_1d(np.asarray(k_eV, dtype=float))
    out = np.zeros((kernel.theta_deg.size, k.size))
    H = _kernel_at_T(kernel, T_keV)
    keep = (g >= kernel.g_invang[0]) & (g <= kernel.g_invang[-1]) & (weights > 0)
    if H is None or not keep.any():
        return out
    g, weights = g[keep], weights[keep]
    lg = np.log(kernel.g_invang)
    ig = np.clip(np.searchsorted(lg, np.log(g)) - 1, 0, lg.size - 2)
    wg = (np.log(g) - lg[ig]) / (lg[ig + 1] - lg[ig])
    edges = kernel.x_edges
    centres = 0.5 * (edges[:-1] + edges[1:])
    rho = H / np.diff(edges)  # density per unit x
    _, beta = _kinematics(T_keV)
    th = np.deg2rad(kernel.theta_deg)
    scale = HBARC_EV_ANG * g[:, None] * beta / (1.0 - beta * np.cos(th))[None, :]  # (S, Th) eV
    x = k[None, None, :] / scale[..., None]  # (S, Th, K)
    j = np.clip(np.searchsorted(centres, x) - 1, 0, centres.size - 2)
    f = np.clip((x - centres[j]) / (centres[j + 1] - centres[j]), 0.0, 1.0)
    inside = x <= edges[-1]
    t_idx = np.arange(th.size)[None, :, None]

    def at(rows):
        r = rows[:, None, None]
        return (1 - f) * rho[r, t_idx, j] + f * rho[r, t_idx, j + 1]

    dens = ((1 - wg)[:, None, None] * at(ig) + wg[:, None, None] * at(ig + 1)) * inside
    return np.einsum("s,stk->tk", weights, dens / scale[..., None])


@dataclass(frozen=True)
class LatticeShare:
    """Bookkeeping of one corrected table: what was removed and what was left alone."""

    element: str
    removed_fraction: float
    clamped_nodes: int
    uncorrected_T_keV: tuple[float, float] | None


def lattice_share_table(table, kernel, element, n_atoms_per_ang3, g, weights):
    """BremsLib table with the lattice-coherent share subtracted from its DDCS.

    At every table node ``(T1, k/T1, theta)`` the scaled DDCS loses
    ``k S_j / (n_j Z^2)`` (``S_j`` from :func:`lattice_share_density`, in mb/sr),
    clamped at zero; the SDCS follows the node's linear-in-``theta`` solid-angle
    integral so the table stays self-consistent. Incident energies outside the
    kernel's range keep their values (reported). The returned table keeps its
    parent's ``key``/``digest``: they name the BremsLib source, which coupled
    transport checks; the correction is recorded separately by the caller.

    Validation: brem-lattice-coherent-fraction
    """
    from ..._bremslib_table import solid_angle_integral

    T = np.asarray(table.incident_energy_keV, dtype=float)
    kap = np.asarray(table.nominal_reduced_energy, dtype=float)
    theta_deg = np.rad2deg(np.asarray(table.theta_rad, dtype=float))
    base = np.asarray(table.scaled_ddcs_mb_sr, dtype=float)
    ddcs = base.copy()
    z2 = float(table.atomic_number) ** 2
    lo, hi = float(kernel.T_keV[0]), float(kernel.T_keV[-1])
    for i, Ti in enumerate(T):
        if not lo <= Ti <= hi:
            continue
        kr = kap.copy()
        kr[-1] = float(table.top_reduced_energy[i])
        k = kr * Ti * 1e3
        S = lattice_share_density(kernel, g, weights, Ti, k)  # (Th_kernel, K)
        delta = _MB_PER_ANG2 * k[None, :] * S / (float(n_atoms_per_ang3) * z2)
        ddcs[i] -= np.array(
            [np.interp(theta_deg, kernel.theta_deg, delta[:, m]) for m in range(k.size)]
        )
    clamped = int((ddcs < 0).sum())
    ddcs = np.maximum(ddcs, 0.0)
    old = solid_angle_integral(table.theta_rad, base)
    new = solid_angle_integral(table.theta_rad, ddcs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(old > 0, new / old, 1.0)
    sdcs = np.asarray(table.scaled_sdcs_mb, dtype=float) * ratio
    in_range = (T >= lo) & (T <= hi)
    removed = 1.0 - float(new[in_range].sum() / old[in_range].sum()) if in_range.any() else 0.0
    outside = None if in_range.all() else (float(T[~in_range].min()), float(T[~in_range].max()))
    corrected = replace(
        table,
        scaled_ddcs_mb_sr=ddcs.astype(np.asarray(table.scaled_ddcs_mb_sr).dtype),
        scaled_sdcs_mb=sdcs.astype(np.asarray(table.scaled_sdcs_mb).dtype),
    )
    return corrected, LatticeShare(element, removed, clamped, outside)
