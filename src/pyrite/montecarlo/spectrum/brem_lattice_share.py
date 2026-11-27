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
    "BREM_LATTICE_SHARE_ISOTROPIC_MODEL",
    "KERNEL_SCHEMA",
    "IsotropicLatticeShare",
    "LatticeShare",
    "LatticeShareKernel",
    "kernel_x",
    "lattice_share_density",
    "lattice_share_model",
    "isotropic_share_sigma",
    "isotropic_shares_of",
    "lattice_share_table",
    "reflection_shell_weights",
    "shell_kernel_x",
    "without_isotropic_share",
]

#: Schema of the released kernel table; bump on any change to its contents.
KERNEL_SCHEMA = "pyrite.brem-lattice-share-kernel.v1"

#: Identity marker of crystalline BremsLib continua scored with the share
#: removed (#1250); bump with the kernel release tag or the subtraction rule.
#: v2: no subtraction below the line route's electron cutoff ``E_cut_lines_keV``.
BREM_LATTICE_SHARE_MODEL = "born-dw-cbs-share-v2/tables-brem-lattice-share-1/scored-set"

#: Identity marker of crystalline isotropic (EEDL / Bethe--Heitler) continua
#: scored with the theta-integrated share removed (#1261).
BREM_LATTICE_SHARE_ISOTROPIC_MODEL = f"{BREM_LATTICE_SHARE_MODEL}/isotropic-theta-integrated-v1"

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
    return _reflection_shell_weights(crystal, hkls, lambda _hkl: B_ang2, composition)


def _reflection_shell_weights(crystal, hkls, b_for_hkl, composition):
    """Internal per-reflection thermal policy; scalar wrapper is unchanged."""
    from ...materials.crystal import CRYSTALS, U_g, reciprocal_g_vector

    lattice = CRYSTALS[crystal]["lattice"]
    shells: dict[float, float] = {}
    for hkl in hkls:
        g = round(float(reciprocal_g_vector(tuple(hkl), lattice)[1]), 9)
        b = b_for_hkl(tuple(hkl))
        u = complex(np.asarray(U_g(crystal, tuple(hkl), 5000.0, B_ang2=b)).ravel()[0])
        shells[g] = shells.get(g, 0.0) + abs(u) ** 2
    g = np.array(sorted(shells))
    total = np.array([shells[x] for x in g])
    incoherent = {el: float(n) * _incoherent_u2(el, g) for el, n in composition}
    norm = sum(incoherent.values())
    return g, {el: total * w / norm for el, w in incoherent.items()}


def _kernel_at_T(
    kernel: LatticeShareKernel, T_keV: float, rows: np.ndarray | None = None
) -> np.ndarray | None:
    """``H(g_nodes, theta_nodes, x)`` at ``T``; ``None`` outside the kernel's ``T`` range.

    ``rows`` restricts the result to those ``|g|`` nodes. Each ``(g, theta)``
    row is interpolated on its own, so a subset equals the same rows of the
    full result.
    """
    lt = np.log(kernel.T_keV)
    t = float(np.log(T_keV))
    if not lt[0] <= t <= lt[-1]:
        return None
    i = int(np.clip(np.searchsorted(lt, t) - 1, 0, lt.size - 2))
    w = (t - lt[i]) / (lt[i + 1] - lt[i])
    g_nodes = slice(None) if rows is None else rows
    a, b = kernel.mass[g_nodes, i], kernel.mass[g_nodes, i + 1]
    ta, tb = a.sum(-1, keepdims=True), b.sum(-1, keepdims=True)
    both = (ta > 0) & (tb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.where(both, np.exp((1 - w) * np.log(ta) + w * np.log(tb)), (1 - w) * ta + w * tb)
        shape_a = np.where(ta > 0, a / ta, 0.0)
        shape_b = np.where(tb > 0, b / tb, 0.0)
    shape = np.where(both, (1 - w) * shape_a + w * shape_b, np.where(ta > 0, shape_a, shape_b))
    return total * shape


def _share_profiles(kernel, g, T_keV, k, used=None):
    """Per-reflection share density at unit weight, ``(rows, D)``, or ``None``.

    ``rows`` indexes the ``g`` inside the kernel's ``|g|`` range and ``D`` is
    their ``(S, theta_node, K)`` density per unit ``|U_g|^2``. Weights enter
    only through :func:`_share_from_profiles`, so elements sharing a node
    reuse one profile. Every ``g`` row is computed on its own: a row equals
    the one a smaller set of ``g`` would give. ``None`` outside the kernel's
    ``T`` range or with no ``g`` in range. ``used`` (a mask over ``g``) skips
    rows no weight set uses.

    Validation: brem-lattice-coherent-fraction
    """
    in_range = (g >= kernel.g_invang[0]) & (g <= kernel.g_invang[-1])
    rows = np.flatnonzero(in_range if used is None else in_range & used)
    if rows.size == 0:
        return None
    g = g[rows]
    lg = np.log(kernel.g_invang)
    ig = np.clip(np.searchsorted(lg, np.log(g)) - 1, 0, lg.size - 2)
    wg = (np.log(g) - lg[ig]) / (lg[ig + 1] - lg[ig])
    # Interpolate in T only the |g| nodes the shells bracket (#1281): a
    # selector shell spans a few of the kernel's |g| nodes.
    nodes = np.unique(np.concatenate([ig, ig + 1]))
    H = _kernel_at_T(kernel, T_keV, nodes)
    if H is None:
        return None
    ig = np.searchsorted(nodes, ig)
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

    def at(nodes_):
        r = nodes_[:, None, None]
        return (1 - f) * rho[r, t_idx, j] + f * rho[r, t_idx, j + 1]

    dens = ((1 - wg)[:, None, None] * at(ig) + wg[:, None, None] * at(ig + 1)) * inside
    return rows, dens / scale[..., None]


def _share_from_profiles(profiles, weights, shape):
    """``S(theta_node, k)`` of one weight set from :func:`_share_profiles`."""
    if profiles is None:
        return np.zeros(shape)
    rows, D = profiles
    w = weights[rows]
    use = w > 0
    if not use.any():
        return np.zeros(shape)
    if not use.all():
        w, D = w[use], D[use]
    return np.einsum("s,stk->tk", w, D)


def lattice_share_density(kernel, g, weights, T_keV, k_eV):
    """Share density ``S(theta_node, k)`` [photons/(eV sr Angstrom)] per unit path.

    ``g`` and ``weights`` from :func:`reflection_shell_weights` (one element's
    weights, or their sum); rows are the kernel's ``theta`` nodes. Zero
    outside the kernel's ``T`` range and for ``|g|`` above its last node.

    Validation: brem-lattice-coherent-fraction
    """
    k = np.atleast_1d(np.asarray(k_eV, dtype=float))
    g = np.asarray(g, dtype=float)
    weights = np.asarray(weights, dtype=float)
    used = weights > 0
    profiles = _share_profiles(kernel, g, T_keV, k, used) if used.any() else None
    return _share_from_profiles(profiles, weights, (kernel.theta_deg.size, k.size))


@dataclass(frozen=True)
class LatticeShare:
    """Bookkeeping of one corrected table: what was removed and what was left alone."""

    element: str
    removed_fraction: float
    clamped_nodes: int
    uncorrected_T_keV: tuple[float, float] | None


def lattice_share_table(
    table, kernel, element, n_atoms_per_ang3, g, weights, *, T_floor_keV: float = 0.0
):
    """BremsLib table with the lattice-coherent share subtracted from its DDCS.

    At every table node ``(T1, k/T1, theta)`` the scaled DDCS loses
    ``k S_j / (n_j Z^2)`` (``S_j`` from :func:`lattice_share_density`, in mb/sr),
    clamped at zero; the SDCS follows the node's linear-in-``theta`` solid-angle
    integral so the table stays self-consistent. Incident energies outside the
    kernel's range, or below ``T_floor_keV`` (the line route's electron cutoff:
    no CBS line is scored there, so nothing would replace the removed share),
    keep their values (reported). The returned table keeps its
    parent's ``key``/``digest``: they name the BremsLib source, which coupled
    transport checks; the correction is recorded separately by the caller.

    Validation: brem-lattice-coherent-fraction
    """
    entry = (table, element, n_atoms_per_ang3, weights)
    return _lattice_share_tables([entry], kernel, g, T_floor_keV=T_floor_keV)[0]


def _grid_signature(table) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()
        for a in (
            table.incident_energy_keV,
            table.nominal_reduced_energy,
            table.top_reduced_energy,
            table.theta_rad,
        )
    )


def _lattice_share_tables(entries, kernel, g, *, T_floor_keV: float = 0.0):
    """:func:`lattice_share_table` of several ``(table, element, n, weights)`` entries.

    Entries on one node grid share each node's :func:`_share_profiles` (the
    kernel interpolation and resonance gathers); only the weight contraction
    is per element. Each result equals the entry's own
    :func:`lattice_share_table`.

    Validation: brem-lattice-coherent-fraction
    """
    g = np.asarray(g, dtype=float)
    groups: dict[tuple[bytes, ...], list[int]] = {}
    for index, (table, *_) in enumerate(entries):
        groups.setdefault(_grid_signature(table), []).append(index)
    out: list = [None] * len(entries)
    for members in groups.values():
        for index, result in zip(
            members,
            _corrected_group([entries[i] for i in members], kernel, g, float(T_floor_keV)),
            strict=True,
        ):
            out[index] = result
    return out


def _corrected_group(entries, kernel, g, T_floor_keV):
    """Corrected tables of entries sharing one node grid (see :func:`_lattice_share_tables`).

    Validation: brem-lattice-coherent-fraction
    """
    first = entries[0][0]
    T = np.asarray(first.incident_energy_keV, dtype=float)
    kap = np.asarray(first.nominal_reduced_energy, dtype=float)
    theta_deg = np.rad2deg(np.asarray(first.theta_rad, dtype=float))
    lo, hi = max(float(kernel.T_keV[0]), T_floor_keV), float(kernel.T_keV[-1])
    ddcs = [np.asarray(table.scaled_ddcs_mb_sr, dtype=float).copy() for table, *_ in entries]
    weights = [np.asarray(w, dtype=float) for *_, w in entries]
    norms = [float(n) * float(table.atomic_number) ** 2 for table, _, n, _ in entries]
    shape = (kernel.theta_deg.size, kap.size)
    used = np.logical_or.reduce([w > 0 for w in weights])
    for i, Ti in enumerate(T):
        if not lo <= Ti <= hi:
            continue
        kr = kap.copy()
        kr[-1] = float(first.top_reduced_energy[i])
        k = kr * Ti * 1e3
        profiles = _share_profiles(kernel, g, Ti, k, used) if used.any() else None
        for d, w, norm in zip(ddcs, weights, norms, strict=True):
            S = _share_from_profiles(profiles, w, shape)  # (Th_kernel, K)
            delta = _MB_PER_ANG2 * k[None, :] * S / norm
            d[i] -= np.array(
                [np.interp(theta_deg, kernel.theta_deg, delta[:, m]) for m in range(k.size)]
            )
    in_range = (T >= lo) & (T <= hi)
    return [
        _finish_table(table, element, d, in_range)
        for (table, element, _, _), d in zip(entries, ddcs, strict=True)
    ]


def _finish_table(table, element, ddcs, in_range):
    """Clamp ``ddcs``, rescale the SDCS by the solid-angle ratio, and record it.

    Only rows the correction or the clamp can change are integrated: any other
    row keeps its DDCS, so its ratio is exactly one.
    """
    from ..._bremslib_table import solid_angle_integral

    T = np.asarray(table.incident_energy_keV, dtype=float)
    base = np.asarray(table.scaled_ddcs_mb_sr, dtype=float)
    clamped = int((ddcs < 0).sum())
    ddcs = np.maximum(ddcs, 0.0)
    touched = in_range | (base < 0).any(axis=(1, 2))
    old = np.zeros(base.shape[:-1])
    new = np.zeros(base.shape[:-1])
    old[touched] = solid_angle_integral(table.theta_rad, base[touched])
    new[touched] = solid_angle_integral(table.theta_rad, ddcs[touched])
    ratio = np.ones(base.shape[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[touched] = np.where(old[touched] > 0, new[touched] / old[touched], 1.0)
    sdcs = np.asarray(table.scaled_sdcs_mb, dtype=float) * ratio
    removed = 1.0 - float(new[in_range].sum() / old[in_range].sum()) if in_range.any() else 0.0
    outside = None if in_range.all() else (float(T[~in_range].min()), float(T[~in_range].max()))
    corrected = replace(
        table,
        scaled_ddcs_mb_sr=ddcs.astype(np.asarray(table.scaled_ddcs_mb_sr).dtype),
        scaled_sdcs_mb=sdcs.astype(np.asarray(table.scaled_sdcs_mb).dtype),
    )
    return corrected, LatticeShare(element, removed, clamped, outside)


def lattice_share_model(bremsstrahlung_model: str | None) -> str:
    """Identity marker of the correction for a continuum source (``"bremslib"`` or EEDL).

    Validation: brem-lattice-coherent-fraction
    """
    if bremsstrahlung_model == "bremslib":
        return BREM_LATTICE_SHARE_MODEL
    return BREM_LATTICE_SHARE_ISOTROPIC_MODEL


#: 1 Angstrom^2 in cm^2.
_CM2_PER_ANG2 = 1.0e-16

#: ``ln T`` sub-intervals per kernel ``T`` interval of the isotropic share table.
_ISOTROPIC_T_SUBNODES = 4

#: Log photon-energy nodes per decade of the isotropic share table; finer grids
#: are interpolated linearly in ``ln k`` (the share is smooth in ``k``).
_ISOTROPIC_K_PER_DECADE = 96


def isotropic_share_sigma(kernel, g, weights, n_atoms_per_ang3, T_keV, k_eV):
    """Theta-integrated share per atom, ``d sigma / dk`` [cm^2/eV] at ``T`` and each ``k``.

    ``2 pi int S_j(theta, k) sin theta d theta / n_j``: the solid-angle integral
    (linear in ``theta`` between the kernel nodes) of the direction-resolved
    share :func:`lattice_share_density`, per atom of element ``j``. It is
    exactly the SDCS the BremsLib route removes when its DDCS loses
    ``k S_j / (n_j Z^2)`` on the kernel's ``theta`` nodes, so an isotropic
    continuum loses the same photons per unit path.

    Validation: brem-lattice-share-isotropic
    """
    from ..._bremslib_table import solid_angle_integral

    S = lattice_share_density(kernel, g, weights, T_keV, k_eV)  # (theta, K)
    sigma = solid_angle_integral(np.deg2rad(kernel.theta_deg), S.T)
    return sigma * _CM2_PER_ANG2 / float(n_atoms_per_ang3)


class IsotropicLatticeShare:
    """Theta-integrated share of one element, subtracted from an isotropic continuum (#1261).

    EEDL and Bethe--Heitler continua emit ``d sigma/dk / (4 pi)`` toward every
    direction, so the direction-resolved BremsLib subtraction has no node to act
    on. Their ``d sigma / dk`` instead loses :func:`isotropic_share_sigma`, the
    absolute Born--Debye--Waller CBS-only share integrated over photon
    direction, and the remainder is emitted isotropically. Orientation-averaged
    lines plus continuum then hold the isotropic model's angle-integrated yield
    (the absolute-subtraction rule of ``brem-lattice-coherent-fraction``).

    The share is tabulated on ``ln T`` nodes (the kernel's own, each split into
    :data:`_ISOTROPIC_T_SUBNODES`) from ``max(kernel T_min, T_floor_keV)`` to the
    kernel top, and interpolated linearly in ``ln T`` per segment. Incident
    energies outside that range lose nothing: below ``T_floor_keV`` the line
    route scores no CBS.

    Validation: brem-lattice-share-isotropic
    """

    def __init__(self, kernel, g, weights, n_atoms_per_ang3, *, T_floor_keV: float = 0.0):
        self.kernel = kernel
        self.g = np.asarray(g, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.n_atoms_per_ang3 = float(n_atoms_per_ang3)
        lo = max(float(kernel.T_keV[0]), float(T_floor_keV))
        hi = float(kernel.T_keV[-1])
        inner = [float(t) for t in kernel.T_keV if lo < t < hi]
        coarse = np.log(np.array([lo, *inner, hi])) if lo < hi else np.zeros(0)
        if coarse.size >= 2:
            fine = [
                np.linspace(a, b, _ISOTROPIC_T_SUBNODES + 1)[:-1]
                for a, b in zip(coarse[:-1], coarse[1:], strict=True)
            ]
            self.log_T = np.concatenate([*fine, coarse[-1:]])
        else:
            self.log_T = np.zeros(0)
        self._tables: dict[bytes, np.ndarray] = {}

    @property
    def T_keV(self) -> np.ndarray:
        return np.exp(self.log_T)

    def sigma_nodes(self, k_eV) -> np.ndarray:
        """Share per atom on the ``T`` nodes, ``(T, K)`` [cm^2/eV]."""
        k = np.ascontiguousarray(np.asarray(k_eV, dtype=float))
        key = k.tobytes()
        cached = self._tables.get(key)
        if cached is not None:
            return cached
        positive = k[k > 0]
        if self.log_T.size == 0 or positive.size == 0 or self.weights.size == 0:
            table = np.zeros((self.log_T.size, k.size))
        else:
            lo, hi = float(positive.min()), float(positive.max())
            count = max(2, int(np.ceil(_ISOTROPIC_K_PER_DECADE * np.log10(hi / lo))) + 1)
            direct = k.size <= count
            nodes = k if direct else np.geomspace(lo, hi, count)
            table = np.empty((self.log_T.size, k.size))
            for i, T in enumerate(self.T_keV):
                row = isotropic_share_sigma(
                    self.kernel, self.g, self.weights, self.n_atoms_per_ang3, T, nodes
                )
                if direct:
                    table[i] = row
                else:
                    table[i] = np.where(
                        k > 0, np.interp(np.log(np.where(k > 0, k, lo)), np.log(nodes), row), 0.0
                    )
        table.setflags(write=False)
        self._tables[key] = table
        return table

    def sigma(self, T_keV, k_eV, xp=np):
        """Share per atom for each segment energy and photon energy, ``(M, K)`` [cm^2/eV]."""
        from ..._backend import _to_cpu

        T = xp.asarray(T_keV)
        k = np.asarray(_to_cpu(k_eV), dtype=float)
        if self.log_T.size < 2:
            return xp.zeros((T.size, k.size), dtype=T.dtype)
        table = xp.asarray(self.sigma_nodes(k), dtype=T.dtype)
        nodes = xp.asarray(self.log_T, dtype=xp.float64)
        lt = xp.log(xp.maximum(T.astype(xp.float64), 1e-300))
        i = xp.clip(xp.searchsorted(nodes, lt) - 1, 0, self.log_T.size - 2)
        w = ((lt - nodes[i]) / (nodes[i + 1] - nodes[i])).astype(T.dtype)
        inside = ((lt >= nodes[0]) & (lt <= nodes[-1])).astype(T.dtype)
        return ((1 - w)[:, None] * table[i] + w[:, None] * table[i + 1]) * inside[:, None]


def isotropic_shares_of(shares, bremslib_staged) -> dict:
    """Lattice shares of the elements scored isotropically (no staged BremsLib table)."""
    return {el: s for el, s in (shares or {}).items() if el not in bremslib_staged}


def without_isotropic_share(dsig, share, T_keV, E_grid, xp):
    """Isotropic ``dsig`` [cm^2/eV] less the theta-integrated share, clamped at zero.

    ``xp`` is the caller's array module (backends bind at module scope).

    Validation: brem-lattice-share-isotropic
    """
    if share is None:
        return dsig
    return xp.maximum(dsig - share.sigma(T_keV, E_grid, xp=xp), 0)
