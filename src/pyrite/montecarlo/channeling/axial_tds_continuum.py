r"""Independent-string (continuum) TDS jumps for the carrier-fibred unraveling (#623).

Research CPU reference. Each atomic string at ``r`` vibrates independently
with Gaussian in-plane displacement ``u`` (variance ``a^2`` per axis). Summing
the single-string jumps over all lattice images turns the TDS dissipator into
a continuous family of carrier-moving jumps

$$
\sum_{\mathbf R}\lambda\,\mathbb E_u\,J_{\mathbf R,u}\rho J_{\mathbf R,u}^\dagger
=\int\frac{d^2Q}{(2\pi)^2}\,A\,\mathbb E_u\,J_{Q,u}\rho J_{Q,u}^\dagger ,
$$

where $J_{Q,u}$ moves the carrier by $Q$ (folded) and multiplies by a
cell-periodic function whose order-$\mathbf n$ harmonic is
$\sqrt\lambda\,c(\mathbf t)w_{\mathbf t}(u)$ at transfer
$\mathbf t=\boldsymbol\delta+\mathbf g_{\mathbf n}$, with
$c(\mathbf t)=\varphi(|\mathbf t|)e^{-i\mathbf t\cdot\mathbf r}/A$ and
$w_{\mathbf t}(u)=e^{-i\mathbf t\cdot u}-e^{-a^2t^2/2}$. The displacement
average is the Kraus decomposition, so no covariance factorization is
needed. For a window state $\chi$ the joint density of output momentum $Q$
(relative to the carrier) and displacement is
$\propto|\sum_p\chi_pc(Q-\mathbf g_p)w_{Q-\mathbf g_p}(u)|^2N(u)$, sampled
exactly by rejection from the incoherent bound
$(\sum|\chi|)\sum_p|\chi_p|\,|c|^2|w|^2$ (weighted Cauchy--Schwarz). Its loss
effect is the Toeplitz matrix
$\Gamma_{ab}=\lambda e^{i(\mathbf g_b-\mathbf g_a)\cdot\mathbf r}D(|\mathbf g_a-\mathbf g_b|)/(A(2\pi)^2)$
with $D(G)=\int d^2k\,\varphi(|k-G/2|)\varphi(|k+G/2|)[e^{-a^2G^2/2}-e^{-a^2(k^2+G^2/4)}]$.
Transfers beyond ``q_max_inv_ang`` are excluded from both, consistently.

Validation: cr-axial-continuum-tds-jumps
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np

_GL_RADIAL = np.polynomial.legendre.leggauss(32)
_GL_ANGULAR = np.polynomial.legendre.leggauss(192)
_MAX_TRIALS = 1_000_000


def order_extents(value, name: str) -> tuple[int, int]:
    """Per-axis nonnegative order extents ``(M_x, M_y)``; an integer means ``(M, M)``."""
    pair = (value, value) if isinstance(value, Integral) else tuple(value)
    if len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in pair):
        raise ValueError(f"{name} must be an integer or a pair of integers")
    if min(pair) < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(pair[0]), int(pair[1])


@dataclass(frozen=True)
class TdsString:
    """One independently vibrating atomic string of the cell.

    ``phase(q)`` is the isotropic per-atom phase transform [Å^2] (real); its
    modulus must be nonincreasing in ``q`` (checked on the sampling grid).
    ``displacement_ang2`` is the in-plane mean-square displacement per axis
    and ``linear_density_inv_ang`` the atoms per Å along the axis.
    Validation: cr-axial-continuum-tds-jumps
    """

    position_ang: tuple[float, float]
    displacement_ang2: float
    linear_density_inv_ang: float
    phase: Callable[[np.ndarray], np.ndarray]
    q_max_inv_ang: float = 1.0e4


def overlap_transform(phase, a2: float, g_values, q_max: float) -> np.ndarray:
    """``D(G)`` [Å^4 x phase units^2] by midpoint-centred polar quadrature.

    Factors with ``|k -+ G/2| > q_max`` are zero, matching the sampler.
    Validation: cr-axial-continuum-tds-jumps
    """
    edges = np.concatenate([[0.0], 0.05 * 2.0 ** np.arange(int(np.ceil(np.log2(q_max / 0.05))))])
    edges = np.append(edges[edges < q_max], q_max)
    x, w = _GL_RADIAL
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    radius = (edges[:-1, None] + half * (x + 1.0)).ravel()
    w_r = (half * w).ravel()
    t, w_t = _GL_ANGULAR
    angle, w_a = 0.5 * np.pi * (t + 1.0), 0.5 * np.pi * w_t
    out = []
    for g in np.atleast_1d(np.asarray(g_values, dtype=float)):
        base = radius[:, None] ** 2 + 0.25 * g * g
        cross = radius[:, None] * g * np.cos(angle)[None, :]
        values = []
        for sign in (1.0, -1.0):
            q = np.sqrt(np.maximum(base + sign * cross, 0.0))
            values.append(np.where(q <= q_max, phase(q), 0.0))
        bracket = np.exp(-0.5 * a2 * g * g) - np.exp(-a2 * base)
        out.append(2.0 * (radius * w_r) @ (values[0] * values[1] * bracket) @ w_a)
    return np.array(out)


class StringChannel:
    """Sampler and loss of one string on a fixed order window.

    ``orders`` (K, 2) are the window orders, ``reciprocal_inv_ang`` the
    rectangular reciprocal vectors.
    Validation: cr-axial-continuum-tds-jumps
    """

    def __init__(self, string: TdsString, reciprocal_inv_ang, orders, bins: int = 900):
        self.s = string
        self.g = np.asarray(reciprocal_inv_ang, dtype=float)
        self.area = 4.0 * np.pi**2 / float(self.g[0] * self.g[1])
        self.r = np.asarray(string.position_ang, dtype=float)
        self.a2, self.lam = float(string.displacement_ang2), float(string.linear_density_inv_ang)
        for name, value in (("displacement", self.a2), ("linear density", self.lam)):
            if not (np.isfinite(value) and value >= 0):
                raise ValueError(f"string {name} must be finite and nonnegative")
        self.q_max = float(string.q_max_inv_ang)
        self.momenta = np.asarray(orders, dtype=float) * self.g
        self.edges = np.concatenate([[0.0], np.geomspace(1e-3, self.q_max, bins)])
        fine = np.sort(np.concatenate([self.edges, 0.5 * (self.edges[1:] + self.edges[:-1])]))
        modulus = np.abs(np.asarray(string.phase(fine), dtype=float))
        if not np.all(np.isfinite(modulus)) or np.any(np.diff(modulus) > 1e-12 * modulus[:-1]):
            raise ValueError("|phase(q)| must be finite and nonincreasing for the envelope")
        left = np.abs(np.asarray(string.phase(self.edges[:-1]), dtype=float))
        self.bound = left**2 * -np.expm1(-self.a2 * self.edges[1:] ** 2)
        mass = self.bound * np.pi * np.diff(self.edges**2)
        self.cdf = np.cumsum(mass) / mass.sum() if mass.sum() > 0 else None
        diff = np.asarray(orders)[:, None, :] - np.asarray(orders)[None, :, :]
        g_ab = diff * self.g
        lengths = np.hypot(g_ab[..., 0], g_ab[..., 1])
        unique, inverse = np.unique(np.round(lengths, 12), return_inverse=True)
        d = overlap_transform(string.phase, self.a2, unique, self.q_max)[
            inverse.reshape(lengths.shape)
        ]
        phase = np.exp(-1j * (g_ab @ self.r))  # e^{i(g_b - g_a).r}, a = row
        self.loss = self.lam / (self.area * (2 * np.pi) ** 2) * phase * d

    def rate(self, chi: np.ndarray) -> float:
        """``<chi|Gamma|chi>`` [1/Å]."""
        return float(np.vdot(chi, self.loss @ chi).real)

    def _terms(self, q_rel: np.ndarray, u: np.ndarray, chi: np.ndarray):
        t = q_rel[None, :] - self.momenta
        size = np.hypot(t[:, 0], t[:, 1])
        live = size <= self.q_max
        phi = np.where(live, self.s.phase(np.minimum(size, self.q_max)), 0.0)
        damp = np.exp(-0.5 * self.a2 * size**2)
        w = np.exp(-1j * (t @ u)) - damp
        amplitude = np.sum(chi * phi * np.exp(-1j * (t @ self.r)) * w)
        index = np.clip(np.searchsorted(self.edges, size, side="right") - 1, 0, len(self.bound) - 1)
        one_minus = -np.expm1(-self.a2 * size**2)
        ratio = np.divide(np.abs(w) ** 2, one_minus, out=np.zeros(size.shape), where=one_minus > 0)
        envelope = np.sum(np.abs(chi) * np.where(live, self.bound[index], 0.0) * ratio)
        return abs(amplitude) ** 2, envelope

    def _displacement(self, q_vec: np.ndarray, q: float, rng) -> np.ndarray:
        """``u`` from ``|e^{-iq.u} - chi_q|^2 N(u) / (1 - chi_q^2)``.

        ``|w|^2 = (1-chi)^2 + 4 chi sin^2(s/2)`` with ``s = q.u``: a positive mixture.
        """
        a = np.sqrt(self.a2)
        sigma = a * q
        one_minus = -np.expm1(-0.5 * sigma * sigma)  # 1 - chi
        if rng.random() * (2.0 - one_minus) < one_minus:  # (1-chi)/(1+chi)
            s = sigma * rng.standard_normal()
        else:
            while True:
                if sigma < 2.0:
                    z = rng.standard_normal(3)
                    s = sigma * np.sqrt(z @ z) * (1.0 if rng.random() < 0.5 else -1.0)
                    if rng.random() * 0.25 * s * s <= np.sin(0.5 * s) ** 2:
                        break
                else:
                    s = sigma * rng.standard_normal()
                    if rng.random() <= np.sin(0.5 * s) ** 2:
                        break
        unit = q_vec / q
        return (s / q) * unit + a * rng.standard_normal() * np.array([-unit[1], unit[0]])

    def sample(self, chi: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``(Q, u)``: output momentum relative to the carrier and displacement.

        Validation: cr-axial-continuum-tds-jumps
        """
        if self.cdf is None:
            raise ValueError("string has no TDS rate")
        weights = np.abs(chi)
        total = float(weights.sum())
        for _ in range(_MAX_TRIALS):
            p = int(rng.choice(len(weights), p=weights / total))
            i = int(np.searchsorted(self.cdf, rng.random(), side="right"))
            i = min(i, len(self.bound) - 1)
            lo, hi = self.edges[i] ** 2, self.edges[i + 1] ** 2
            q = float(np.sqrt(lo + rng.random() * (hi - lo)))
            theta = 2.0 * np.pi * rng.random()
            q_vec = q * np.array([np.cos(theta), np.sin(theta)])
            u = self._displacement(q_vec, q, rng)
            q_rel = self.momenta[p] + q_vec
            target, envelope = self._terms(q_rel, u, chi)
            if target > envelope * total * (1.0 + 1e-9):
                raise FloatingPointError("TDS envelope violated: phase not monotone?")
            if rng.random() * envelope * total <= target:
                return q_rel, u
        raise FloatingPointError("TDS rejection sampler exceeded its trial budget")

    def column(self, delta: np.ndarray, u: np.ndarray, half) -> np.ndarray:
        """Harmonics ``phi(|t|) e^{-it.r} w_t(u)`` at ``t = delta + g n``, ``|n_i| <= half_i``.

        ``half`` is an integer or per-axis ``(h_x, h_y)``; the column is (2h_x+1, 2h_y+1).
        Validation: cr-axial-continuum-tds-jumps
        """
        hx, hy = order_extents(half, "half")
        nx, ny = np.arange(-hx, hx + 1), np.arange(-hy, hy + 1)
        tx = delta[0] + nx[:, None] * self.g[0] + 0.0 * ny[None, :]
        ty = delta[1] + ny[None, :] * self.g[1] + 0.0 * nx[:, None]
        size = np.hypot(tx, ty)
        phi = np.where(size <= self.q_max, self.s.phase(np.minimum(size, self.q_max)), 0.0)
        w = np.exp(-1j * (tx * u[0] + ty * u[1])) - np.exp(-0.5 * self.a2 * size**2)
        return phi * np.exp(-1j * (tx * self.r[0] + ty * self.r[1])) * w


def string_jump(channel: StringChannel, chi, kappa, order_max, box, rng, fold):
    """One continuum TDS jump from window state ``chi`` at carrier ``kappa``.

    ``order_max`` = (M_x, M_y) and ``box`` = (B_x, B_y) are per-axis extents
    (an integer means square). Returns ``None`` when the sampled output
    momentum lies outside the box (unresolved), else ``(kappa', box_output)``
    with the (2B_x+1)(2B_y+1) box output in centred C order. ``fold(moved)``
    returns the integer shift into the zone. Given ``(Q, u)``,
    ``P(Q in box) = ||B J chi||^2 / ||J chi||^2``, so deciding by where ``Q``
    lands is exact.
    Validation: cr-axial-continuum-tds-jumps
    """
    m = np.array(order_extents(order_max, "order_max"))
    b = np.array(order_extents(box, "box"))
    q_rel, u = channel.sample(chi, rng)
    moved = np.asarray(kappa, dtype=float) + q_rel
    shift = fold(moved)
    if np.any(np.abs(shift) > b):
        return None
    new_kappa = moved - shift * channel.g
    half = b + m
    column = channel.column(new_kappa - np.asarray(kappa, dtype=float), u, tuple(half))
    width = tuple(2 * m + 1)
    size = tuple(np.array(width) + 2 * half)
    full = np.fft.ifft2(np.fft.fft2(chi.reshape(width), s=size) * np.fft.fft2(column, s=size))
    # Output order o sits at index o + M + half per axis; the box is |o_i| <= B_i.
    lo = m + half - b
    out = full[lo[0] : lo[0] + 2 * b[0] + 1, lo[1] : lo[1] + 2 * b[1] + 1]
    return (float(new_kappa[0]), float(new_kappa[1])), out.ravel()
