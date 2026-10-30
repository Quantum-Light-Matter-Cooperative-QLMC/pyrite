"""User-definable transverse entry states for channeling (ADR-0018, #508).

A transverse state is the electron's wavefunction across the beam at the
crystal entrance, ``psi(x)`` (planar: ``x`` along the plane normal) or
``psi(x, y)`` (``y`` in the plane, or the second transverse axis of an axial
run). Every state carries a normalized momentum amplitude

``psi~(k) = int psi(r) exp(-i k . r) d^d r``, ``int |psi~|^2 d^d k / (2 pi)^d = 1``,

which is what the solvers consume; :meth:`TransverseState.wavefunction`
samples ``psi`` itself on a grid. States are defined in the beam's transverse
plane and carry no tilt; the mean incidence is applied when a state is
projected (``incidence_rad``).

**Presets.** :func:`gaussian`, :func:`hermite_gauss`, :func:`laguerre_gauss`
(vortex, OAM ``l``; :func:`donut` is ``p = 0``), :func:`bessel_gauss`,
:func:`airy` (finite-energy Airy, 1D or separable 2D) and
:func:`aperture_probe` (a focused probe: hard or annular aperture, polar
aberration coefficients in the abTEM convention, optional vortex charge and a
user phase plate). :func:`from_array` wraps any sampled array and
:func:`from_abtem` imports abTEM ``Waves``. :func:`superpose` (coherent),
:func:`mixture` (incoherent) and :func:`translate` combine them.

**Planar projection.** The planar solver's state is block-diagonal in Bloch
``kappa``: the crystal and its dissipator are invariant under a lattice
translation ``T_d``, so averaging a state over ``T_d`` copies leaves every
lattice-periodic observable (band populations, line rates) and its evolution
unchanged. :func:`planar_entry_density` gives that average,

``rho_kappa[nu, nu'] = int_bin dkappa / 2 pi  psi~(kappa + nu g) psi~*(kappa + nu' g)``,

integrated over each Bloch-grid bin (``impact="fixed"``: the probe centre's
position relative to the planes, modulo ``d``, matters). With
``impact="averaged"`` (the default; a beam that cannot be positioned to a
fraction of ``d``) the entry point is also uniform over one period, which
removes the ``nu != nu'`` coherences and keeps ``|psi~(k)|^2`` only: the
plane-wave/divergence entry limit. A 2D state enters through its reduced
``x`` density (``int dk_y / 2 pi``), so a vortex becomes an ``x`` mixture.

Lengths in angstroms, momenta in 1/angstrom, angles in radians.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import airy as _airy
from scipy.special import eval_genlaguerre, eval_hermite, ive, jv

from ..._units import normalize_unit
from ...materials.channeling.transverse import longitudinal_kinematics
from ...materials.crystal import HBARC_EV_ANG, M_E_EV

#: Envelope level treated as negligible when sizing momentum quadratures.
_TAIL = 1e-8


def _physical(value, unit: str, name: str, expected: str, *, kind: str | None = None):
    """Normalize physical inputs before constructing runtime state objects."""
    return normalize_unit(value, unit, field=f"pyrite.states.{name}", expected=expected, kind=kind)


def _length(value, name: str):
    return _physical(value, "angstrom", name, "length")


def _angle(value, name: str):
    return _physical(value, "radian", name, "angle", kind="angle")


def momentum_inv_ang(kinetic_energy_eV: float) -> float:
    """Electron momentum ``p / hbar`` [1/angstrom]."""
    gamma, beta = longitudinal_kinematics(float(kinetic_energy_eV))
    return gamma * beta * M_E_EV / HBARC_EV_ANG


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


class TransverseState:
    """Base class: a normalized transverse state of dimension 1 or 2.

    Subclasses implement :meth:`amplitude` on a tensor grid of momenta and set
    the quadrature hints ``k_extent`` (per-axis ``|k|`` beyond which the
    amplitude is negligible) and ``k_resolution`` (finest momentum feature).
    """

    if TYPE_CHECKING:
        # Read-only: the frozen dataclass subclasses define them as fields.
        @property
        def dimension(self) -> int: ...
        @property
        def k_extent(self) -> tuple[float, ...]: ...
        @property
        def k_resolution(self) -> tuple[float, ...]: ...

    def amplitude(self, kx, ky=None) -> np.ndarray:
        """``psi~`` on the tensor grid ``kx`` (x ``ky``), shape ``(nkx[, nky])``."""
        raise NotImplementedError

    def wavefunction(self, x, y=None) -> np.ndarray:
        """``psi`` sampled on the tensor grid ``x`` (x ``y``) by numerical inversion."""
        axes = [np.asarray(x, dtype=float)] + ([] if y is None else [np.asarray(y, float)])
        if len(axes) != self.dimension:
            raise ValueError(f"state is {self.dimension}D; pass {self.dimension} coordinate axes")
        nodes, weights = zip(*(_uniform_nodes(e, r) for e, r in self._hints()), strict=True)
        _check_tensor(*(k.size for k in nodes))
        amp = self.amplitude(*nodes)
        out = amp
        for axis, (k, w, r) in enumerate(zip(nodes, weights, axes, strict=True)):
            kernel = np.exp(1j * np.outer(r, k)) * (w / (2.0 * np.pi))
            out = np.moveaxis(np.tensordot(kernel, out, axes=([1], [axis])), 0, axis)
        return out

    def _hints(self):
        return zip(self.k_extent, self.k_resolution, strict=True)

    @property
    def centre(self) -> tuple[float, ...]:
        """Representative position per axis [angstrom]; sets superposition fringe sampling."""
        return (0.0,) * self.dimension

    @property
    def centre_position(self):
        """Fresh angstrom quantity for the representative centre of each axis.

        This is a quadrature/fringe hint, not an invertible preset definition.
        """
        from ...units import Q_

        return Q_(np.asarray(self.centre, dtype=float).copy(), "angstrom")

    @property
    def momentum_extent(self):
        """Fresh per-axis momentum-extent quantity in inverse angstroms."""
        from ...units import Q_

        return Q_(np.asarray(self.k_extent, dtype=float).copy(), "1/angstrom")

    @property
    def momentum_resolution(self):
        """Fresh per-axis quadrature-resolution quantity in inverse angstroms."""
        from ...units import Q_

        return Q_(np.asarray(self.k_resolution, dtype=float).copy(), "1/angstrom")


#: Largest uniform momentum quadrature per axis; finer requests raise.
MAX_NODES = 65537
#: Largest amplitude tensor (product of nodes over axes) one evaluation may build.
MAX_TENSOR_NODES = 2**24


def _check_tensor(*sizes: int) -> None:
    total = math.prod(int(n) for n in sizes)
    if total > MAX_TENSOR_NODES:
        raise ValueError(
            f"momentum quadrature needs a {' x '.join(map(str, sizes))} tensor ({total} nodes, "
            f"limit {MAX_TENSOR_NODES}): the state's extent/resolution ratio is too large"
        )


def _uniform_nodes(extent: float, resolution: float, max_nodes: int = MAX_NODES):
    step = resolution / 4.0
    count = max(2 * math.ceil(extent / step) + 1, 33)
    if count > max_nodes:
        raise ValueError(
            f"momentum quadrature needs {count} nodes per axis (limit {max_nodes}): the state's "
            "extent/resolution ratio is too large for this grid-based evaluation"
        )
    nodes = np.linspace(-extent, extent, count)
    return nodes, np.full(count, nodes[1] - nodes[0])


def _center_phase(k, center):
    return np.exp(-1j * k * float(center))


@dataclass(frozen=True)
class _Separable(TransverseState):
    """Product of 1D factors ``psi~(kx) psi~(ky)``."""

    factors: tuple
    dimension: int = field(init=False)
    k_extent: tuple[float, ...] = field(init=False)
    k_resolution: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", len(self.factors))
        object.__setattr__(self, "k_extent", tuple(f[1] for f in self.factors))
        object.__setattr__(self, "k_resolution", tuple(f[2] for f in self.factors))

    @property
    def centre(self) -> tuple[float, ...]:
        return tuple(float(f[3]) for f in self.factors)

    def amplitude(self, kx, ky=None) -> np.ndarray:
        axes = [kx] + ([] if ky is None else [ky])
        if len(axes) != self.dimension:
            raise ValueError(f"state is {self.dimension}D")
        out = np.asarray(self.factors[0][0](np.asarray(kx, dtype=float)), dtype=complex)
        if self.dimension == 2:
            out = np.multiply.outer(out, self.factors[1][0](np.asarray(ky, dtype=float)))
        return out


@dataclass(frozen=True)
class _Radial(TransverseState):
    """2D state ``psi~(k) = A(|k|) exp(i l phi_k) exp(-i k . r0)``."""

    radial: Callable[[np.ndarray], np.ndarray]
    ell: int
    center: tuple[float, float]
    extent: float
    resolution: float
    real_space: Callable[[np.ndarray], np.ndarray] | None = None
    dimension: int = field(init=False, default=2)
    k_extent: tuple[float, ...] = field(init=False)
    k_resolution: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_extent", (self.extent, self.extent))
        object.__setattr__(self, "k_resolution", (self.resolution, self.resolution))

    @property
    def centre(self) -> tuple[float, ...]:
        return self.center

    def amplitude(self, kx, ky=None) -> np.ndarray:
        if ky is None:
            raise ValueError("state is 2D")
        kx, ky = np.meshgrid(np.asarray(kx, float), np.asarray(ky, float), indexing="ij")
        k, phi = np.hypot(kx, ky), np.arctan2(ky, kx)
        shift = np.exp(-1j * (kx * self.center[0] + ky * self.center[1]))
        return self.radial(k) * np.exp(1j * self.ell * phi) * shift

    def wavefunction(self, x, y=None) -> np.ndarray:
        if self.real_space is None or y is None:
            return super().wavefunction(x, y)
        gx, gy = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="ij")
        dx, dy = gx - self.center[0], gy - self.center[1]
        r, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
        return self.real_space(r) * np.exp(1j * self.ell * phi)


@dataclass(frozen=True)
class SampledState(TransverseState):
    """A state given by samples on a uniform grid (:func:`from_array`).

    ``psi~`` is the rectangle-rule Fourier integral of the samples, normalized
    by their discrete norm. ``kinetic_energy_eV`` records the energy the
    samples were made for (e.g. by abTEM); it is metadata only.
    """

    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    kinetic_energy_eV: float | None = None
    dimension: int = field(init=False)
    k_extent: tuple[float, ...] = field(init=False)
    k_resolution: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(a, dtype=float) for a in self.axes)
        values = np.asarray(self.values, dtype=complex)
        if len(axes) not in (1, 2) or values.shape != tuple(a.size for a in axes):
            raise ValueError("values must have shape (nx,) or (nx, ny) matching the axes")
        steps = []
        for a in axes:
            d = np.diff(a)
            if a.size < 2 or not np.allclose(d, d[0], rtol=1e-9) or d[0] <= 0:
                raise ValueError("sample axes must be uniform and increasing")
            steps.append(float(d[0]))
        norm = float(np.sum(np.abs(values) ** 2) * np.prod(steps))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("samples must have a finite nonzero norm")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "values", values / np.sqrt(norm))
        object.__setattr__(self, "dimension", len(axes))
        object.__setattr__(self, "k_extent", tuple(np.pi / s for s in steps))
        object.__setattr__(
            self,
            "k_resolution",
            tuple(2 * np.pi / (a.size * s) for a, s in zip(axes, steps, strict=True)),
        )

    @property
    def centre(self) -> tuple[float, ...]:
        return tuple(float(0.5 * (a[0] + a[-1])) for a in self.axes)

    @property
    def sample_positions(self):
        """Detached position-axis quantities (one per transverse dimension)."""
        from ...units import Q_

        return tuple(Q_(axis.copy(), "angstrom") for axis in self.axes)

    @property
    def kinetic_energy(self):
        """Optional kinetic-energy metadata, in eV; not a solver control."""
        from ...units import Q_

        return None if self.kinetic_energy_eV is None else Q_(self.kinetic_energy_eV, "eV")

    def amplitude(self, kx, ky=None) -> np.ndarray:
        """Band-limited transform: zero beyond the Nyquist momentum ``pi / spacing``."""
        ks = [np.asarray(kx, dtype=float)] + ([] if ky is None else [np.asarray(ky, float)])
        if len(ks) != self.dimension:
            raise ValueError(f"state is {self.dimension}D")
        out = self.values
        for axis, (k, a) in enumerate(zip(ks, self.axes, strict=True)):
            step = a[1] - a[0]
            kernel = np.exp(-1j * np.outer(k, a)) * step
            # Keep the discrete Nyquist bin.  ``fftfreq`` can round that bin
            # just beyond ``pi / step`` on an even grid, which otherwise
            # breaks exact array-state round trips.
            kernel[np.abs(k) > np.pi / step * (1.0 + 1e-12)] = 0.0
            out = np.moveaxis(np.tensordot(kernel, out, axes=([1], [axis])), 0, axis)
        return out

    def wavefunction(self, x, y=None) -> np.ndarray:
        requested = [np.asarray(x, float)] + ([] if y is None else [np.asarray(y, float)])
        if len(requested) == self.dimension and all(
            r.shape == a.shape and np.allclose(r, a)
            for r, a in zip(requested, self.axes, strict=True)
        ):
            return self.values.copy()
        return super().wavefunction(x, y)


@dataclass(frozen=True)
class Superposition(TransverseState):
    """Coherent normalized sum ``sum_i c_i psi_i`` (:func:`superpose`)."""

    terms: tuple[tuple[complex, TransverseState], ...]
    norm: float
    dimension: int = field(init=False)
    k_extent: tuple[float, ...] = field(init=False)
    k_resolution: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        dims = {s.dimension for _, s in self.terms}
        object.__setattr__(self, "dimension", dims.pop())
        object.__setattr__(
            self,
            "k_extent",
            tuple(max(s.k_extent[i] for _, s in self.terms) for i in range(self.dimension)),
        )
        resolution = []
        for i in range(self.dimension):
            centres = [s.centre[i] for _, s in self.terms]
            spread = max(centres) - min(centres)
            # cross terms oscillate as exp(-i k (c_a - c_b)); resolve a quarter period
            fringe = np.pi / (2.0 * spread) if spread > 0 else np.inf
            resolution.append(min(min(s.k_resolution[i] for _, s in self.terms), fringe))
        object.__setattr__(self, "k_resolution", tuple(resolution))

    @property
    def centre(self) -> tuple[float, ...]:
        return tuple(
            float(np.mean([s.centre[i] for _, s in self.terms])) for i in range(self.dimension)
        )

    def amplitude(self, kx, ky=None) -> np.ndarray:
        total = sum(c * s.amplitude(kx, ky) for c, s in self.terms)
        return np.asarray(total) / self.norm


@dataclass(frozen=True)
class Translated(TransverseState):
    """``psi(r - d)``: the state shifted by ``offset`` (:func:`translate`)."""

    state: TransverseState
    offset: tuple[float, ...]
    dimension: int = field(init=False)
    k_extent: tuple[float, ...] = field(init=False)
    k_resolution: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.offset) != self.state.dimension:
            raise ValueError("offset must have one entry per state dimension")
        object.__setattr__(self, "dimension", self.state.dimension)
        object.__setattr__(self, "k_extent", self.state.k_extent)
        object.__setattr__(self, "k_resolution", self.state.k_resolution)

    @property
    def centre(self) -> tuple[float, ...]:
        return tuple(c + o for c, o in zip(self.state.centre, self.offset, strict=True))

    @property
    def displacement(self):
        """A fresh angstrom quantity for the applied translation vector."""
        from ...units import Q_

        return Q_(np.asarray(self.offset, dtype=float).copy(), "angstrom")

    def amplitude(self, kx, ky=None) -> np.ndarray:
        phase = _center_phase(np.asarray(kx, float), self.offset[0])
        if self.dimension == 2:
            phase = np.multiply.outer(phase, _center_phase(np.asarray(ky, float), self.offset[1]))
        return self.state.amplitude(kx, ky) * phase


@dataclass(frozen=True)
class MixedState:
    """Incoherent mixture ``rho = sum_i w_i |psi_i><psi_i|``, ``sum w_i = 1`` (:func:`mixture`).

    Use it for partial coherence, finite source size or averaging over
    uncontrolled probe positions.
    """

    components: tuple[tuple[float, TransverseState], ...]

    @property
    def dimension(self) -> int:
        return self.components[0][1].dimension


EntryState = TransverseState | MixedState


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def _hg_factor(m: int, waist: float, center: float):
    """Normalized 1D Hermite-Gauss ``psi~`` (FT of ``HG_m`` is ``(-i)^m HG_m`` at waist ``2/w``)."""
    w_k = 2.0 / waist
    scale = np.sqrt(2.0 * np.pi) * (-1j) ** m
    norm = (2.0 / np.pi) ** 0.25 / np.sqrt(2.0**m * math.factorial(m) * w_k)

    def amp(k):
        u = np.sqrt(2.0) * k / w_k
        return (
            scale * norm * eval_hermite(m, u) * np.exp(-((k / w_k) ** 2)) * _center_phase(k, center)
        )

    extent = (np.sqrt(2.0 * np.log(1.0 / _TAIL)) + 2.0 * np.sqrt(m + 1.0)) * w_k / np.sqrt(2.0)
    return amp, float(extent), float(w_k / (2.0 * np.sqrt(m + 1.0))), center


def hermite_gauss(m: int, n: int | None = None, *, waist_ang: float, center_ang=(0.0, 0.0)):
    """Hermite-Gauss mode ``HG_m`` (1D, ``n is None``) or ``HG_mn`` (2D) at its waist.

    ``psi(x) prop H_m(sqrt 2 x / w) exp(-x^2 / w^2)``.

    Validation: cr-entry-state-projection
    """
    waist_ang = _length(waist_ang, "waist_ang")
    center_ang = _length(center_ang, "center_ang")
    if waist_ang <= 0 or m < 0 or (n is not None and n < 0):
        raise ValueError("waist_ang must be positive and mode orders non-negative")
    center = np.atleast_1d(np.asarray(center_ang, dtype=float))
    factors = [_hg_factor(int(m), float(waist_ang), float(center[0]))]
    if n is not None:
        factors.append(_hg_factor(int(n), float(waist_ang), float(center[-1])))
    return _Separable(tuple(factors))


def gaussian(*, waist_ang: float, dimension: int = 2, center_ang=(0.0, 0.0)):
    """Gaussian beam ``exp(-r^2 / w^2)`` at its waist (momentum RMS ``1 / w`` per axis)."""
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    return hermite_gauss(
        0, 0 if dimension == 2 else None, waist_ang=waist_ang, center_ang=center_ang
    )


def laguerre_gauss(ell: int, p: int = 0, *, waist_ang: float, center_ang=(0.0, 0.0)):
    """Laguerre-Gauss vortex ``LG_p^l`` at its waist; orbital angular momentum ``l hbar``.

    ``psi prop (sqrt 2 r / w)^|l| L_p^|l|(2 r^2 / w^2) exp(-r^2 / w^2) exp(i l phi)``.
    Its Fourier transform is ``2 pi (-1)^p (-i)^|l|`` times the same mode at waist
    ``2 / w``.

    Validation: cr-entry-state-projection
    """
    waist_ang = _length(waist_ang, "waist_ang")
    center_ang = _length(center_ang, "center_ang")
    if waist_ang <= 0 or p < 0:
        raise ValueError("waist_ang must be positive and p non-negative")
    a, w = abs(int(ell)), float(waist_ang)
    norm = np.sqrt(2.0 * math.factorial(p) / (np.pi * math.factorial(p + a)))

    def profile(r, width):
        u = 2.0 * (r / width) ** 2
        return norm / width * (np.sqrt(u)) ** a * eval_genlaguerre(p, a, u) * np.exp(-u / 2.0)

    w_k = 2.0 / w
    phase = 2.0 * np.pi * (-1.0) ** p * (-1j) ** a
    spread = np.sqrt(2.0 * p + a + 1.0)
    return _Radial(
        radial=lambda k: phase * profile(k, w_k),
        ell=int(ell),
        center=_pair(center_ang),
        extent=float((np.sqrt(np.log(1.0 / _TAIL)) + 2.0 * spread) * w_k),
        resolution=float(w_k / (2.0 * spread)),
        real_space=lambda r: profile(r, w),
    )


def donut(ell: int = 1, *, waist_ang: float, center_ang=(0.0, 0.0)):
    """Doughnut (hollow) vortex: :func:`laguerre_gauss` with ``p = 0`` and ``|l| >= 1``."""
    if ell == 0:
        raise ValueError("a donut needs |ell| >= 1")
    return laguerre_gauss(ell, 0, waist_ang=waist_ang, center_ang=center_ang)


def bessel_gauss(ell: int, *, k_radial_inv_ang: float, waist_ang: float, center_ang=(0.0, 0.0)):
    """Bessel-Gauss beam ``J_l(k_r r) exp(-r^2 / w^2) exp(i l phi)``.

    ``psi~ prop exp(-(k - k_r)^2 w^2 / 4) ive(l, k k_r w^2 / 2)``: a ring of radius
    ``k_r`` and width ``~2 / w``; ``w -> inf`` is the ideal Bessel beam.

    Validation: cr-entry-state-projection
    """
    waist_ang = _length(waist_ang, "waist_ang")
    center_ang = _length(center_ang, "center_ang")
    k_radial_inv_ang = _physical(
        k_radial_inv_ang, "1 / angstrom", "k_radial_inv_ang", "inverse length"
    )
    if waist_ang <= 0 or k_radial_inv_ang <= 0:
        raise ValueError("waist_ang and k_radial_inv_ang must be positive")
    a, kr, w = abs(int(ell)), float(k_radial_inv_ang), float(waist_ang)
    norm = 1.0 / np.sqrt(2.0 * np.pi * w**2 / 4.0 * ive(a, kr**2 * w**2 / 4.0))
    phase = 2.0 * np.pi * (-1j) ** a * norm * w**2 / 2.0

    def radial(k):
        return phase * np.exp(-((k - kr) ** 2) * w**2 / 4.0) * ive(a, k * kr * w**2 / 2.0)

    return _Radial(
        radial=radial,
        ell=int(ell),
        center=_pair(center_ang),
        extent=float(kr + 2.0 * np.sqrt(np.log(1.0 / _TAIL)) / w),
        resolution=float(0.5 / w),
        real_space=lambda r: norm * jv(a, kr * r) * np.exp(-((r / w) ** 2)),
    )


def _airy_factor(x0: float, a: float, center: float):
    norm = 1.0 / np.sqrt(x0 * np.exp(2.0 * a**3 / 3.0) / np.sqrt(8.0 * np.pi * a))

    def amp(k):
        b = a - 1j * k * x0
        return norm * x0 * np.exp(b**3 / 3.0) * _center_phase(k, center)

    q = np.sqrt(np.log(1.0 / _TAIL) / a)
    return amp, float(q / x0), float(1.0 / (x0 * max(1.0, q * q)) / 2.0), center


def airy(*, scale_ang: float, truncation: float, dimension: int = 1, center_ang=(0.0, 0.0)):
    """Finite-energy Airy beam ``Ai(x / x0) exp(a x / x0)`` (1D, or separable 2D).

    ``psi~ = x0 exp((a - i k x0)^3 / 3)`` (normalized): a Gaussian envelope with a
    cubic phase. In free space its intensity peak follows the self-accelerating
    parabola ``x = z^2 / (4 k^2 x0^3)`` (``k = p / hbar``).

    Validation: cr-entry-state-projection
    """
    scale_ang = _length(scale_ang, "scale_ang")
    center_ang = _length(center_ang, "center_ang")
    if scale_ang <= 0 or truncation <= 0:
        raise ValueError("scale_ang and truncation must be positive")
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    center = np.atleast_1d(np.asarray(center_ang, dtype=float))
    factors = [_airy_factor(float(scale_ang), float(truncation), float(center[0]))]
    if dimension == 2:
        factors.append(_airy_factor(float(scale_ang), float(truncation), float(center[-1])))
    return _Separable(tuple(factors))


def airy_wavefunction(x, *, scale_ang: float, truncation: float, center_ang: float = 0.0):
    """Analytic 1D :func:`airy` samples, for checks."""
    s = (np.asarray(x, dtype=float) - center_ang) / scale_ang
    norm = 1.0 / np.sqrt(
        scale_ang * np.exp(2.0 * truncation**3 / 3.0) / np.sqrt(8.0 * np.pi * truncation)
    )
    return norm * _airy(s)[0] * np.exp(truncation * s)


#: Polar aberration coefficients accepted by :func:`aperture_probe`: ``C_nm`` with
#: ``n + m`` odd excluded is not required; ``m = 0`` terms have no angle.
ABERRATION_KEYS = (
    "C10", "C12", "C21", "C23", "C30", "C32", "C34", "C41", "C43", "C45", "C50", "C52", "C54", "C56",
)  # fmt: skip


def aberration_phase(alpha, phi, momentum: float, aberrations: Mapping[str, float]):
    """Aberration phase ``chi = (2 pi / lambda) sum C_nm alpha^(n+1) cos(m (phi - phi_nm)) / (n+1)``.

    abTEM/Krivanek polar convention: ``C10`` is minus the defocus [angstrom],
    ``C30`` is ``Cs``; angles ``phi_nm`` [rad] are given as ``"phi12"`` etc.
    ``2 pi / lambda = p / hbar`` [1/angstrom].

    Validation: cr-entry-state-projection
    """
    unknown = set(aberrations) - set(ABERRATION_KEYS) - {f"phi{k[1:]}" for k in ABERRATION_KEYS}
    if unknown:
        raise ValueError(f"unknown aberration keys: {sorted(unknown)}")
    chi = np.zeros(np.broadcast(alpha, phi).shape)
    for key in ABERRATION_KEYS:
        value = float(aberrations.get(key, 0.0))
        if value == 0.0:
            continue
        n, m = int(key[1]), int(key[2])
        angle = float(aberrations.get(f"phi{n}{m}", 0.0))
        chi = chi + value * alpha ** (n + 1) * np.cos(m * (phi - angle)) / (n + 1)
    return momentum * chi


@dataclass(frozen=True)
class ApertureProbe(TransverseState):
    """Focused probe ``psi~ = A(k) exp(-i chi(k)) exp(i l phi_k) exp(-i phase(k)) exp(-i k . r0)``.

    See :func:`aperture_probe`.
    """

    momentum: float
    k_inner: float
    k_outer: float
    aberrations: Mapping[str, float]
    ell: int
    center: tuple[float, float]
    phase_plate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    dimension: int = field(init=False, default=2)
    k_extent: tuple[float, ...] = field(init=False)
    k_resolution: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        theta = self.k_outer / self.momentum
        walk = sum(
            abs(float(self.aberrations.get(k, 0.0))) * theta ** int(k[1]) for k in ABERRATION_KEYS
        )
        object.__setattr__(self, "k_extent", (self.k_outer, self.k_outer))
        resolution = 1.0 / (4.0 * (walk + 10.0 / self.k_outer))
        object.__setattr__(self, "k_resolution", (resolution, resolution))

    @property
    def centre(self) -> tuple[float, ...]:
        return self.center

    @property
    def kinetic_energy(self):
        """Kinetic energy recovered from the stored relativistic momentum (eV)."""
        from ...units import Q_

        ratio = self.momentum * HBARC_EV_ANG / M_E_EV
        # (gamma - 1) * mc², evaluated without cancellation at low energy.
        energy = M_E_EV * ratio**2 / (math.hypot(1.0, ratio) + 1.0)
        return Q_(energy, "eV")

    @property
    def semiangle(self):
        """Outer aperture semi-angle as a fresh angular quantity."""
        from ...units import Q_

        return Q_(self.k_outer / self.momentum, "radian")

    @property
    def inner_semiangle(self):
        """Inner aperture semi-angle as a fresh angular quantity."""
        from ...units import Q_

        return Q_(self.k_inner / self.momentum, "radian")

    @property
    def aberration_quantities(self):
        """Copy of the polar aberration coefficients (Å) and phase angles (rad)."""
        from ...units import Q_

        return {
            key: Q_(value, "radian" if key.startswith("phi") else "angstrom")
            for key, value in self.aberrations.items()
        }

    def amplitude(self, kx, ky=None) -> np.ndarray:
        if ky is None:
            raise ValueError("state is 2D")
        gx, gy = np.meshgrid(np.asarray(kx, float), np.asarray(ky, float), indexing="ij")
        k, phi = np.hypot(gx, gy), np.arctan2(gy, gx)
        inside = (k >= self.k_inner) & (k <= self.k_outer)
        scale = np.sqrt(4.0 * np.pi / (self.k_outer**2 - self.k_inner**2))
        chi = aberration_phase(k / self.momentum, phi, self.momentum, self.aberrations)
        if self.phase_plate is not None:
            chi = chi + np.asarray(self.phase_plate(gx, gy), dtype=float)
        shift = gx * self.center[0] + gy * self.center[1]
        return np.where(inside, scale * np.exp(1j * (self.ell * phi - chi - shift)), 0.0)


def aperture_probe(
    kinetic_energy_eV: float,
    *,
    semiangle_rad: float,
    inner_semiangle_rad: float = 0.0,
    aberrations: Mapping[str, float] | None = None,
    ell: int = 0,
    phase_plate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    center_ang=(0.0, 0.0),
) -> ApertureProbe:
    """Aperture-limited focused probe (STEM-like), optionally a vortex or shaped probe.

    Parameters
    ----------
    kinetic_energy_eV
        Electron kinetic energy; sets ``k = p theta / hbar`` of the aperture.
    semiangle_rad, inner_semiangle_rad
        Outer and inner (annular, ``> 0`` gives a hollow-cone probe) aperture
        semi-angles.
    aberrations
        Polar coefficients (:data:`ABERRATION_KEYS`, angles ``"phiNM"``), abTEM
        convention: ``{"C10": -defocus, "C30": Cs}`` [angstrom, rad].
    ell
        Vortex charge of an ideal spiral phase plate ``exp(i l phi_k)``.
    phase_plate
        Extra phase ``f(kx, ky)`` [rad] subtracted like ``chi``: any holographic
        or programmable phase plate.
    center_ang
        Probe position ``(x, y)``.

    Validation: cr-entry-state-projection
    """
    kinetic_energy_eV = _physical(kinetic_energy_eV, "eV", "kinetic_energy_eV", "energy")
    semiangle_rad = _angle(semiangle_rad, "semiangle_rad")
    inner_semiangle_rad = _angle(inner_semiangle_rad, "inner_semiangle_rad")
    center_ang = _length(center_ang, "center_ang")
    aberrations = {
        key: (
            _angle(value, f"aberrations.{key}")
            if key.startswith("phi")
            else _length(value, f"aberrations.{key}")
        )
        for key, value in (aberrations or {}).items()
    }
    p = momentum_inv_ang(kinetic_energy_eV)
    if not 0.0 <= inner_semiangle_rad < semiangle_rad:
        raise ValueError("need 0 <= inner_semiangle_rad < semiangle_rad")
    return ApertureProbe(
        momentum=p,
        k_inner=p * float(inner_semiangle_rad),
        k_outer=p * float(semiangle_rad),
        aberrations=dict(aberrations or {}),
        ell=int(ell),
        center=_pair(center_ang),
        phase_plate=phase_plate,
    )


# ---------------------------------------------------------------------------
# Construction from data and combination
# ---------------------------------------------------------------------------


def from_array(values, *, spacing_ang, origin_ang=0.0, kinetic_energy_eV=None) -> SampledState:
    """Wrap a sampled ``psi`` (1D ``(nx,)`` or 2D ``(nx, ny)``) on a uniform grid.

    ``spacing_ang`` and ``origin_ang`` are scalars or one value per axis; the
    samples are normalized. Momenta beyond ``pi / spacing`` are not represented.

    Validation: cr-entry-state-projection
    """
    spacing_ang = _length(spacing_ang, "spacing_ang")
    origin_ang = _length(origin_ang, "origin_ang")
    if kinetic_energy_eV is not None:
        kinetic_energy_eV = _physical(kinetic_energy_eV, "eV", "kinetic_energy_eV", "energy")
    values = np.asarray(values, dtype=complex)
    dims = values.ndim
    spacing = np.broadcast_to(np.asarray(spacing_ang, dtype=float), (dims,))
    origin = np.broadcast_to(np.asarray(origin_ang, dtype=float), (dims,))
    axes = tuple(
        o + d * np.arange(n) for o, d, n in zip(origin, spacing, values.shape, strict=True)
    )
    return SampledState(axes, values, kinetic_energy_eV)


def from_abtem(waves, *, origin_ang=(0.0, 0.0), swap_axes: bool = False) -> EntryState:
    """Import abTEM ``Waves`` (``.array``, ``.sampling`` [angstrom], ``.energy`` [eV]).

    abTEM's first array axis is its ``x``; pass ``swap_axes`` when that axis is
    not the channeling plane normal. Leading ensemble axes (frozen phonons,
    positions) become an equally weighted :func:`mixture`. Requires the
    optional ``oracle`` dependency group only for building the waves.

    Validation: cr-entry-state-projection
    """
    array = np.asarray(waves.array if not hasattr(waves.array, "get") else waves.array.get())
    sampling = tuple(float(s) for s in waves.sampling)
    energy = float(waves.energy) if getattr(waves, "energy", None) is not None else None
    if array.ndim < 2:
        raise ValueError("abTEM waves must have two trailing spatial axes")
    if swap_axes:
        array, sampling = np.swapaxes(array, -1, -2), sampling[::-1]
    frames = array.reshape((-1,) + array.shape[-2:])
    states = [
        from_array(f, spacing_ang=sampling, origin_ang=origin_ang, kinetic_energy_eV=energy)
        for f in frames
    ]
    return states[0] if len(states) == 1 else mixture([(1.0, s) for s in states])


def superpose(terms: Sequence[tuple[complex, TransverseState]]) -> Superposition:
    """Coherent superposition ``sum c_i psi_i``, renormalized by quadrature.

    Validation: cr-entry-state-projection
    """
    terms = tuple((complex(c), s) for c, s in terms)
    if not terms or len({s.dimension for _, s in terms}) != 1:
        raise ValueError("superpose needs at least one term, all of one dimension")
    draft = Superposition(terms, 1.0)
    nodes, weights = zip(*(_uniform_nodes(e, r) for e, r in draft._hints()), strict=True)
    _check_tensor(*(k.size for k in nodes))
    density = np.abs(draft.amplitude(*nodes)) ** 2
    for w in weights:
        density = np.tensordot(w / (2.0 * np.pi), density, axes=([0], [0]))
    return Superposition(terms, float(np.sqrt(density)))


def mixture(components: Sequence[tuple[float, TransverseState]]) -> MixedState:
    """Incoherent mixture with weights normalized to one.

    Validation: cr-entry-state-projection
    """
    weights = np.array([float(w) for w, _ in components])
    if not components or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("mixture needs non-negative weights with a positive sum")
    if len({s.dimension for _, s in components}) != 1:
        raise ValueError("mixture components must share one dimension")
    weights = weights / weights.sum()
    return MixedState(tuple((float(w), s) for w, (_, s) in zip(weights, components, strict=True)))


def translate(state: TransverseState, offset_ang) -> Translated:
    """Shift a state by ``offset_ang`` (one value per dimension).

    Validation: cr-entry-state-projection
    """
    offset_ang = _length(offset_ang, "offset_ang")
    return Translated(state, tuple(np.atleast_1d(np.asarray(offset_ang, dtype=float))))


def _pair(value) -> tuple[float, float]:
    v = np.broadcast_to(np.asarray(value, dtype=float), (2,))
    return float(v[0]), float(v[1])


# ---------------------------------------------------------------------------
# Planar projection
# ---------------------------------------------------------------------------


def planar_entry_density(
    solver,
    state: EntryState,
    *,
    incidence_rad: float = 0.0,
    impact: str = "averaged",
    trace_tolerance: float = 2e-3,
) -> tuple[np.ndarray, float]:
    """Plane-wave-basis entry density of ``state`` on the Bloch grid.

    Returns ``(plane, outside)``: ``plane[r, i, j]`` over plane-wave orders
    ``i, j`` of block ``kappa_r`` (the input of
    :meth:`~.density.DensityMatrixSolver.entry_from_plane`) and the weight
    outside the plane-wave band, ``1 - tr(plane)``. ``incidence_rad`` tilts
    the state by ``k_x -> k_x + p psi``. ``impact`` is ``"fixed"`` (the state's
    own position relative to the planes, modulo ``d``; use :func:`translate`)
    or ``"averaged"`` (uniform entry point: only ``|psi~|^2``).

    Each Bloch bin is split into ``M`` equal sub-intervals, with ``M`` set by
    ``state.k_resolution`` and no cap. Only sub-intervals whose momentum lies
    in the state's support ``|k_x - p psi| <= k_extent`` are evaluated, so a
    wide (narrow-momentum) beam costs a few hundred amplitude evaluations. If
    the support lies inside the plane-wave band, a trace below
    ``1 - trace_tolerance`` (or above ``1 + trace_tolerance``) means the
    quadrature under-resolves the state, and is refused.

    Validation: cr-entry-state-projection
    """
    if impact not in ("fixed", "averaged"):
        raise ValueError("impact must be 'fixed' or 'averaged'")
    components = state.components if isinstance(state, MixedState) else ((1.0, state),)
    p = momentum_inv_ang(solver.energy_eV)
    g = solver.profile.g_inv_ang
    n_cells, orders, top = solver.n_cells, solver.orders, solver.max_order
    width = g / n_cells
    plane = np.zeros((n_cells, orders.size, orders.size), dtype=complex)
    shift = p * float(incidence_rad)
    # Bins l = r + nu n_cells cover [l width - width/2, l width + width/2).
    l_lo, l_hi = -top * n_cells, top * n_cells + n_cells - 1
    band = ((l_lo - 0.5) * width, (l_hi + 0.5) * width)
    inside = True
    for weight, component in components:
        extent, resolution = component.k_extent[0], component.k_resolution[0]
        lo, hi = max(shift - extent, band[0]), min(shift + extent, band[1])
        inside &= shift - extent >= band[0] and shift + extent <= band[1]
        if hi <= lo:
            continue
        count = max(4, math.ceil(4.0 * width / resolution))
        step = width / count
        # Global sub-interval J = l count + s has midpoint (J + 1/2) step - width/2.
        first = math.floor((lo + width / 2.0) / step)
        last = math.ceil((hi + width / 2.0) / step)
        index = np.arange(first, last)
        bins, s = np.divmod(index, count)
        r = np.mod(bins, n_cells)
        if impact == "averaged":
            nu = (bins - r) // n_cells
            k = (index + 0.5) * step - width / 2.0
            values = _amplitude_sq(component, k - shift)
            np.add.at(plane, (r, nu + top, nu + top), weight * step / (2.0 * np.pi) * values)
            continue
        # fixed: every (block, sub-interval) pair present needs all orders at the same delta
        pairs = np.unique(np.stack((r, s), axis=1), axis=0)
        k = (pairs[:, 0:1] * width - width / 2.0 + (pairs[:, 1:2] + 0.5) * step) + (orders * g)[
            None, :
        ]
        flat = k.reshape(-1) - shift
        scale = weight * step / (2.0 * np.pi)
        if component.dimension == 1:
            amp = component.amplitude(flat).reshape(pairs.shape[0], orders.size)
            outer = np.einsum("pi,pj->pij", amp, amp.conj())
        else:
            ky, wy = _uniform_nodes(component.k_extent[1], component.k_resolution[1])
            _check_tensor(flat.size, ky.size)
            amp = component.amplitude(flat, ky).reshape(pairs.shape[0], orders.size, ky.size)
            outer = np.einsum("pit,pjt,t->pij", amp, amp.conj(), wy / (2.0 * np.pi))
        np.add.at(plane, pairs[:, 0], scale * outer)
    trace = float(np.einsum("rii->", plane).real)
    if trace > 1.0 + trace_tolerance or (inside and trace < 1.0 - trace_tolerance):
        raise ValueError(
            f"projected trace {trace:.6f} with the state's support inside the band: the "
            "momentum quadrature under-resolves the state (check its k_extent/k_resolution)"
        )
    return plane, max(0.0, 1.0 - trace)


def _amplitude_sq(component: TransverseState, kx: np.ndarray) -> np.ndarray:
    """``int |psi~(k_x, k_y)|^2 dk_y / 2 pi`` (or ``|psi~(k_x)|^2`` in 1D)."""
    if component.dimension == 1:
        return np.abs(component.amplitude(kx)) ** 2
    ky, wy = _uniform_nodes(component.k_extent[1], component.k_resolution[1])
    _check_tensor(kx.size, ky.size)
    return (np.abs(component.amplitude(kx, ky)) ** 2) @ (wy / (2.0 * np.pi))


__all__ = [
    "ABERRATION_KEYS",
    "MAX_NODES",
    "MAX_TENSOR_NODES",
    "ApertureProbe",
    "EntryState",
    "MixedState",
    "SampledState",
    "Superposition",
    "TransverseState",
    "Translated",
    "aberration_phase",
    "airy",
    "airy_wavefunction",
    "aperture_probe",
    "bessel_gauss",
    "donut",
    "from_abtem",
    "from_array",
    "gaussian",
    "hermite_gauss",
    "laguerre_gauss",
    "mixture",
    "momentum_inv_ang",
    "planar_entry_density",
    "superpose",
    "translate",
]
