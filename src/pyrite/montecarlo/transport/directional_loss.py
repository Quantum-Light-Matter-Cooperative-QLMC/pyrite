"""Direction-resolved valence oscillators from a uniaxial dielectric function.

A uniaxial crystal responds to a momentum transfer ``q`` at angle ``alpha`` to
its optic axis ``c`` through ``eps(q_hat, w) = eps_perp(w) sin^2 alpha +
eps_par(w) cos^2 alpha`` (Bassani--Tosatti; Tarrio--Schnatterly 1989, Sec. II).
This module turns tabulated ``eps_perp`` and ``eps_par`` into valence-band
delta oscillators ``(W_k, f_k(cos^2 alpha))`` on one shared energy grid, the
inputs of the direction-resolved channels in
:mod:`~pyrite.montecarlo.channeling.directional`.

The oscillator strength per bin follows the optical-limit loss function
``ELF = -Im 1/eps``: ``f_k = N (2 / pi Omega^2) int_bin w ELF dw``, with
``Omega^2 = 4 pi n r_e (hbar c)^2`` the plasma energy of the ``N`` valence
electrons. Strengths are renormalized per direction to ``N`` exactly; the
unrenormalized closure, the static sum rule ``(2/pi) int ELF/w dw =
1 - 1/eps1(0)`` and the logarithmic sum are returned as diagnostics so a
dataset that does not close is visible rather than silently rescaled.
"""

from __future__ import annotations

import csv
import tomllib
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ... import DATA_DIR
from ...materials.crystal import HBARC_EV_ANG

DIRECTIONAL_LOSS_PATH = DATA_DIR / "directional_loss.toml"
# 4 pi r_e (hbar c)^2 [eV^2 angstrom^3]; Omega^2 = this * n [1/angstrom^3].
_PLASMA_COEFFICIENT = 4.0 * np.pi * 2.8179403262e-5 * HBARC_EV_ANG**2
DEFAULT_COS2_NODES = 33
DEFAULT_BINS = 96
DEFAULT_CLOSURE_TOLERANCE = 0.2
_TAIL_DECADES = 2.5
_TAIL_BINS = 12


class ProvisionalModelWarning(UserWarning):
    """A directional loss model is in use that has not reached ``anchored``."""


@dataclass(frozen=True)
class UniaxialDielectric:
    """Complex ``eps_perp(w)``, ``eps_par(w)`` of a uniaxial crystal in the optical limit.

    ``energy_eV`` is strictly increasing; ``eps2`` must be non-negative within
    ``tolerance`` so every mixed loss function is a positive spectral density.
    ``eps1_static_*`` anchor ``eps1(0)`` for the static sum rule (``None`` when
    the table starts too high to extrapolate).
    """

    energy_eV: np.ndarray
    eps_perp: np.ndarray
    eps_par: np.ndarray
    label: str
    eps1_static_perp: float | None = None
    eps1_static_par: float | None = None

    def __post_init__(self):
        e = np.asarray(self.energy_eV, dtype=float)
        if e.ndim != 1 or e.size < 4 or not np.all(np.diff(e) > 0.0) or e[0] <= 0.0:
            raise ValueError("energies must be positive, strictly increasing, at least 4 points")
        for name in ("eps_perp", "eps_par"):
            value = np.asarray(getattr(self, name), dtype=complex)
            if value.shape != e.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and match the energy grid")
            if np.any(value.imag < -1e-9):
                raise ValueError(f"{name} has Im eps < 0: the loss function would be negative")

    def elf(self, cos2_alpha: float, energy_eV: np.ndarray) -> np.ndarray:
        """``-Im 1/eps`` of the mixed tensor at ``cos^2 alpha``; tabulated range only."""
        e = np.asarray(energy_eV, dtype=float)
        eps = _mix(
            _interp(self.energy_eV, self.eps_perp, e),
            _interp(self.energy_eV, self.eps_par, e),
            cos2_alpha,
        )
        return -np.imag(1.0 / eps)


def _interp(x, y, xi):
    return np.interp(xi, x, y.real) + 1j * np.interp(xi, x, y.imag)


def _mix(eps_perp, eps_par, cos2_alpha):
    return eps_perp * (1.0 - cos2_alpha) + eps_par * cos2_alpha


@dataclass(frozen=True)
class DirectionalValence:
    """Valence oscillators on a shared energy grid, strengths per ``cos^2 alpha`` node.

    Attributes
    ----------
    energies_eV
        Oscillator energies ``W_k``, common to every direction.
    cos2_nodes
        Uniform ``cos^2 alpha`` nodes spanning ``[0, 1]``.
    strengths
        ``f_k`` [electrons per formula] with shape ``(nodes, K)``; each row sums to
        ``electrons``.
    mean_strengths
        Strengths averaged over the sphere of ``q`` directions (uniform in
        ``cos alpha``), the isotropic part used for close collisions.
    closure
        Per node, the raw ``(2 / pi Omega^2) int w ELF dw / N`` before renormalization.
    static_sum
        Per node, ``(2/pi) int ELF/w dw`` of the data plus tail, against
        ``static_target`` from ``eps1(0)`` (``nan`` without an anchor).
    tail_fraction
        Per node, the share of the closure above the last tabulated energy.
    label
        Provenance string, including the flagged tail assumption.
    """

    energies_eV: np.ndarray
    cos2_nodes: np.ndarray
    strengths: np.ndarray
    mean_strengths: np.ndarray
    electrons: float
    closure: np.ndarray
    static_sum: np.ndarray
    static_target: np.ndarray
    tail_fraction: np.ndarray
    label: str
    status: str = "provisional"
    log_sums: np.ndarray = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "log_sums", self.strengths @ np.log(self.energies_eV))

    def ratio(self, cos2_alpha):
        """``f_k(cos^2 alpha_k) / mean f_k`` per oscillator; ``cos2_alpha`` has last axis ``K``."""
        n = self.cos2_nodes.size
        u = np.clip(np.asarray(cos2_alpha, dtype=float), 0.0, 1.0) * (n - 1)
        lo = np.minimum(u.astype(int), n - 2)
        w = u - lo
        k = np.arange(self.energies_eV.size)
        f = self.strengths[lo, k] * (1.0 - w) + self.strengths[lo + 1, k] * w
        return f / self.mean_strengths

    @property
    def mean_log_energy_eV(self) -> float:
        """Orientation-averaged ``exp(sum f ln W / N)``: the ``W_cb`` of the Bethe sum."""
        return float(np.exp(self.mean_strengths @ np.log(self.energies_eV) / self.electrons))

    def warn_if_provisional(self):
        if self.status != "anchored":
            warnings.warn(
                f"{self.label}: directional valence loss is {self.status}, not an anchored "
                "channeling model (see #536, #541)",
                ProvisionalModelWarning,
                stacklevel=3,
            )


def _sphere_mean(strengths: np.ndarray) -> np.ndarray:
    """Mean over ``cos alpha`` uniform in ``[0, 1]`` of the ``cos^2``-node interpolant."""
    nodes = np.linspace(0.0, 1.0, strengths.shape[0])
    c = np.linspace(0.0, 1.0, 2049)
    lo = np.minimum((c**2 * (nodes.size - 1)).astype(int), nodes.size - 2)
    w = (c**2 * (nodes.size - 1) - lo)[:, None]
    f = strengths[lo] * (1.0 - w) + strengths[lo + 1] * w
    return np.trapezoid(f, c, axis=0)


def directional_valence(
    dielectric: UniaxialDielectric,
    electrons: float,
    plasma_energy_eV: float,
    *,
    status: str = "provisional",
    nodes: int = DEFAULT_COS2_NODES,
    bins: int = DEFAULT_BINS,
    tolerance: float = DEFAULT_CLOSURE_TOLERANCE,
    tail: bool = True,
) -> DirectionalValence:
    """Bin the mixed loss function into per-direction valence oscillators.

    ``plasma_energy_eV`` is ``Omega`` of the ``electrons`` valence electrons. The
    loss function beyond the last tabulated energy follows the free-electron
    tail ``ELF(w) = Im eps(w_max)(w_max/w)^3``, the ``eps -> 1`` asymptote matched
    to ``Im eps`` rather than to the still-screened ``ELF(w_max)`` (``tail=False``
    drops it); this is an assumption, flagged in ``tail_fraction``. A direction whose raw closure
    differs from 1 by more than ``tolerance`` raises; otherwise its strengths are
    rescaled to ``electrons`` exactly.

    Source: optical-limit f-sum ``int w Im(-1/eps) dw = (pi/2) Omega^2`` and
    static sum ``int Im(-1/eps)/w dw = (pi/2)(1 - 1/eps1(0))`` (Nozieres--Pines);
    uniaxial mixing as :class:`UniaxialDielectric`.

    Validation: cr-valence-directional-loss
    """
    if not (np.isfinite(electrons) and electrons > 0.0 and plasma_energy_eV > 0.0):
        raise ValueError("electrons and plasma energy must be finite and positive")
    e0, e1 = float(dielectric.energy_eV[0]), float(dielectric.energy_eV[-1])
    data_edges = np.geomspace(e0, e1, bins + 1)
    tail_edges = (
        e1 * np.geomspace(1.0, 10.0**_TAIL_DECADES, _TAIL_BINS + 1)[1:] if tail else np.empty(0)
    )
    edges = np.concatenate([data_edges, tail_edges])
    cos2 = np.linspace(0.0, 1.0, nodes)
    fine = np.geomspace(e0, edges[-1], 40 * edges.size)
    weight = np.empty((nodes, edges.size - 1))
    weight_log = np.empty_like(weight)
    static = np.empty(nodes)
    tail_frac = np.empty(nodes)
    for j, c2 in enumerate(cos2):
        # Tail ELF -> Im eps once eps -> 1: matched to Im eps(e1), not to ELF(e1),
        # which is still screened by the plasmon shoulder at the table end.
        eps2_end = _mix(dielectric.eps_perp[-1], dielectric.eps_par[-1], c2).imag
        elf = np.where(
            fine <= e1,
            dielectric.elf(c2, np.minimum(fine, e1)),
            eps2_end * (e1 / np.maximum(fine, e1)) ** 3,
        )
        if not tail:
            elf = np.where(fine <= e1, elf, 0.0)
        # ELF is sampled on a geometric grid: integrate w ELF dw = w^2 ELF dln w.
        integrand = fine**2 * elf
        cum = np.concatenate(
            [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(np.log(fine)))]
        )
        at_edges = np.interp(np.log(edges), np.log(fine), cum)
        weight[j] = np.diff(at_edges)
        cum_log = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5
                    * (integrand[1:] * np.log(fine[1:]) + integrand[:-1] * np.log(fine[:-1]))
                    * np.diff(np.log(fine))
                ),
            ]
        )
        weight_log[j] = np.diff(np.interp(np.log(edges), np.log(fine), cum_log))
        stat = np.trapezoid(elf, np.log(fine))
        static[j] = 2.0 / np.pi * stat
        tail_frac[j] = weight[j, bins:].sum() / weight[j].sum() if tail else 0.0
    norm = 2.0 * electrons / (np.pi * plasma_energy_eV**2)
    raw = norm * weight
    closure = raw.sum(axis=1) / electrons
    if np.any(np.abs(closure - 1.0) > tolerance):
        raise ValueError(
            f"{dielectric.label}: f-sum closure {closure.min():.3f}..{closure.max():.3f} "
            f"outside 1 +/- {tolerance:g}"
        )
    strengths = raw * (electrons / raw.sum(axis=1, keepdims=True))
    centres = np.sqrt(edges[:-1] * edges[1:])
    mean = _sphere_mean(strengths)
    # Spectral-weight log centroid of each bin (mean over nodes), so W_k tracks the peak.
    total = weight.mean(axis=0)
    centroid = np.where(
        total > 0.0, np.exp(weight_log.mean(axis=0) / np.where(total > 0.0, total, 1.0)), centres
    )
    eps1_perp, eps1_par = dielectric.eps1_static_perp, dielectric.eps1_static_par
    if eps1_perp is None or eps1_par is None:
        target = np.full(nodes, np.nan)
    else:
        target = 1.0 - 1.0 / (eps1_perp * (1.0 - cos2) + eps1_par * cos2)
    return DirectionalValence(
        energies_eV=centroid,
        cos2_nodes=cos2,
        strengths=strengths,
        mean_strengths=np.maximum(mean, 1e-300),
        electrons=float(electrons),
        closure=closure,
        static_sum=static,
        static_target=target,
        tail_fraction=tail_frac,
        label=dielectric.label
        + ("; free-electron tail above last tabulated energy" if tail else ""),
        status=status,
    )


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        r
        for r in csv.reader(
            line for line in path.read_text().splitlines() if not line.startswith("#")
        )
        if r and r[0].strip().lower() != "energy_ev"
    ]
    data = np.array([[float(r[0]), float(r[1])] for r in rows])
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def load_directional_loss(
    key: str, path: str | Path | None = None
) -> tuple[UniaxialDielectric, dict]:
    """Load the registered uniaxial dielectric ``key`` and its registry entry.

    Reads ``directional_loss.toml``: each entry names a directory under
    ``data/`` holding ``eps1_perp.csv``, ``eps2_perp.csv``, ``eps1_par.csv`` and
    ``eps2_par.csv`` (columns ``energy_eV,value``; ``#`` lines are provenance;
    scaled figure branches already divided). Curves are interpolated onto the
    union of their energies; ``eps2`` is clipped at zero below
    ``eps2_floor`` and rejected below ``-eps2_floor``.
    """
    registry = tomllib.loads((Path(path) if path else DIRECTIONAL_LOSS_PATH).read_bytes().decode())
    if key not in registry:
        raise ValueError(
            f"no directional loss model {key!r}; registered: {', '.join(sorted(registry))}"
        )
    entry = registry[key]
    folder = DATA_DIR / entry["directory"]
    curves = {}
    for name in ("eps1_perp", "eps2_perp", "eps1_par", "eps2_par"):
        file = folder / f"{name}.csv"
        if not file.is_file():
            raise FileNotFoundError(
                f"{key}: digitized curve {file.name} is missing from {folder}; see {entry['protocol']}"
            )
        curves[name] = _read_curve(file)
    grid = np.unique(np.concatenate([x for x, _ in curves.values()]))
    grid = grid[grid >= max(x[0] for x, _ in curves.values())]
    grid = grid[grid <= min(x[-1] for x, _ in curves.values())]
    floor = float(entry.get("eps2_floor", 0.0))
    out = {name: np.interp(grid, *curve) for name, curve in curves.items()}
    for name in ("eps2_perp", "eps2_par"):
        if np.any(out[name] < -floor):
            raise ValueError(f"{key}: {name} is negative beyond the stated digitization floor")
        out[name] = np.maximum(out[name], 0.0)
    dielectric = UniaxialDielectric(
        grid,
        out["eps1_perp"] + 1j * out["eps2_perp"],
        out["eps1_par"] + 1j * out["eps2_par"],
        entry["label"],
        entry.get("eps1_static_perp"),
        entry.get("eps1_static_par"),
    )
    return dielectric, entry


def catalog_directional_oscillators(key: str, valence: DirectionalValence):
    """Catalog shell oscillators whose conduction band closes on ``valence``.

    As :func:`~.shell_rates.catalog_shell_oscillators`, but the conduction-band
    resonance is the orientation-averaged logarithmic mean energy of
    ``valence``, so the bound shells' Sternheimer closure still reproduces the
    catalog mean excitation energy when the single ``W_cb`` oscillator is
    replaced by the direction-resolved set.

    Validation: cr-valence-directional-loss
    """
    from .shell_oscillators import load_conduction_bands
    from .shell_rates import catalog_shell_oscillators

    band = load_conduction_bands().get(key)
    if band is None:
        raise ValueError(f"{key!r} has no conduction-band shell data")
    if abs(band.electrons_per_formula - valence.electrons) > 1e-9:
        raise ValueError("valence electron count differs from the catalog conduction band")
    return catalog_shell_oscillators(key, conduction_resonance_eV=valence.mean_log_energy_eV)


def load_directional_valence(key: str) -> DirectionalValence:
    """Registered dataset ``key`` binned into valence oscillators of its catalog material.

    The plasma energy is that of the conduction-band electrons,
    ``Omega_p sqrt(N_cb / N_total)`` of the catalog oscillators.
    """
    from .shell_rates import catalog_shell_oscillators

    dielectric, entry = load_directional_loss(key)
    electrons = float(entry["valence_electrons"])
    shells = catalog_shell_oscillators(key)
    omega = shells.plasma_energy_eV * float(np.sqrt(electrons / shells.electrons_per_formula))
    return directional_valence(
        dielectric,
        electrons,
        omega,
        status=entry["status"],
        tolerance=float(entry.get("closure_tolerance", DEFAULT_CLOSURE_TOLERANCE)),
    )


@lru_cache(maxsize=4)
def registered_directional_valence(crystal: str) -> tuple[DirectionalValence, tuple] | None:
    """``(valence, optic_axis)`` of a crystal with a registered dataset, else ``None``.

    ``optic_axis`` is the ``c`` direction in the catalog Cartesian frame. A
    registered crystal whose curves are missing raises ``FileNotFoundError``.
    """
    registry = tomllib.loads(DIRECTIONAL_LOSS_PATH.read_bytes().decode())
    if crystal not in registry:
        return None
    return load_directional_valence(crystal), tuple(registry[crystal].get("optic_axis", (0, 0, 1)))
