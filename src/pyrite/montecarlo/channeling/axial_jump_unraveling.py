"""Trajectory unraveling of the supplied absorbing axial jump model (#623).

Research CPU reference. It samples the same retained/outgoing/unresolved law
as :mod:`.axial_jump_instrument` with work linear in the retained density
dimension, so material-sized retained spaces fit. Material jumps, matching,
return flux, stopping and admission remain caller obligations under #623/#624.

Validation: cr-axial-absorbing-jump-unraveling
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from .axial_phase_space import phase_space_stream, sample_phase_space

# Versioned domain, independent of MCWF, entry-selection and measurement streams.
_UNRAVEL_DOMAIN = 0x6230001
_KINDS = ("outgoing", "unresolved", "retained")


@dataclass(frozen=True)
class AxialTrajectoryEvent:
    """Terminal record of one incident trajectory.

    ``kind`` is ``"outgoing"`` (represented outgoing jump, measured on the
    supplied window), ``"unresolved"`` (absorbed by ``E_O``) or ``"retained"``
    (still retained at ``max_depth_ang``, then ``depth_ang`` is that depth).
    ``weight`` is the incident trace carried by every trajectory; an estimator
    of a tally is ``weight`` times the fraction of matching events. Indices are
    Weyl-measurement indices, as in :class:`.PhaseSpaceOutcome`.
    """

    incident_id: int
    kind: str
    depth_ang: float
    weight: float
    retained_jumps: int
    position_index: tuple[int, int] | None = None
    momentum_index: tuple[int, int] | None = None


def unraveling_stream(seed: int, incident_id: int) -> np.random.Generator:
    """Version-1 jump-draw stream for one incident identity (order independent).

    Validation: cr-axial-absorbing-jump-unraveling
    """
    for name, value in (("seed", seed), ("incident_id", incident_id)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    key = np.random.SeedSequence((int(seed), _UNRAVEL_DOMAIN, 1, int(incident_id)))
    return np.random.Generator(np.random.Philox(key))


@dataclass(frozen=True)
class _Model:
    h_eff: np.ndarray  # H - i hbar_v Gamma / 2 [eV]
    hbar_v: float
    retained: np.ndarray  # (a, n, n)
    outgoing: np.ndarray  # (b, m, n)
    unresolved: np.ndarray  # (n, n)
    shape: tuple[int, int]
    components: np.ndarray  # (k, n) normalized pure entry components
    probabilities: np.ndarray
    weight: float
    # Eigen form H_eff = R diag(lam) R^-1, or None when R is ill conditioned.
    eigen: tuple[np.ndarray, np.ndarray, np.ndarray] | None

    def evolve(self, psi: np.ndarray, depth: float) -> np.ndarray:
        """No-jump amplitude ``exp[-i H_eff z / hbar v] psi`` (unnormalized).

        Validation: cr-axial-absorbing-jump-unraveling
        """
        if self.eigen is None:
            return expm(-1j * depth / self.hbar_v * self.h_eff) @ psi
        lam, right, inverse = self.eigen
        return right @ (np.exp(-1j * depth / self.hbar_v * lam) * (inverse @ psi))


#: Eigenvector condition number above which propagation falls back to expm.
_MAX_CONDITION = 1e8


def _eigen(h_eff: np.ndarray):
    lam, right = np.linalg.eig(h_eff)
    if not np.all(np.isfinite(right)) or np.linalg.cond(right) > _MAX_CONDITION:
        return None
    return lam, right, np.linalg.inv(right)


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=complex)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _hermitian_psd(value, n, name, *, positive=True):
    matrix = _matrix(value, (n, n), name)
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1e-300)
    tolerance = 64 * np.finfo(float).eps * n * scale
    if not np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=0):
        raise ValueError(f"{name} must be Hermitian")
    matrix = 0.5 * (matrix + matrix.conj().T)
    if positive and np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _prepare(h, rho, retained, outgoing, unresolved, shape, hbar_v) -> _Model:
    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian_eV must be a nonempty square matrix")
    n = h.shape[0]
    if len(shape) != 2 or any(
        isinstance(i, bool) or not isinstance(i, Integral) or i < 1 for i in shape
    ):
        raise ValueError("outgoing_shape must contain two positive integers")
    m = int(shape[0]) * int(shape[1])
    if not np.isfinite(hbar_v) or hbar_v <= 0:
        raise ValueError("hbar_v_eV_ang must be finite and positive")
    h = _hermitian_psd(h, n, "hamiltonian_eV", positive=False)
    rho = _hermitian_psd(rho, n, "initial_density")
    effect = _hermitian_psd(np.zeros((n, n)) if unresolved is None else unresolved, n, "E_O")
    a = np.ascontiguousarray(
        np.stack([_matrix(x, (n, n), "retained jump") for x in retained] or [np.zeros((n, n))])
    )
    b = np.ascontiguousarray(
        np.stack([_matrix(x, (m, n), "outgoing jump") for x in outgoing] or [np.zeros((m, n))])
    )
    loss = effect + np.einsum("jpa,jpb->ab", a.conj(), a) + np.einsum("jpa,jpb->ab", b.conj(), b)
    values, vectors = np.linalg.eigh(rho)
    weight = float(np.sum(np.maximum(values, 0.0)))
    if not 0 < weight <= 1 + 1e-12:
        raise ValueError("initial_density trace must lie in (0, 1]")
    keep = values > 0
    h_eff = h - 0.5j * hbar_v * loss
    return _Model(
        h_eff,
        float(hbar_v),
        a,
        b,
        effect,
        (int(shape[0]), int(shape[1])),
        vectors[:, keep].T.copy(),
        values[keep] / weight,
        weight,
        _eigen(h_eff),
    )


def _waiting_depth(model, psi, threshold, start, stop, step, tolerance):
    """First depth where the no-jump norm reaches ``threshold``, else ``None``.

    The norm is nonincreasing because the loss effect is positive, so the
    first step bracketing the threshold contains the unique crossing.

    Validation: cr-axial-absorbing-jump-unraveling
    """
    depth, current = start, psi
    while depth < stop:
        span = min(step, stop - depth)
        trial = model.evolve(current, span)
        if float(np.vdot(trial, trial).real) <= threshold:

            def excess(s, base=current):
                state = model.evolve(base, s)
                return float(np.vdot(state, state).real) - threshold

            offset = brentq(excess, 0.0, span, xtol=tolerance)
            return depth + offset, model.evolve(current, offset)
        depth, current = depth + span, trial
    return None, current


def _trajectory(model, window, max_depth, seed, incident, step, tolerance):
    """One terminal-event history; law in :func:`unravel_absorbing_outgoing`.

    Validation: cr-axial-absorbing-jump-unraveling
    """
    rng = unraveling_stream(seed, incident)
    psi = model.components[rng.choice(len(model.probabilities), p=model.probabilities)]
    depth, recycled = 0.0, 0
    while True:
        threshold = rng.random()
        depth, state = _waiting_depth(model, psi, threshold, depth, max_depth, step, tolerance)
        if depth is None:
            return AxialTrajectoryEvent(incident, "retained", max_depth, model.weight, recycled)
        state = state / np.linalg.norm(state)
        retained = (model.retained.reshape(-1, state.size) @ state).reshape(
            model.retained.shape[:2]
        )  # (a, n)
        outgoing = (model.outgoing.reshape(-1, state.size) @ state).reshape(
            model.outgoing.shape[:2]
        )  # (b, m)
        rates = np.concatenate(
            [
                np.sum(np.abs(retained) ** 2, axis=1),
                np.sum(np.abs(outgoing) ** 2, axis=1),
                [max(float(np.vdot(state, model.unresolved @ state).real), 0.0)],
            ]
        )
        total = float(rates.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("jump at a state with no finite jump rate")
        channel = int(rng.choice(rates.size, p=rates / total))
        if channel < len(retained):
            psi = retained[channel] / np.linalg.norm(retained[channel])
            recycled += 1
            continue
        if channel == rates.size - 1:
            return AxialTrajectoryEvent(incident, "unresolved", depth, model.weight, recycled)
        phi = outgoing[channel - len(retained)].reshape(model.shape)
        outcome = sample_phase_space(phi, window, phase_space_stream(seed, incident))
        return AxialTrajectoryEvent(
            incident,
            "outgoing",
            depth,
            model.weight,
            recycled,
            outcome.position_index,
            outcome.momentum_index,
        )


def unravel_absorbing_outgoing(
    hamiltonian_eV,
    initial_density,
    *,
    retained_jumps_per_sqrt_ang=(),
    outgoing_jumps_per_sqrt_ang=(),
    unresolved_effect_inv_ang=None,
    outgoing_shape: tuple[int, int],
    hbar_v_eV_ang: float,
    window,
    max_depth_ang: float,
    incident_ids,
    seed: int = 0,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
) -> list[AxialTrajectoryEvent]:
    r"""Unravel the absorbing model into terminal events, one per incident.

    Castin, Dalibard & Mølmer, arXiv:0805.4002, Eqs. (4), (8): between jumps
    $\psi$ evolves under $H-i\hbar v\Gamma/2$, with
    $\Gamma=\sum A_a^\dagger A_a+\sum B_b^\dagger B_b+E_O$; the jump depth
    solves $\lVert\psi(z)\rVert^2=u$ for uniform $u$; the channel is chosen
    with probability $\lVert A_a\psi\rVert^2$, $\lVert B_b\psi\rVert^2$ or
    $\langle\psi|E_O|\psi\rangle$ over $\langle\psi|\Gamma|\psi\rangle$.
    Retained jumps continue the trajectory; an outgoing jump measures
    $B_b\psi$ with the Weyl phase-space instrument and terminates; the
    unresolved channel terminates. Averaging reproduces the dense
    :func:`.absorbing_axial_model` tallies:
    $\mathrm{tr}Q_F(z)$ and $U_O(z)$ are ``weight`` times the fraction of
    outgoing/unresolved events by $z$, and the joint outgoing outcomes follow
    $\int_0^Z\langle g_{ab}|\sigma_F|g_{ab}\rangle dz/M$.

    Jump depths are root-found to ``depth_tolerance_ang`` inside
    ``search_step_ang`` brackets (the norm is nonincreasing), not step
    quantized. Units and assumptions as :func:`.absorbing_axial_model`; the
    entry component is drawn from the eigen-decomposition of the initial
    density. Zero jumps and zero unresolved effect give only retained events.
    Each incident uses :func:`unraveling_stream` for jump draws and
    :func:`.phase_space_stream` for its one measurement, so results do not
    depend on processing order. No outgoing event is a physical escape or
    native launch.

    Validation: cr-axial-absorbing-jump-unraveling
    """
    model = _prepare(
        hamiltonian_eV,
        initial_density,
        retained_jumps_per_sqrt_ang,
        outgoing_jumps_per_sqrt_ang,
        unresolved_effect_inv_ang,
        outgoing_shape,
        hbar_v_eV_ang,
    )
    for name, value in (
        ("max_depth_ang", max_depth_ang),
        ("search_step_ang", search_step_ang),
        ("depth_tolerance_ang", depth_tolerance_ang),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    g = np.asarray(window, dtype=complex)
    if g.shape != model.shape or not np.all(np.isfinite(g)):
        raise ValueError("window must be finite and match outgoing_shape")
    if not np.isclose(np.vdot(g, g).real, 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("window must have unit discrete squared norm")
    return [
        _trajectory(
            model, g, float(max_depth_ang), seed, int(i), search_step_ang, depth_tolerance_ang
        )
        for i in incident_ids
    ]


def event_tallies(events, depths_ang) -> dict[str, np.ndarray]:
    """Incident-weighted cumulative ``outgoing``/``unresolved`` and ``retained`` at depths.

    ``retained`` is weight minus both cumulative tallies. Validation:
    cr-axial-absorbing-jump-unraveling
    """
    depths = np.asarray(depths_ang, dtype=float)
    if not events:
        raise ValueError("no events")
    horizon = min((e.depth_ang for e in events if e.kind == "retained"), default=np.inf)
    if np.any(depths > horizon):
        # Histories still retained at the cutoff were not followed further.
        raise ValueError("tally depths exceed the unraveling max_depth_ang")
    weight = events[0].weight
    out = {}
    for kind in _KINDS[:2]:
        hits = np.array([e.depth_ang for e in events if e.kind == kind])
        out[kind] = weight * np.array([np.sum(hits <= z) for z in depths]) / len(events)
    out["retained"] = weight - out["outgoing"] - out["unresolved"]
    return out
