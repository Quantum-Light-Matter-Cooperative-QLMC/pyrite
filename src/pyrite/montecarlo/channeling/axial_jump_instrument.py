"""Depth-resolved absorbing jump instrument for supplied finite axial models.

Bounded CPU reference only. Material jumps, matching, omitted return flux,
stopping and physical admission remain caller obligations under #623/#624.

Validation: cr-axial-absorbing-jump-instrument
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from .axial_phase_space import sample_density_phase_space


@dataclass(frozen=True)
class AxialJumpSnapshot:
    """Incident-weight densities at a depth, without survivor normalization.

    ``outgoing_density`` integrates represented outgoing gain from zero to
    ``depth_ang``. ``outgoing_rate_inv_ang`` is its local derivative. Both
    use the discrete position basis declared by the model's outgoing shape.
    ``unresolved_weight`` is a separate absorbing tally, never outgoing flux.
    """

    depth_ang: float
    retained_density: np.ndarray
    outgoing_density: np.ndarray
    outgoing_rate_inv_ang: np.ndarray
    unresolved_weight: float


@dataclass(frozen=True)
class AxialOutgoingJump:
    """One research outgoing event, conditional on the declared depth window.

    The joint indices have the Weyl measurement's resolution and FFT order.
    ``outgoing_weight`` is cumulative incident weight in the window;
    ``rate_inv_ang`` is the trace of local gain at the sampled depth. Neither
    is a native launch probability independent of the conditioning window.
    Energy, carrier/frame, position origin, stopping and clocks are absent.
    """

    depth_ang: float
    position_index: tuple[int, int]
    momentum_index: tuple[int, int]
    outgoing_weight: float
    rate_inv_ang: float


def _positive_matrix(value, name, *, positive=True):
    matrix = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    scale = float(np.max(np.abs(matrix)))
    tolerance = 64 * np.finfo(float).eps * matrix.shape[0] * scale
    if not np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=0):
        raise ValueError(f"{name} must be Hermitian")
    if positive and np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _admit_inputs(
    hamiltonian, initial_density, retained_jumps, outgoing_jumps, effect, shape, budget
):
    """Bound dense superoperators before complex conversions or Kronecker work."""
    budget = _positive_integer(budget, "max_working_bytes")
    if len(shape) != 2:
        raise ValueError("outgoing_shape must contain two positive integers")
    shape = tuple(_positive_integer(i, "outgoing_shape entries") for i in shape)
    h = np.asarray(hamiltonian)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian_eV must be a nonempty square matrix")
    n, m = h.shape[0], int(shape[0]) * int(shape[1])
    dimension = n**2 + m**2 + 1
    estimate = 16 * 16 * dimension**2
    if estimate > budget:
        raise ValueError("absorbing instrument exceeds working-memory budget")
    rho = np.asarray(initial_density)
    retained = tuple(np.asarray(a) for a in retained_jumps)
    outgoing = tuple(np.asarray(b) for b in outgoing_jumps)
    unresolved = np.zeros((n, n)) if effect is None else np.asarray(effect)
    matrices = ((rho, (n, n)), (unresolved, (n, n)))
    matrices += tuple((a, (n, n)) for a in retained) + tuple((b, (m, n)) for b in outgoing)
    if any(a.shape != expected for a, expected in matrices):
        raise ValueError("density, effects and jumps must match retained/outgoing dimensions")
    estimate += 32 * sum(a.size for a, _ in matrices)
    if estimate > budget:
        raise ValueError("absorbing instrument exceeds working-memory budget")
    return h, rho, retained, outgoing, unresolved, estimate


@dataclass(frozen=True)
class AbsorbingAxialModel:
    """Prepared constant finite model; construct with :func:`absorbing_axial_model`.

    Arrays store a column-major augmented generator for retained density,
    cumulative represented outgoing density and unresolved weight. No outgoing
    state propagates back into the retained space. This is an absorbing model
    assumption, not evidence that physical return flux is small.
    """

    _generator: np.ndarray = field(repr=False)
    _initial: np.ndarray = field(repr=False)
    _gain: np.ndarray = field(repr=False)
    retained_dimension: int
    outgoing_shape: tuple[int, int]
    initial_weight: float
    working_bytes: int
    max_working_bytes: int

    def at_depth(self, depth_ang: float) -> AxialJumpSnapshot:
        r"""Evaluate retained density, local outgoing gain and cumulative weights.

        The augmented exponential integrates $\sigma_F(z)=\sum_b B_b\rho(z)B_b^\dagger$
        and $\mathrm{tr}[E_O\rho(z)]$ without depth quadrature. This preserves
        the supplied initial trace, not unit trace. Source/assumptions:
        :func:`absorbing_axial_model`. Zero depth gives the initial retained
        density and zero cumulative tallies. Roundoff is not a physical sink.

        Validation: cr-axial-absorbing-jump-instrument
        """
        if not np.isfinite(depth_ang) or depth_ang < 0:
            raise ValueError("depth_ang must be finite and nonnegative")
        with np.errstate(over="raise", invalid="raise"):
            state = expm(self._generator * depth_ang) @ self._initial
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("absorbing evolution is not finite")
        n = self.retained_dimension
        m = self.outgoing_shape[0] * self.outgoing_shape[1]
        retained = state[: n**2].reshape((n, n), order="F")
        outgoing = state[n**2 : -1].reshape((m, m), order="F")
        with np.errstate(over="raise", invalid="raise"):
            gain = (self._gain @ state[: n**2]).reshape((m, m), order="F")
        if not np.all(np.isfinite(gain)):
            raise FloatingPointError("outgoing gain is not finite")
        return AxialJumpSnapshot(float(depth_ang), retained, outgoing, gain, float(state[-1].real))


def absorbing_axial_model(
    hamiltonian_eV,
    initial_density,
    *,
    retained_jumps_per_sqrt_ang=(),
    outgoing_jumps_per_sqrt_ang=(),
    unresolved_effect_inv_ang=None,
    outgoing_shape: tuple[int, int],
    hbar_v_eV_ang: float,
    max_working_bytes: int = 64 * 1024**2,
) -> AbsorbingAxialModel:
    r"""Prepare one consistent retained/outgoing/unresolved finite jump model.

    For fixed retained Hamiltonian $H$, retained jumps $A_a$, represented
    outgoing jumps $B_b$, and unresolved positive effect $E_O$, set
    $\Gamma=\sum_a A_a^\dagger A_a+\sum_b B_b^\dagger B_b+E_O$ and
    $\mathcal L\rho=-i[H,\rho]/(\hbar v)+\sum_a A_a\rho A_a^\dagger
    -\{\Gamma,\rho\}/2$. Then
    $\mathrm{tr}\rho(z)+\int_0^z\mathrm{tr}\sigma_F(s)ds+
    \int_0^z\mathrm{tr}[E_O\rho(s)]ds=\mathrm{tr}\rho(0)$.

    Source: Castin, Dalibard & Mølmer, arXiv:0805.4002, Eqs. (2), (4), (8),
    projected into the fixed absorbing model derived in the escape research
    note. Assumptions: constant finite basis, positive initial density of
    trace at most one, no outgoing return or coherent retained/outgoing
    coupling. Jumps carry Å^-1/2; H is eV and hbar_v is eV Å. Zero jumps and
    unresolved effect leave unitary retained evolution and no outgoing event.

    Outgoing jump rows use C-order discrete *position* indices of the supplied
    periodic shape. Provide the full differential jump maps, not a diagonal
    loss or a relabelled numerical sink. The unresolved effect is caller-supplied
    and does not certify completeness of omitted material transfers. No
    per-case depth budget or native transport/source admission follows.
    Fourth-power dense memory is admitted before constructing superoperators;
    ``working_bytes`` is a conservative estimate, not measured peak RSS.

    Validation: cr-axial-absorbing-jump-instrument
    """
    h, rho, retained, outgoing, unresolved, estimate = _admit_inputs(
        hamiltonian_eV,
        initial_density,
        retained_jumps_per_sqrt_ang,
        outgoing_jumps_per_sqrt_ang,
        unresolved_effect_inv_ang,
        outgoing_shape,
        max_working_bytes,
    )
    if not np.isfinite(hbar_v_eV_ang) or hbar_v_eV_ang <= 0:
        raise ValueError("hbar_v_eV_ang must be finite and positive")
    h = _positive_matrix(h, "hamiltonian_eV", positive=False)
    rho = _positive_matrix(rho, "initial_density")
    weight = float(np.trace(rho).real)
    if not 0 <= weight <= 1 + 1e-12:
        raise ValueError("initial_density trace must lie between zero and one")
    unresolved = _positive_matrix(unresolved, "unresolved_effect_inv_ang")
    retained = tuple(np.asarray(a, dtype=complex) for a in retained)
    outgoing = tuple(np.asarray(b, dtype=complex) for b in outgoing)
    if any(not np.all(np.isfinite(a)) for a in (*retained, *outgoing)):
        raise ValueError("jump operators must be finite")
    n, m = h.shape[0], int(outgoing_shape[0]) * int(outgoing_shape[1])
    identity = np.eye(n)
    loss = unresolved.copy()
    for jump in (*retained, *outgoing):
        with np.errstate(over="raise", invalid="raise"):
            loss += jump.conj().T @ jump
    dimension = n**2 + m**2 + 1
    generator = np.zeros((dimension, dimension), dtype=complex)
    with np.errstate(over="raise", invalid="raise"):
        generator[: n**2, : n**2] = -1j * (
            np.kron(identity, h) - np.kron(h.T, identity)
        ) / hbar_v_eV_ang - 0.5 * (np.kron(identity, loss) + np.kron(loss.T, identity))
        for jump in retained:
            generator[: n**2, : n**2] += np.kron(jump.conj(), jump)
        for jump in outgoing:
            generator[n**2 : -1, : n**2] += np.kron(jump.conj(), jump)
    generator[-1, : n**2] = unresolved.T.ravel(order="F")
    if not np.all(np.isfinite(generator)):
        raise FloatingPointError("absorbing generator is not finite")
    initial = np.zeros(dimension, complex)
    initial[: n**2] = rho.ravel(order="F")
    gain = generator[n**2 : -1, : n**2].copy()
    # Keep the prepared model independent of subsequent edits to caller inputs.
    for array in (generator, initial, gain):
        array.setflags(write=False)
    return AbsorbingAxialModel(
        generator,
        initial,
        gain,
        n,
        (int(outgoing_shape[0]), int(outgoing_shape[1])),
        weight,
        estimate,
        max_working_bytes,
    )


def sample_absorbing_outgoing(
    model: AbsorbingAxialModel,
    window,
    max_depth_ang: float,
    rng: np.random.Generator,
    *,
    depth_tolerance_ang: float = 1e-8,
) -> AxialOutgoingJump:
    r"""Sample a correlated depth/position/momentum outcome in [0, max_depth].

    Let $C(z)=\int_0^z\mathrm{tr}\sigma_F(s)ds$. Draw $z$ from
    $C(z)/C(z_{max})$, then measure the instantaneous mixed $\sigma_F(z)$.
    The conditional joint density is $q(z,a,b)/C(z_{max})$, not a product of
    independently sampled depth and integrated phase-space marginals.
    Source: the positive outgoing measure in the escape research note and
    :func:`absorbing_axial_model`; zero cumulative gain refuses before draws.

    One uniform depth draw plus the two Weyl conditional draws are consumed.
    Retain the caller's incident-addressed generator across events. Numerical
    CDF inversion has absolute depth tolerance ``depth_tolerance_ang`` and
    requires separate tolerance convergence for rapidly varying models.
    Vanishing/nonfinite local rate at a numerically selected root refuses;
    no synthetic outgoing state is substituted. Event clocks/energy and
    physical escape interpretation remain outside this reference.

    Validation: cr-axial-absorbing-jump-instrument
    """
    if not np.isfinite(depth_tolerance_ang) or depth_tolerance_ang <= 0:
        raise ValueError("depth_tolerance_ang must be finite and positive")
    raw_window = np.asarray(window)
    if raw_window.shape != model.outgoing_shape:
        raise ValueError("window must be finite and match outgoing_shape")
    g = np.asarray(raw_window, dtype=complex)
    if not np.all(np.isfinite(g)):
        raise ValueError("window must be finite and match outgoing_shape")
    if not np.isclose(np.vdot(g, g).real, 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("window must have unit discrete squared norm")
    final = model.at_depth(max_depth_ang)
    weight = float(np.trace(final.outgoing_density).real)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("cannot sample a zero-weight outgoing depth window")
    target = rng.random() * weight

    def cumulative(depth):
        return float(np.trace(model.at_depth(depth).outgoing_density).real) - target

    depth = brentq(cumulative, 0.0, max_depth_ang, xtol=depth_tolerance_ang)
    local = model.at_depth(depth).outgoing_rate_inv_ang
    outcome = sample_density_phase_space(local, g, rng, max_working_bytes=model.max_working_bytes)
    return AxialOutgoingJump(
        float(depth), outcome.position_index, outcome.momentum_index, weight, outcome.state_weight
    )
