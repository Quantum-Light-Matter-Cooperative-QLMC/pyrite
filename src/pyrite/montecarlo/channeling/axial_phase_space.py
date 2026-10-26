"""Finite periodic phase-space measurement for axial matching research.

This supplies a positive joint position/momentum *measurement*, not a physical
escape criterion or a native launch prescription. The caller supplies an already
selected outgoing state. Neither the grid sink nor this measurement establishes
irreversible dechanneling. No main-program axial selector is enabled here.

Validation: cr-axial-phase-space-instrument
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

# Versioned domain, independent of MCWF jump and entry-selection generators.
_PHASE_SPACE_DOMAIN = 0x6205053


@dataclass(frozen=True)
class PhaseSpaceOutcome:
    """Joint grid indices and the supplied outgoing state's squared norm.

    Position indices refer to the window centre under periodic translation;
    momentum indices use NumPy FFT ordering. The caller must supply the origin,
    carrier, physical spacing, frame, energy shell and clock before interpreting
    these indices as a classical launch. ``state_weight`` is not a loss rate.
    """

    position_index: tuple[int, int]
    momentum_index: tuple[int, int]
    state_weight: float


def phase_space_stream(seed: int, incident_id: int) -> np.random.Generator:
    """Return the version-1 measurement stream for one incident identity.

    Keys depend on incident identity rather than processing order or batch size.
    Recreating a stream restarts it; retain the generator for successive events.
    """
    for name, value in (("seed", seed), ("incident_id", incident_id)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    key = np.random.SeedSequence((int(seed), _PHASE_SPACE_DOMAIN, 1, int(incident_id)))
    return np.random.Generator(np.random.Philox(key))


def _inputs(state, window) -> tuple[np.ndarray, np.ndarray, float]:
    phi = np.asarray(state, dtype=complex)
    g = np.asarray(window, dtype=complex)
    if phi.ndim != 2 or not phi.size or g.shape != phi.shape:
        raise ValueError("state and window must have the same nonempty 2D shape")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(g)):
        raise ValueError("state and window must be finite")
    norm = float(np.vdot(g, g).real)
    if not np.isclose(norm, 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("window must have unit discrete squared norm")
    weight = float(np.vdot(phi, phi).real)
    if not np.isfinite(weight):
        raise ValueError("state squared norm must be finite")
    return phi, g, weight


def phase_space_weights(state, window, *, max_outcomes: int = 1_048_576) -> np.ndarray:
    r"""Return subnormalized joint weights ``q[x,y,kx,ky]`` for a grid state.

    For a normalized discrete window $g$ on $M$ sites, the Weyl translations
    $g_{ab}=T_a M_b g$ resolve the identity as
    $M^{-1}\sum_{ab}|g_{ab}\rangle\langle g_{ab}|=I$. Hence
    $q_{ab}=|\langle g_{ab}|\phi\rangle|^2/M$ is positive and sums to
    $\langle\phi|\phi\rangle$. FFTs use the unnormalized forward convention.
    Source: finite Weyl orthogonality; continuous coherent-state POVM analogue
    in Brody, Graefe & Melanathuru, PRL 134, 120201 (2025).
    Assumptions: periodic finite grid, fixed normalized window; no continuum or
    physical escape limit is asserted. Zero state gives zero weights. A delta
    window gives exact position weights and uniform conditional momentum.

    This diagnostic allocates $M^2$ real entries and refuses excess before
    allocation. Use :func:`sample_phase_space` for linear-memory sampling.

    Validation: cr-axial-phase-space-instrument
    """
    phi, g, _ = _inputs(state, window)
    if isinstance(max_outcomes, bool) or not isinstance(max_outcomes, Integral) or max_outcomes < 1:
        raise ValueError("max_outcomes must be a positive integer")
    if phi.size**2 > max_outcomes:
        raise ValueError("joint phase-space weights exceed max_outcomes; use sample_phase_space")
    weights = np.empty((*phi.shape, *phi.shape), dtype=float)
    for index in np.ndindex(phi.shape):
        shifted = np.roll(g, index, axis=(0, 1))
        weights[index] = np.abs(np.fft.fft2(shifted.conj() * phi)) ** 2 / phi.size
    return weights


def sample_phase_space(state, window, rng: np.random.Generator) -> PhaseSpaceOutcome:
    r"""Sample the joint Weyl-frame measurement with linear working memory.

    The position marginal is
    $p_a=\sum_n|g_{n-a}|^2|\phi_n|^2/\|\phi\|^2$; conditional momentum weights
    are $|\mathrm{FFT}(g_a^*\phi)_b|^2$. Sampling these conditionals preserves
    position/momentum correlations. Source/assumptions: :func:`phase_space_weights`.
    A zero state has no outcome and raises ``ValueError``. FFT roundoff in the
    nonnegative correlation is clipped only within a declared machine tolerance.

    Validation: cr-axial-phase-space-instrument
    """
    phi, g, weight = _inputs(state, window)
    if weight <= 0:
        raise ValueError("cannot sample a zero-weight state")
    # Normalize before convolution to avoid overflow in intermediate products.
    phi = phi / np.sqrt(weight)
    marginal = np.fft.ifft2(np.fft.fft2(np.abs(phi) ** 2) * np.fft.fft2(np.abs(g) ** 2).conj()).real
    tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(marginal)))
    if np.min(marginal) < -tolerance:
        raise FloatingPointError("negative position marginal exceeds FFT roundoff tolerance")
    marginal = np.maximum(marginal, 0.0).ravel()
    flat_position = rng.choice(phi.size, p=marginal / np.sum(marginal))
    px, py = np.unravel_index(flat_position, phi.shape)
    position = (int(px), int(py))
    shifted = np.roll(g, position, axis=(0, 1))
    conditional = (np.abs(np.fft.fft2(shifted.conj() * phi)) ** 2).ravel()
    total = float(np.sum(conditional))
    if total <= 0 or not np.isfinite(total):
        raise FloatingPointError("selected position has no finite momentum probability")
    flat_momentum = rng.choice(phi.size, p=conditional / total)
    kx, ky = np.unravel_index(flat_momentum, phi.shape)
    momentum = (int(kx), int(ky))
    return PhaseSpaceOutcome(position, momentum, weight)


def _density_inputs(density, window, max_working_bytes):
    """Admit the dense measurement before complex copies or eigensolvers."""
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, Integral)
        or max_working_bytes < 1
    ):
        raise ValueError("max_working_bytes must be a positive integer")
    raw = np.asarray(window)
    if raw.ndim != 2 or not raw.size:
        raise ValueError("window must have a nonempty 2D shape")
    # Includes density conversion, Hermiticity/PSD checks, FFT temporaries and
    # a full joint output. This is an admission estimate, not measured peak RSS.
    if 16 * 16 * raw.size**2 > max_working_bytes:
        raise ValueError("density measurement exceeds working-memory budget")
    g, _, _ = _inputs(raw, raw)
    raw_density = np.asarray(density)
    if raw_density.shape != (g.size, g.size):
        raise ValueError("density must be a finite square matrix matching the window size")
    rho = np.asarray(raw_density, dtype=complex)
    if not np.all(np.isfinite(rho)):
        raise ValueError("density must be a finite square matrix matching the window size")
    scale = float(np.max(np.abs(rho)))
    if not np.isfinite(np.trace(rho)):
        raise ValueError("density trace must be finite")
    tolerance = 64 * np.finfo(float).eps * g.size * scale
    if not np.allclose(rho, rho.conj().T, atol=tolerance, rtol=0):
        raise ValueError("density must be Hermitian positive semidefinite")
    if np.linalg.eigvalsh(rho).min() < -tolerance:
        raise ValueError("density must be Hermitian positive semidefinite")
    weight = float(np.trace(rho).real)
    if weight < 0:
        raise ValueError("density trace must be nonnegative")
    return rho, g, weight


def _density_momentum_weights(rho, window, position):
    r"""Weyl weights for one position, including the factor 1/M.

    Validation: cr-axial-phase-space-instrument
    """
    shifted = np.roll(window, position, axis=(0, 1)).ravel()
    localized = shifted.conj()[:, None] * rho * shifted[None, :]
    transformed = np.fft.fft2(localized.reshape(*window.shape, *window.shape), axes=(0, 1))
    transformed = np.fft.ifft2(transformed, axes=(2, 3))
    weights = np.diag(transformed.reshape(rho.shape)).real.reshape(window.shape)
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("momentum weights must be finite")
    tolerance = 128 * np.finfo(float).eps * window.size * float(np.max(np.abs(rho)))
    if np.min(weights) < -tolerance:
        raise FloatingPointError("negative momentum weights exceed FFT roundoff tolerance")
    return np.maximum(weights, 0.0)


def density_phase_space_weights(
    density, window, *, max_outcomes: int = 1_048_576, max_working_bytes: int = 64 * 1024**2
) -> np.ndarray:
    r"""Measure a supplied mixed outgoing density without discarding coherence.

    For the Weyl modes of :func:`phase_space_weights`, return
    $q_{ab}=\langle g_{ab}|\rho|g_{ab}\rangle/M$. Finite Fourier
    orthogonality gives $\sum_{ab}q_{ab}=\mathrm{tr}\rho$; this is linear
    in the incident-normalized density, not its diagonal or a normalized
    survivor state. Source: finite Weyl identity, as in the pure-state owner.
    Assumptions: finite periodic grid, supplied positive density in the discrete
    position basis and a unit window. A rank-one density recovers the pure
    measurement; zero density gives zero. No physical escape map is inferred.

    ``density[n,m]`` uses flattened C-order position indices. Physical
    coordinate densities require the cell-area factor before this boundary.
    A gain density per angstrom instead produces weights per angstrom; this
    function neither integrates depth nor supplies event probabilities.
    Dense input and FFT workspace cost O(M²). Limits precede those allocations.

    Validation: cr-axial-phase-space-instrument
    """
    if isinstance(max_outcomes, bool) or not isinstance(max_outcomes, Integral) or max_outcomes < 1:
        raise ValueError("max_outcomes must be a positive integer")
    if np.size(window) ** 2 > max_outcomes:
        raise ValueError("joint phase-space weights exceed max_outcomes")
    rho, g, _ = _density_inputs(density, window, max_working_bytes)
    weights = np.empty((*g.shape, *g.shape), dtype=float)
    for position in np.ndindex(g.shape):
        weights[position] = _density_momentum_weights(rho, g, position)
    return weights


def sample_density_phase_space(
    density, window, rng: np.random.Generator, *, max_working_bytes: int = 64 * 1024**2
) -> PhaseSpaceOutcome:
    r"""Sample a mixed-state joint law with two conditional draws and O(M²) memory.

    The position marginal is $p_a=\sum_n|g_{n-a}|^2\rho_{nn}$.
    Conditional momentum uses the full localized density, including its
    off-diagonal elements, rather than separate position/momentum marginals.
    Source/assumptions: :func:`density_phase_space_weights`. A delta window
    resolves the position diagonal with uniform conditional momentum. Zero
    trace refuses before any draw. ``state_weight`` preserves the input trace;
    no pure-state decomposition or additional mixture draw is performed.
    Retain the incident-addressed generator across successive events.

    Validation: cr-axial-phase-space-instrument
    """
    rho, g, weight = _density_inputs(density, window, max_working_bytes)
    if weight <= 0:
        raise ValueError("cannot sample a zero-weight density")
    # Scaling preserves conditional probabilities and avoids FFT overflow.
    rho = rho / weight
    marginal = np.empty(g.shape)
    diagonal = np.diag(rho).real.reshape(g.shape)
    for position in np.ndindex(g.shape):
        marginal[position] = np.sum(np.abs(np.roll(g, position, axis=(0, 1))) ** 2 * diagonal)
    marginal = np.maximum(marginal, 0.0).ravel()
    flat_position = rng.choice(g.size, p=marginal / marginal.sum())
    px, py = np.unravel_index(flat_position, g.shape)
    position = (int(px), int(py))
    conditional = _density_momentum_weights(rho, g, position).ravel()
    total = float(conditional.sum())
    if total <= 0 or not np.isfinite(total):
        raise FloatingPointError("selected position has no finite momentum probability")
    flat_momentum = rng.choice(g.size, p=conditional / total)
    kx, ky = np.unravel_index(flat_momentum, g.shape)
    momentum = (int(kx), int(ky))
    return PhaseSpaceOutcome(position, momentum, weight)
