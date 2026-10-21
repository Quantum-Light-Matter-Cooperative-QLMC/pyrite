"""Bounded dense quantum-regression reference for source-model research.

These correlations retain all supplied finite-Hilbert-space coherences. They
are not a photon source, a claim of Hilbert-space completeness, or production
axial run support. The caller still owes physical current operators, photon
normalization, frame mapping, and convergence beyond its represented space.

Validation: cr-quantum-regression-correlation
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class RegressionCorrelation:
    """Recorded complex correlation and the represented density's trace.

    ``values[i,j,a,b] = <J_b†(z_j) J_a(z_i)>``. Units are the product of
    the supplied current units. The density has incident normalization,
    without conditioning on a surviving subspace. ``working_bytes`` is the
    conservative admission estimate, not a measured allocator peak.
    """

    depths_ang: np.ndarray
    values: np.ndarray
    trace: np.ndarray
    working_bytes: int


def _matrix(value, name, size=None):
    array = np.asarray(value, dtype=complex)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or not array.size
        or (size is not None and array.shape != (size, size))
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite square matrix of the declared dimension")
    return array


def _hermitian(matrix, name):
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-13):
        raise ValueError(f"{name} must be Hermitian")


def _generator(hamiltonian, jumps, hbar_v):
    """Column-major GKSL generator; derivation in the public reference owner.

    Validation: cr-quantum-regression-correlation
    """
    identity = np.eye(hamiltonian.shape[0])
    generator = -1j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity)) / hbar_v
    for jump in jumps:
        loss = jump.conj().T @ jump
        generator += np.kron(jump.conj(), jump)
        generator -= 0.5 * (np.kron(identity, loss) + np.kron(loss.T, identity))
    return generator


def _problem_inputs(hamiltonian, jumps, density, currents, depths_ang, hbar_v):
    """Validate the finite reference model before constructing its generator."""
    h = _matrix(hamiltonian, "hamiltonian_eV")
    _hermitian(h, "hamiltonian_eV")
    dimension = h.shape[0]
    rho = _matrix(density, "initial_density", dimension)
    _hermitian(rho, "initial_density")
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.min(eigenvalues) < -1e-12 or not 0 <= np.trace(rho).real <= 1 + 1e-12:
        raise ValueError("initial_density must be positive with trace between zero and one")
    jump_arrays = tuple(_matrix(jump, "jump", dimension) for jump in jumps)
    operators = np.asarray(currents, dtype=complex)
    if operators.ndim != 3 or operators.shape[0] < 1:
        raise ValueError("currents must have shape (component, state, state)")
    for current in operators:
        _matrix(current, "current", dimension)
    depths = np.asarray(depths_ang, dtype=float)
    if (
        depths.ndim != 1
        or not depths.size
        or not np.all(np.isfinite(depths))
        or np.any(depths < 0)
        or np.any(np.diff(depths) <= 0)
    ):
        raise ValueError("depths_ang must be nonempty, finite, nonnegative and increasing")
    if not np.isfinite(hbar_v) or hbar_v <= 0:
        raise ValueError("hbar_v_eV_ang must be finite and positive")
    return h, jump_arrays, rho, operators, depths


def regression_correlation(
    hamiltonian_eV,
    jumps_per_sqrt_ang,
    initial_density,
    currents,
    depths_ang,
    *,
    hbar_v_eV_ang: float,
    max_working_bytes: int = 64 * 1024**2,
    max_recorded_depths: int = 256,
) -> RegressionCorrelation:
    r"""Compute a finite GKSL two-time current correlation by quantum regression.

    Source: Gorini, Kossakowski & Sudarshan, JMP 17, 821 (1976), and Lindblad,
    CMP 48, 119 (1976) (Markov generator); Lax, Phys. Rev. 129, 2342 (1963)
    (quantum regression). For $z\ge z'$, use
    $G_{ab}(z,z')=\mathrm{Tr}[J_a e^{\mathcal L(z-z')}(\rho(z')J_b^\dagger)]$,
    with $\mathcal L\rho=-i[H,\rho]/\hbar v+
    \sum_r(L_r\rho L_r^\dagger-\{L_r^\dagger L_r,\rho\}/2)$.
    The reversed pair follows by conjugation and component transposition.

    Assumptions: supplied time-independent finite Markov model, paraxial depth
    as evolution coordinate with constant positive $\hbar v$ [eV Å]. Jumps
    have units Å$^{-1/2}$; density is positive with trace at most one. Currents
    have shape ``(component, state, state)``. A two-level lowering current has
    $G=e^{-i\Delta(z-z')/\hbar v}$ without damping; a ground state gives zero.

    The intermediate $\rho J^\dagger$ is generally non-Hermitian. It must
    never be Hermitized, normalized or clipped. Full dense superoperators have
    fourth-power memory cost; refuse the conservative working-set estimate
    before constructing them. This is a small-reference algorithm.

    Validation: cr-quantum-regression-correlation
    """
    h, jumps, rho, operators, depths = _problem_inputs(
        hamiltonian_eV, jumps_per_sqrt_ang, initial_density, currents, depths_ang, hbar_v_eV_ang
    )
    dimension = h.shape[0]
    for name, value in (
        ("max_working_bytes", max_working_bytes),
        ("max_recorded_depths", max_recorded_depths),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    count, components = depths.size, operators.shape[0]
    # expm work arrays and each distinct one-step propagator; no pairwise expms.
    estimate = 16 * (
        (12 + count) * dimension**4
        + 4 * count * dimension**2
        + count**2 * components**2
        + 4 * components * dimension**2
    )
    if count > max_recorded_depths or estimate > max_working_bytes:
        raise ValueError(
            "quantum-regression reference exceeds recording/working-memory budget "
            f"({count} depths, estimated {estimate} bytes)"
        )
    generator = _generator(h, jumps, hbar_v_eV_ang)
    steps = [expm(generator * delta) for delta in np.diff(np.r_[0.0, depths])]
    densities = []
    state = rho.ravel(order="F")
    for propagator in steps:
        state = propagator @ state
        densities.append(state.reshape((dimension, dimension), order="F"))
    values = np.empty((count, count, components, components), dtype=complex)
    for earlier, density in enumerate(densities):
        intermediate = np.stack(
            [(density @ current.conj().T).ravel(order="F") for current in operators], axis=1
        )
        for later in range(earlier, count):
            if later > earlier:
                intermediate = steps[later] @ intermediate
            for b in range(components):
                matrix = intermediate[:, b].reshape((dimension, dimension), order="F")
                values[later, earlier, :, b] = np.einsum("aij,ji->a", operators, matrix)
            values[earlier, later] = values[later, earlier].conj().T
    traces = np.asarray([np.trace(density).real for density in densities])
    return RegressionCorrelation(depths.copy(), values, traces, estimate)


def spectral_quadratic_form(
    correlation: RegressionCorrelation,
    phase_inv_ang,
    *,
    amplitudes=None,
    max_working_bytes: int = 64 * 1024**2,
):
    r"""Integrate the complex correlation with phase and amplitude weights.

    For trapezoid weights $w_i$ and $v_{ia}=w_i A_{ia}e^{+i\kappa z_i}$,
    return $\sum_{ijab}v_{ia}G_{ijab}v_{jb}^*$. Units are current$^2$ Å$^2$
    for dimensionless amplitudes, not photons/eV/sr. Source: the correlation
    definition and a squared integrated-current amplitude. A lowering current
    of Bohr gap $\Delta$ peaks at positive $\kappa=\Delta/\hbar v$.

    ``amplitudes`` has shape ``(phase, depth, component)``; complex angular
    weights and square-root *intensity* transmission belong here, at amplitude
    level at both depths. Default unit amplitude requires one current component.
    Sampling/quadrature convergence and physical photon prefactors are caller
    responsibilities. Zero density gives zero. Negative weights beyond roundoff
    are errors; values within the roundoff bound are returned as zero. A separate
    working-memory budget admits the frequency/depth/component products before
    allocating amplitude/phase products; use bounded frequency slices for more.

    Validation: cr-quantum-regression-correlation
    """
    depths, values = correlation.depths_ang, correlation.values
    phase = np.atleast_1d(np.asarray(phase_inv_ang, dtype=float))
    if phase.ndim != 1 or not np.all(np.isfinite(phase)):
        raise ValueError("phase_inv_ang must be a finite one-dimensional array")
    if depths.size < 2:
        raise ValueError("spectral quadrature needs at least two recorded depths")
    shape = (phase.size, depths.size, values.shape[2])
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, Integral)
        or max_working_bytes < 1
    ):
        raise ValueError("max_working_bytes must be a positive integer")
    estimate = 16 * (8 * int(np.prod(shape)) + 3 * phase.size + 2 * values.size)
    if estimate > max_working_bytes:
        raise ValueError(
            "spectral quadrature exceeds working-memory budget; use bounded frequency slices "
            f"(estimated {estimate} bytes)"
        )
    if amplitudes is None:
        if values.shape[2] != 1:
            raise ValueError("multiple current components require explicit amplitudes")
        amplitude = np.ones(shape)
    else:
        amplitude = np.asarray(amplitudes, dtype=complex)
        if amplitude.shape != shape or not np.all(np.isfinite(amplitude)):
            raise ValueError("amplitudes must be finite with shape (phase, depth, component)")
    spacing = np.diff(depths)
    weights = np.r_[spacing[0], spacing[:-1] + spacing[1:], spacing[-1]] / 2
    v = amplitude * (weights[None, :] * np.exp(1j * phase[:, None] * depths))[..., None]
    result = np.einsum("fia,ijab,fjb->f", v, values, v.conj(), optimize=True)
    bound = np.einsum("fia,ijab,fjb->f", abs(v), abs(values), abs(v), optimize=True)
    tolerance = 256 * np.finfo(float).eps * np.maximum(1.0, bound)
    if np.any(abs(result.imag) > tolerance) or np.any(result.real < -tolerance):
        raise FloatingPointError(
            "spectral quadratic form violates reality/positivity beyond roundoff"
        )
    return np.maximum(result.real, 0)
