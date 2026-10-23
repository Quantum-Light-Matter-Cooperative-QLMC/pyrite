"""Bounded complex128 Chebyshev subspace iteration for measurement-only H(q).

FFT matvecs are batched on device; small Rayleigh--Ritz problems run on host.
Unconverged sectors are returned as ``None`` for the canonical host fallback.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics

# Batching sectors amortizes per-round host Rayleigh--Ritz and launch costs on
# large grids; beyond ~64 sectors small grids slow down again.
SOLVER_BUDGET_BYTES = 4 << 30
MAX_BATCH_SECTORS = 64
_RESIDUAL_TOL_EV = 1e-10


def _block(n, count):
    return min(n, max(2 * count, count + 8))


def _sector_bytes(n, count):
    return 12 * _block(n, count) * n * 16


def solver_workspace_bytes(n, count):
    """Device workspace for up to ``MAX_BATCH_SECTORS`` sectors per batch.

    Capped by ``SOLVER_BUDGET_BYTES`` and a quarter of device memory, so smaller
    GPUs keep admission headroom for propagation.
    """
    cap = SOLVER_BUDGET_BYTES
    try:
        import cupy

        cap = min(cap, cupy.cuda.Device().mem_info[1] // 4)
    except Exception:  # no usable CUDA device: only CUDA admission reads this
        pass
    return min(cap, MAX_BATCH_SECTORS * _sector_bytes(n, count))


def _host(xp, value):
    return value if xp is np else xp.asnumpy(value)


def _eigh(xp, matrix):
    matrix = _host(xp, matrix)
    values, vectors = np.linalg.eigh((matrix + matrix.conj().swapaxes(-1, -2)) / 2)
    return xp.asarray(values), xp.asarray(vectors)


_CHEBYSHEV = None


def _chebyshev(xp, vrows, rows, previous, t, center, scale, weight):
    """Return ``(V rows + (T - center) rows) * scale - weight * previous`` in one pass."""
    if xp is np:
        return (vrows + (t - center) * rows) * scale - weight * previous
    global _CHEBYSHEV
    if _CHEBYSHEV is None:
        _CHEBYSHEV = xp.ElementwiseKernel(
            "complex128 v, complex128 x, complex128 p, float64 t, float64 c, float64 s, float64 w",
            "complex128 y",
            "y = (v + (t - c) * x) * s - w * p",
            "pyrite_axial_chebyshev",
        )
    return _CHEBYSHEV(vrows, rows, previous, t, center, scale, weight)


def _residual_tolerances(xp, energies):
    """Keep energy error below 1e-9 eV and resolve retained/guard projectors."""
    if energies.shape[-1] == 1:
        return xp.full(energies.shape[:-1], _RESIDUAL_TOL_EV)
    gap = energies[..., -1] - energies[..., -2]
    return xp.where(gap > 1e-8, xp.minimum(_RESIDUAL_TOL_EV, gap * 1e-9), _RESIDUAL_TOL_EV)


def _iterate(xp, potential, kinetic, initial, count, *, degree=20, iterations=40):
    """Return lowest momentum-space rows, energies, and per-sector residuals."""
    batch, block, n = initial.shape
    degree = min(degree, max(2, n // block))
    shape = potential.shape
    t = kinetic.reshape(batch, 1, n)
    upper = (kinetic.reshape(batch, n).max(axis=1) + potential.max())[:, None, None]

    def potential_rows(rows):
        real = xp.fft.ifft2(rows.reshape(-1, block, *shape), axes=(-2, -1), norm="ortho")
        real *= potential
        return xp.fft.fft2(real, axes=(-2, -1), norm="ortho").reshape(-1, block, n)

    def apply(rows):
        return potential_rows(rows) + t * rows

    def ritz(rows):
        s, u = _eigh(xp, rows.conj() @ rows.swapaxes(-1, -2))
        s = xp.maximum(s, s[:, -1:] * 1e-14)
        rows = (u / xp.sqrt(s)[:, None, :]).swapaxes(-1, -2) @ rows
        hrows = apply(rows)
        energies, rotation = _eigh(xp, rows.conj() @ hrows.swapaxes(-1, -2))
        rotation = rotation.swapaxes(-1, -2)
        return energies, rotation @ rows, rotation @ hrows

    energies, rows, hrows = ritz(initial)
    remaining = np.arange(batch)
    out_energies = xp.empty((batch, count))
    out_rows = xp.empty((batch, count, n), dtype=initial.dtype)
    out_residual = xp.empty((batch, count))
    for _ in range(iterations):
        residual = xp.linalg.norm(
            hrows[:, :count] - energies[:, :count, None] * rows[:, :count], axis=-1
        )
        # Retire converged sectors rather than making an entire batch wait for
        # the slowest one. This also reduces host Rayleigh--Ritz work each round.
        gram = rows[:, :count].conj() @ rows[:, :count].swapaxes(-1, -2)
        orthogonal = xp.max(xp.abs(gram - xp.eye(count)), axis=(-2, -1)) <= 1e-12
        tolerance = _residual_tolerances(xp, energies[:, :count])
        done = _host(xp, xp.all(residual <= tolerance[:, None], axis=1) & orthogonal)
        if np.any(done):
            out_energies[remaining[done]] = energies[done, :count]
            out_rows[remaining[done]] = rows[done, :count]
            out_residual[remaining[done]] = residual[done]
            remaining = remaining[~done]
            if not len(remaining):
                return out_energies, out_rows, out_residual
            energies, rows, hrows = energies[~done], rows[~done], hrows[~done]
            t, upper = t[~done], upper[~done]
            batch = len(remaining)
        edge = energies[:, -1, None, None]
        radius, center = (upper - edge) / 2, (upper + edge) / 2
        # A full subspace already spans H. Zero-width filters need no iteration.
        if bool(xp.any(radius <= 0)):
            break
        sigma = radius / (energies[:, 0, None, None] - center)
        tau = 2 / sigma
        previous = rows
        filtered = (hrows - center * rows) * (sigma / radius)
        for _ in range(2, degree + 1):
            next_sigma = 1 / (tau - sigma)
            # Fused recursion: FFTs dominate, so avoid extra full-block passes.
            next_rows = _chebyshev(
                xp,
                potential_rows(filtered),
                filtered,
                previous,
                t,
                center,
                2 * next_sigma / radius,
                sigma * next_sigma,
            )
            previous, filtered, sigma = filtered, next_rows, next_sigma
        energies, rows, hrows = ritz(filtered)
    residual = xp.linalg.norm(
        hrows[:, :count] - energies[:, :count, None] * rows[:, :count], axis=-1
    )
    out_energies[remaining] = energies[:, :count]
    out_rows[remaining] = rows[:, :count]
    out_residual[remaining] = residual
    return out_energies, out_rows, out_residual


def transverse_spectra_batch(
    grid, potential, energy_eV, tilts, count, base, *, base_tilt=(0.0, 0.0), xp=None
):
    """Solve exact sectors with deterministic starts and bounded device workspace.

    Returns host ``DiscreteSpectrum`` objects or ``None`` per sector. Absolute
    residuals <= 1e-10 eV are required; a nondegenerate retained/guard gap also
    bounds the projector error. No precision reduction or tilt rounding.
    """
    from .axial_state_diagnostics import DiscreteSpectrum

    if xp is None:
        import cupy as xp
    n = int(np.prod(grid.shape))
    block = _block(n, count)
    budget = solver_workspace_bytes(n, count)
    if _sector_bytes(n, count) > budget:
        return [None] * len(tilts)
    size = budget // _sector_bytes(n, count)
    gamma, _ = longitudinal_kinematics(energy_eV)
    dk = 2 * np.pi / np.asarray(grid.extent_ang)
    basek = np.fft.fft2(base.vectors.T.reshape(-1, *grid.shape), axes=(-2, -1), norm="ortho")
    potential = xp.asarray(potential)
    result = []
    for start in range(0, len(tilts), size):
        qs = np.asarray(tilts[start : start + size])
        initial = []
        for q in qs:
            shift = np.rint((q - base_tilt) / dk).astype(int)
            seed = np.roll(basek, tuple(-shift), axis=(1, 2)).reshape(len(basek), n)
            fill = np.random.default_rng(653).standard_normal((block - len(seed), n, 2))
            initial.append(np.concatenate([seed, fill[..., 0] + 1j * fill[..., 1]]))
        q = xp.asarray(qs)[:, :, None, None]
        kinetic = (
            HBAR2_2M_EV_ANG2
            / gamma
            * (
                (xp.asarray(grid.kx)[None, :, None] + q[:, 0]) ** 2
                + (xp.asarray(grid.ky)[None, None, :] + q[:, 1]) ** 2
            )
        )
        try:
            # Enter the BLAS limit once: per-eigh entry rescans loaded libraries.
            with threadpool_limits(1, user_api="blas"):
                energies, rows, residual = _iterate(
                    xp, potential, kinetic, xp.asarray(np.stack(initial)), count
                )
        except np.linalg.LinAlgError:
            result.extend([None] * len(qs))
            continue
        energies, residual = _host(xp, energies), _host(xp, residual)
        vectors = _host(
            xp, xp.fft.ifft2(rows.reshape(len(qs), count, *grid.shape), axes=(-2, -1), norm="ortho")
        ).reshape(len(qs), count, n)
        tolerances = _residual_tolerances(np, energies)
        for e, v, r, tolerance in zip(energies, vectors, residual, tolerances, strict=True):
            orthogonal = np.linalg.norm(v.conj() @ v.T - np.eye(count), ord=2) <= 1e-10
            accepted = orthogonal and np.all(np.isfinite(r)) and np.all(r <= tolerance)
            result.append(DiscreteSpectrum(e, v.T) if accepted else None)
    return result
