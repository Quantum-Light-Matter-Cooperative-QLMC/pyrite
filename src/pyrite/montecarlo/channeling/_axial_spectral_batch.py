"""Bounded complex128 Chebyshev subspace iteration for measurement-only H(q).

FFT matvecs are batched on device; small Rayleigh--Ritz problems run on host.
Unconverged sectors are returned as ``None`` for the canonical host fallback.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics

SOLVER_BUDGET_BYTES = 512 << 20


def _host(xp, value):
    return value if xp is np else xp.asnumpy(value)


def _eigh(xp, matrix):
    matrix = _host(xp, matrix)
    with threadpool_limits(1, user_api="blas"):
        values, vectors = np.linalg.eigh((matrix + matrix.conj().swapaxes(-1, -2)) / 2)
    return xp.asarray(values), xp.asarray(vectors)


def _iterate(xp, potential, kinetic, initial, count, *, degree=20, iterations=40):
    """Return lowest momentum-space rows, energies, and per-sector residuals."""
    batch, block, n = initial.shape
    degree = min(degree, max(2, n // block))
    shape = potential.shape
    t = kinetic.reshape(batch, 1, n)
    upper = (kinetic.reshape(batch, n).max(axis=1) + potential.max())[:, None, None]

    def apply(rows):
        real = xp.fft.ifft2(rows.reshape(batch, block, *shape), axes=(-2, -1), norm="ortho")
        transformed = xp.fft.fft2(real * potential, axes=(-2, -1), norm="ortho")
        return transformed.reshape(batch, block, n) + t * rows

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
        done = _host(xp, xp.all(residual <= 1e-11, axis=1) & orthogonal)
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
            next_rows = (apply(filtered) - center * filtered) * (2 * next_sigma / radius) - (
                sigma * next_sigma
            ) * previous
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
    residuals <= 1e-11 eV are required; a nondegenerate retained/guard gap also
    bounds the projector error. No precision reduction or tilt rounding.
    """
    from .axial_state_diagnostics import DiscreteSpectrum

    if xp is None:
        import cupy as xp
    n = int(np.prod(grid.shape))
    block = min(n, max(2 * count, count + 8))
    if 12 * block * n * 16 > SOLVER_BUDGET_BYTES:
        return [None] * len(tilts)
    size = max(1, SOLVER_BUDGET_BYTES // (12 * block * n * 16))
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
        for e, v, r in zip(energies, vectors, residual, strict=True):
            gap = e[-1] - e[-2] if count > 1 else 1.0
            tolerance = min(1e-11, gap * 1e-9) if gap > 1e-8 else 1e-11
            orthogonal = np.linalg.norm(v.conj() @ v.T - np.eye(count), ord=2) <= 1e-10
            accepted = orthogonal and np.all(np.isfinite(r)) and np.all(r <= tolerance)
            result.append(DiscreteSpectrum(e, v.T) if accepted else None)
    return result
