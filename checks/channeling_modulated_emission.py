"""Prototype: channeling-modulated incoherent emission factor (#501).

Incoherent emission whose rate is local in the transverse position (nuclear
bremsstrahlung, large-``q`` inner-shell ionization) is, for a translation-
averaged planar beam, weighted by the transverse probability density
``rho(x, x; z)``. Its yield relative to a uniform (random-orientation)
distribution is

``F(z) = int_cell rho(x, x; z) n(x) dx / (n_bar tr rho)
       = sum_r tr(T_n P_r) / (n_0 tr rho)``,

with ``P_r = V_r rho_r V_r^dagger`` the Bloch block in plane waves and
``T_n[h, h'] = n_{h - h'}`` the Toeplitz matrix of the density harmonics.
Population that has left the subspace (dechanneled) is assigned ``F = 1``,
so the depth-averaged yield factor is
``Y(L) = (1/L) int_0^L [tr rho F + (1 - tr rho)] dz``.

Entry states (all through the #504 ``DensityMatrixSolver``, TDS + electronic):

- ``plane`` -- a plane wave at incidence ``psi`` (scan in units of ``psi_c``);
- ``on-plane`` / ``between`` -- a d-periodic Gaussian comb of width
  ``--sigma`` centred on (between) the atomic planes, a planar analogue of the
  wavefunction-shaped beams of Wong et al., Light Sci. Appl. 13, 29 (2024).
  A d-periodic state is one Bloch block (``kappa = 0``) and keeps its
  registration to the lattice under the translation average; the converter
  below is the prototype psi(x) -> basis interface proposed for #56.

    PYRITE_MC_BACKEND=cpu uv run python checks/channeling_modulated_emission.py

Exit 0 when the built-in limits hold: an aligned plane wave (uniform
density) has ``F(0) = 1`` up to ``2 (n_max / n_bar) sqrt(w)``, where ``w`` is the
entry weight above the subspace cutoff whose interference the collapse rule
drops; and the comb's ``F(0)`` from the converter equals a direct
position-space quadrature of ``|psi|^2 n / n_bar`` (to 1e-6) when nothing is
lost at entry. About 2 minutes on CPU (multithreaded BLAS).

Validation: channeling-modulated-incoherent-emission
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pyrite.materials.channeling import planar_profile
from pyrite.materials.channeling.transverse import longitudinal_kinematics
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import DensityMatrixSolver, electronic_kernel, tds_kernel


def _toeplitz(coefficients: np.ndarray, orders: np.ndarray) -> np.ndarray:
    lag = orders[:, None] - orders[None, :]
    padded = np.zeros(2 * orders.size, dtype=complex)
    count = min(coefficients.size, padded.size)
    padded[:count] = coefficients[:count]
    return np.where(lag >= 0, padded[np.abs(lag)], np.conj(padded[np.abs(lag)]))


def overlap_factor(solver: DensityMatrixSolver, rho: np.ndarray, coefficients: np.ndarray):
    """Return ``(F, tr rho)`` of a subspace density matrix on density harmonics."""
    vectors = solver.basis.vectors
    plane = vectors @ rho @ np.conj(np.swapaxes(vectors, 1, 2))
    toeplitz = _toeplitz(coefficients, solver.orders)
    weighted = float(np.einsum("hk,rkh->", toeplitz, plane).real)
    trace = float(np.einsum("rnn->", rho).real)
    return weighted / (float(coefficients[0].real) * trace), trace


def comb_state(solver: DensityMatrixSolver, centre_ang: float, sigma_ang: float):
    """Prototype psi(x) -> basis converter for a d-periodic Gaussian comb.

    ``psi(x) = sum_m exp(-(x - centre - m d)^2 / (4 sigma^2))`` has plane-wave
    amplitudes ``c_h ∝ exp(-sigma^2 (h g)^2 - i h g centre)`` in Bloch block
    ``r = 0``. Returns ``(rho_S, entry_flux)`` for ``propagate(initial=...)``.
    """
    g = solver.profile.g_inv_ang
    k = solver.orders * g
    amplitude = np.exp(-(sigma_ang**2) * k**2 - 1j * k * centre_ang)
    amplitude /= np.linalg.norm(amplitude)
    size = solver.orders.size
    plane = np.zeros((solver.n_cells, size, size), dtype=complex)
    plane[0] = np.outer(amplitude, amplitude.conj())
    vectors = solver.basis.vectors
    eigen = np.conj(np.swapaxes(vectors, 1, 2)) @ plane @ vectors
    keep = solver.basis.subspace
    entry = float(np.einsum("rnn->rn", eigen).real[~keep].sum())
    return eigen * (keep[:, :, None] & keep[:, None, :]), entry


def depth_profile(solver, initial, coefficients, thickness_ang, chunk_ang, dz_ang):
    """Chain constant-energy solves; return depths, F(z) and tr rho(z)."""
    rho, entry = initial
    depths, factors, traces = [0.0], [], []
    f0, t0 = overlap_factor(solver, rho, coefficients)
    factors.append(f0)
    traces.append(t0)
    z = 0.0
    while z < thickness_ang - 1e-9:
        step = min(chunk_ang, thickness_ang - z)
        result = solver.propagate(step, dz_ang, initial=(rho, entry))
        rho, entry = result.final_density, result.entry_flux
        z += step
        f, t = overlap_factor(solver, rho, coefficients)
        depths.append(z)
        factors.append(f)
        traces.append(t)
    return np.array(depths), np.array(factors), np.array(traces)


def yield_factor(depths, factors, traces) -> float:
    """``Y(L)``: retained population at ``F``, dechanneled population at 1."""
    integrand = traces * factors + (1.0 - traces)
    return float(np.trapezoid(integrand, depths) / depths[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crystal", default="silicon")
    parser.add_argument("--energies-eV", type=float, nargs="+", default=[3.0e5, 3.0e6, 1.7e7])
    parser.add_argument("--orders", type=int, default=16)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--dz", type=float, default=100.0)
    parser.add_argument("--thickness-um", type=float, default=2.0)
    parser.add_argument("--chunk-ang", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.15, help="comb width [angstrom]")
    args = parser.parse_args(argv)

    profile = planar_profile(args.crystal, (1, 1, 0), 293.15)
    x = np.linspace(0.0, profile.period_ang, 4096, endpoint=False)
    plane_x = float(x[np.argmin(profile.potential(x))])
    nuclear, electrons = profile.nuclear_density, profile.electron_density
    peak = float(profile.nuclear(x).max() / nuclear[0].real)
    print(f"{args.crystal}(110): d = {profile.period_ang:.4f} A, plane at x = {plane_x:.3f} A")
    print(f"perfect-focus planar bound: n_max / n_bar = {peak:.2f} (nuclear, thermal smear)")
    failures: list[str] = []
    thickness = args.thickness_um * 1e4

    for energy in args.energies_eV:
        grid = {"n_cells": args.cells, "max_order": args.orders}
        kernel = tds_kernel(profile, energy, **grid) + electronic_kernel(profile, energy, **grid)
        solver = DensityMatrixSolver(profile, energy, kernel)
        gamma, beta = longitudinal_kinematics(energy)
        pv = gamma * M_E_EV * beta**2
        minimum, barrier = profile.extrema()
        psi_c = np.sqrt(2.0 * (barrier - minimum) / pv)
        tag = f"{energy / 1e6:g} MeV"
        print(
            f"\n== {tag}: psi_c = {psi_c * 1e3:.3f} mrad, hbar v = {HBARC_EV_ANG * beta:.1f} eV A"
        )

        rho, free, outside = solver.entry_state(0.0, 0.0)
        f_plane, _ = overlap_factor(solver, rho, nuclear)
        plane_entry = float(free.sum()) + outside
        print(f"aligned plane wave: F(0) = {f_plane:.6f}, entry flux {plane_entry:.2e}")
        # Dropping the entry's above-cutoff part removes its interference with the
        # retained part, an O(sqrt(entry)) change of the density (collapse rule).
        if abs(f_plane - 1.0) > 1e-10 + 2.0 * peak * np.sqrt(plane_entry):
            failures.append(f"{tag}: aligned plane wave F(0) = {f_plane}")

        states = {
            "plane psi=0": (rho, plane_entry),
            "on-plane comb": comb_state(solver, plane_x, args.sigma),
            "between comb": comb_state(solver, plane_x + 0.5 * profile.period_ang, args.sigma),
        }
        # Direct quadrature of the on-plane comb's entry overlap.
        comb = sum(
            np.exp(-((x - plane_x - m * profile.period_ang) ** 2) / (4 * args.sigma**2))
            for m in range(-3, 4)
        )
        # Periodic grid: the rectangle rule is exact for these band-limited functions.
        direct = float(np.mean(comb**2 * profile.nuclear(x)) / np.mean(comb**2) / nuclear[0].real)
        full_rho, entry = states["on-plane comb"]
        retained = float(np.einsum("rnn->", full_rho).real)
        print(
            f"on-plane comb sigma = {args.sigma} A: retained {retained:.4f}, entry flux {entry:.2e}"
        )
        f_comb, _ = overlap_factor(solver, full_rho, nuclear)
        print(f"  converter F(0) {f_comb:.6f}, direct quadrature {direct:.6f}")
        if entry < 1e-12:
            if abs(f_comb / direct - 1.0) > 1e-6:
                failures.append(f"{tag}: comb F(0) {f_comb:.6f} != quadrature {direct:.6f}")

        print("  state            F_nuc(0)  F_el(0)  Y_nuc(0.1um) Y_nuc(1um) Y_nuc(L)  tr rho(L)")
        for label, initial in states.items():
            depths, f_nuc, traces = depth_profile(
                solver, initial, nuclear, thickness, args.chunk_ang, args.dz
            )
            f_el0, _ = overlap_factor(solver, initial[0], electrons)
            cut = [np.searchsorted(depths, length) + 1 for length in (1e3, 1e4)]
            y01 = yield_factor(depths[: cut[0]], f_nuc[: cut[0]], traces[: cut[0]])
            y1 = yield_factor(depths[: cut[1]], f_nuc[: cut[1]], traces[: cut[1]])
            y_l = yield_factor(depths, f_nuc, traces)
            print(
                f"  {label:15s} {f_nuc[0]:8.3f} {f_el0:8.3f} {y01:11.3f} {y1:10.3f} "
                f"{y_l:9.3f} {traces[-1]:9.3f}"
            )

        # Orientation scan: plane-wave entry, tilt in units of psi_c.
        tilts = np.linspace(-2.0, 2.0, 9)
        scan = []
        for tilt in tilts:
            rho_t, free_t, outside_t = solver.entry_state(tilt * psi_c, 0.0)
            depths, f_nuc, traces = depth_profile(
                solver,
                (rho_t, float(free_t.sum()) + outside_t),
                nuclear,
                min(thickness, 1e4),
                args.chunk_ang,
                args.dz,
            )
            scan.append(yield_factor(depths, f_nuc, traces))
        print("  tilt/psi_c " + " ".join(f"{t:6.2f}" for t in tilts))
        print("  Y_nuc(1um) " + " ".join(f"{y:6.3f}" for y in scan))

    for failure in failures:
        print(f"FAIL: {failure}")
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
