"""Is the single-``n_hat`` coherent spectrum an observable, or one speckle draw?

``mc_spectrum(coherent=True)`` phase-sums segment fields against a single fixed
far-field direction ``n_hat`` and squares once, so the stored spectrum is
``|E_tot(E, n_hat)|^2`` at one point in angle.  Two approximations ride on that
single direction, and this script separates them.

- ``fresnel``   The far-field expansion keeps the linear retardation
                ``-k n_hat.r`` exactly and drops the curvature term
                ``k r_perp^2 / (2R)``.  Pure separation ALONG ``n_hat`` carries
                no error at all, so slab depth is exact; only the transverse
                extent of the emitting volume matters.  The neglected phase
                reaches one radian at ``r_perp = sqrt(lambda R / pi)``.  This is
                geometry alone -- no Monte Carlo needed.

- ``speckle``   The cross-electron terms carry ``exp[-i(omega n_hat + g).dr]``
                for the per-electron transverse offset ``dr`` drawn from
                ``beam_fwhm_mm``.  The implementation evaluates ONE realization
                of those offsets rather than averaging over their distribution,
                so the coherent peak height fluctuates from seed to seed with a
                contrast that does NOT fall as the electron count grows --
                speckle contrast is independent of the number of randomly
                phased emitters.  The incoherent path, whose seed scatter is
                ordinary Monte Carlo counting noise, is the control.

The ensemble average that is missing is analytic.  Averaging the cross-electron
term over a Gaussian transverse spot gives the form factor
``exp[-(q_perp sigma_perp)^2]`` with ``q_perp`` the transverse part of
``omega n_hat + g`` -- the transverse partner of the longitudinal
``exp[-(omega sigma_z)^2]`` the coherent model already carries.  With
``q_perp ~ 1 Ang^-1`` and any spot above a nanometre that factor is
numerically zero, so the ensemble answer keeps only the intra-electron sum
``sum_e |sum_{j in e} E_j|^2``.  Section 4 checks that claim the slow way: the
seed-averaged coherent peak should be independent of the spot size, and equal to
the intra-electron enhancement over the incoherent path.

Section 3 measures whether tiling the detector face recovers the ensemble
average numerically instead.  It does not, at any affordable tile count: the
speckle angular scale ``lambda / D_perp`` is orders of magnitude finer than the
tile spacing, so the face integral plateaus well short of convergence.

Numbers from this script back
``docs/validation/radiation-physics/transverse-bunch-form-factor.md``.

Validation: transverse-bunch-form-factor

Run:  uv run python checks/coherent_transverse_coherence.py
      uv run python checks/coherent_transverse_coherence.py --quick
"""

import argparse
import time

import numpy as np

from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.montecarlo import (
    beta_from_keV,
    detector_directions,
    mc_spectrum,
    simulate_trajectories,
)

THETA_OBS_RAD = np.deg2rad(119.0)
E0_KEV = 25.0
B_ANG2 = 0.8
HKL = ((0, 0, 2), (0, 0, -2))
THICKNESS_ANG = 1.0e7
E_GRID = np.arange(850.0, 1100.0, 0.5)

# Timepix geometry (docs/physics/detectors/detector-solid-angle.md).
CHIP_MM = 14.0
DIST_MM = 400.0
OMEGA_SR = 9.5e-4

# Working distances spanned by section 1.  30 mm is the ``detector_directions``
# default, 400 mm the Timepix operating point.
DISTANCES_MM = (10.0, 30.0, 100.0, 400.0)

# Transverse spot sizes for sections 2 and 4.  ``None`` is the point-source
# degenerate limit: every electron starts at the origin, so the cross-electron
# terms are maximally constructive and the result is not a physical bunch.
SPOTS_MM = (None, 0.001, 0.05, 1.0)


def _traj(n_electrons, seed, spot_mm, n_atoms):
    return simulate_trajectories(
        E0_KEV,
        n_electrons,
        THICKNESS_ANG,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=5.0,
        seed=seed,
        beam_fwhm_mm=spot_mm,
    )


def _spec(segs, n_hat, coherent):
    return mc_spectrum(segs, E_GRID, "hopg", HKL, n_hat=n_hat, B_ang2=B_ANG2, coherent=coherent)


def _face_integral(segs, n_side, coherent, dist_mm=DIST_MM):
    n_hats, weights = detector_directions(
        THETA_OBS_RAD, n_side=n_side, chip_mm=CHIP_MM, dist_mm=dist_mm, domega_sr=OMEGA_SR
    )
    total = None
    for n_i, w_i in zip(n_hats, weights, strict=True):
        contrib = float(w_i) * _spec(segs, n_i, coherent)
        total = contrib if total is None else total + contrib
    return total


def _transverse_extent_ang(segs, n_hat):
    """95th-percentile diameter of the emitting volume transverse to ``n_hat``."""
    r = segs["r_mid"]
    perp = r - np.outer(r @ n_hat, n_hat)
    return float(np.percentile(np.linalg.norm(perp - perp.mean(axis=0), axis=1), 95) * 2)


def _line_wavelength_ang(beta, n_hat):
    """Resonance wavelength of the (0,0,2) forward harmonic, in Angstrom."""
    g_vec, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    v = beta * np.array([0.0, 0.0, 1.0])
    omega = float(v @ g_vec) / (1.0 - float(n_hat @ v))
    return 2.0 * np.pi / omega


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="fewer electrons, seeds, and tile counts"
    )
    args = parser.parse_args(argv)

    n_electrons = 80 if args.quick else 300
    n_seeds = 4 if args.quick else 8
    tile_sides = (1, 3, 5) if args.quick else (1, 3, 5, 7, 9)
    ref_side = 7 if args.quick else 11

    info = CRYSTALS["hopg"]
    n_atoms = len(info["basis"]) / info["V_cell"]
    beta = beta_from_keV(E0_KEV)
    n_hat0 = np.array([np.sin(THETA_OBS_RAD), 0.0, np.cos(THETA_OBS_RAD)])
    t_start = time.time()

    segs_point = _traj(n_electrons, 7, None, n_atoms)
    d_perp_ang = _transverse_extent_ang(segs_point, n_hat0)
    lam_ang = _line_wavelength_ang(beta, n_hat0)
    dtheta = 2.0 * np.arctan(0.5 * CHIP_MM / DIST_MM)
    speckle_rad = lam_ang / d_perp_ang

    # --- 1. far-field (Fraunhofer) validity ----------------------------------
    print("[1] neglected Fresnel curvature vs working distance")
    print(
        f"    line wavelength {lam_ang:.2f} A; emitting volume D_perp "
        f"{d_perp_ang / 1e4:.2f} um (95th pct diameter, transverse to n_hat)"
    )
    for dist_mm in DISTANCES_MM:
        R_ang = dist_mm * 1e7
        patch_um = np.sqrt(lam_ang * R_ang / np.pi) / 1e4
        phi_rad = np.pi * (d_perp_ang / 2.0) ** 2 / (lam_ang * R_ang)
        print(
            f"    dist={dist_mm:6.1f} mm: one-radian patch sqrt(lam R/pi)={patch_um:6.2f} um  "
            f"neglected phase at D_perp/2 = {phi_rad:6.3f} rad"
        )

    # --- 2. speckle: seed-to-seed scatter of the single-n_hat peak -----------
    print(f"\n[2] single-n_hat peak height over {n_seeds} transport seeds")
    print("    (incoherent scatter is the Monte Carlo counting-noise control)")
    ensemble = {}
    for spot in SPOTS_MM:
        pk_coh, pk_inc = [], []
        for seed in range(n_seeds):
            segs = _traj(n_electrons, seed, spot, n_atoms)
            pk_coh.append(float(_spec(segs, n_hat0, True).max()))
            pk_inc.append(float(_spec(segs, n_hat0, False).max()))
        pk_coh = np.array(pk_coh)
        pk_inc = np.array(pk_inc)
        ensemble[spot] = (pk_coh.mean(), pk_inc.mean())
        print(
            f"    spot={str(spot):>6s} mm: coherent std/mean={pk_coh.std() / pk_coh.mean():.3f} "
            f"min/max={pk_coh.min() / pk_coh.max():.3f}   |   "
            f"incoherent std/mean={pk_inc.std() / pk_inc.mean():.3f}"
        )

    # --- 3. does tiling the face recover the ensemble average? ---------------
    print(f"\n[3] face-integral convergence, L1 error vs an n_side={ref_side} reference")
    print(
        f"    speckle angular scale lam/D_perp = {speckle_rad:.2e} rad against "
        f"Delta-theta = {np.degrees(dtheta):.2f} deg\n"
        f"    -> {dtheta / speckle_rad:.0f} fringes across the face; convergence needs "
        f"n_side ~ {dtheta / speckle_rad:.0f} ({(dtheta / speckle_rad) ** 2:.0f} directions)"
    )
    for spot in (None, 0.05):
        segs = _traj(n_electrons, 7, spot, n_atoms)
        print(f"    -- beam_fwhm_mm={spot} --")
        for coherent in (False, True):
            ref = _face_integral(segs, ref_side, coherent)
            denom = np.trapezoid(ref, E_GRID)
            cells = []
            for n_side in tile_sides:
                spec = _face_integral(segs, n_side, coherent)
                err = np.trapezoid(np.abs(spec - ref), E_GRID) / denom
                cells.append(f"n={n_side}:{err * 100:6.2f}%")
            label = "coherent" if coherent else "incoherent"
            print(f"       {label:11s} " + "  ".join(cells))

    # --- 4. the ensemble mean the form factor would give deterministically ---
    print("\n[4] seed-averaged coherent enhancement (should be spot-independent)")
    for spot in SPOTS_MM:
        coh_mean, inc_mean = ensemble[spot]
        tag = "  <- degenerate point source, not a physical bunch" if spot is None else ""
        print(
            f"    spot={str(spot):>6s} mm: <coherent peak>/<incoherent peak> = "
            f"{coh_mean / inc_mean:8.2f}{tag}"
        )
    finite = [ensemble[s][0] / ensemble[s][1] for s in SPOTS_MM if s is not None]
    print(
        f"    finite-spot enhancement spread: {min(finite):.2f}-{max(finite):.2f} "
        f"(the intra-electron floor; cross-electron terms average away)"
    )
    print(f"\nelapsed {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
