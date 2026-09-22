"""Is it safe to drop the cross-reflection terms of the coherent segment sum?

``mc_spectrum(coherent=True)`` accumulates a complex field per ``(reflection,
mosaic orientation)`` row and squares each row separately, so distinct
reflections add as INTENSITIES.  Writing the total field as
``F = sum_g F_g`` the exact intensity is
``|F|^2 = sum_g |F_g|^2 + sum_{g != g'} F_g F_g'^*``; the implementation keeps
the first sum and drops the second.  This script measures what is dropped.

Two independent mechanisms suppress the cross term, and the script measures one
number for each:

- ``bound``  Cauchy--Schwarz.  ``|2 Re(F_g F_g'^*)| <= 2 sqrt(S_g S_g')`` is the
             tightest pointwise bound available from the per-reflection
             intensities alone, and needs no access to the fields.  It is what
             spectral separation buys: where one line is strong the others sit
             far out on their own sinc tails, so the geometric mean collapses.
- ``|<e^{i dg.r}>|``  reciprocal-lattice decorrelation.  The same-segment part
             of ``F_g F_g'^*`` carries a residual phase ``exp[-i(g-g').r_j]``
             that the diagonal ``|F_g|^2`` does not: its ``j = k`` terms are
             exactly ``|c_j|^2``.  Segment midpoints are spread over a sample
             thousands of lattice spacings deep and are uncorrelated with the
             lattice period, so this factor random-walks to ``~1/sqrt(n_seg)``
             while the diagonal keeps a non-cancelling floor.  The cross term is
             therefore zero-mean over the ensemble, not merely small.

Per-reflection spectra are produced by calling the production coherent kernel
once per reflection; the script first checks that those pieces re-sum to the
all-reflections call, which is what makes the split a faithful decomposition of
the shipped result rather than a separate model.

Numbers from this script back
``docs/validation/radiation-physics/cross-reflection-coherence.md``.

Validation: cross-reflection-coherence

Run:  uv run python checks/cross_reflection_coherence.py
      uv run python checks/cross_reflection_coherence.py --quick
"""

import argparse
import itertools
import time

import numpy as np

from pyrite.materials.catalog import load_material_catalog
from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    beta_from_Ee,
    reciprocal_g_vector,
)
from pyrite.montecarlo import mc_spectrum, simulate_trajectories

THETA_OBS_DEG = 119.0

# (label, crystal, beam keV, thickness Ang, grid lo/hi/step eV, electron scale)
# The scale divides ``--electrons``: a thick, high-energy slab makes far more
# segments per electron, and the coherent kernel holds an (n_seg, n_bin)
# complex grid.
CONFIGS = (
    ("hopg 30 keV / 500 nm", "hopg", 30.0, 5.0e3, (300.0, 3000.0, 2.0), 1),
    ("hbn 30 keV / 921 nm", "hbn", 30.0, 9.21e3, (300.0, 3000.0, 2.0), 1),
    ("hopg 100 keV / 10 um", "hopg", 100.0, 1.0e5, (300.0, 11000.0, 5.0), 40),
)


def _n_hat(theta_obs_deg: float) -> np.ndarray:
    """Far-field observation direction in the sample frame."""
    theta = np.deg2rad(theta_obs_deg)
    return np.array([np.sin(theta), 0.0, np.cos(theta)])


def _sinc_zero_width_eV(segments, denom: float) -> np.ndarray:
    """First-zero half-width of each segment's ``sinc^2`` line profile [eV].

    The kernel's ``a_width = denom * t_L / (2 hbar c)`` puts the first zero of
    ``sinc(a_width (E - E_res)/pi)`` at ``a_width |E - E_res| = pi``.
    """
    beta = beta_from_Ee(np.asarray(segments["E_keV"], dtype=float) * 1e3)
    t_L = np.asarray(segments["L_ang"], dtype=float) / beta
    return 2.0 * np.pi * HBARC_EV_ANG / (denom * t_L)


def _decorrelation(segments, dg: np.ndarray) -> float:
    """``|<exp(-i dg.r)>|`` over the segment midpoints, ``dg = g - g'``."""
    phase = np.asarray(segments["r_mid"], dtype=float) @ dg
    return float(abs(np.mean(np.exp(-1j * phase))))


def _measure(label, crystal, E0_keV, thickness_ang, grid, n_electrons, seed, chunk):
    """Run one configuration and return its printable rows."""
    catalog_crystal = load_material_catalog().crystal(crystal)
    composition = catalog_crystal.composition
    hkl_list = list(catalog_crystal.hkl_list)
    lattice = CRYSTALS[crystal]["lattice"]
    n_hat = _n_hat(THETA_OBS_DEG)
    E_grid = np.arange(*grid)

    segments = simulate_trajectories(
        E0_keV,
        n_electrons,
        thickness_ang,
        composition=composition,
        seed=seed,
        energy_model="midpoint",
    )
    kwargs = dict(
        crystal=crystal,
        B_ang2=catalog_crystal.B_ang2,
        n_hat=n_hat,
        composition=composition,
        chunk=chunk,
    )
    total = mc_spectrum(segments, E_grid, hkl_list=hkl_list, coherent=True, **kwargs)
    per_hkl = {
        hkl: mc_spectrum(segments, E_grid, hkl_list=[hkl], coherent=True, **kwargs)
        for hkl in hkl_list
    }
    resummed = sum(per_hkl.values())
    peak = float(total.max())
    additivity = float(np.max(np.abs(total - resummed))) / peak

    radiating = [hkl for hkl in hkl_list if per_hkl[hkl].max() > 0.0]
    bound = np.zeros_like(E_grid)
    pairs = []
    for hkl, hkl2 in itertools.combinations(radiating, 2):
        bound += 2.0 * np.sqrt(per_hkl[hkl] * per_hkl[hkl2])
        g, _ = reciprocal_g_vector(hkl, lattice)
        g2, _ = reciprocal_g_vector(hkl2, lattice)
        pairs.append((hkl, hkl2, _decorrelation(segments, g - g2)))

    fraction = np.divide(bound, resummed, out=np.zeros_like(bound), where=resummed > 0.0)
    significant = resummed > 1e-3 * peak
    denom = 1.0 - beta_from_Ee(E0_keV * 1e3) * float(n_hat[2])
    widths = np.percentile(_sinc_zero_width_eV(segments, denom), [5, 50, 95])

    summary = {
        "label": label,
        "n_seg": int(np.asarray(segments["L_ang"]).size),
        "additivity": additivity,
        "widths": widths,
        "worst": float(fraction.max()),
        "worst_significant": float(fraction[significant].max()),
        "worst_significant_eV": float(E_grid[significant][np.argmax(fraction[significant])]),
        "worst_significant_share": float(
            resummed[significant][np.argmax(fraction[significant])] / peak
        ),
        "integrated": float(np.trapezoid(bound, E_grid) / np.trapezoid(resummed, E_grid)),
    }
    lines = []
    for hkl in radiating:
        index = int(np.argmax(per_hkl[hkl]))
        lines.append(
            (
                hkl,
                float(E_grid[index]),
                float(per_hkl[hkl][index] / peak),
                float(fraction[index]),
            )
        )
    return summary, lines, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer electrons, first config only")
    parser.add_argument("--electrons", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=4000, help="segments per kernel chunk")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = CONFIGS[:1] if args.quick else CONFIGS
    n_electrons = 200 if args.quick else args.electrons

    for label, crystal, E0_keV, thickness_ang, grid, scale in configs:
        started = time.time()
        summary, lines, pairs = _measure(
            label,
            crystal,
            E0_keV,
            thickness_ang,
            grid,
            max(1, n_electrons // scale),
            args.seed,
            args.chunk,
        )
        print(f"{summary['label']}   {summary['n_seg']} segments   {time.time() - started:.1f} s")
        print(
            f"  per-reflection pieces re-sum to the shipped total to "
            f"{summary['additivity']:.2e} of peak"
        )
        w5, w50, w95 = summary["widths"]
        print(f"  sinc first-zero half-width [eV]  p5 {w5:.1f}  p50 {w50:.1f}  p95 {w95:.1f}")
        print("  reflection   E_res [eV]   share of peak   bound at its own peak")
        for hkl, E_res, share, at_peak in lines:
            print(f"  {str(hkl):>12}   {E_res:9.1f}   {share:13.3e}   {at_peak:21.3e}")
        print("  pair                        |<exp(-i dg.r)>|")
        for hkl, hkl2, factor in pairs:
            print(f"  {str(hkl):>12} x {str(hkl2):<12}  {factor:.3e}")
        print(f"  worst-case bound, all bins                    {summary['worst']:.3e}")
        print(
            f"  worst-case bound, bins above 1e-3 of peak     "
            f"{summary['worst_significant']:.3e}"
            f"  (at {summary['worst_significant_eV']:.0f} eV, "
            f"{summary['worst_significant_share']:.1e} of peak)"
        )
        print(f"  bound on the integrated yield                 {summary['integrated']:.3e}")
        print()


if __name__ == "__main__":
    main()
