"""Transition-radiation omission bound per detected observable (#102).

For representative scenes, bound the exact classical TR of every boundary
crossing that can reach the detector and compare it with the production
bremsstrahlung density the detector already scores (the continuum alone, so
lines and CXR only make the comparison more favourable). The decision rule is
``bias <= target_rse / 3`` (``omission_tolerance``) on every band an
observable can draw from. Small runs; the TR solve follows PYRITE_MC_BACKEND
(NumPy or CuPy, complex128 on both). No remote compute needed.

    PYRITE_MC_BACKEND=cpu uv run python checks/transition_radiation_omission.py \
        [--trials 100] [--json out.json]

Validation: tr-detected-omission-bound
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

import pyrite as pr
from pyrite._backend import xp
from pyrite._photon_continuum_floor import continuum_medium_key, photon_continuum_floor_eV
from pyrite.montecarlo.geometry import sample_to_lab_R
from pyrite.montecarlo.spectrum.transition_radiation import (
    beta_from_kinetic_eV,
    medium_dielectric,
    omission_tolerance,
    planar_transition_radiation,
)

TARGET_RSE = 0.05  # catalog adaptive-precision default
# A primary crosses a flat face at most twice; entry and exit through the same
# face add as amplitudes (<= 4x one crossing), plus escaping secondaries (< 1).
CROSSINGS = 5.0
HALF_SPAN_DEG = 8.3  # half the 16.6 deg EDS polar span used by the reference detector
BANDS = {"euv": (0.0, 300.0), "soft": (300.0, 1000.0), "kev": (1000.0, math.inf)}
# Documented decisions (docs/physics/radiation-physics/transition-radiation.md):
# the check fails if any of them stops holding.
DOCUMENTED_OMIT = {"kev": "tens-of-keV"}
DOCUMENTED_ESCALATE = ("euv",)


@dataclass(frozen=True)
class Scene:
    name: str
    row: str
    layers: tuple[tuple[str, float], ...]  # beam-entrance order
    kinetic_keV: float
    theta_obs_deg: float
    tilt_deg: float = 30.0
    # None: diffusive transport, any crossing direction. Otherwise electrons
    # stay within this half-angle of the beam (3x the Highland width).
    cone_deg: float | None = None


SCENES = (
    Scene("si-10keV-bulk-front", "tens-of-keV", (("silicon", 1e5),), 10.0, 150.0),
    Scene("si-30keV-film-rear", "tens-of-keV", (("silicon", 2e4),), 30.0, 90.0),
    Scene("si-30keV-bulk-front", "tens-of-keV", (("silicon", 1e5),), 30.0, 150.0),
    Scene("si-60keV-film-rear", "tens-of-keV", (("silicon", 2e4),), 60.0, 90.0),
    Scene("hopg-30keV-film-rear", "tens-of-keV", (("hopg", 2e4),), 30.0, 90.0),
    Scene("mos2-30keV-film-rear", "tens-of-keV", (("mos2", 2e4),), 30.0, 90.0),
    Scene("wse2-30keV-film-rear", "tens-of-keV", (("wse2", 2e4),), 30.0, 90.0),
    Scene(
        "mos2-on-sapphire-30keV-front",
        "tens-of-keV",
        (("mos2", 2e3), ("sapphire", 5e5)),
        30.0,
        150.0,
    ),
    Scene("si-3MeV-film-offaxis", "MeV thin film", (("silicon", 2e4),), 3e3, 90.0, cone_deg=2.1),
    Scene("si-3MeV-film-forward", "MeV thin film", (("silicon", 2e4),), 3e3, 1.0, cone_deg=2.1),
    Scene("si-10MeV-film-forward", "MeV thin film", (("silicon", 2e4),), 1e4, 1.0, cone_deg=0.7),
    Scene("si-30MeV-film-forward", "30-100 MeV", (("silicon", 2e4),), 3e4, 1.0, cone_deg=0.21),
    Scene("si-100MeV-film-offaxis", "30-100 MeV", (("silicon", 2e4),), 1e5, 90.0, cone_deg=0.07),
    Scene("si-100MeV-film-forward", "30-100 MeV", (("silicon", 2e4),), 1e5, 1.0, cone_deg=0.07),
    Scene("si-100MeV-bulk-forward", "30-100 MeV", (("silicon", 1e6),), 1e5, 1.0, cone_deg=0.6),
)


def _simulate(scene: Scene, grid: np.ndarray, trials: int):
    beam = pr.Beam(
        energy_keV=scene.kinetic_keV,
        transverse=pr.beams.TransverseSpot(fwhm_x_mm=1.0),
        longitudinal=pr.beams.PointBunch(),
    )
    footprint = pr.geometry.Footprint(5.0, 5.0)
    if len(scene.layers) == 1:
        material, thickness = scene.layers[0]
        target = pr.geometry.Slab(
            material, thickness_ang=thickness, tilt_deg=scene.tilt_deg, footprint=footprint
        )
    else:
        target = pr.geometry.Stack(
            tuple(pr.geometry.Layer(m, thickness_ang=t) for m, t in scene.layers),
            tilt_deg=scene.tilt_deg,
            footprint=footprint,
        )
    # The study reads only the continuum; above 10 MeV the automatic line grid
    # exceeds its point budget, so give the lines a fixed coarse grid there.
    cxr = np.linspace(1e3, 20e3, 2000) if scene.kinetic_keV > 1e4 else None
    result = pr.simulate(
        beam,
        target,
        pr.Detector(scene.theta_obs_deg),
        physics=pr.Physics(),
        numerics=pr.Numerics(
            cxr_trials=trials,
            brem_trials=trials,
            energy_bins=pr.detectors.EnergyBins(cxr=cxr, brem=grid),
        ),
    )
    return result.brem.photon_energy_eV, result.brem.density


def _detector_cone(scene: Scene) -> tuple[np.ndarray, bool]:
    """Sample-frame acceptance directions, mirrored so the facing face is ``+z``.

    Centre plus four points at the half span in polar angle and in azimuth
    about the beam axis. Returns the directions and whether the detector faces
    the beam-entrance face.
    """
    R = sample_to_lab_R(math.radians(scene.tilt_deg))
    half = math.radians(HALF_SPAN_DEG)
    theta = math.radians(scene.theta_obs_deg)
    lab = [
        (theta + d_theta, d_phi)
        for d_theta, d_phi in ((0, 0), (half, 0), (-half, 0), (0, half), (0, -half))
    ]
    dirs = np.array(
        [R.T @ [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t, p in lab]
    )
    front = bool(dirs[0, 2] < 0)
    if np.any(np.sign(dirs[:, 2]) != np.sign(dirs[0, 2])):
        raise ValueError(f"{scene.name}: acceptance straddles the slab plane")
    return (dirs * [1, 1, -1] if front else dirs), front


def _velocities(beta0: float, axis: np.ndarray | None, cone_deg: float | None) -> np.ndarray:
    """Crossing velocities: speeds up to ``beta0``, both crossing senses.

    Without a cone, every direction to 88 deg from the normal; with one, the
    beam ``axis`` and directions at the cone half-angle around it.
    """
    out = []
    if cone_deg is None:
        for psi in np.radians(np.arange(0.0, 89.0, 4.0)):
            for az in np.radians(np.arange(0.0, 360.0, 20.0)):
                for sign in (+1, -1):
                    out.append(
                        [np.sin(psi) * np.cos(az), np.sin(psi) * np.sin(az), sign * np.cos(psi)]
                    )
    else:
        u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        out.append(axis)
        for frac in (0.5, 1.0):
            cone = math.radians(cone_deg) * frac
            for az in np.radians(np.arange(0.0, 360.0, 30.0)):
                out.append(np.cos(cone) * axis + np.sin(cone) * (np.cos(az) * u + np.sin(az) * w))
    unit = np.array(out)
    return np.concatenate([f * beta0 * unit for f in (0.25, 0.5, 0.75, 1.0)])


def transition_radiation_bound(scene: Scene, energy: np.ndarray) -> np.ndarray:
    """``CROSSINGS x`` max TR over speeds <= beta0, all crossing directions, the cone."""
    cone, front = _detector_cone(scene)
    beam_dir = sample_to_lab_R(math.radians(scene.tilt_deg)).T @ [0.0, 0.0, 1.0]
    axis = beam_dir * [1, 1, -1] if front else beam_dir
    velocities = _velocities(beta_from_kinetic_eV(scene.kinetic_keV * 1e3), axis, scene.cone_deg)
    # Stack seen from the facing vacuum: far side first.
    layers = list(scene.layers[::-1] if front else scene.layers)
    out = np.zeros(energy.size)
    for i, e in enumerate(energy):
        eps = [complex(medium_dielectric(m, e)) for m, _ in layers]
        stacks = [([(eps_j, t) for eps_j, (_, t) in zip(eps, layers, strict=True)], 1.0)]
        if scene.cone_deg is None:
            # Reversing (backscattered) electrons see the facing layer alone.
            stacks.append(([], eps[-1]))
        best = 0.0
        for stack, substrate in stacks:
            tr = planar_transition_radiation(
                e, cone[None, :, :], velocities[:, None, :], stack, substrate, xp=xp
            )
            best = max(best, float(tr.max()))
        out[i] = CROSSINGS * best
    return out


def _band_bounds(energy, ratio, tol):
    bands = {}
    for name, (lo, hi) in BANDS.items():
        keep = (energy >= lo) & (energy < hi)
        if np.any(keep):
            bound = float(np.max(ratio[keep]))
            bands[name] = {"bias_bound": bound, "omit": bound <= tol}
    # Lowest energy above which every grid point passes.
    failing = np.flatnonzero(~(np.isfinite(ratio) & (ratio <= tol)))
    if not failing.size:
        return bands, float(energy[0])
    if failing[-1] + 1 == energy.size:
        return bands, math.inf
    return bands, float(energy[failing[-1] + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tr-points", type=int, default=40)
    parser.add_argument("--only", default=None, help="comma-separated scene names")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    tol = omission_tolerance(TARGET_RSE)
    records = []
    broken = []
    for scene in SCENES:
        if args.only and scene.name not in args.only.split(","):
            continue
        floor = photon_continuum_floor_eV(continuum_medium_key(scene.layers[0][0]))
        e_max = min(scene.kinetic_keV * 1e3 * 0.98, 30e3)
        grid = np.geomspace(1.01 * floor, e_max, 160)
        brem_energy, brem = _simulate(scene, grid, args.trials)
        energy = np.geomspace(brem_energy[0], brem_energy[-1], args.tr_points)
        source = np.interp(energy, brem_energy, brem)
        tr = transition_radiation_bound(scene, energy)
        ratio = np.where(source > 0, tr / np.where(source > 0, source, 1.0), np.inf)
        bands, e_star = _band_bounds(energy, ratio, tol)
        record = {
            "scene": scene.name,
            "row": scene.row,
            "layers": scene.layers,
            "kinetic_keV": scene.kinetic_keV,
            "theta_obs_deg": scene.theta_obs_deg,
            "tilt_deg": scene.tilt_deg,
            "cone_deg": scene.cone_deg,
            "trials": args.trials,
            "tolerance": tol,
            "bands": bands,
            "e_star_eV": e_star,
            "max_ratio": float(np.max(ratio)),
            "energy_eV": energy.tolist(),
            "ratio": ratio.tolist(),
        }
        records.append(record)
        summary = "  ".join(
            f"{k}={v['bias_bound']:.1e}{'' if v['omit'] else '!'}" for k, v in bands.items()
        )
        print(f"{scene.name:32s} E*={e_star:8.0f} eV  {summary}", flush=True)
        for band, row in DOCUMENTED_OMIT.items():
            if scene.row == row and not bands[band]["omit"]:
                broken.append(f"{scene.name}: {band} no longer omissible")
        for band in DOCUMENTED_ESCALATE:
            if band in bands and bands[band]["omit"]:
                broken.append(f"{scene.name}: {band} now omissible; revisit the escalation")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(records, handle, indent=1)
    for message in broken:
        print(f"DECISION CHANGED: {message}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
