"""Omitted angular spread of planar zone rows: measured second-order bound (#1148).

Native scorers (PXR/CBS ``spec``, incoherent ``brem``, ``spec_characteristic``)
treat each planar zone row as a straight mean segment along the beam. The
retained (channeled) electrons in fact move at transverse angles: in the
diffraction direction normal to the planes (bounded oscillation, ``|theta_x|
<~ psi_c``) and parallel to them (free, broadened by TDS and electronic kicks).

For a row yield ``Y(v)`` at a fixed detector direction, averaging over the
retained angle distribution gives, to second order,

``<Y> - Y(v0) = sum_a [ Y_a <theta_a> + (1/2) Y_aa <theta_a^2> ]``,

with ``a`` the normal and in-plane-transverse axes (the cross term vanishes for
the mirror-symmetric planar ensemble). The check measures:

- moments: the zone solve's retained density (``propagate_batch`` with the
  case's kernels and entry state) at several depths, weighted along the depth by
  the retained trace. ``theta_x`` moments come from the plane-wave diagonal
  (``density_exit_distribution`` angles). ``<theta_y^2>`` comes from the
  out-of-plane transform ``P~(eta)/P~(0) = <cos(p theta_y eta)>`` at two small
  ``eta`` (``cr-planar-exit-out-of-plane``), plus the beam's own out-of-plane
  divergence;
- derivatives: zone rows rescored with ``v_hat`` tilted by ``+-d`` and
  ``+-2d`` about each axis, ``d`` the measured rms. The ``+-d`` two-point rule
  equals the Gaussian average through third order, and the ``2d`` arm checks
  that the response is quadratic.

The reported shift is ``Delta = [Y(+d) + Y(-d)]/2 - Y(0)`` plus
``<theta> [Y(+d) - Y(-d)]/(2 d)``, per axis, against the zone-row yield and the
all-row (detector) yield of each component. Characteristic emission is isotropic,
so its shift tests the method (only the escape path changes).

    PYRITE_MC_BACKEND=cpu uv run python checks/channeling_zone_angular_spread.py \
        --electrons 16 --json scratch/zone_angular_spread.json

Detector-face arms reuse ``checks/channeling_detector_components.py``
(``--detector-solid-angle-sr``, ``--polar-deg``). Report only: exit 0 when every
number is finite, 1 otherwise. No gate: the decision whether a shift is
negligible is recorded with the evidence.

Validation: cr-zone-row-angular-spread
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np

_COMPONENTS = runpy.run_path(str(Path(__file__).with_name("channeling_detector_components.py")))
SCORED = ("spec", "spec_characteristic", "brem")


def _retained_moments(case, depths_count: int, eta_scale: tuple[float, float]):
    """Depth-averaged retained angle moments [rad, rad^2] of the zone solve.

    Validation: cr-zone-row-angular-spread
    """
    from pyrite.materials.channeling.potential import planar_profile
    from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
    from pyrite.montecarlo.channeling.density import DensityMatrixSolver
    from pyrite.montecarlo.channeling.exit_out_of_plane import (
        out_of_plane_kernel,
        propagate_batch,
    )
    from pyrite.montecarlo.channeling.handoff import case_plane_incidence

    record = case["channeling"]
    kinetic = float(case["E0_keV"]) * 1e3
    plane = tuple(int(v) for v in record["plane_hkl"])
    profile = planar_profile(case["crystal"], plane, float(record["temperature_K"]))
    options = {"n_cells": int(record["bloch_cells"]), "max_order": int(record["basis_size"])}
    base = out_of_plane_kernel(profile, kinetic, 0.0, **options)
    solver = DensityMatrixSolver(profile, kinetic, base, margin_eV=float(record["margin_eV"]))
    p_inv_ang = solver.gamma0 * solver.beta0 * M_E_EV / HBARC_EV_ANG
    incidence = case_plane_incidence(case)
    divergence = float(case.get("divergence_rad", 0.0) or 0.0)
    if p_inv_ang * divergence / (profile.g_inv_ang / solver.n_cells) < 1.0:
        divergence = 0.0  # snapped as in handoff._solve
    psi_c = float(np.sqrt(2.0 * (solver.barrier_eV - profile.extrema()[0]) / (
        solver.gamma0 * solver.beta0**2 * M_E_EV)))  # fmt: skip
    # eta such that p eta sqrt(<theta_y^2>) ~ 0.1, 0.2 for theta_y ~ psi_c.
    eta = np.array([0.0, *(s / (p_inv_ang * psi_c) for s in eta_scale)])
    kernels = np.stack(
        [solver._cell_kernel0]
        + [
            out_of_plane_kernel(profile, kinetic, value, **options).cell_kernel(solver.n_grid)
            for value in eta[1:]
        ]
    )
    rho, _, _ = solver.entry_state(float(incidence.effective_rad), divergence)
    dz = float(record["dz_ang"])
    path = float(case["thickness_ang"]) / float(np.asarray(incidence.beam_dir)[2])
    steps = max(int(np.floor(path / dz)), 1)
    marks = np.unique(np.linspace(1, steps, depths_count).round().astype(int))
    depths = marks * dz
    weights = propagate_batch(solver, kernels, rho, depths, dz)  # (D, B, cells, N)
    angles = (solver.kappa[:, None] + solver.orders[None, :] * profile.g_inv_ang) / p_inv_ang
    rows = []
    for d, depth in enumerate(depths):
        w = weights[d, 0]
        trace = float(w.sum())
        mean_x = float((w * angles).sum() / trace)
        second_x = float((w * angles**2).sum() / trace)
        ratio = weights[d, 1:].sum(axis=(1, 2)) / trace
        second_y = -2.0 * np.log(np.clip(ratio, 1e-300, None)) / (p_inv_ang * eta[1:]) ** 2
        rows.append(
            {
                "depth_ang": float(depth),
                "retained": trace,
                "mean_x_rad": mean_x,
                "second_x_rad2": second_x,
                "second_y_rad2_by_eta": [float(v) for v in second_y],
            }
        )
    retained = np.array([r["retained"] for r in rows])
    # Trapezoid in depth from the entrance (retained = entry flux at z = 0).
    z = np.r_[0.0, depths]
    w_z = np.r_[float(np.real(np.trace(rho.sum(axis=0)))), retained]

    def average(values, at_entry):
        return float(np.trapezoid(w_z * np.r_[at_entry, values], z) / np.trapezoid(w_z, z))

    entry_x = float(incidence.effective_rad)
    mean_x = average([r["mean_x_rad"] for r in rows], entry_x)
    second_x = average([r["second_x_rad2"] for r in rows], entry_x**2 + divergence**2)
    by_eta = np.array([r["second_y_rad2_by_eta"] for r in rows])
    second_y_scatter = [average(by_eta[:, k], 0.0) for k in range(by_eta.shape[1])]
    divergence_y = float(case.get("divergence_y_rad", divergence) or 0.0)
    return {
        "psi_c_rad": psi_c,
        "eta_ang": [float(v) for v in eta[1:]],
        "depths": rows,
        "mean_x_rad": mean_x,
        "rms_x_rad": float(np.sqrt(max(second_x - mean_x**2, 0.0))),
        "second_x_rad2": second_x,
        "second_y_scatter_rad2_by_eta": second_y_scatter,
        "rms_y_rad": float(np.sqrt(second_y_scatter[0] + divergence_y**2)),
        "plane_normal": [float(v) for v in incidence.plane_normal],
    }


def _tilted(segs, zone, axis, angle):
    """Rows with zone ``v_hat`` rotated by ``angle`` towards ``axis``.

    Validation: cr-zone-row-angular-spread
    """
    v = np.asarray(segs["v_hat"], dtype=float)
    a = np.broadcast_to(np.asarray(axis, dtype=float), v.shape)
    perp = a - np.sum(a * v, axis=1)[:, None] * v
    perp /= np.linalg.norm(perp, axis=1)[:, None]
    out = dict(segs)
    turned = v.copy()
    turned[zone] = np.cos(angle) * v[zone] + np.sin(angle) * perp[zone]
    out["v_hat"] = turned
    return out


def main(argv=None) -> int:
    """Measure the retained moments and the tilted-row shifts; print and record them.

    Validation: cr-zone-row-angular-spread
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="channeling_demo")
    parser.add_argument("--material", default="silicon")
    parser.add_argument("--electrons", type=int, default=16, help="Ne = Ne_brem")
    parser.add_argument("--max-bragg", type=int, default=2, help="Bragg reflections kept")
    parser.add_argument("--depths", type=int, default=8, help="retained-moment depths")
    parser.add_argument("--json", default=None, help="write the record here")
    parser.add_argument("--detector-solid-angle-sr", type=float, default=None)
    parser.add_argument("--detector-distance-mm", type=float, default=5000.0)
    parser.add_argument("--detector-order", type=int, default=3)
    parser.add_argument("--polar-deg", type=float, nargs="+", default=None)
    args = parser.parse_args(argv)

    from pyrite.montecarlo import runner
    from pyrite.montecarlo.geometry import tilted_geometry

    case = _COMPONENTS["_profile_case"](args.profile, args.material)
    if case.get("channeling") is None or case["channeling"].get("model") != "planar":
        raise SystemExit(f"profile {args.profile!r} is not a planar channeling case")
    case.update(Ne=args.electrons, Ne_brem=args.electrons, line_quadrature="bin-mean")
    moments = _retained_moments(case, args.depths, (0.1, 0.2))

    transport = runner._transport_case(case, transport_core="per-electron")
    # The expected-value CR source does not use the rows and is not scored here;
    # skip its (dominant) recomputation in each of the rescorings.
    from pyrite.montecarlo.runner import channeling_emission

    channeling_emission.case_channeling_spectrum = lambda *_args, **_kw: None
    zone = np.asarray(transport["segs"]["channeling_zone"], dtype=bool)
    grid = np.asarray(transport["E_grid"], dtype=float)
    beam, _ = tilted_geometry(
        case["theta_obs_rad"], np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )  # fmt: skip
    beam = np.asarray(beam, dtype=float)
    normal = np.asarray(moments["plane_normal"], dtype=float)
    along = np.cross(normal, beam)
    along /= np.linalg.norm(along)

    directions = [("forward", np.asarray(transport["n_hat"], dtype=float), None)]
    bragg = _COMPONENTS["_bragg_directions"](case, beam, float(grid[0]), float(grid[-1]))
    bragg.sort(key=lambda item: -item[2])
    gamma = 1.0 + float(case["E0_keV"]) / 510.99895
    kept: list[float] = []
    for label, n_hat, energy in bragg:
        if len(kept) >= args.max_bragg or any(abs(energy - e) < 1.0 for e in kept):
            continue
        kept.append(energy)
        directions.append((label, n_hat, energy))
        side = np.cross(beam, n_hat)
        side /= np.linalg.norm(side)
        in_plane = np.cross(side, n_hat)
        for tag, axis in (("+in", in_plane), ("+out", side)):
            probe = n_hat + axis / gamma
            directions.append((label + tag, probe / np.linalg.norm(probe), energy))
    if args.polar_deg is not None:
        if not bragg:
            parser.error("--polar-deg needs an in-band Bragg reflection to define the azimuth")
        tangent = bragg[0][1] - (bragg[0][1] @ beam) * beam
        tangent /= np.linalg.norm(tangent)
        directions = [
            (f"polar-{a:g}", np.cos(np.deg2rad(a)) * beam + np.sin(np.deg2rad(a)) * tangent, None)
            for a in args.polar_deg
        ]

    subset = _COMPONENTS["_subset"]
    score = _COMPONENTS["_detector_score"]
    zone_segs = subset(transport["segs"], zone)
    zone_mask = np.ones(int(zone.sum()), dtype=bool)
    axes = {
        "normal": (normal, moments["rms_x_rad"], moments["mean_x_rad"]),
        "in-plane": (along, moments["rms_y_rad"], 0.0),
    }
    record = {
        "profile": args.profile,
        "electrons": args.electrons,
        "zone_rows": int(zone.sum()),
        "moments": moments,
        "units": "photons/electron" if args.detector_solid_angle_sr else "photons/sr/electron",
        "directions": [],
    }
    ok = True
    for label, n_hat, energy in directions:
        all_rows = score(runner, case, transport, beam, n_hat, args)
        base = score(runner, case, {**transport, "segs": zone_segs}, beam, n_hat, args)
        row = {"label": label, "bragg_eV": energy, "all": all_rows, "zone": base, "axes": {}}
        for name, (axis, rms, mean) in axes.items():
            arms = {}
            for k in (1, -1, 2, -2):
                segs = _tilted(zone_segs, zone_mask, axis, k * rms)
                arms[k] = score(runner, case, {**transport, "segs": segs}, beam, n_hat, args)
            shifts = {}
            for key in SCORED:
                y0, yp, ym = base[key], arms[1][key], arms[-1][key]
                y2p, y2m = arms[2][key], arms[-2][key]
                even = 0.5 * (yp + ym) - y0
                odd = mean * (yp - ym) / (2.0 * rms) if rms > 0 else 0.0
                even_2 = 0.5 * (y2p + y2m) - y0
                shift = even + odd
                shifts[key] = {
                    "shift": shift,
                    "rel_zone": shift / y0 if y0 else float("nan"),
                    "rel_all": shift / all_rows[key] if all_rows[key] else float("nan"),
                    "quadratic_ratio": even_2 / (4.0 * even) if even else float("nan"),
                }
                ok &= bool(np.isfinite(shift))
            row["axes"][name] = {"rms_rad": rms, "mean_rad": mean, "shifts": shifts}
        record["directions"].append(row)

    m = moments
    print(
        f"{args.profile}: Ne={args.electrons}, zone rows {record['zone_rows']}, "
        f"psi_c={m['psi_c_rad']:.3e} rad"
    )
    print(
        f"retained rms: normal {m['rms_x_rad']:.3e} (mean {m['mean_x_rad']:.2e}), "
        f"in-plane {m['rms_y_rad']:.3e} rad; eta arms "
        + ", ".join(f"{v:.3e}" for v in m["second_y_scatter_rad2_by_eta"])
    )
    print(
        f"{'direction':>14} {'axis':>9} {'component':>20} {'rel_zone':>10} {'rel_all':>10} {'q2/q1':>6}"
    )
    for row in record["directions"]:
        name = row["label"] + ("" if row["bragg_eV"] is None else f"@{row['bragg_eV']:.0f}eV")
        for axis, data in row["axes"].items():
            for key, s in data["shifts"].items():
                print(
                    f"{name:>14} {axis:>9} {key:>20} {s['rel_zone']:10.2e} "
                    f"{s['rel_all']:10.2e} {s['quadratic_ratio']:6.2f}"
                )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
