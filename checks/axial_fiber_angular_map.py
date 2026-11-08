"""Si axial off-axis photon map of the fibred absorbing model (#992). Research only.

Same channels and incidence as ``checks/axial_fiber_source.py`` (its ``build``),
scored with
:func:`pyrite.montecarlo.channeling.axial_fiber_angular.fiber_angular_source`
on a polar grid ``theta = theta_over_gamma / gamma`` and ``n_phi`` uniform
azimuths from the profile's first reciprocal axis. Chunks merge exactly:
the source is linear in the correlation, so the merged Stokes map is the
count-weighted mean of chunk maps and its SEM the chunk (batch) spread.

Usage::

    python checks/axial_fiber_angular_map.py chunk '<json kwargs>' START COUNT OUT.json
    python checks/axial_fiber_angular_map.py merge OUT.json PART.json [PART.json ...]

Kwargs: those of ``axial_fiber_source.py`` plus ``theta_over_gamma`` (list) and
``n_phi``. The depth step must keep the Doppler phase step at the largest
energy and angle within 1 rad (library refusal). Summary per polar angle: line
peak of the azimuth-averaged intensity, its azimuthal ``cos(2 phi)`` and
``cos(4 phi)`` Fourier contrasts (the latter vanishes identically at the
implemented dipole order; a nonzero ``cos 2 phi`` for a C4-symmetric incidence
is sampling noise), and the intensity-weighted linear and circular
polarization degrees.

Validation: cr-axial-fiber-angular
"""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

_SOURCE = runpy.run_path(str(Path(__file__).with_name("axial_fiber_source.py")))
ANGULAR_DEFAULTS = {"theta_over_gamma": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], "n_phi": 8}


def chunk(options: dict, start: int, count: int) -> dict:
    """One merge-ready chunk: count times the chunk-mean Stokes map."""
    from pyrite.materials.attenuation import linear_attenuation_inv_mm
    from pyrite.montecarlo.channeling.axial_fiber_angular import fiber_angular_source
    from pyrite.montecarlo.channeling.bloch import longitudinal_kinematics

    angular = {k: options.pop(k, v) for k, v in ANGULAR_DEFAULTS.items()}
    channels, entry, hbar_v, info, o = _SOURCE["build"](options)
    o.update(angular)
    gamma, _ = longitudinal_kinematics(float(o["energy_eV"]))
    theta = np.asarray(o["theta_over_gamma"], dtype=float) / gamma
    phi = 2 * np.pi * np.arange(int(o["n_phi"])) / int(o["n_phi"])
    depths = np.linspace(
        0.0, o["thickness_ang"], int(round(o["thickness_ang"] / o["step_ang"])) + 1
    )
    energies = np.linspace(o["e_min_eV"], o["e_max_eV"], int(o["n_energy"]))
    extra = {}
    if o["max_working_bytes"] is not None:
        extra["max_working_bytes"] = int(o["max_working_bytes"])
    if o["absorption"]:
        extra.update(
            absorption_inv_ang=linear_attenuation_inv_mm("silicon", energies) * 1e-7,
            exit_path_ang=depths[-1] - depths,
        )
    clock = time.perf_counter()
    source = fiber_angular_source(
        channels,
        entry,
        kinetic_energy_eV=float(o["energy_eV"]),
        hbar_v_eV_ang=hbar_v,
        depths_ang=depths,
        photon_energy_eV=energies,
        theta_rad=theta,
        phi_rad=phi,
        incident_ids=range(start, start + count),
        seed=int(o["seed"]),
        boundary_eV=o["boundary_eV"],
        carrier_inv_ang=tuple(o["carrier_inv_ang"]),
        batches=2,
        search_step_ang=200.0,
        **extra,
    )
    n = len(source.events)
    return {
        "status": "research-only; fibred absorbing model, off-axis dipole order, retained only",
        "settings": o,
        "info": {k: v for k, v in info.items() if k != "events"},
        "start": start,
        "count": n,
        "seconds": time.perf_counter() - clock,
        "photon_energy_eV": energies.tolist(),
        "theta_rad": theta.tolist(),
        "phi_rad": phi.tolist(),
        "stokes_sum": (source.stokes * n).tolist(),
        "trace_sum": (source.correlation.trace * n).tolist(),
        "kick_jumps_sum": int(sum(e.kick_jumps for e in source.events)),
        "tds_jumps_sum": int(sum(e.lattice_jumps for e in source.events)),
    }


def merge(parts: list[dict]) -> dict:
    """Count-weighted mean Stokes map; SEM from the chunk spread (>= 2 chunks)."""
    first = parts[0]
    if any(p["settings"] != first["settings"] for p in parts[1:]):
        raise ValueError("chunks were run with different settings")
    if len(parts) < 2:
        raise ValueError("at least two chunks are needed for an error estimate")
    starts = sorted((p["start"], p["count"]) for p in parts)
    for (a, n), (b, _) in zip(starts, starts[1:], strict=False):
        if a + n != b:
            raise ValueError("chunks overlap or leave a gap")
    counts = np.array([p["count"] for p in parts], dtype=float)
    n = counts.sum()
    maps = np.array([p["stokes_sum"] for p in parts]) / counts[:, None, None, None, None]
    mean = np.tensordot(counts, maps, axes=1) / n
    w = counts[:, None, None, None, None]
    sem = np.sqrt(np.sum(w * (maps - mean) ** 2, axis=0) / (len(parts) - 1) / n)
    energies = np.asarray(first["photon_energy_eV"])
    theta = np.asarray(first["theta_rad"])
    phi = np.asarray(first["phi_rad"])
    return {
        "status": first["status"],
        "settings": first["settings"],
        "info": first["info"],
        "incidents": int(n),
        "cpu_seconds": float(sum(p["seconds"] for p in parts)),
        "photon_energy_eV": energies.tolist(),
        "theta_rad": theta.tolist(),
        "phi_rad": phi.tolist(),
        "stokes_per_eV_sr": mean.tolist(),
        "stokes_sem_per_eV_sr": sem.tolist(),
        "retained_weight": float(np.sum([p["trace_sum"][-1] for p in parts]) / n),
        "mean_kick_jumps": sum(p["kick_jumps_sum"] for p in parts) / n,
        "mean_tds_jumps": sum(p["tds_jumps_sum"] for p in parts) / n,
        "summary": summary(energies, theta, phi, mean, sem),
    }


def summary(energies, theta, phi, mean, sem) -> list[dict]:
    """Per polar angle: azimuth-averaged line peak, azimuthal contrasts, polarization."""
    rows = []
    for t, angle in enumerate(theta):
        i_map = mean[t, :, 0]  # (phi, energy)
        average = i_map.mean(axis=0)
        k = int(np.argmax(average))
        integrated = np.trapezoid(mean[t], energies, axis=-1)  # (phi, 4)
        integrated_sem = np.sqrt(np.trapezoid(sem[t] ** 2, energies, axis=-1))
        total = integrated[:, 0]
        rows.append(
            {
                "theta_rad": float(angle),
                "peak_eV": float(energies[k]),
                "peak_per_eV_sr": float(average[k]),
                "integrated_per_sr_by_phi": total.tolist(),
                "integrated_sem_per_sr_by_phi": integrated_sem[:, 0].tolist(),
                "cos4phi_contrast": float(2 * np.mean(total * np.cos(4 * phi)) / np.mean(total)),
                "cos2phi_contrast": float(2 * np.mean(total * np.cos(2 * phi)) / np.mean(total)),
                "linear_degree": float(
                    np.mean(np.hypot(integrated[:, 1], integrated[:, 2])) / np.mean(total)
                ),
                "circular_degree": float(np.mean(integrated[:, 3]) / np.mean(total)),
            }
        )
    return rows


def main(argv: list[str]) -> None:
    if argv[0] == "chunk":
        options, start, count, out = json.loads(argv[1]), int(argv[2]), int(argv[3]), argv[4]
        Path(out).write_text(json.dumps(chunk(options, start, count)))
    elif argv[0] == "merge":
        merged = merge([json.loads(Path(p).read_text()) for p in argv[2:]])
        Path(argv[1]).write_text(json.dumps(merged, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
