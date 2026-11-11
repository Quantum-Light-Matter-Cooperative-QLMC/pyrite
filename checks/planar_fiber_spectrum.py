"""Research study: planar two-time source vs the secular line model (#1056).

Heavy arms run on the lab box (remote-gpu-jobs skill); local pilots::

    uv run python checks/planar_fiber_spectrum.py run demo_17 0 20 --out a.npz
    uv run python checks/planar_fiber_spectrum.py analyze demo_17 a*.npz --out report.json

Model: :func:`pyrite.montecarlo.channeling.planar_fiber.planar_on_axis_source`
(claim ``cr-planar-fiber-regression``): the on-axis two-time source of the
planar fibre model, per primary, for slabs ending at every listed thickness
(:func:`~pyrite.montecarlo.channeling.axial_fiber_regression.fiber_thickness_spectra`;
thicknesses are paired exactly within each primary). Reference:
:func:`pyrite.montecarlo.channeling.emission.channeling_spectrum` on
:class:`~pyrite.montecarlo.channeling.density.DensityMatrixSolver` with the
same profile, energy, kernels (TDS + electronic) and 300 eV margin, at
``theta = 0``, zero aperture and divergence: the production secular line
model (``cr-line-shape``, ``cr-yield-assembly``, ``cr-capture-yield``).

Both sides: Si at 293.15 K, fixed energy, plane-wave entry at zero
incidence, photon absorption off (an emission-model comparison, not a
detector prediction). Arms: ``control`` (depth step ``dz``) and ``fine``
(``dz / 2``), paired by seed and incident ID; recording depths never change
the draws, so the fine arm isolates the depth-quadrature error of the
Doppler phase.

Per line window (bounded by the midpoints between Doppler-shifted Bohr
energies at ``kappa = 0``), the analysis reports the integrated yield
[photons/(sr electron)], the centroid and the RMS width of both spectra,
the production capture share, and fibre statistics from the per-primary
spread (batch-means over 20 batches for the ratios).

Validation: cr-planar-fiber-regression
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TEMPERATURE_K = 293.15
MARGIN_EV = 300.0


@dataclasses.dataclass(frozen=True)
class SpectrumCase:
    """One planar configuration of the comparison."""

    key: str
    hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    thicknesses_um: tuple[float, ...]
    order_max: int
    dz_ang: float
    photon_eV: tuple[float, float, int]
    lines: tuple[tuple[int, int], ...]


CASES = {
    # channeling_demo: 17 MeV, 2 µm, (1-10) = (110) planes.
    "demo_17": SpectrumCase(
        "demo_17",
        (1, 1, 0),
        17.0e6,
        (0.5, 1.0, 2.0),
        32,
        50.0,
        (500.0, 26000.0, 1021),
        ((1, 0), (2, 1), (3, 2), (4, 3)),
    ),
    "si100_17": SpectrumCase(
        "si100_17",
        (1, 0, 0),
        16.92e6,
        (0.5, 1.0, 2.0),
        24,
        50.0,
        (500.0, 26000.0, 1021),
        ((1, 0), (2, 1), (3, 2)),
    ),
    "si110_5": SpectrumCase(
        "si110_5",
        (1, 1, 0),
        5.0e6,
        (0.5, 1.0, 2.0),
        16,
        25.0,
        (100.0, 4000.0, 781),
        ((1, 0), (2, 1)),
    ),
}
ARMS = {"control": 1.0, "fine": 0.5}


def photon_grid(case: SpectrumCase) -> np.ndarray:
    lo, hi, n = case.photon_eV
    return np.linspace(lo, hi, int(n))


def depth_grid(case: SpectrumCase, arm: str) -> tuple[np.ndarray, list[int]]:
    """Depths [Å] and the indices of the listed thicknesses."""
    dz = case.dz_ang * ARMS[arm]
    top = max(case.thicknesses_um) * 1e4
    z = np.arange(0.0, top + 0.5 * dz, dz)
    stops = [int(round(t * 1e4 / dz)) for t in case.thicknesses_um]
    if any(abs(z[k] - t * 1e4) > 1e-6 for k, t in zip(stops, case.thicknesses_um, strict=True)):
        raise ValueError("thicknesses must be multiples of the depth step")
    return z, stops


def _inputs(case: SpectrumCase):
    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling.planar_fiber import planar_fiber_inputs

    profile = planar_profile("silicon", case.hkl, TEMPERATURE_K)
    return profile, planar_fiber_inputs(
        profile, case.kinetic_energy_eV, order_max=case.order_max, margin_eV=MARGIN_EV
    )


def _revision() -> str:
    if os.environ.get("PYRITE_SOURCE_REVISION"):
        return os.environ["PYRITE_SOURCE_REVISION"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_chunk(key: str, arm: str, start: int, count: int, seed: int = 0) -> dict:
    """Per-primary fibre spectra ``(primary, thickness, energy)`` for one chunk."""
    from pyrite.montecarlo.channeling.axial_fiber_regression import fiber_thickness_spectra

    case = CASES[key]
    _, inputs = _inputs(case)
    z, stops = depth_grid(case, arm)
    energies = photon_grid(case)
    clock = time.perf_counter()
    spectra, events = fiber_thickness_spectra(
        inputs.channels,
        inputs.entry_orders,
        kinetic_energy_eV=inputs.kinetic_energy_eV,
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
        depths_ang=z,
        thickness_indices=stops,
        photon_energy_eV=energies,
        incident_ids=range(start, start + count),
        seed=seed,
        carrier_inv_ang=inputs.carrier_inv_ang,
        boundary_eV=inputs.barrier_eV,
        max_working_bytes=2 * 1024**3,
    )
    info = {
        "case": dataclasses.asdict(case),
        "arm": arm,
        "seed": seed,
        "start": start,
        "count": count,
        "dz_ang": float(z[1] - z[0]),
        "barrier_eV": inputs.barrier_eV,
        "cutoff_eV": inputs.cutoff_eV,
        "seconds": time.perf_counter() - clock,
        "source_sha": _revision(),
    }
    return {
        "spectra": spectra.astype(np.float64),
        "depth": np.array([e.depth_ang for e in events]),
        "kind": np.array([e.kind for e in events]),
        "photon_eV": energies,
        "info": json.dumps(info),
    }


def load(paths) -> dict:
    """Concatenate chunks of one arm, sorted by first incident."""
    chunks = [dict(np.load(p, allow_pickle=False)) for p in paths]
    infos = [json.loads(str(c["info"])) for c in chunks]
    keys = {(i["case"]["key"], i["arm"], i["seed"]) for i in infos}
    if len(keys) != 1:
        raise ValueError(f"chunks mix configurations: {sorted(keys)}")
    order = np.argsort([i["start"] for i in infos])
    starts = [infos[j]["start"] for j in order]
    ids = np.concatenate(
        [np.arange(infos[j]["start"], infos[j]["start"] + infos[j]["count"]) for j in order]
    )
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicate incidents across chunks")
    return {
        "spectra": np.concatenate([chunks[j]["spectra"] for j in order]),
        "kind": np.concatenate([chunks[j]["kind"] for j in order]),
        "ids": ids,
        "photon_eV": chunks[0]["photon_eV"],
        "info": infos[order[0]],
        "starts": starts,
        "seconds": float(sum(i["seconds"] for i in infos)),
        "source_shas": sorted({i["source_sha"] for i in infos}),
    }


def line_windows(case: SpectrumCase, profile, gamma: float, beta: float) -> list[dict]:
    """Doppler-shifted Bohr centres at ``kappa = 0`` and midpoint window edges [eV]."""
    from pyrite.montecarlo.channeling.bloch import transverse_states

    states = transverse_states(profile, gamma, 0.0, n_plane_waves=case.order_max)
    doppler = 1.0 / (1.0 - beta)
    centres = sorted(
        ((states.energies_eV[u] - states.energies_eV[lower]) * doppler, (u, lower))
        for u, lower in case.lines
    )
    lo, hi, _ = case.photon_eV
    edges = [lo] + [0.5 * (a[0] + b[0]) for a, b in zip(centres, centres[1:], strict=False)] + [hi]
    return [
        {"line": list(line), "centre_eV": c, "window_eV": [edges[i], edges[i + 1]]}
        for i, (c, line) in enumerate(centres)
    ]


def production(case: SpectrumCase, energies: np.ndarray, cells: int = 16) -> dict:
    """Production secular spectra at every thickness (absorption off)."""
    from pyrite.montecarlo.channeling.density import DensityMatrixSolver
    from pyrite.montecarlo.channeling.emission import channeling_spectrum
    from pyrite.montecarlo.channeling.operators import electronic_kernel, tds_kernel

    profile, _ = _inputs(case)
    e0 = case.kinetic_energy_eV
    kernel = tds_kernel(profile, e0, n_cells=cells, max_order=case.order_max) + electronic_kernel(
        profile, e0, n_cells=cells, max_order=case.order_max
    )
    solver = DensityMatrixSolver(profile, e0, kernel, margin_eV=MARGIN_EV)
    out = {"spectrum": [], "capture": []}
    for t in case.thicknesses_um:
        spec = channeling_spectrum(
            solver,
            t * 1e4,
            energies,
            0.0,
            absorption=False,
            dz_ang=case.dz_ang / 2,
            sample_ang=case.dz_ang,
        )
        out["spectrum"].append(np.asarray(spec.spectrum, dtype=float).ravel())
        out["capture"].append(np.asarray(spec.capture_spectrum, dtype=float).ravel())
    return {k: np.asarray(v) for k, v in out.items()}


def _moments(energies, spectrum, mask):
    de = energies[1] - energies[0]
    w = spectrum[..., mask]
    total = w.sum(axis=-1) * de
    centroid = (w * energies[mask]).sum(axis=-1) * de / total
    spread = (w * (energies[mask] - centroid[..., None]) ** 2).sum(axis=-1) * de / total
    return total, centroid, np.sqrt(np.maximum(spread, 0.0))


def _batch(values: np.ndarray, batches: int = 20):
    """Batch means of per-primary rows; returns (mean, batch SEM) of a derived function."""
    groups = np.array_split(values, batches)
    return np.asarray([g.mean(axis=0) for g in groups])


def _sem(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def compare(fibre: dict, reference: dict, case: SpectrumCase) -> dict:
    """Per thickness and line window: fibre vs production yield, centroid and width."""
    from pyrite.montecarlo.channeling.bloch import longitudinal_kinematics

    profile, _ = _inputs(case)
    gamma, beta = longitudinal_kinematics(case.kinetic_energy_eV)
    energies = fibre["photon_eV"]
    de = energies[1] - energies[0]
    windows = line_windows(case, profile, gamma, beta)
    spectra = fibre["spectra"]
    batches = _batch(spectra)
    out = []
    for t_index, t in enumerate(case.thicknesses_um):
        mean = spectra[:, t_index].mean(axis=0)
        sem = spectra[:, t_index].std(axis=0, ddof=1) / np.sqrt(spectra.shape[0])
        prod = reference["spectrum"][t_index]
        capture = reference["capture"][t_index]
        row = {
            "thickness_um": t,
            "total": {
                "fibre": float(mean.sum() * de),
                "fibre_sem": float(
                    spectra[:, t_index].sum(axis=-1).std(ddof=1) * de / np.sqrt(spectra.shape[0])
                ),
                "production": float(prod.sum() * de),
                "production_capture": float(capture.sum() * de),
            },
            "lines": [],
        }
        for window in windows:
            lo, hi = window["window_eV"]
            mask = (energies >= lo) & (energies < hi)
            f_total, f_centroid, f_width = _moments(energies, mean, mask)
            p_total, p_centroid, p_width = _moments(energies, prod, mask)
            b_total, b_centroid, b_width = _moments(energies, batches[:, t_index], mask)
            row["lines"].append(
                {
                    **window,
                    "fibre_yield": float(f_total),
                    "fibre_yield_sem": float(np.sqrt((sem[mask] ** 2).sum()) * de),
                    "production_yield": float(p_total),
                    "production_capture_yield": float(capture[mask].sum() * de),
                    "yield_ratio": float(f_total / p_total),
                    "yield_ratio_sem": _sem(b_total / p_total),
                    "fibre_centroid_eV": float(f_centroid),
                    "production_centroid_eV": float(p_centroid),
                    "centroid_shift_eV": float(f_centroid - p_centroid),
                    "centroid_shift_sem_eV": _sem(b_centroid),
                    "fibre_rms_width_eV": float(f_width),
                    "production_rms_width_eV": float(p_width),
                    "width_ratio": float(f_width / p_width),
                    "width_ratio_sem": _sem(b_width / p_width),
                }
            )
        out.append(row)
    return {"windows": windows, "thicknesses": out}


def quadrature(control: dict, fine: dict) -> dict:
    """Paired control-minus-fine total per thickness (depth-step error of the Doppler phase)."""
    common = np.intersect1d(control["ids"], fine["ids"])
    a = np.searchsorted(control["ids"], common)
    b = np.searchsorted(fine["ids"], common)
    de = control["photon_eV"][1] - control["photon_eV"][0]
    diff = (control["spectra"][a] - fine["spectra"][b]).sum(axis=-1) * de
    base = fine["spectra"][b].sum(axis=-1).mean(axis=0) * de
    return {
        "incidents": int(common.size),
        "relative_difference": (diff.mean(axis=0) / base).tolist(),
        "relative_sem": (diff.std(axis=0, ddof=1) / np.sqrt(common.size) / base).tolist(),
        "max_abs_bin_difference_over_peak": (
            np.abs(control["spectra"][a].mean(axis=0) - fine["spectra"][b].mean(axis=0)).max(
                axis=-1
            )
            / fine["spectra"][b].mean(axis=0).max(axis=-1)
        ).tolist(),
    }


def analyze(key: str, control_paths, fine_paths=()) -> dict:
    case = CASES[key]
    control = load(control_paths)
    reference = production(case, control["photon_eV"])
    kinds, counts = np.unique(control["kind"], return_counts=True)
    report = {
        "case": dataclasses.asdict(case),
        "incidents": int(control["ids"].size),
        "source_shas": control["source_shas"],
        "cpu_seconds": control["seconds"],
        "fates": {str(k): int(c) for k, c in zip(kinds, counts, strict=True)},
        "comparison": compare(control, reference, case),
    }
    if fine_paths:
        report["quadrature"] = quadrature(control, load(fine_paths))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("case", choices=sorted(CASES))
    run.add_argument("start", type=int)
    run.add_argument("count", type=int)
    run.add_argument("--arm", choices=sorted(ARMS), default="control")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", type=Path, required=True)
    an = sub.add_parser("analyze")
    an.add_argument("case", choices=sorted(CASES))
    an.add_argument("control", nargs="+", type=Path)
    an.add_argument("--fine", nargs="*", type=Path, default=())
    an.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_chunk(args.case, args.arm, args.start, args.count, args.seed)
        np.savez_compressed(args.out, **result)
        print(json.dumps({"out": str(args.out), "seconds": json.loads(result["info"])["seconds"]}))
        return 0
    report = analyze(args.case, args.control, args.fine)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
