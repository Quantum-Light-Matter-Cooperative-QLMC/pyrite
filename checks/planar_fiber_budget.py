"""Research study: planar paired-arm source budgets (#1056), following #993.

Arms (all chunk-paired: same seed, incident IDs and entrance draws as the
``control``; ``checks/planar_fiber_study.py`` runs each one):

==================  ==========================================================
control             absorbing cut ``barrier + 300 eV``, window ``M``, box
                    ``B = M``, kick harmonics ``B + M``, search step 200 Å
return_flux         ``P = R`` (whole window retained; tails still terminate)
unresolved          box ``B = M + dM``
matching_surface    cut ``barrier + 600 eV``
grid_supercell      window ``M + dM``, search step 100 Å
source_model        kick harmonics ``1.5 (B + M)``
all_on              all of the above
==================  ==========================================================

``dM = max(4, M // 4)``. The planar fibre has no supercell or transverse
position grid; its discretisation is the order window and the event search
step, so ``grid_supercell`` probes those.

Observable (one bin per upper band ``u`` of the configuration's lines):
the forward-detector line source at crystal thickness ``c``,
``Y_u(c) = w_u int_0^c W |c_u(z)|^2 (1 + gamma^2 theta_y^2)^-2 / D_0 dz``
per incident, where ``w_u`` is the dipole weight of ``line_weights``
(rest-frame ``|x_ul|^2 dE^3``, summed over the lines leaving ``u``), ``W`` the
event weight and ``D_0`` the incident-mean detector factor at entrance.
It is proportional to the on-axis photon yield of those lines per incident in
the secular line model with constant line rates (fixed energy, no stopping);
the proportionality constant is common to every arm and cancels in ``ε``.
Integration is trapezoidal on the 241 recorded checkpoints, identical in
every arm.

Statistics, drift, interaction and fate caps are
:func:`pyrite.montecarlo.channeling.axial_depth_calibration.calibrate_axial_depth`
unchanged (claim ``cr-axial-depth-calibration``): batch-``t`` bounds over
100-incident chunks, Bonferroni over every bound, drift factor 2 over the
previous recorded thickness (``span`` = one checkpoint spacing), Hoeffding
fate fractions with outgoing and tail terminations counted as
``physical_outgoing`` and box overflow as ``unresolved``. Joint confidence
0.95. ``ε`` is the summed absolute bound over the summed reference lower
bound of all bins.

Usage::

    uv run python checks/planar_fiber_budget.py run demo_17 0 100 --out-dir scratch/budget
    uv run python checks/planar_fiber_budget.py analyze demo_17 scratch/budget --out report.json

Validation: cr-planar-fiber-unraveling
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import runpy
import sys
from pathlib import Path

import numpy as np

_STUDY = runpy.run_path(str(Path(__file__).with_name("planar_fiber_study.py")))
CASES = _STUDY["CASES"]
KEPHART_CASES = _STUDY["KEPHART_CASES"]
line_weights = _STUDY["line_weights"]
run_arm = _STUDY["run_arm"]
load = _STUDY["load"]
KINDS = _STUDY["KINDS"]

ARMS = (
    "control",
    "return_flux",
    "unresolved",
    "matching_surface",
    "grid_supercell",
    "source_model",
    "all_on",
)
#: Lines of each configuration's observable (upper, lower).
LINES = {
    "demo_17": ((1, 0), (2, 1), (3, 2)),
    "si110_17": ((1, 0), (2, 1), (3, 0)),
    "si100_17": ((1, 0),),
    "si100_54": ((1, 0), (2, 1)),
    "si110_54": ((1, 0), (2, 1), (3, 2)),
}
#: Budget checkpoints c [µm]: crystal thicknesses within each configuration's slab.
BUDGET_UM = {
    "demo_17": (2.0,),
    "si110_17": (20.0, 60.0, 120.0),
    "si100_17": (20.0, 60.0, 120.0),
    "si100_54": (50.0, 200.0, 500.0),
    "si110_54": (50.0, 200.0, 500.0),
}
CHUNK = 100


def probe_settings(key: str, probe: str) -> dict:
    """``run_arm`` keyword arguments of one budget arm."""
    m = CASES[key].order_max
    dm = max(4, m // 4)
    on = {name: probe in (name, "all_on") for name in ARMS[1:6]}
    order = m + dm if on["grid_supercell"] else m
    box = order + dm if on["unresolved"] else order
    return {
        "arm": "reference" if on["return_flux"] else "absorbing",
        "margin_eV": 600.0 if on["matching_surface"] else 300.0,
        "order_max": order,
        "outgoing_order_max": box,
        "kick_half": int(np.ceil(1.5 * (box + order))) if on["source_model"] else None,
        "search_step_ang": 100.0 if on["grid_supercell"] else 200.0,
        "probe": probe,
    }


def band_weights(key: str) -> dict[int, float]:
    """Dipole weight per upper band of the configuration's lines."""
    case = CASES[key]
    template = KEPHART_CASES["si100_17"]
    lines = LINES[key]
    spec = dataclasses.replace(
        template,
        key=key,
        hkl=case.hkl,
        kinetic_energy_eV=case.kinetic_energy_eV,
        lines=lines,
    )
    if len(lines) == 1:
        return {lines[0][0]: 1.0}
    return line_weights(spec, 2 * case.order_max + 1)


def line_source(arm: dict, weights: dict[int, float]) -> np.ndarray:
    """Per-incident cumulative line source ``Y[i, t, u]`` [Å] at every checkpoint."""
    info = arm["info"]
    detector = (1.0 + (info["gamma"] * arm["theta_y"]) ** 2) ** -2
    norm = float(np.mean((1.0 + (info["gamma"] * arm["entry_y"]) ** 2) ** -2))
    w = arm["weight"][:, None]
    density = np.stack(
        [weights[u] * w * arm["pops"][:, :, u] * detector / norm for u in sorted(weights)],
        axis=-1,
    )
    dz = np.diff(arm["checkpoints"])[None, :, None]
    steps = 0.5 * (density[:, 1:] + density[:, :-1]) * dz
    return np.concatenate([np.zeros_like(density[:, :1]), np.cumsum(steps, axis=1)], axis=1)


def _chunk_paths(directory: Path, key: str, probe: str) -> list[Path]:
    return sorted(directory.glob(f"{key}-{probe}-*.npz"))


def analyze(key: str, directory: Path) -> dict:
    """#993 calibration of one configuration from its seven arms' chunks."""
    from pyrite.montecarlo.channeling.axial_depth_calibration import (
        ArmBatches,
        calibrate_axial_depth,
    )
    from pyrite.montecarlo.channeling.axial_outgoing_launch import AxialHandoffRecord

    weights = band_weights(key)
    arms, loaded = {}, {}
    for probe in ARMS:
        paths = _chunk_paths(directory, key, probe)
        if not paths:
            raise FileNotFoundError(f"no chunks for {key} {probe} in {directory}")
        arm = load(paths)
        expected = probe_settings(key, probe)
        if arm["info"].get("probe") != probe or arm["info"]["margin_eV"] != expected["margin_eV"]:
            raise ValueError(f"{probe} chunks do not carry the probe's settings")
        y = line_source(arm, weights)
        starts = np.arange(arm["ids"][0], arm["ids"][-1] + 1, CHUNK)
        if not np.array_equal(arm["ids"], np.arange(starts[0], starts[-1] + CHUNK)):
            raise ValueError(f"{probe}: incident IDs are not whole contiguous chunks")
        means = y.reshape(starts.size, CHUNK, *y.shape[1:]).mean(axis=1)
        arms["return_flux" if probe == "return_flux" else probe] = ArmBatches(
            thickness_ang=tuple(float(z) for z in arm["checkpoints"]),
            starts=tuple(int(s) for s in starts),
            counts=np.full(starts.size, CHUNK),
            yields=means,
        )
        loaded[probe] = arm
    grid = loaded["control"]["checkpoints"]
    for probe, arm in loaded.items():
        if not np.array_equal(arm["checkpoints"], grid):
            raise ValueError(f"{probe} records a different checkpoint grid")
    span = float(grid[1] - grid[0])
    marks = [float(grid[int(np.argmin(np.abs(grid - c * 1e4)))]) for c in BUDGET_UM[key]]

    def fates(arm):
        names = {"outgoing": "physical_outgoing", "tail": "physical_outgoing"}
        return [
            AxialHandoffRecord(int(i), names.get(KINDS[k], KINDS[k]), float(d))
            for i, k, d in zip(arm["ids"], arm["kind"], arm["depth"], strict=True)
        ]

    case = CASES[key]
    settings = probe_settings(key, "control")
    case_key = (
        f"planar-fiber:silicon{case.hkl}:{case.kinetic_energy_eV:g}eV:"
        f"div{case.divergence_rad:g}:M{settings['order_max']}:B{settings['outgoing_order_max']}:"
        "cut300eV:293.15K"
    )
    observable_key = f"line-source:{LINES[key]}:forward-detector"
    evidence, report = calibrate_axial_depth(
        arms,
        fates(loaded["control"]),
        fates(loaded["all_on"]),
        checkpoints_ang=marks,
        span_ang=span,
        sampled_max_depth_ang=float(grid[-1]),
        case_key=case_key,
        observable_key=observable_key,
        model_id="cr-planar-fiber-unraveling",  # planar_fiber.py unchanged since 986fc3849
        evidence_id=f"planar-budget-1056:{key}",
    )
    return {
        "case": key,
        "case_key": case_key,
        "observable_key": observable_key,
        "band_weights": {str(u): w for u, w in weights.items()},
        "incidents": int(loaded["control"]["ids"].size),
        "chunks": len(arms["control"].starts),
        "source_shas": sorted({s for arm in loaded.values() for s in arm["source_shas"]}),
        "settings": {probe: probe_settings(key, probe) for probe in ARMS},
        "cpu_seconds": {probe: arm["seconds"] for probe, arm in loaded.items()},
        "budget_um": list(BUDGET_UM[key]),
        "epsilon": [i.relative_error_upper for i in evidence.intervals],
        **report,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run every arm for one chunk")
    run.add_argument("case", choices=sorted(LINES))
    run.add_argument("start", type=int)
    run.add_argument("count", type=int, default=CHUNK, nargs="?")
    run.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    run.add_argument("--out-dir", type=Path, required=True)
    an = sub.add_parser("analyze")
    an.add_argument("case", choices=sorted(LINES))
    an.add_argument("directory", type=Path)
    an.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.start % CHUNK or args.count != CHUNK:
            raise SystemExit(f"chunks are {CHUNK} incidents starting at multiples of {CHUNK}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for probe in args.arms:
            settings = probe_settings(args.case, probe)
            arm = settings.pop("arm")
            result = run_arm(args.case, arm, args.start, args.count, **settings)
            out = args.out_dir / f"{args.case}-{probe}-{args.start}.npz"
            np.savez_compressed(out, **result)
            print(json.dumps({"out": str(out), "seconds": json.loads(result["info"])["seconds"]}))
        return 0
    report = analyze(args.case, args.directory)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"case": args.case, "epsilon": report["epsilon"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
