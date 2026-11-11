"""Research study: planar fibre handoff angles, return deficit and occupation lengths (#1056).

Heavy arms run on the lab box (remote-gpu-jobs skill); local pilots::

    uv run python checks/planar_fiber_study.py run si110_5 absorbing 0 50 --out a.npz
    uv run python checks/planar_fiber_study.py analyze si110_5 a*.npz --reference r*.npz

Model: :mod:`pyrite.montecarlo.channeling.planar_fiber` (claim
``cr-planar-fiber-unraveling``), Si at 293.15 K, independent-string TDS plus
electronic kicks, fixed energy. Arms:

* ``absorbing``: retained cut ``barrier + margin`` (default the declared
  300 eV of ``DensityMatrixSolver``);
* ``reference``: ``P = R``, the whole order window retained; only outputs
  outside the window terminate (string outputs are recorded as tails).

Both arms use the same seed and incident IDs, so entrance draws coincide and
per-incident differences are paired (``cr-axial-fiber-return-deficit``).

Entrance: per incident, beam angles ``(psi_x, psi_y) ~ N(0, sigma^2)`` from
an incident-keyed stream (``sigma`` the Table II divergence of
``checks/kephart_occupation.py``). ``k_x = k psi_x`` folds into carrier and
entrance order; ``kappa_y = k psi_y`` starts the out-of-plane label, which
the fibre then carries exactly through every jump.

Per incident and checkpoint ``z`` (alive only): band populations
``|c_n|^2`` of the ``n`` lowest ``H(kappa)`` states (``n < N_BANDS``, the
solver's ``band_populations``), the below-barrier population, and the
out-of-plane angle ``theta_y = kappa_y / k``. Terminal records: kind, depth,
exit angles. Outgoing outputs are measured with a uniform position window,
which makes the Weyl momentum outcome the exact box order, so
``psi_x = (kappa_x + n g) / k``; tails carry the exact ``(k_x, q_y)``.

Forward-detector weight per trajectory: ``(1 + gamma^2 theta_y^2)^-2``, the
boosted-dipole photon density at angle ``theta_y`` relative to forward
(``detector_acceptance`` of the Kephart harness averages the same factor over
a modelled ``theta_y`` distribution).

Validation: cr-planar-fiber-unraveling
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_KEPHART = runpy.run_path(str(Path(__file__).with_name("kephart_occupation.py")))
KEPHART_CASES = _KEPHART["CASES"]
TEMPERATURE_K = _KEPHART["TEMPERATURE_K"]
fit_occupation_length = _KEPHART["fit_occupation_length"]
line_weights = _KEPHART["line_weights"]

N_BANDS = 6
KINDS = ("retained", "outgoing", "unresolved", "tail")
_ENTRY_DOMAIN = 0x1056


@dataclasses.dataclass(frozen=True)
class StudyCase:
    """One planar configuration of the study."""

    key: str
    hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    divergence_rad: float
    thickness_um: float
    order_max: int


# Window orders reach >= 2x the 300 eV cut in kinetic energy (17 MeV: M = 32
# reaches ~1.3 keV, 54.5 MeV: M = 48 reaches ~0.9 keV for Si(110)).
CASES = {
    "si110_5": StudyCase("si110_5", (1, 1, 0), 5.0e6, 0.0, 20.0, 16),
    "si110_17": StudyCase("si110_17", (1, 1, 0), 16.92e6, 2.2e-3, 120.0, 32),
    "si100_17": StudyCase("si100_17", (1, 0, 0), 16.92e6, 2.2e-3, 120.0, 24),
    "si100_54": StudyCase("si100_54", (1, 0, 0), 54.5e6, 1.0e-3, 500.0, 36),
    "si110_54": StudyCase("si110_54", (1, 1, 0), 54.5e6, 1.0e-3, 500.0, 48),
}
#: Kephart Table I entries fitted from each configuration.
FITS = {
    "si110_17": ("si110_17_n1", "si110_17_n2", "si110_17_n3"),
    "si100_17": ("si100_17",),
    "si100_54": ("si100_54",),
    "si110_54": ("si110_54",),
}


def checkpoints(case: StudyCase, count: int = 241) -> np.ndarray:
    """Uniform checkpoint grid [Å] over the slab (0.5 µm steps at 120 µm)."""
    return np.linspace(0.0, case.thickness_um * 1e4, count)


def _inputs(case: StudyCase, arm: str, margin_eV: float, order_max: int | None):
    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling.planar_fiber import planar_fiber_inputs

    profile = planar_profile("silicon", case.hkl, TEMPERATURE_K)
    inputs = planar_fiber_inputs(
        profile, case.kinetic_energy_eV, order_max=order_max or case.order_max, margin_eV=margin_eV
    )
    channels = inputs.channels
    if arm == "reference":
        channels = dataclasses.replace(channels, cut=lambda energies: np.inf)
    elif arm != "absorbing":
        raise ValueError(f"unknown arm {arm!r}")
    # Paraxial cap: every string output outside the window is a tail record.
    channels = dataclasses.replace(channels, tail_q_max_inv_ang=0.5 * inputs.k_inv_ang)
    return profile, inputs, channels


def entrance_angles(seed: int, incident: int, sigma: float) -> tuple[float, float]:
    """Beam angles ``(psi_x, psi_y)`` of one incident [rad], order independent."""
    rng = np.random.Generator(
        np.random.Philox(np.random.SeedSequence((seed, _ENTRY_DOMAIN, incident)))
    )
    return tuple(float(x) for x in sigma * rng.standard_normal(2))


def run_arm(key, arm, start, count, *, seed=0, margin_eV=300.0, order_max=None, n_checkpoints=241):
    """Run ``count`` incidents from ``start``; per-incident arrays (see module doc)."""
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import _Engine, _trajectory

    case = CASES[key]
    profile, inputs, channels = _inputs(case, arm, margin_eV, order_max)
    engine = _Engine(channels, inputs.hbar_v_eV_ang)
    m = int(engine.m[0])
    box = int(engine.box[0])
    g, k = profile.g_inv_ang, inputs.k_inv_ang
    window = np.ones((2 * box + 1, 1), dtype=complex) / np.sqrt(2 * box + 1)
    cps = checkpoints(case, n_checkpoints)
    index = {float(z): j for j, z in enumerate(cps)}
    shape = (count, cps.size)
    pops = np.zeros((*shape, N_BANDS))
    bound = np.zeros(shape)
    theta_y = np.zeros(shape)
    alive = np.zeros(shape, dtype=bool)
    out = {
        name: np.zeros(count)
        for name in ("weight", "depth", "psi_x", "psi_y", "entry_x", "entry_y")
    }
    kind = np.zeros(count, dtype=np.int8)
    jumps = np.zeros((count, 2), dtype=np.int32)
    orders = inputs.orders
    clock = time.perf_counter()
    for row, incident in enumerate(range(start, start + count)):
        psi_x, psi_y = entrance_angles(seed, incident, case.divergence_rad)
        k_x = k * psi_x
        order = int(np.floor(k_x / g + 0.5))
        if abs(order) > m:
            raise ValueError("entrance order outside the window; raise order_max")
        entry = (orders == order).astype(complex)

        def observe(_incident, depth, kappa, energies, c, row=row):
            j = index[depth]
            alive[row, j] = True
            p = np.abs(c) ** 2
            n = min(N_BANDS, p.size)
            pops[row, j, :n] = p[:n]
            bound[row, j] = float(p[np.asarray(energies) < inputs.barrier_eV].sum())
            theta_y[row, j] = kappa[1] / k

        event = _trajectory(
            engine,
            entry,
            (k_x - order * g, k * psi_y),
            window,
            float(cps[-1]),
            seed,
            incident,
            200.0,
            1e-6,
            observe,
            tuple(float(z) for z in cps),
        )
        kind[row] = KINDS.index(event.kind)
        jumps[row] = (event.lattice_jumps, event.kick_jumps)
        out["weight"][row], out["depth"][row] = event.weight, event.depth_ang
        out["entry_x"][row], out["entry_y"][row] = psi_x, psi_y
        if event.kind == "tail":
            out["psi_x"][row] = event.tail_momentum_inv_ang[0] / k
            out["psi_y"][row] = event.tail_momentum_inv_ang[1] / k
        elif event.kind == "outgoing":
            b = event.momentum_index[0]
            n_out = b if b <= box else b - (2 * box + 1)
            out["psi_x"][row] = (event.carrier_inv_ang[0] + n_out * g) / k
            out["psi_y"][row] = event.carrier_inv_ang[1] / k
        else:
            out["psi_x"][row] = out["psi_y"][row] = np.nan
    u_min, u_max = (float(x) for x in profile.extrema())
    info = {
        "case": dataclasses.asdict(case),
        "arm": arm,
        "seed": seed,
        "start": start,
        "count": count,
        "margin_eV": margin_eV,
        "order_max": m,
        "outgoing_order_max": box,
        "cutoff_eV": None if arm == "reference" else inputs.cutoff_eV,
        "barrier_eV": u_max,
        "well_bottom_eV": u_min,
        "k_inv_ang": k,
        "gamma": inputs.gamma,
        "g_inv_ang": g,
        "seconds": time.perf_counter() - clock,
        "source_sha": _revision(),
    }
    return {
        "kind": kind,
        "jumps": jumps,
        "alive": alive,
        "pops": pops,
        "bound": bound,
        "theta_y": theta_y,
        "checkpoints": cps,
        "info": json.dumps(info),
        **out,
    }


def _revision() -> str:
    """Source revision: ``PYRITE_SOURCE_REVISION`` (isolated remote copies), else git."""
    if os.environ.get("PYRITE_SOURCE_REVISION"):
        return os.environ["PYRITE_SOURCE_REVISION"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load(paths) -> dict:
    """Concatenate chunks of one arm (same case, arm, seed and grid), sorted by incident."""
    chunks = [dict(np.load(p, allow_pickle=False)) for p in paths]
    infos = [json.loads(str(c["info"])) for c in chunks]
    keys = {(i["case"]["key"], i["arm"], i["seed"], i["margin_eV"], i["order_max"]) for i in infos}
    if len(keys) != 1:
        raise ValueError(f"chunks mix configurations: {sorted(keys)}")
    order = np.argsort([i["start"] for i in infos])
    merged = {
        name: np.concatenate([chunks[j][name] for j in order])
        for name in chunks[0]
        if name not in ("info", "checkpoints")
    }
    merged["ids"] = np.concatenate(
        [np.arange(infos[j]["start"], infos[j]["start"] + infos[j]["count"]) for j in order]
    )
    merged["checkpoints"] = chunks[0]["checkpoints"]
    merged["info"] = infos[order[0]]
    merged["seconds"] = float(sum(i["seconds"] for i in infos))
    return merged


def _mean_sem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def signals(arm: dict, fit_key: str) -> dict[str, np.ndarray]:
    """Per-incident radiation-weighted band signal (raw and forward-detector weighted)."""
    info = arm["info"]
    case = KEPHART_CASES[fit_key]
    weights = line_weights(case, 2 * info["order_max"] + 1)
    band = sum(w * arm["pops"][:, :, b] for b, w in weights.items())
    w = arm["weight"][:, None]
    detector = (1.0 + (info["gamma"] * arm["theta_y"]) ** 2) ** -2
    norm = float(np.mean(detector[:, 0]))
    return {"mean": w * band, "detector": w * band * detector / norm}


def occupation(arm: dict) -> dict:
    """Fitted lengths [µm] and Kephart ratios per Table I entry of this configuration."""
    key = arm["info"]["case"]["key"]
    z_um = arm["checkpoints"] / 1e4
    out = {}
    for fit_key in FITS.get(key, ()):
        case = KEPHART_CASES[fit_key]
        row = {}
        for name, per_incident in signals(arm, fit_key).items():
            length, _, residual = fit_occupation_length(
                z_um, per_incident.mean(axis=0), case.thickness_range_um
            )
            # Batch-means SEM of the length: refit 10 disjoint incident batches.
            batches = np.array_split(per_incident, 10)
            spread = [
                fit_occupation_length(z_um, b.mean(axis=0), case.thickness_range_um)[0]
                for b in batches
            ]
            row[name] = {
                "length_um": length,
                "length_sem_um": float(np.std(spread, ddof=1) / np.sqrt(len(spread))),
                "ratio": length / case.occupation_length_um,
                "fit_rms_relative": residual,
            }
        out[fit_key] = row
    return out


def handoff(arm: dict) -> dict:
    """Terminal angle statistics in units of the planar critical angle."""
    info = arm["info"]
    case = info["case"]
    from pyrite.materials.crystal import HBARC_EV_ANG

    pv = info["k_inv_ang"] * HBARC_EV_ANG * (1.0 - 1.0 / info["gamma"] ** 2) ** 0.5
    depth = info["barrier_eV"] - info["well_bottom_eV"]
    psi_c = float(np.sqrt(2.0 * depth / pv))
    w = arm["weight"]
    result = {"psi_c_rad": psi_c, "incidents": int(w.size)}
    for name in KINDS:
        mask = arm["kind"] == KINDS.index(name)
        result[f"{name}_fraction"] = float(np.sum(w[mask]) / np.sum(w))
        if name in ("outgoing", "tail") and mask.any():
            ratio = np.abs(arm["psi_x"][mask]) / psi_c
            result[f"{name}_abs_psi_over_psi_c"] = {
                "mean": float(ratio.mean()),
                "sem": float(ratio.std(ddof=1) / np.sqrt(ratio.size)) if ratio.size > 1 else None,
                "quantiles_10_50_90": [float(q) for q in np.quantile(ratio, (0.1, 0.5, 0.9))],
            }
    if info["cutoff_eV"] is not None:
        cut_kinetic = info["cutoff_eV"] - info["well_bottom_eV"]
        result["cut_angle_over_psi_c"] = float(np.sqrt(2.0 * cut_kinetic / pv) / psi_c)
    g = info["g_inv_ang"]
    result["band_edge_over_psi_c"] = float(
        (info["order_max"] + 0.5) * g / info["k_inv_ang"] / psi_c
    )
    result["depth_um_terminal_mean"] = float(np.mean(arm["depth"][arm["kind"] != 0]) / 1e4)
    _ = case
    return result


def return_deficit(absorbing: dict, reference: dict, depths_um) -> dict:
    """Paired ``N_B(P=R) - N_B(absorbing)`` and band-signal deficits at ``depths_um``."""
    common = np.intersect1d(absorbing["ids"], reference["ids"])
    a = np.searchsorted(absorbing["ids"], common)
    r = np.searchsorted(reference["ids"], common)
    z_um = absorbing["checkpoints"] / 1e4
    picks = [int(np.argmin(np.abs(z_um - d))) for d in depths_um]
    out = {"incidents": int(common.size), "depth_um": [float(z_um[j]) for j in picks]}
    bound_a = absorbing["weight"][a, None] * absorbing["bound"][a]
    bound_r = reference["weight"][r, None] * reference["bound"][r]
    for name, (x, y) in {"bound": (bound_a, bound_r)}.items():
        mean, sem = _mean_sem(y - x)
        base, _ = _mean_sem(x)
        out[name] = {
            "absorbing": [float(base[j]) for j in picks],
            "deficit": [float(mean[j]) for j in picks],
            "deficit_sem": [float(sem[j]) for j in picks],
        }
    return out


def analyze(absorbing_paths, reference_paths=(), depths_um=(1.0, 5.0, 20.0, 100.0)) -> dict:
    absorbing = load(absorbing_paths)
    report = {
        "info": absorbing["info"],
        "cpu_seconds": absorbing["seconds"],
        "handoff": handoff(absorbing),
        "occupation": occupation(absorbing),
    }
    if reference_paths:
        reference = load(reference_paths)
        report["reference_handoff"] = handoff(reference)
        report["reference_occupation"] = occupation(reference)
        top = absorbing["checkpoints"][-1] / 1e4
        report["return_deficit"] = return_deficit(
            absorbing, reference, [d for d in depths_um if d <= top] + [top]
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("case", choices=sorted(CASES))
    run.add_argument("arm", choices=("absorbing", "reference"))
    run.add_argument("start", type=int)
    run.add_argument("count", type=int)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--margin", type=float, default=300.0)
    run.add_argument("--order-max", type=int, default=None)
    run.add_argument("--checkpoints", type=int, default=241)
    run.add_argument("--out", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("absorbing", nargs="+")
    an.add_argument("--reference", nargs="*", default=())
    an.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_arm(
            args.case,
            args.arm,
            args.start,
            args.count,
            seed=args.seed,
            margin_eV=args.margin,
            order_max=args.order_max,
            n_checkpoints=args.checkpoints,
        )
        np.savez_compressed(args.out, **result)
        info = json.loads(result["info"])
        print(json.dumps({"out": args.out, "seconds": info["seconds"]}))
        return 0
    report = analyze(args.absorbing, args.reference)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
