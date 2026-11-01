"""Research study: paired convergence and return-flux deficit of the fibre unraveling (#882).

Heavy arms run on the lab box (remote-gpu-jobs skill); local pilots::

    uv run python checks/axial_outgoing_convergence.py run control 0 200 --out c.npz
    uv run python checks/axial_outgoing_convergence.py analyze c.npz other.npz ...
    uv run python checks/axial_outgoing_convergence.py record <source-sha> runs/out/*.npz

Si [001], 5 MeV, independent-string continuum TDS + electronic kicks,
absorbing no-return fibre model (``axial_fiber_unraveling``). Each arm changes
one numerical choice of the control (M = 6, provisional 95 eV target, box
B = 18, 200 Å search step). Every arm uses the same seed and incident IDs, so
per-incident differences are paired; the SEM of a difference is the sample
standard deviation of the per-incident differences over sqrt(N).

Observables per incident (weight ``w`` = entrance weight, ``a(z)`` = alive at
``z``, ``c(z)`` the normalized retained amplitude in the ``H(kappa)``
eigenbasis):

    retained R = w a,  bound N_B = w a sum_{E < U_max} |c|^2,
    exposure X_B(z) = int_0^z N_B dz' (trapezoid on the checkpoint grid),

plus terminal outgoing/unresolved tallies. Their ensemble means are the
corresponding traces of the model density (waiting-time unraveling,
Castin, Dalibard & Molmer, arXiv:0805.4002, Eqs. (4), (8)).

Return-flux deficit. The reference arm ``ref`` retains the whole order window
(``P = R``); outputs outside the window still terminate. Its bound population
includes every path that left the energy cut and came back, the coherent
no-jump coupling across the cut, and jump interference of P-Q coherences
recycled into P; the absorbing arm drops all three (the last two have no
fixed sign). The paired
deficit D_B = N_B(ref) - N_B(control) (and the exposure deficit) is the
weight the absorbing model loses to return at a fixed window, as in the
secular P = R study of docs/research/physics/axial-escape-instrument.md. It
is numerical accounting, not a physical escape fraction; the reference itself
is window dependent (arm ``ref-m8``).

Validation: cr-axial-fiber-return-deficit
"""

from __future__ import annotations

import dataclasses
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

_MAPS = runpy.run_path(str(Path(__file__).with_name("axial_outgoing_maps.py")))
material_fiber_channels = _MAPS["material_fiber_channels"]
balanced_window = _MAPS["balanced_window"]

#: Checkpoint grid [Å]; the exposure trapezoid uses it.
CHECKPOINTS_ANG = np.linspace(0.0, 3.0e4, 61)
REPORT_DEPTHS_ANG = (1.0e3, 1.0e4, 3.0e4)
_KINDS = ("retained", "outgoing", "unresolved")

CONTROL = {
    "max_order": 6,
    "target_eV": 95.0,
    "outgoing_order_max": 18,
    "search_step_ang": 200.0,
    "reference": False,
}
#: One change from the control per arm.
ARMS = {
    "control": {},
    "ref": {"reference": True},
    "ref-m8": {"reference": True, "max_order": 8},
    "target-50": {"target_eV": 50.0},
    "target-140": {"target_eV": 140.0},
    "target-180": {"target_eV": 180.0},
    "m5": {"max_order": 5},
    "m7": {"max_order": 7},
    "m8": {"max_order": 8},
    "b12": {"outgoing_order_max": 12},
    "b24": {"outgoing_order_max": 24},
    "step-100": {"search_step_ang": 100.0},
    "step-400": {"search_step_ang": 400.0},
}


def arm_settings(name: str) -> dict:
    """Control settings with the arm's single change applied."""
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {sorted(ARMS)}")
    return {**CONTROL, **ARMS[name]}


def bound_weight(energies: np.ndarray, c: np.ndarray, u_max: float) -> float:
    """``sum_{E < U_max} |c|^2`` of a normalized eigenbasis amplitude.

    Validation: cr-axial-fiber-return-deficit
    """
    return float(np.sum(np.abs(c[np.asarray(energies) < u_max]) ** 2))


def run_arm(
    name: str,
    start: int,
    count: int,
    *,
    seed: int = 0,
    electronic: bool = True,
    checkpoints_ang=CHECKPOINTS_ANG,
) -> dict:
    """Run ``count`` incidents from ``start`` of one arm; per-incident arrays.

    Returns ``kind`` (index into ``retained/outgoing/unresolved``), ``depth``,
    ``alive`` and ``bound`` (n, checkpoints), and arm metadata. The reference
    arm replaces the cut by ``+inf`` (``P = R``).
    Validation: cr-axial-fiber-return-deficit
    """
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import unravel_fiber_outgoing

    s = arm_settings(name)
    channels, entry, hbar_v, info = material_fiber_channels(
        s["max_order"],
        s["target_eV"],
        electronic=electronic,
        outgoing_order_max=s["outgoing_order_max"],
        tds="strings",
    )
    if s["reference"]:
        channels = dataclasses.replace(channels, cut=lambda energies: np.inf)
    u_min, u_max = axial_profile("silicon", (0, 0, 1)).extrema()
    cut_gamma = channels.cut(channels.spectrum((0.0, 0.0))[0])
    cps = np.asarray(checkpoints_ang, dtype=float)
    index = {float(z): k for k, z in enumerate(cps)}
    ids = range(start, start + count)
    row = {i: k for k, i in enumerate(ids)}
    alive = np.zeros((count, len(cps)), dtype=bool)
    bound = np.zeros((count, len(cps)))

    def observe(incident, depth, _kappa, energies, c):
        k = row[incident]
        alive[k, index[depth]] = True
        bound[k, index[depth]] = bound_weight(energies, c, u_max)

    clock = time.perf_counter()
    events = unravel_fiber_outgoing(
        channels,
        entry,
        hbar_v_eV_ang=hbar_v,
        window=balanced_window(2 * info["outgoing_order_max"] + 1),
        max_depth_ang=float(cps.max()),
        incident_ids=ids,
        seed=seed,
        search_step_ang=s["search_step_ang"],
        observe=observe,
        checkpoints_ang=cps,
    )
    info.update(
        arm=name,
        settings=s,
        seed=seed,
        start=start,
        count=count,
        seconds=time.perf_counter() - clock,
        u_min_max_eV=[float(u_min), float(u_max)],
        entrance_weight=events[0].weight,
        cut_at_gamma_eV=None if s["reference"] else float(cut_gamma),
    )
    return {
        "kind": np.array([_KINDS.index(e.kind) for e in events], dtype=np.int8),
        "depth": np.array([e.depth_ang for e in events]),
        "lattice_jumps": np.array([e.lattice_jumps for e in events]),
        "kick_jumps": np.array([e.kick_jumps for e in events]),
        "alive": alive,
        "bound": bound,
        "checkpoints": cps,
        "info": json.dumps(info),
    }


def _consistent(chunk: dict) -> None:
    """Observed survival must equal the terminal-event survival at every checkpoint."""
    cps, kind, depth = chunk["checkpoints"], chunk["kind"], chunk["depth"]
    alive = (kind[:, None] == 0) | (depth[:, None] > cps[None, :])
    if not np.array_equal(alive, chunk["alive"]):
        raise ValueError("observed survival disagrees with the terminal events")


def load_arm(paths) -> dict:
    """Concatenate chunk files of one arm (sorted by incident ID); refuse gaps/overlaps."""
    chunks = [dict(np.load(p, allow_pickle=False)) for p in paths]
    infos = [json.loads(str(c["info"])) for c in chunks]
    order = np.argsort([i["start"] for i in infos])
    chunks, infos = [chunks[k] for k in order], [infos[k] for k in order]
    expected = infos[0]["start"]
    for chunk, info in zip(chunks, infos, strict=True):
        if info["start"] != expected or info["arm"] != infos[0]["arm"]:
            raise ValueError("arm chunks must be one arm with contiguous incident ranges")
        if info["seed"] != infos[0]["seed"]:
            raise ValueError("arm chunks must share one seed")
        _consistent(chunk)
        expected += info["count"]
    merged = {
        k: np.concatenate([c[k] for c in chunks])
        for k in ("kind", "depth", "alive", "bound", "lattice_jumps", "kick_jumps")
    }
    merged["checkpoints"] = chunks[0]["checkpoints"]
    merged["weight"] = infos[0]["entrance_weight"]
    merged["ids"] = (infos[0]["start"], expected)
    merged["info"] = {**infos[0], "seconds": sum(i["seconds"] for i in infos)}
    return merged


def observables(arm: dict, depths_ang=REPORT_DEPTHS_ANG) -> dict[str, np.ndarray]:
    """Per-incident observables (n, depths): R, O, U, N_B, X_B [Å].

    Validation: cr-axial-fiber-return-deficit
    """
    cps, w = arm["checkpoints"], arm["weight"]
    cols = [int(np.flatnonzero(np.isclose(cps, z))[0]) for z in depths_ang]
    n_b = w * arm["bound"]
    increments = 0.5 * (n_b[:, 1:] + n_b[:, :-1]) * np.diff(cps)[None, :]
    exposure = np.concatenate([np.zeros((len(n_b), 1)), np.cumsum(increments, axis=1)], axis=1)
    z = np.asarray(depths_ang, dtype=float)
    terminal = arm["depth"][:, None] <= z[None, :]
    return {
        "retained": w * arm["alive"][:, cols],
        "outgoing": w * ((arm["kind"][:, None] == 1) & terminal),
        "unresolved": w * ((arm["kind"][:, None] == 2) & terminal),
        "bound": n_b[:, cols],
        "bound_exposure_ang": exposure[:, cols],
    }


def _mean_sem(x: np.ndarray) -> list[list[float]]:
    n = len(x)
    sem = x.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(x.shape[1], np.nan)
    return [[float(m), float(s)] for m, s in zip(x.mean(axis=0), sem, strict=True)]


def common(arm: dict, other: dict) -> tuple[dict, dict]:
    """Both arms restricted to their shared contiguous incident-ID range.

    Validation: cr-axial-fiber-return-deficit
    """
    lo, hi = max(arm["ids"][0], other["ids"][0]), min(arm["ids"][1], other["ids"][1])
    if lo >= hi:
        raise ValueError("arms share no incident IDs")

    def cut(x):
        rows = slice(lo - x["ids"][0], hi - x["ids"][0])
        keys = ("kind", "depth", "alive", "bound", "lattice_jumps", "kick_jumps")
        return {**x, **{k: x[k][rows] for k in keys if k in x}, "ids": (lo, hi)}

    return cut(arm), cut(other)


def paired(arm: dict, base: dict, depths_ang=REPORT_DEPTHS_ANG) -> dict:
    """Paired difference ``arm - base`` per observable and depth: [mean, SEM].

    Also the relative exposure deficit ``1 - X(arm)/X(base)`` with a
    delta-method SEM of the paired ratio of means.
    Validation: cr-axial-fiber-return-deficit
    """
    if arm["ids"] != base["ids"] or arm["info"]["seed"] != base["info"]["seed"]:
        raise ValueError("paired arms need the same seed and incident IDs")
    a, b = observables(arm, depths_ang), observables(base, depths_ang)
    out = {k: _mean_sem(a[k] - b[k]) for k in a}
    for key in ("bound", "bound_exposure_ang"):
        xa, xb = a[key], b[key]
        ma, mb = xa.mean(0), xb.mean(0)
        safe = np.where(mb > 0, mb, np.nan)
        ratio = ma / safe
        resid = (xa - ratio[None, :] * xb) / safe[None, :]
        sem = resid.std(axis=0, ddof=1) / np.sqrt(len(xa))
        out[key + "_relative_deficit"] = [
            [float(1 - r), float(s)] for r, s in zip(ratio, sem, strict=True)
        ]
    return out


def return_deficit(base: dict, reference: dict, depths_ang=REPORT_DEPTHS_ANG) -> dict:
    """Weight lost by absorption: ``reference - base`` and ``1 - X(base)/X(reference)``.

    Positive values are bound population/exposure the absorbing cut drops
    relative to the ``P = R`` reference; paired SEMs as in :func:`paired`.
    Validation: cr-axial-fiber-return-deficit
    """
    lost = paired(base, reference, depths_ang)
    out = {k: [[-m, e] for m, e in v] for k, v in lost.items() if "relative" not in k}
    out.update({k: v for k, v in lost.items() if "relative" in k})
    return out


def analyze(paths, *, base: str = "control", reference: str = "ref") -> dict:
    """Group chunk files by arm; tallies per arm and paired differences.

    Pairs use the shared incident-ID range (:func:`common`).
    ``convergence`` pairs every arm against ``base`` (``arm - base``;
    relative entries ``1 - X(arm)/X(base)``); ``return_deficit`` is
    :func:`return_deficit` of ``base`` against ``reference`` (and of ``m8``
    against ``ref-m8`` when both are present).
    Validation: cr-axial-fiber-return-deficit
    """
    groups: dict[str, list] = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as data:
            groups.setdefault(json.loads(str(data["info"]))["arm"], []).append(p)
    arms = {name: load_arm(files) for name, files in groups.items()}
    result = {
        "depths_ang": list(REPORT_DEPTHS_ANG),
        "arms": {},
        "convergence": {},
    }
    for name, arm in arms.items():
        obs = observables(arm)
        info = arm["info"]
        result["arms"][name] = {
            "incidents": int(len(arm["kind"])),
            "ids": list(arm["ids"]),
            "seconds": info["seconds"],
            "settings": info["settings"],
            "cut_at_gamma_eV": info["cut_at_gamma_eV"],
            "entrance_weight": arm["weight"],
            "mean_lattice_jumps": float(arm["lattice_jumps"].mean()),
            "mean_kick_jumps": float(arm["kick_jumps"].mean()),
            **{k: _mean_sem(v) for k, v in obs.items()},
        }
        if name != base and base in arms:
            result["convergence"][name] = paired(*common(arm, arms[base]))
    if reference in arms and base in arms:
        result["return_deficit"] = return_deficit(*common(arms[base], arms[reference]))
    if "ref-m8" in arms and "m8" in arms:
        result["return_deficit_m8"] = return_deficit(*common(arms["m8"], arms["ref-m8"]))
    return result


def record(paths, source_sha: str) -> dict:
    """Analysis plus provenance for a check record (research only)."""
    import hashlib

    root = Path(__file__).parents[1]
    sources = (
        "checks/axial_outgoing_convergence.py",
        "checks/axial_outgoing_maps.py",
        "src/pyrite/montecarlo/channeling/axial_fiber_unraveling.py",
        "src/pyrite/montecarlo/channeling/axial_tds_continuum.py",
    )
    return {
        "status": (
            "research-only; Si [001] 5 MeV, continuum TDS + electronic, absorbing no-return "
            "fibre model; numerical accounting, not a physical escape fraction or depth budget"
        ),
        "source_commit": source_sha,
        "source_sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in sources},
        "control": CONTROL,
        "arms": ARMS,
        "checkpoints_ang": CHECKPOINTS_ANG.tolist(),
        "result": analyze(paths),
    }


def main(argv) -> None:
    if argv[:1] == ["run"] and len(argv) >= 4:
        name, start, count = argv[1], int(argv[2]), int(argv[3])
        out = argv[argv.index("--out") + 1] if "--out" in argv else f"{name}-{start}.npz"
        np.savez_compressed(out, **run_arm(name, start, count, electronic="--tds" not in argv))
    elif argv[:1] == ["analyze"]:
        print(json.dumps(analyze(argv[1:]), indent=1))
    elif argv[:1] == ["record"] and len(argv) >= 3:
        print(json.dumps(record(argv[2:], argv[1]), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
