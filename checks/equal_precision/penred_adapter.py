"""Run PenRed (PENELOPE-2018 physics) with the #510 per-history tally.

Arms: ``source`` (terminal fractions, transmitted energy and creation-level
hard-photon counts) and ``transport`` (the same run reduced to terminal
fractions and transmitted energy; bremsstrahlung energy loss stays coupled and
photons are absorbed at creation, matching the PyRITE ``bremslib-soft-hard``
transport headline). Scenes are pilot ``manifest.json`` ids or frozen
``matrix.json`` case ids. The executable must come from ``prepare_penred.py``.
"""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

ARMS = ("source", "transport")
# PenRed 1.15.2's generic-source loop leaves the last sampled particle of a
# request unsimulated, so a request of N + 1 histories transports N primaries.
EXTRA_REQUESTED = 1
SEED_PAIRS = 1001
UNSUPPORTED_BEAM = ("spot_rms_mm", "divergence_rms_mrad", "energy_spread_rms_frac")
UNSUPPORTED_PHYSICS = ("secondary_transport", "photon_transport", "characteristic", "pxr", "cbs")
MISMATCHES = [
    "PENELOPE always samples soft energy-loss straggling; compare with PyRITE straggling variants",
    "PENELOPE mixed elastic scattering (C1/C2) and its own stopping/bremsstrahlung models, "
    "not ELSEPA/SBETHE/BremsLib tables",
    "WCC = E0 suppresses delta rays, so all inelastic loss is soft and step control differs",
    "inner-shell impact ionization and relaxation stay on; unscored secondaries are transported "
    "and charged to the timing",
    "material written by PenRed with an 8-digit mass density",
]


def case_unsupported(case):
    """Matrix-case features this adapter cannot express; empty when runnable."""
    out = [f"beam {k}={case['beam'][k]}" for k in UNSUPPORTED_BEAM if case["beam"][k]]
    out += [k for k in UNSUPPORTED_PHYSICS if case["physics"][k]]
    if case["physics"]["inelastic_model"] != "continuous":
        out.append(f"inelastic_model={case['physics']['inelastic_model']}")
    return out


def case_arms(case):
    """Adapter arms that can run a matrix case; the cone detector arm is not ported."""
    return [] if case_unsupported(case) else list(ARMS)


def parse_log_timings(text):
    """PenRed's own phase times [s]; real time has one-second resolution."""
    out = {}
    for key, label in [
        ("initialization", "Initialization processing time:"),
        ("simulation_real", "Simulation real time:"),
        ("simulation_user", "Simulation user time:"),
    ]:
        match = re.search(re.escape(label) + r"\s*([-+0-9.Ee]+)", text)
        out[key] = float(match.group(1)) if match else None
    seeds = re.search(r"initial history \d+ and seeds (\d+) (\d+)", text)
    return out, [int(s) for s in seeds.groups()] if seeds else None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help=f"seed pair in [0, {SEED_PAIRS})")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="source")
    parser.add_argument("--c1", type=float, default=0.05)
    parser.add_argument("--c2", type=float, default=0.05)
    parser.add_argument("--wcr-ev", type=float, default=None)
    parser.add_argument("--dsmax-cm", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    args = parser.parse_args()
    if args.histories < 2 or not 0 <= args.seed < SEED_PAIRS:
        parser.error("need at least two histories and a seed pair index in range")
    return args


def write_record(args, record):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": record["status"]}))


def main():
    args = parse_args()
    import numpy as np
    from pyrite_adapter import load_scene

    from pyrite.perf.comparison import comparison_record
    from pyrite.perf.penred import config_text, geometry_text, parse_tally, primary_scores

    raw, spec, raw_path = load_scene(args.scene)
    base = dict(
        case_id=spec["id"],
        engine="penred",
        arm="cpu-serial",
        kind=args.arm,
        n_primaries=args.histories,
        energy_keV=spec["energy_keV"],
        seed=args.seed,
    )
    unsupported = case_unsupported(raw) if spec["source"] == "matrix" else []
    if unsupported:
        reason = "adapter cannot express: " + ", ".join(unsupported)
        write_record(args, comparison_record(status="unsupported-physics", reason=reason, **base))
        return
    work = args.output.with_suffix("")
    work.mkdir(parents=True, exist_ok=False)
    thickness_cm = spec["thickness_um"] * 1e-4
    width_cm = spec["width_mm"] * 0.1
    wcr_eV = spec["hard_photon_cutoff_eV"] if args.wcr_ev is None else args.wcr_ev
    (work / "slab.geo").write_text(geometry_text(thickness_cm, width_cm))
    (work / "config.in").write_text(
        config_text(
            composition=spec["composition"],
            density_g_cm3=raw["density_g_cm3"],
            energy_keV=spec["energy_keV"],
            thickness_cm=thickness_cm,
            width_cm=width_cm,
            angle_deg=spec["angle_deg"],
            histories=args.histories + EXTRA_REQUESTED,
            seed_pair=args.seed,
            electron_eabs_eV=spec["electron_cutoff_keV"] * 1e3,
            wcr_eV=wcr_eV,
            c1=args.c1,
            c2=args.c2,
            dsmax_cm=args.dsmax_cm,
        )
    )
    start = time.perf_counter()
    status, reason = "ok", None
    try:
        with (work / "stdout.log").open("w") as log:
            completed = subprocess.run(
                [str(args.executable.resolve()), "config.in"],
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=args.timeout_s,
            )
        if completed.returncode:
            status, reason = "failed", f"pen_main exit {completed.returncode}; see {work}"
    except subprocess.TimeoutExpired:
        status, reason = "timeout", f"pen_main exceeded {args.timeout_s} s"
    elapsed = time.perf_counter() - start
    log_text = (work / "simulation.log").read_text() if (work / "simulation.log").exists() else ""
    timings, initial_seeds = parse_log_timings(log_text)
    tally = work / "primary-th0-pyrite-primary.dat"
    provenance = dict(
        scene=spec,
        scene_file=raw_path.name,
        scene_file_sha256=sha256(raw_path),
        executable_sha256=sha256(args.executable),
        engine_label="PenRed (PENELOPE-2018 physics)",
        requested_histories=args.histories + EXTRA_REQUESTED,
        seed_pair=args.seed,
        initial_seeds=initial_seeds,
        controls={
            "eabs_electron_eV": spec["electron_cutoff_keV"] * 1e3,
            "wcr_eV": wcr_eV,
            "wcc_eV": spec["energy_keV"] * 1e3,
            "c1": args.c1,
            "c2": args.c2,
            "dsmax_cm": args.dsmax_cm or 0.1 * thickness_cm,
            "threads": 1,
        },
        cache_state="fresh-process",
        timings_s={"fresh_process": elapsed, **timings},
        inputs_sha256={name: sha256(work / name) for name in ("config.in", "slab.geo")},
        mismatches=MISMATCHES,
        numerical_gate="unmeasured",
        equivalence_gate="unmeasured",
    )
    if args.arm == "transport":
        provenance["radiative_energy_loss"] = (
            "PENELOPE soft + hard bremsstrahlung loss kept; photons absorbed at creation, unscored"
        )
    if status != "ok":
        write_record(
            args, comparison_record(status=status, reason=reason, provenance=provenance, **base)
        )
        return
    provenance["material_sha256"] = sha256(work / "slab.mat")
    provenance["tally_sha256"] = sha256(tally)
    parsed = parse_tally(tally.read_text(), args.histories)
    provenance["unscored_secondaries"] = parsed["unscored_secondaries"]
    scores = primary_scores(parsed, args.histories, kind=args.arm)
    raw_scores = args.output.with_suffix(".npz")
    np.savez_compressed(raw_scores, **scores)
    provenance["raw_scores_sha256"] = sha256(raw_scores)
    write_record(args, comparison_record(status="ok", scores=scores, provenance=provenance, **base))


if __name__ == "__main__":
    main()
