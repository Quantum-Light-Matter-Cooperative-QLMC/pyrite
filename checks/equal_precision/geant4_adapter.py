"""Run instrumented Geant4 TestEm5 and retain per-primary raw scoring logs."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from pyrite.perf.geant4 import parse_primary_log
from pyrite.perf.precision import grouped_scores, primary_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics", choices=["empenelope", "emstandardSS"], default="empenelope")
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    scene = next(s for s in manifest["scenes"] if s["id"] == args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.with_suffix("")
    work.mkdir(exist_ok=False)
    angle = math.radians(scene["angle_deg"])
    material = "Silicon" if scene["element"] == "Si" else "Tungsten"
    macro = f"""/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds {args.seed} 67890
/testem/det/setAbsMat {material}
/testem/det/setAbsThick {scene["thickness_um"]} um
/testem/det/setAbsYZ 1 cm
/testem/phys/addPhysics {args.physics}
/process/eLoss/fluct false
/process/em/fluo false
/process/em/lowestElectronEnergy 10 keV
/run/setCut 1 nm
/run/setCutForAGivenParticle e- 10 mm
/run/initialize
/run/dumpCouples
/testem/gun/setDefault
/gun/particle e-
/gun/energy {scene["energy_keV"]} keV
/gun/position {-scene["thickness_um"] / 2 - 1e-6} {-math.tan(angle) * 1e-6} 0 um
/gun/direction {math.cos(angle)} {math.sin(angle)} 0
/analysis/setFileName score.csv
/run/beamOn {args.histories}
"""
    (work / "input.mac").write_text(macro)
    start = time.perf_counter()
    with (work / "raw.log").open("w") as log:
        completed = subprocess.run(
            [str(args.executable.resolve()), "input.mac"],
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    text = (work / "raw.log").read_text()
    if completed.returncode or "COMMAND NOT FOUND" in text or "***** Illegal" in text:
        raise RuntimeError(f"Geant4 failed; inspect {work / 'raw.log'}")
    ends, pids, k = parse_primary_log(text, args.histories)
    scores = {}
    ids = np.array([e[0] for e in ends], dtype=int)
    ev = np.array([e[1] for e in ends])
    for name, code in [
        ("transmitted", 5),
        ("backscattered", 4),
        ("side_exited", 6),
        ("stopped", 7),
    ]:
        scores[name] = grouped_scores(
            ids[ev == code], np.ones(np.count_nonzero(ev == code)), args.histories
        )
    scores["transmitted_energy_keV_per_primary"] = grouped_scores(
        ids[ev == 5], np.array([e[2] for e in ends])[ev == 5], args.histories
    )
    for name, low, high in [
        ("photons_ge10kev", 10000, float("inf")),
        ("photons_10_50kev", 10000, 50000),
        ("photons_50_100kev", 50000, 100000),
        ("photons_ge100kev", 100000, float("inf")),
    ]:
        mask = (k >= low) & (k < high)
        scores[name] = grouped_scores(pids[mask], np.ones(np.count_nonzero(mask)), args.histories)
    record = dict(
        schema="pyrite.equal-precision.sample.v1",
        engine="geant4",
        physics=args.physics,
        scene=scene,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        histories=args.histories,
        seed=args.seed,
        timings_s={"fresh_process_with_logging": elapsed},
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        raw_log_sha256=hashlib.sha256((work / "raw.log").read_bytes()).hexdigest(),
        observables={k: primary_moments(v) for k, v in scores.items()},
        numerical_gate="unmeasured",
        equivalence_gate="unmeasured",
        mismatches=[
            "Material definitions are patched to the frozen manifest; inspect raw log",
            "Geant4 photon production threshold is a material-dependent range-cut conversion",
            "per-emission logging overhead is included",
        ],
    )
    raw_scores = args.output.with_suffix(".npz")
    np.savez_compressed(raw_scores, **scores)
    record["raw_scores_sha256"] = hashlib.sha256(raw_scores.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "timings_s": record["timings_s"]}))


if __name__ == "__main__":
    main()
