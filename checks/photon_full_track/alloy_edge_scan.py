"""Geant4 Penelope/Livermore K-line yield scan near K edges (issue #341).

Report-only diagnostic for the GaAs comparison (``alloy_fluorescence.py``).
Normal-incidence photons enter 0.2 mm of a pure NIST element. For each
energy, the script reports reflected (R aperture) K-alpha photons per
incident photon from the Geant4 ``empenelope`` and ``emlivermore``
constructors and their ratio. The GaAs settings are otherwise unchanged
(complete vacancy cascades, charged secondaries killed, 1 nm upstream
start). It locates the energy range of the Penelope Ga K deficit and checks
neighbouring elements as controls. It has no gate and makes no PyRITE claim.

Validation: photon-alloy-secondary-fluorescence
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from cases import LATERAL_MM, apertures, hits
from compare import _mean_se, _parse

THICKNESS_MM = 0.2
#: element -> (NIST material, K edge eV, K-alpha window eV, scan energies keV)
ELEMENTS = {
    "Ga": ("G4_Ga", 10367.1, (9000.0, 9500.0), (10.5, 11.0, 11.5, 12.5, 14.0, 17.0, 20.0, 30.0)),
    "Ge": ("G4_Ge", 11103.1, (9500.0, 10000.0), (11.5, 12.5, 14.0, 20.0, 30.0)),
    "As": ("G4_As", 11866.7, (10500.0, 11000.0), (12.2, 13.0, 15.0, 20.0, 30.0)),
}
PHYSICS = ("empenelope", "emlivermore")
#: Geant4 NIST density (g/cm^3) and mean atomic weight for the PyRITE arm
NIST = {"Ga": (5.904, 69.723)}
#: (element, energy keV) pairs with a PyRITE photon-only arm
PYRITE_RUNS = (("Ga", 11.5), ("Ga", 15.0))
BLOCKS = 100


def label(element: str, energy_keV: float, physics: str) -> str:
    return f"{element}_{energy_keV:g}kev_{physics}".replace(".", "p")


def macro(material: str, energy_keV: float, physics: str, events: int, seed: int) -> str:
    return f"""/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds {seed} 341
/testem/det/setAbsMat {material}
/testem/det/setAbsThick {THICKNESS_MM} mm
/testem/det/setAbsYZ {LATERAL_MM} mm
/testem/phys/addPhysics {physics}
/process/em/fluo true
/process/em/auger true
/process/em/deexcitationIgnoreCut true
/process/em/pixe false
/run/setCut 1 nm
/run/setCutForAGivenParticle e- 1 km
/run/setCutForAGivenParticle e+ 1 km
/run/initialize
/testem/gun/setDefault
/gun/particle gamma
/gun/energy {energy_keV:g} keV
/gun/position {-(0.5 * THICKNESS_MM + 1e-6):.7f} 0 0 mm
/gun/direction 1 0 0
/testem/stack/killSecondaries 3
/run/printProgress 0
/run/beamOn {events}
"""


def runs():
    for element, (material, _, _, energies) in ELEMENTS.items():
        for energy in energies:
            for physics in PHYSICS:
                yield element, material, energy, physics


def k_alpha_yield(log: Path, element: str, events: int) -> tuple[float, float]:
    """R-aperture K-alpha photons per incident photon and history-block SE."""
    with gzip.open(log, "rt") as stream:
        if f"The run was {events} gamma" not in stream.read():
            raise ValueError(f"{log}: incomplete Geant4 run")
    event, energy, position, direction, _, _ = _parse(
        [log], THICKNESS_MM, events, source_creator="primary"
    )
    lo, hi = ELEMENTS[element][2]
    hit = hits(apertures(THICKNESS_MM)["R"], position, direction)
    selected = hit & (energy >= lo) & (energy < hi)
    counts = np.bincount(event[selected] * BLOCKS // events, minlength=BLOCKS)
    return _mean_se(counts, np.full(BLOCKS, events // BLOCKS))


def pyrite_k_alpha_yield(element: str, energy_keV: float, events: int, seed: int):
    """PyRITE photon-only R-aperture K-alpha yield on the same slab and footprint."""
    from cases import MM

    from pyrite.montecarlo.transport.photon_histories import track_photon_histories

    rho, atomic_weight = NIST[element]
    layer = [(element, rho * 6.02214076e23 / atomic_weight / 1e24)]
    lo, hi = ELEMENTS[element][2]
    aperture = apertures(THICKNESS_MM)["R"]
    counts = np.zeros(BLOCKS, dtype=np.int64)
    for start in range(0, events, 1000):
        n = min(1000, events - start)
        ids = np.arange(start, start + n)
        result = track_photon_histories(
            np.tile([0.0, 0.0, -1e-6 * MM], (n, 1)),
            np.tile([0.0, 0.0, 1.0], (n, 1)),
            np.full(n, energy_keV * 1000.0),
            np.ones(n),
            ids,
            np.zeros(n, dtype=np.int64),
            [(0.0, THICKNESS_MM * MM, layer)],
            seed=seed,
            photon_cutoff_eV=1000.0,
            electron_cutoff_eV=1000.0,
            width_ang=LATERAL_MM * MM,
            height_ang=LATERAL_MM * MM,
        )
        if not np.all(result.ledger["complete"]):
            raise RuntimeError("unresolved photon histories")
        x = result.exits
        energy = np.asarray(x["energy_eV"])
        selected = (
            hits(aperture, x["position_ang"], x["direction"]) & (energy >= lo) & (energy < hi)
        )
        np.add.at(counts, np.asarray(x["history_id"])[selected] * BLOCKS // events, 1)
    return _mean_se(counts, np.full(BLOCKS, events // BLOCKS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("macros", "pyrite", "report"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=34130)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "macros":
        args.directory.mkdir(parents=True, exist_ok=True)
        names = []
        for i, (element, material, energy, physics) in enumerate(runs()):
            name = label(element, energy, physics)
            seed = args.seed + i // len(PHYSICS)  # same seed for both constructors
            (args.directory / f"{name}.mac").write_text(
                macro(material, energy, physics, args.events, seed)
            )
            names.append(name)
        (args.directory / "runs.txt").write_text("\n".join(names) + "\n")
        return 0
    if args.action == "pyrite":
        out = {}
        for element, energy in PYRITE_RUNS:
            mean, se = pyrite_k_alpha_yield(element, energy, args.events, args.seed + 1)
            out[label(element, energy, "pyrite")] = [mean, se]
            print(f"{element} {energy:5.1f} keV  PyRITE = {mean:.5e} ± {se:.1e}")
        (args.directory / "pyrite.json").write_text(json.dumps(out, indent=2) + "\n")
        return 0
    pyrite_path = args.directory / "pyrite.json"
    pyrite = json.loads(pyrite_path.read_text()) if pyrite_path.exists() else {}
    rows = []
    for element, _, energy, physics in runs():
        if physics != PHYSICS[0]:
            continue
        pen = k_alpha_yield(
            args.directory / f"{label(element, energy, PHYSICS[0])}.log.gz", element, args.events
        )
        liv = k_alpha_yield(
            args.directory / f"{label(element, energy, PHYSICS[1])}.log.gz", element, args.events
        )
        ratio = pen[0] / liv[0] if liv[0] else None
        se = ratio * np.hypot(pen[1] / pen[0], liv[1] / liv[0]) if ratio and pen[0] else None
        rows.append(
            dict(
                element=element,
                energy_keV=energy,
                above_edge_keV=energy - ELEMENTS[element][1] / 1e3,
                penelope=pen,
                livermore=liv,
                ratio=ratio,
                ratio_se=se,
                pyrite=pyrite.get(label(element, energy, "pyrite")),
            )
        )
        extra = rows[-1]["pyrite"]
        line = f"{element} {energy:5.1f} keV  Pen/Liv = {ratio:.3f} ± {se:.3f}"
        if extra:
            line += f"  PyRITE/Liv = {extra[0] / liv[0]:.3f}"
        print(line)
    if args.output:
        args.output.write_text(json.dumps(dict(events=args.events, rows=rows), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
