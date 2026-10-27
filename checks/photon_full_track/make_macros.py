"""TestEm5 macros for the #341 thick-target photon full-track check.

Electron side as in #439 (``../mev_full_track``): ``empenelope``,
Goudsmit-Saunderson, no loss fluctuations, a 10 keV electron stop, photons
produced above the 990 eV PenBrem floor (1 nm gamma range cut) and no delta
rays (1 km e-/e+ range cut). Photons are transported with fluorescence after
photoabsorption; historical defaults disable Auger emission and PIXE.
``killSecondaries 3`` (``patch_testem5.py``) kills every charged secondary at
creation and keeps every photon, matching PyRITE's photon-only mode, in which
photon-born charged daughters are recorded but not transported. The explicit
``--complete-cascade`` arm enables Auger and ignores deexcitation production
cuts, preserving vacancy daughters before charged secondaries are killed.
"""

import argparse
from pathlib import Path

from cases import CASES, LATERAL_MM

#: events per macro; each case runs SHARDS[case] macros with distinct seeds,
#: sized for >= 2e4 hits on its weaker aperture (100k-event pilot rates)
EVENTS = 1_000_000
SHARDS = dict(
    si_0p3mev=30, si_1mev=10, si_3mev=3, si_10mev=1, w_0p3mev=10, w_1mev=3, w_3mev=1, w_10mev=1
)


def macro(case, shard, *, events=EVENTS, complete_cascade=False):
    """Keep historical seeds/settings; explicitly select complete vacancy cascades."""
    _, material, thickness, energy = CASES[case]
    i = list(CASES).index(case)
    name = f"{case}_s{shard:02d}" + ("_cascade" if complete_cascade else "")
    auger = "true" if complete_cascade else "false"
    ignore_cut = "/process/em/deexcitationIgnoreCut true\n" if complete_cascade else ""
    return f"""# Geant4 TestEm5 v11.4.2 (8cc04f65), issue #341 photon full-track reference.
# {energy:g} MeV e- pencil beam on {thickness:g} mm {material}; photon-only secondaries.
/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds {34100 + shard} {i + 1}
/testem/det/setAbsMat {material}
/testem/det/setAbsThick {thickness:g} mm
/testem/det/setAbsYZ {LATERAL_MM:g} mm
/testem/phys/addPhysics empenelope
/process/eLoss/fluct false
/process/em/fluo true
/process/em/auger {auger}
{ignore_cut}/process/em/pixe false
/process/em/lowestElectronEnergy 10 keV
/run/setCut 1 nm
/run/setCutForAGivenParticle e- 1 km
/run/setCutForAGivenParticle e+ 1 km
/run/initialize
/run/dumpCouples
/testem/gun/setDefault
/gun/particle e-
/gun/energy {energy:g} MeV
/testem/stack/killSecondaries 3
/analysis/setFileName {name}
/run/printProgress 0
/run/beamOn {events}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--complete-cascade", action="store_true")
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.events <= 0 or (args.shards is not None and args.shards <= 0):
        parser.error("events and shards must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    names = []
    for case in args.case or CASES:
        for shard in range(args.shards or SHARDS[case]):
            text = macro(case, shard, events=args.events, complete_cascade=args.complete_cascade)
            suffix = "_cascade" if args.complete_cascade else ""
            name = f"{case}_s{shard:02d}{suffix}.mac"
            (args.output / name).write_text(text)
            names.append(name)
    (args.output / "cases.txt").write_text("".join(f"{name}\n" for name in names))


if __name__ == "__main__":
    main()
