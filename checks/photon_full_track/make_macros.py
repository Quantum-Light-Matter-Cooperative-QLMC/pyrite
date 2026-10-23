"""TestEm5 macros for the #341 thick-target photon full-track check.

Electron side as in #439 (``../mev_full_track``): ``empenelope``,
Goudsmit-Saunderson, no loss fluctuations, a 10 keV electron stop, photons
produced above the 990 eV PenBrem floor (1 nm gamma range cut) and no delta
rays (1 km e-/e+ range cut). Photons are transported with fluorescence after
photoabsorption; Auger emission and charged-particle PIXE are off.
``killSecondaries 3`` (``patch_testem5.py``) kills every charged secondary at
creation and keeps every photon, matching PyRITE's photon-only mode, in which
photon-born charged daughters are recorded but not transported.
"""

from cases import CASES, LATERAL_MM

#: events per macro; each case runs SHARDS[case] macros with distinct seeds,
#: sized for >= 2e4 hits on its weaker aperture (100k-event pilot rates)
EVENTS = 1_000_000
SHARDS = dict(
    si_0p3mev=30, si_1mev=10, si_3mev=3, si_10mev=1, w_0p3mev=10, w_1mev=3, w_3mev=1, w_10mev=1
)

if __name__ == "__main__":
    names = []
    for i, (case, (_, material, thickness, energy)) in enumerate(CASES.items()):
        for shard in range(SHARDS[case]):
            name = f"{case}_s{shard:02d}"
            names.append(name)
            with open(f"{name}.mac", "w") as f:
                f.write(f"""# Geant4 TestEm5 v11.4.2 (8cc04f65), issue #341 photon full-track reference.
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
/process/em/auger false
/process/em/pixe false
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
/run/beamOn {EVENTS}
""")
    with open("cases.txt", "w") as f:
        f.write("".join(f"{name}.mac\n" for name in names))
