"""TestEm5 macros: mono-energetic photon pencil beams on thick C and Pb (issue #275)."""

CASES = {  # name: (material, thickness, energy MeV, events)
    "c_2mev": ("Graphite", "30 cm", 2.0, 6_000_000),
    "c_5mev": ("Graphite", "30 cm", 5.0, 1_000_000),
    "pb_2mev": ("Lead", "3 cm", 2.0, 1_000_000),
    "pb_5mev": ("Lead", "3 cm", 5.0, 500_000),
}
PHYSICS = ("empenelope", "emstandard_opt0", "emstandard_opt4")
for i, (name, (material, thickness, energy, events)) in enumerate(CASES.items()):
    for j, physics in enumerate(PHYSICS):
        with open(f"{name}_{physics}.mac", "w") as f:
            f.write(f"""# Geant4 TestEm5 (v11.4.2), issue #275 pair-conversion reference.
# {energy} MeV photon pencil beam on {thickness} {material}; secondaries killed.
/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds 27500 {100 * i + j + 1}
/testem/det/setAbsMat {material}
/testem/det/setAbsThick {thickness}
/testem/det/setAbsYZ 1 m
/testem/phys/addPhysics {physics}
/process/em/fluo false
/run/initialize
/testem/gun/setDefault
/gun/particle gamma
/gun/energy {energy} MeV
/testem/stack/killSecondaries 2
/run/printProgress 0
/run/beamOn {events}
""")
