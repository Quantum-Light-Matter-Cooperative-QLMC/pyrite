"""TestEm5 macros for the 3-100 MeV full-track benchmark (issue #439).

Each case is an electron pencil beam at normal incidence on a 1 cm lateral
slab. Physics matches PyRITE's production scope: ``empenelope``, no loss
fluctuations or fluorescence, a 10 keV electron stop, photons produced above
the 990 eV PenBrem floor (1 nm gamma range cut), no delta rays (1 km electron
range cut), and every secondary killed at creation without depositing energy
(``killSecondaries 2``), so only the primary is transported and the deposit
is its continuous loss.
"""

#: case -> (TestEm5 material, thickness, energy MeV); events per case below.
CASES = {
    f"{el}_{e}mev": (mat, thick, float(e))
    for el, mat, thick in (("w", "Tungsten", "100 um"), ("si", "Silicon", "1 mm"))
    for e in (3, 10, 30, 50, 100)
}
EVENTS = 100_000

if __name__ == "__main__":
    for i, (name, (material, thickness, energy)) in enumerate(CASES.items()):
        with open(f"{name}.mac", "w") as f:
            f.write(f"""# Geant4 TestEm5 v11.4.2 (8cc04f65), issue #439 MeV full-track reference.
# {energy:g} MeV e- pencil beam on {thickness} {material}; primary only.
/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds 43900 {i + 1}
/testem/det/setAbsMat {material}
/testem/det/setAbsThick {thickness}
/testem/det/setAbsYZ 1 cm
/testem/phys/addPhysics empenelope
/process/eLoss/fluct false
/process/em/fluo false
/process/em/lowestElectronEnergy 10 keV
/run/setCut 1 nm
/run/setCutForAGivenParticle e- 1 km
/run/initialize
/run/dumpCouples
/testem/gun/setDefault
/gun/particle e-
/gun/energy {energy:g} MeV
/testem/stack/killSecondaries 2
/analysis/setFileName {name}.csv
/analysis/h1/set 63 1000 0 {energy:g} MeV
/analysis/h1/setAscii 63
/run/printProgress 0
/run/beamOn {EVENTS}
""")
