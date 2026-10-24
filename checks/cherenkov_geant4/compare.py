"""Geant4 G4Cerenkov reference cases and reduction for issue #587.

``python compare.py cases DIR`` writes the RINDEX tables and ``cases.txt``
(one ``cherenkov`` argument line per case) into DIR. ``python compare.py
reduce DIR`` turns the raw ``*.steps``/``*.hist`` outputs into
``reference/<case>.json``: the photon energy and polar-angle histograms and
the primary path length grouped by step-mean beta. The comparison with
``frank_tamm_yield`` is ``tests/montecarlo/test_cherenkov_geant4_reference.py``.
"""

import json
import sys
from pathlib import Path

import numpy as np

HC_EV_UM = 1.23984198
BAND_EV = (1.5, 5.9)
N_EBINS = 88
#: Step-mean beta grouping width. Near threshold the band yield changes by
#: ~100 relative per unit beta, so rounding moves a step by <= 5e-5 relative.
BETA_QUANTUM = 1e-6

# name: (material, rindex, particle, beta, thickness mm, events, seed)
CASES = {
    "water_mu_b099": ("G4_WATER", "n133", "mu-", 0.99, 10.0, 20_000, 58701),
    "silica_mu_threshold": ("G4_SILICON_DIOXIDE", "silica", "mu-", 1 / 1.47, 0.5, 300_000, 58702),
    "n150_alpha_b095": ("G4_WATER", "n150", "alpha", 0.95, 5.0, 10_000, 58703),
    "water_p_below": ("G4_WATER", "n133", "proton", 0.74, 10.0, 2_000, 58704),
}
#: Geant4 11.4.2 rest energies [MeV] (G4MuonMinus, G4Proton, G4Alpha).
MASS_MEV = {"mu-": 105.6583755, "proton": 938.27208816, "alpha": 3727.3794066}


def silica_index(energy_eV):
    """Fused silica, Malitson, JOSA 55, 1205 (1965), 0.21--3.71 um."""
    lam2 = (HC_EV_UM / np.asarray(energy_eV)) ** 2
    terms = ((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161))
    return np.sqrt(1.0 + sum(b * lam2 / (lam2 - c**2) for b, c in terms))


def rindex_table(kind):
    """Tabulated (energy eV, n): constant indices need only the band ends."""
    if kind == "silica":
        energy = np.linspace(*BAND_EV, 441)
        return energy, silica_index(energy)
    energy = np.array(BAND_EV)
    return energy, np.full(2, {"n133": 1.33, "n150": 1.50}[kind])


def write_cases(out):
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("n133", "n150", "silica"):
        np.savetxt(out / f"rindex_{kind}.txt", np.column_stack(rindex_table(kind)), fmt="%.12g")
    lines = []
    for name, (material, kind, particle, beta, thick, events, seed) in CASES.items():
        kinetic = MASS_MEV[particle] * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
        lines.append(
            f"{material} rindex_{kind}.txt {particle} {kinetic:.10g} {thick} {events} {seed} "
            f"{BAND_EV[0]} {BAND_EV[1]} {N_EBINS} {name}"
        )
    (out / "cases.txt").write_text("\n".join(lines) + "\n")


def _read_hist(path):
    rows = dict(line.split(maxsplit=1) for line in path.read_text().splitlines())
    return {key: np.array(value.split(), dtype=float) for key, value in rows.items()}


def reduce(raw, reference):
    reference.mkdir(parents=True, exist_ok=True)
    for name, (material, kind, particle, beta, thick, events, seed) in CASES.items():
        steps = np.loadtxt(raw / f"{name}.steps", ndmin=2)
        hist = _read_hist(raw / f"{name}.hist")
        keys, inverse = np.unique(np.round(steps[:, 0] / BETA_QUANTUM), return_inverse=True)
        energy_table, index_table = rindex_table(kind)
        record = {
            "case": name,
            "geant4": "11.4.2 G4Cerenkov, G4EmStandardPhysics_option4, secondaries killed",
            "material": material,
            "particle": particle,
            "charge": {"mu-": -1, "proton": 1, "alpha": 2}[particle],
            "nominal_beta": beta,
            "thickness_mm": thick,
            "events": events,
            "seed": seed,
            "rindex_energy_eV": energy_table.tolist(),
            "rindex_n": index_table.tolist(),
            "step_beta": (keys * BETA_QUANTUM).tolist(),
            "step_path_mm": np.bincount(inverse, weights=steps[:, 1]).tolist(),
            "steps": int(len(steps)),
            "photons": int(hist["photons"][0]),
            "photons_from_steps": int(steps[:, 2].sum()),
            "angle_outside": int(hist["angle_outside"][0]),
            "energy_edges": list(hist["energy_edges"]),
            "energy_counts": hist["energy_counts"].astype(int).tolist(),
            "theta_edges_deg": list(hist["theta_edges_deg"]),
            "theta_counts": hist["theta_counts"].astype(int).tolist(),
        }
        (reference / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(name, record["photons"], "photons in", record["steps"], "steps")


if __name__ == "__main__":
    command, directory = sys.argv[1], Path(sys.argv[2])
    if command == "cases":
        write_cases(directory)
    elif command == "reduce":
        reduce(directory, Path(__file__).resolve().parent / "reference")
    else:
        raise SystemExit("usage: compare.py cases|reduce DIR")
