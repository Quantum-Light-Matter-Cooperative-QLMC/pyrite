"""PyRITE side of the 3-100 MeV full-track benchmark against Geant4 TestEm5 (#439).

Usage:
  pyrite_bench.py CASE NE FIRST_BATCH OUT_NPZ [--elastic-npz TABLE_NPZ]

Runs the production transport stack on one case of ``make_macros.CASES``:
ELSEPA elastic scattering, SBETHE collision stopping, coupled BremsLib
radiative loss at a 1 keV hard cutoff (dielectric suppression on, as in every
coupled run), Kawrakow atomic-electron deflection, midpoint energy, no
straggling, a 10 keV electron stop and the per-electron CPU core, on a
1 cm x 1 cm footprint as in the macros (``setAbsYZ``). Like the Geant4
reference, only the primary is transported.

Batch ``b`` (50 primaries) uses seed ``43900 + b``. Elastic scattering is
analog, so a primary scattered to grazing can need ~1e5 steps (W at 3 MeV);
``max_steps`` is 300,000 and batches stay small because rows are
preallocated as ``batch x max_steps``. The output holds, per
primary, the fate (0 transmitted, 1 reflected, 2 absorbed, 3 side exit), exit kinetic
energy and exit cosine to the beam axis; per hard photon, the primary index,
photon energy, pre-emission primary energy and cosine to the parent
direction; and the primary track length binned by row-mean energy on the
macros' h63 grid (1000 bins over ``[0, E0]``). ``--elastic-npz`` replaces
the production ELSEPA arrays with a table written by
``point_nucleus_table.py`` (single-element targets only).

Validation: mev-full-track-transport
"""

import json
import sys
import time

import numpy as np

from pyrite.materials import CATALOG
from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from pyrite.montecarlo.transport.events import (
    EVENT_CUTOFF,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_HARD_RADIATIVE,
)
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
from pyrite.xsgen.sbethe.catalog import resolve_composition_table

AVOGADRO = 6.02214076e23
BATCH = 50
MAX_STEPS = 300_000
#: element -> (catalog/SBETHE key, density g/cm^3, atomic weight, thickness Angstrom)
TARGETS = {"w": ("W", 19.3, 183.84, 1.0e6), "si": ("silicon", 2.329, 28.085, 1.0e7)}
ENERGIES_MEV = (3, 10, 30, 50, 100)
CASES = {f"{t}_{e}mev": (t, float(e)) for t in TARGETS for e in ENERGIES_MEV}
TRACK_BINS = 1000


def _composition(target):
    key, rho, A, _ = TARGETS[target]
    if key == "silicon":
        # Catalog silicon: its SBETHE and muffin-tin ELSEPA tables are
        # identity-matched to this density (Geant4 G4_Si: 2.33 g/cm^3).
        return key, "Si", CATALOG.crystal("silicon").composition[0][1]
    return key, key, rho * AVOGADRO / A / 1e24


def run_case(case, ne, first_batch, elastic_npz=None):
    target, E0_MeV = CASES[case]
    key, element, density = _composition(target)
    thickness = TARGETS[target][3]
    composition = [(element, density)]
    E0_keV = E0_MeV * 1e3
    sbethe = resolve_composition_table(key, composition)
    elsepa = resolve_layer_tables(composition)
    elastic = [dict(e.arrays) for e in elsepa]
    if elastic_npz is not None:
        with np.load(elastic_npz) as table:
            if [t.key for t in elsepa[0].tables] != [str(table["production_key"])]:
                raise RuntimeError(f"{elastic_npz} does not replace the {element} production table")
            elastic = [{name: table[name] for name in elastic[0]}]
    brem = load_bremsstrahlung_tables([element])
    kwargs = dict(
        elastic_model="elsepa",
        elastic_tables=[elastic],
        stopping_tables=[sbethe.arrays()],
        atomic_electron_deflection="kawrakow",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        bremslib_tables=brem,
        energy_model="midpoint",
        straggling=False,
        E_cut_keV=10.0,
        crystal_width_mm=10.0,
        crystal_height_mm=10.0,
        transport_core="per-electron",
        max_steps=MAX_STEPS,
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    edges = np.linspace(0.0, E0_keV, TRACK_BINS + 1)
    track_um = np.zeros(TRACK_BINS)
    fate, E_exit, cos_exit = [], [], []
    ph_primary, ph_k, ph_T, ph_cos = [], [], [], []
    radiative = None
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, ne, BATCH), start=first_batch):
        n = min(BATCH, ne - start)
        r = simulate_trajectories(
            E0_keV=E0_keV,
            Ne=n,
            composition=composition,
            thickness_ang=thickness,
            seed=43900 + b,
            **kwargs,
        )
        radiative = r["radiative"]
        eid = np.asarray(r["electron_id"])
        kind = np.asarray(r["event_kind"])
        Es, Ee = np.asarray(r["E_start_keV"]), np.asarray(r["E_end_keV"])
        v = np.asarray(r["v_hat"])
        axis = np.asarray(r["initial_v_hat"])[0]
        L = np.asarray(r["L_ang"])
        if np.any(np.diff(eid) < 0):
            raise RuntimeError("rows are not grouped by electron")
        last = np.r_[np.flatnonzero(np.diff(eid)), eid.size - 1]
        if last.size != n:
            raise RuntimeError(f"{last.size} terminal rows for {n} primaries")
        codes = np.full(n, -1, dtype=np.int8)
        codes[kind[last] == EVENT_EXIT_BOTTOM] = 0
        codes[kind[last] == EVENT_EXIT_TOP] = 1
        codes[kind[last] == EVENT_CUTOFF] = 2
        codes[kind[last] == EVENT_EXIT_SIDE] = 3
        if np.any(codes < 0):
            raise RuntimeError(f"unexpected terminal events {np.unique(kind[last])}")
        fate.append(codes)
        E_exit.append(np.where(codes == 2, 0.0, Ee[last] * 1e3))
        cos_exit.append(v[last] @ axis)
        hard = np.flatnonzero(kind == EVENT_HARD_RADIATIVE)
        k = np.asarray(r["hard_radiative_k_eV"])[hard]
        n_hat = np.asarray(r["hard_radiative_direction"])[hard]
        ph_primary.append(start + eid[hard] - eid[0])
        ph_k.append(k)
        ph_T.append(Ee[hard] * 1e3)  # hard rows end at the pre-emission energy
        ph_cos.append(np.sum(n_hat * v[hard], axis=1))
        track_um += np.histogram(0.5 * (Es + Ee), edges, weights=L * 1e-4)[0]
        print(f"{case} batch {b}: {n} primaries, {kind.size} rows", flush=True)
    return dict(
        fate=np.concatenate(fate),
        E_exit_eV=np.concatenate(E_exit),
        cos_exit=np.concatenate(cos_exit),
        photon_primary=np.concatenate(ph_primary),
        photon_k_eV=np.concatenate(ph_k),
        photon_T_eV=np.concatenate(ph_T),
        photon_cos=np.concatenate(ph_cos),
        track_length_um=track_um,
        track_edges_keV=edges,
        meta=json.dumps(
            dict(
                case=case,
                Ne=ne,
                first_batch=first_batch,
                density_per_ang3=density,
                thickness_ang=thickness,
                sbethe=sbethe.key,
                elsepa=[[t.key for t in e.tables] for e in elsepa],
                elastic_override=None if elastic_npz is None else "point-nucleus",
                bremslib=brem[element].key,
                radiative={k: v for k, v in radiative.items() if k != "bremslib_tables"},
                runtime_s=time.perf_counter() - t0,
            ),
            default=str,
        ),
    )


def main(argv):
    elastic_npz = None
    if "--elastic-npz" in argv:
        i = argv.index("--elastic-npz")
        elastic_npz = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    case, ne, first, out = argv[0], int(argv[1]), int(argv[2]), argv[3]
    np.savez_compressed(out, **run_case(case, ne, first, elastic_npz))


if __name__ == "__main__":
    main(sys.argv[1:])
