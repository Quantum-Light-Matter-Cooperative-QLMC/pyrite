"""Primary terminal fractions of PyRITE production transport vs Geant4 TestEm5 (#183, #317).

Usage:
  prod_bench.py CASE ELASTIC STOPPING INELASTIC NE OUT_JSON [FIRST_BATCH [ATOMIC]]

  CASE: w_300kev | si_300kev | w_800kev | si_800kev
  ELASTIC: sr | elsepa | elsepa-free (Si: free-atom tables, no muffin-tin) |
    elsepa-z1 (#183 diagnostic only: ELSEPA rate x (Z+1)/Z by table edit)
  STOPPING: sbethe | legacy
  INELASTIC: continuous | shell (soft/hard, W_c = 50 eV) | shell-sec (+ secondaries >= 10 keV)
  ATOMIC: none (default, the #183 runs) | kawrakow (#317 atomic-electron deflection)

Batch b uses seed 12345 + b (10,000 primaries per batch), as the #182 driver.
Run on the lab box via run_issue317.sbatch; see README.md.

Validation: inelastic-angular-deflection
"""

import json
import sys
import time

import numpy as np

from pyrite.materials import CATALOG
from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from pyrite.montecarlo.transport.events import EVENT_EXIT_BOTTOM, EVENT_EXIT_TOP
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
from pyrite.xsgen.sbethe.catalog import resolve_composition_table

AVOGADRO = 6.02214076e23
CASES = {
    "w_300kev": ("W", 19.3, 183.84, 100_000.0, 300.0),
    "si_300kev": ("Si", 2.329, 28.085, 1_000_000.0, 300.0),
    "w_800kev": ("W", 19.3, 183.84, 100_000.0, 800.0),
    "si_800kev": ("Si", 2.329, 28.085, 1_000_000.0, 800.0),
}


def main(argv):
    case, elastic, stopping, inelastic, ne, out_path = argv[:6]
    ne = int(ne)
    first = int(argv[6]) if len(argv) > 6 else 0
    atomic = argv[7] if len(argv) > 7 else "none"
    el, rho, A, thick, E0 = CASES[case]
    n = rho * AVOGADRO / A / 1e24
    key = el
    if el == "Si":
        # Catalog silicon (2.3290 g/cm3 vs 2.329 here, 3e-5 relative): its SBETHE,
        # ELSEPA muffin-tin and shell tables are identity-matched to this density.
        key = "silicon"
        n = CATALOG.crystal("silicon").composition[0][1]
    comp = [(el, n)]

    kw = dict(elastic_model="elsepa" if elastic.startswith("elsepa") else elastic)
    if atomic != "none":
        kw["atomic_electron_deflection"] = atomic
    record = {}
    if stopping == "sbethe":
        table = resolve_composition_table(key, comp)
        kw["stopping_tables"] = [table.arrays()]
        record["sbethe"] = table.key
    if elastic.startswith("elsepa"):
        # elsepa-free: free-atom tables only (the #182 driver density matches no
        # catalog solid, so no muffin-tin table is selected).
        n_el = rho * AVOGADRO / A / 1e24 if elastic == "elsepa-free" else n
        entries = resolve_layer_tables([(el, n_el)])
        arrays = [dict(e.arrays) for e in entries]
        if elastic == "elsepa-z1":
            z = {"Si": 14, "W": 74}[el]
            for a in arrays:
                a["total_elastic_cm2"] = np.asarray(a["total_elastic_cm2"]) * (z + 1) / z
                a["dcs_cm2_sr"] = np.asarray(a["dcs_cm2_sr"]) * (z + 1) / z
        kw["elastic_tables"] = [arrays]
        record["elsepa"] = [[t.key for t in e.tables] for e in entries]
    if inelastic.startswith("shell"):
        kw.update(
            inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0, inelastic_materials=[key]
        )
        if inelastic == "shell-sec":
            kw["secondary_threshold_eV"] = 10_000.0
    tables = load_bremsstrahlung_tables([el])
    record["bremslib"] = tables[el].key

    tot = dict.fromkeys(
        (
            "n_transmitted",
            "n_backscattered",
            "n_cutoff_stopped",
            "n_side_exited",
            "n_photons_ge_10kev",
            "n_secondary_tracks",
            "n_secondary_exit_top",
            "n_secondary_exit_bottom",
        ),
        0,
    )
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, ne, 10_000), start=first):
        r = simulate_trajectories(
            E0_keV=E0,
            Ne=min(10_000, ne - start),
            composition=comp,
            thickness_ang=thick,
            E_cut_keV=10.0,
            seed=12345 + b,
            energy_model="midpoint",
            transport_lut_config=TransportLUTConfig(enabled=False),
            straggling=False,
            radiative_model="bremslib-soft-hard",
            radiative_cutoff_eV=1000.0,
            bremslib_tables=tables,
            transport_core="per-electron",
            **kw,
        )
        for k in ("n_transmitted", "n_backscattered", "n_cutoff_stopped", "n_side_exited"):
            tot[k] += int(r.get(k, 0))
        k_eV = np.asarray(r["hard_radiative_k_eV"], dtype=float)
        gen = np.asarray(r["generation"]) if "generation" in r else np.zeros(k_eV.size, int)
        tot["n_photons_ge_10kev"] += int(np.count_nonzero((k_eV >= 10_000.0) & (gen == 0)))
        if "generation" in r:
            ev = np.asarray(r["event_kind"])
            sec = gen > 0
            tot["n_secondary_tracks"] += int(np.unique(np.asarray(r["track_id"])[sec]).size)
            tot["n_secondary_exit_top"] += int(np.count_nonzero(sec & (ev == EVENT_EXIT_TOP)))
            tot["n_secondary_exit_bottom"] += int(np.count_nonzero(sec & (ev == EVENT_EXIT_BOTTOM)))
    out = dict(
        case=case,
        elastic=elastic,
        stopping=stopping,
        inelastic=inelastic,
        atomic=atomic,
        Ne=ne,
        first_batch=first,
        tables=record,
        runtime_s=time.perf_counter() - t0,
        **tot,
    )
    print("RESULT=" + json.dumps(out))
    with open(out_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1:])
