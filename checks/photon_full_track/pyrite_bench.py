"""PyRITE side of the #341 thick-target photon full-track check.

Usage:
  pyrite_bench.py CASE NE FIRST_BATCH OUT_NPZ [--samples-per-primary N]

Electron transport repeats the #439 production stack (ELSEPA, SBETHE, coupled
BremsLib at a 1 keV hard cutoff, Kawrakow deflection, midpoint energy, no
straggling, a 10 keV stop, per-electron CPU core), primary only, on the
5 cm x 5 cm slab of ``cases.CASES``. Each 50-primary batch ``b`` (transport
seed ``34100 + b``) then feeds the weighted continuous BremsLib source
(``score_bremslib_continuum``, photon seed ``341000 + b``) into the
photon-only reference tracker, once per aperture in ``cases.apertures``:
photon cutoff 1 keV, source ceiling E0, charged daughters recorded but not
transported. Per batch and aperture the output keeps the total, direct and
redistributed spectra [photons/(primary eV)] and in/out-of-band detections
per primary, so batch-to-batch scatter gives the statistical error.

Validation: photon-full-track-thick-brem
"""

import json
import sys
import time

import numpy as np
from cases import CASES, DENSITY, LATERAL_MM, MM, apertures, edges_eV

from pyrite.materials import CATALOG
from pyrite.montecarlo.runner.photon_emission import score_bremslib_continuum
from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from pyrite.montecarlo.transport.photon_histories import PhotonAperture
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
from pyrite.xsgen.sbethe.catalog import resolve_composition_table

AVOGADRO = 6.02214076e23
BATCH = 50
MAX_STEPS = 300_000


def _composition(element):
    if element == "Si":
        return "silicon", CATALOG.crystal("silicon").composition[0][1]
    rho, A = DENSITY[element]
    return element, rho * AVOGADRO / A / 1e24


def _transport_kwargs(element):
    """The #439 primary-only electron stack on the 5 cm footprint."""
    key, density = _composition(element)
    composition = [(element, density)]
    sbethe = resolve_composition_table(key, composition)
    elsepa = resolve_layer_tables(composition)
    brem = load_bremsstrahlung_tables([element])
    kwargs = dict(
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in elsepa]],
        stopping_tables=[sbethe.arrays()],
        atomic_electron_deflection="kawrakow",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        bremslib_tables=brem,
        energy_model="midpoint",
        straggling=False,
        E_cut_keV=10.0,
        crystal_width_mm=LATERAL_MM,
        crystal_height_mm=LATERAL_MM,
        transport_core="per-electron",
        max_steps=MAX_STEPS,
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    tables = dict(
        density_per_ang3=density,
        sbethe=sbethe.key,
        elsepa=[[t.key for t in e.tables] for e in elsepa],
        bremslib=brem[element].key,
    )
    return kwargs, composition, tables


def run_case(case, ne, first_batch, samples_per_primary=400):
    element, _, thickness_mm, E0_MeV = CASES[case]
    kwargs, composition, tables = _transport_kwargs(element)
    brem = kwargs["bremslib_tables"]
    thickness = thickness_mm * MM
    lateral = LATERAL_MM * MM
    edges = edges_eV(E0_MeV)
    layers = [(0.0, thickness, composition)]
    targets = {
        name: PhotonAperture(a["center"], a["x"], a["y"], a["half"], a["half"])
        for name, a in apertures(thickness_mm).items()
    }
    out = {f"{n}_{f}": [] for n in targets for f in ("total", "direct", "redist", "det", "oob")}
    sizes, t_transport, t_photon = [], 0.0, 0.0
    for b, start in enumerate(range(0, ne, BATCH), start=first_batch):
        n = min(BATCH, ne - start)
        t0 = time.perf_counter()
        r = simulate_trajectories(
            E0_keV=E0_MeV * 1e3,
            Ne=n,
            composition=composition,
            thickness_ang=thickness,
            seed=34100 + b,
            **kwargs,
        )
        t1 = time.perf_counter()
        for name, aperture in targets.items():
            s = score_bremslib_continuum(
                r,
                layers,
                brem,
                aperture,
                edges,
                n_primaries=n,
                n_samples=samples_per_primary * n,
                seed=341000 + b,
                photon_cutoff_eV=1000.0,
                source_ceiling_eV=E0_MeV * 1e6,
                electron_cutoff_eV=1000.0,
                width_ang=lateral,
                height_ang=lateral,
            )
            out[f"{name}_total"].append(s.total_per_primary_eV)
            out[f"{name}_direct"].append(s.direct_per_primary_eV)
            out[f"{name}_redist"].append(s.redistributed_per_primary_eV)
            out[f"{name}_det"].append(s.detected_per_primary)
            out[f"{name}_oob"].append(s.unbinned_detected_per_primary)
        t_transport += t1 - t0
        t_photon += time.perf_counter() - t1
        sizes.append(n)
        print(f"{case} batch {b}: {n} primaries, {len(r['L_ang'])} rows", flush=True)
    return dict(
        {k: np.asarray(v) for k, v in out.items()},
        batch_size=np.asarray(sizes),
        edges_eV=edges,
        meta=json.dumps(
            dict(
                case=case,
                Ne=ne,
                first_batch=first_batch,
                samples_per_primary=samples_per_primary,
                thickness_ang=thickness,
                **tables,
                transport_s=t_transport,
                photon_s=t_photon,
            ),
            default=str,
        ),
    )


def main(argv):
    samples = 400
    if "--samples-per-primary" in argv:
        i = argv.index("--samples-per-primary")
        samples = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    case, ne, first, out = argv[0], int(argv[1]), int(argv[2]), argv[3]
    np.savez_compressed(out, **run_case(case, ne, first, samples))


if __name__ == "__main__":
    main(sys.argv[1:])
