"""Matched incident-photon diagnostic for the unresolved W L-line excess.

No electron source: one normal-incidence photon per history, 0.1 mm W,
the full-track check's finite footprint and apertures. Compare the original
Geant4 Auger-off settings with Auger-on and cuts-ignored vacancy propagation.
Charged secondaries are killed in every arm. This is report-only evidence;
it neither changes the full-track gates nor promotes their ledger claim.
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from cases import DENSITY, LATERAL_MM, MM, apertures, edges_eV, hits
from compare import _mean_se, _parse

ENERGIES_KEV = (12, 20, 30)
SETTINGS = {"auger-off": (False, False), "auger-cut": (True, False), "cascade": (True, True)}
THICKNESS_MM = 0.1
BLOCKS = 100
WINDOWS = {"1-10keV": (1000, 10000), "La": (8000, 8500), "Lb": (9500, 10000)}


def macro(energy_keV, events, setting, seed):
    """Same incident ray in TestEm5's frame; 1 nm upstream of the W face."""
    auger, ignore = SETTINGS[setting]
    return f"""/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds {seed} 341
/testem/det/setAbsMat Tungsten
/testem/det/setAbsThick {THICKNESS_MM} mm
/testem/det/setAbsYZ {LATERAL_MM} mm
/testem/phys/addPhysics empenelope
/process/em/fluo true
/process/em/auger {str(auger).lower()}
/process/em/deexcitationIgnoreCut {str(ignore).lower()}
/process/em/pixe false
/run/setCut 1 nm
/run/setCutForAGivenParticle e- 1 km
/run/setCutForAGivenParticle e+ 1 km
/run/initialize
/testem/gun/setDefault
/gun/particle gamma
/gun/energy {energy_keV} keV
/gun/position -0.050001 0 0 mm
/gun/direction 1 0 0
/testem/stack/killSecondaries 3
/run/printProgress 0
/run/beamOn {events}
"""


def tally(event, energy, position, direction, direct, events, *, creator=None):
    """Raw spectra in independent equal history blocks, including zero-hit blocks."""
    if events < BLOCKS or events % BLOCKS:
        raise ValueError(f"events must be a positive multiple of {BLOCKS}")
    event = np.asarray(event)
    if np.any(event < 0) or np.any(event >= events):
        raise ValueError("photon event id outside the declared history range")
    edges = edges_eV(0.3)
    block = event * BLOCKS // events
    out = {}
    for name, aperture in apertures(THICKNESS_MM).items():
        hit = hits(aperture, position, direction)
        masks = {"total": hit, "direct": hit & direct, "redist": hit & ~direct}
        if creator is not None:
            masks.update({f"creator:{c}": hit & (creator == c) for c in np.unique(creator)})
        out[name] = {}
        for label, mask in masks.items():
            counts = np.zeros((BLOCKS, len(edges) - 1), dtype=np.int64)
            index = np.searchsorted(edges, energy, side="right") - 1
            selected = mask & (index >= 0) & (index < len(edges) - 1)
            np.add.at(counts, (block[selected], index[selected]), 1)
            out[name][label] = counts.tolist()
    return dict(events=events, edges_eV=edges.tolist(), apertures=out)


def run_pyrite(energy_keV, events, seed):
    from pyrite.montecarlo.transport.photon_histories import track_photon_histories

    rho, mass = DENSITY["W"]
    composition = [("W", rho * 6.02214076e23 / mass / 1e24)]
    tables = {k: [] for k in ("history_id", "energy_eV", "position_ang", "direction", "generation")}
    for start in range(0, events, 1000):
        ids = np.arange(start, min(start + 1000, events))
        n = len(ids)
        result = track_photon_histories(
            np.tile([0, 0, -0.000001 * MM], (n, 1)),
            np.tile([0, 0, 1], (n, 1)),
            np.full(n, energy_keV * 1000.0),
            np.ones(n),
            ids,
            np.zeros(n, dtype=np.int64),
            [(0, THICKNESS_MM * MM, composition)],
            seed=seed,
            photon_cutoff_eV=1000,
            electron_cutoff_eV=1000,
            width_ang=LATERAL_MM * MM,
            height_ang=LATERAL_MM * MM,
        )
        if not np.all(result.ledger["complete"]):
            raise RuntimeError("unresolved photon histories")
        for key in tables:
            tables[key].append(result.exits[key])
    rows = {k: np.concatenate(v) for k, v in tables.items()}
    return tally(
        rows["history_id"],
        rows["energy_eV"],
        rows["position_ang"],
        rows["direction"],
        rows["generation"] == 0,
        events,
    )


def summarize(record):
    """Per-incident-photon yields and history-block SE, without significance gates."""
    edges = np.asarray(record["edges_eV"])
    sizes = np.full(BLOCKS, record["events"] // BLOCKS)
    return {
        aperture: {
            label: {
                window: _mean_se(
                    np.asarray(counts)[:, (edges[:-1] >= lo) & (edges[1:] <= hi)].sum(axis=1), sizes
                )
                for window, (lo, hi) in WINDOWS.items()
            }
            for label, counts in labels.items()
        }
        for aperture, labels in record["apertures"].items()
    }


def verify_geant4_run(path, energy_keV, events):
    """Reject truncated logs and mismatched primary counts/energies."""
    with gzip.open(path, "rt") as stream:
        text = stream.read()
    match = re.search(r"The run was\s+(\d+) gamma of\s+([\d.eE+-]+) (\w+)", text)
    if match is None:
        raise ValueError("missing completed Geant4 run summary")
    count, energy, unit = match.groups()
    if int(count) != events or unit != "keV" or float(energy) != energy_keV:
        raise ValueError("Geant4 run summary does not match requested photons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("macros", "pyrite", "reduce-geant4", "summary"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--energy-keV", type=int, choices=ENERGIES_KEV, default=20)
    parser.add_argument("--seed", type=int, default=34101)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.events < BLOCKS or args.events % BLOCKS:
        parser.error(f"--events must be a positive multiple of {BLOCKS}")
    if args.action == "macros":
        args.output.mkdir(parents=True, exist_ok=True)
        for energy in ENERGIES_KEV:
            for setting in SETTINGS:
                (args.output / f"w_{energy}kev_{setting}.mac").write_text(
                    macro(energy, args.events, setting, args.seed)
                )
        return
    if args.action == "summary":
        data = args.output.read_bytes()
        if args.output.suffix == ".gz":
            data = gzip.decompress(data)
        record = json.loads(data)
        print(json.dumps(summarize(record), indent=2))
        return
    if args.action == "pyrite":
        record = run_pyrite(args.energy_keV, args.events, args.seed)
        record.update(engine="pyrite-cpu-photon-only", seed=args.seed)
    else:
        if args.log is None:
            parser.error("reduce-geant4 requires --log")
        verify_geant4_run(args.log, args.energy_keV, args.events)
        event, energy, position, direction, creator, direct = _parse(
            [args.log], THICKNESS_MM, args.events, source_creator="primary"
        )
        record = tally(event, energy, position, direction, direct, args.events, creator=creator)
        record.update(
            engine="geant4-testem5-empenelope",
            log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
        )
    record.update(format="photon-monoenergetic.v1", energy_keV=args.energy_keV)
    args.output.write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
