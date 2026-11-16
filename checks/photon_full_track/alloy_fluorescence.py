"""Binary-compound secondary-fluorescence comparison for issue #341.

Monoenergetic photons enter a 0.2 mm GaAs slab (Geant4 NIST
``G4_GALLIUM_ARSENIDE``: 5.31 g/cm^3, one Ga and one As atom per formula unit)
normally, 1 nm upstream of its entrance face, with the finite footprint and
both ideal apertures of the thick-target check (``cases.apertures``). As
K-alpha (10.51-10.54 keV) and K-beta (11.73 keV) lie above the Ga K edge
(10.367 keV), so As fluorescence re-excites Ga: Ga K emission contains a
secondary-fluorescence component that a primary-only model omits.

- 11.5 keV lies below the As K edge (11.867 keV): Ga K emission is primary
  (plus scattered-photon excitation); it fixes the Ga baseline.
- 15 and 30 keV excite As K, adding As-to-Ga secondary fluorescence.

Both engines kill every charged secondary at creation (PyRITE photon-only;
TestEm5 ``killSecondaries 3``). Geant4 uses the complete-cascade settings that
resolved the W benchmark mismatch (Auger on, deexcitation cuts ignored).

Gates were fixed before any PyRITE spectrum was inspected: for each energy and
aperture, every line window in ``GATED`` holding at least ``MIN_COUNTS``
Geant4 counts passes when ``|P - G| <= max(LINE_TOL * G, 3 sigma)``. Window
counts are raw (no continuum subtraction): adjacent Ga K-beta and As K-alpha
bins leave no clean side windows, and monoenergetic incidence leaves little
continuum under the lines. ``sigma`` combines independent history-block
errors of both engines. PyRITE's indirect Ga K excitation fraction is reported
as a sensitivity diagnostic, not a gate.

Validation: photon-alloy-secondary-fluorescence
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from cases import LATERAL_MM, MM, apertures, hits
from compare import _gate, _mean_se, _parse

AVOGADRO = 6.02214076e23
MATERIAL = "G4_GALLIUM_ARSENIDE"
DENSITY_G_CM3 = 5.31
#: Geant4 NIST mean atomic weights (g/mol)
ATOMIC_WEIGHT = {"Ga": 69.723, "As": 74.9216}
THICKNESS_MM = 0.2
ENERGIES_KEV = (11.5, 15.0, 30.0)
GA_K_EDGE_EV = 10367.1
BLOCKS = 100
EDGES_EV = np.arange(1000.0, 31000.0 + 1.0, 500.0)
#: line windows (eV); each holds one 0.5 keV bin
WINDOWS = {
    "Ga_Ka": (9000.0, 9500.0),
    "Ga_Kb": (10000.0, 10500.0),
    "As_Ka": (10500.0, 11000.0),
    "As_Kb": (11500.0, 12000.0),
}
#: 11.5 keV excites no As K; its As windows hold scattered incident photons
GATED = {11.5: ("Ga_Ka", "Ga_Kb"), 15.0: tuple(WINDOWS), 30.0: tuple(WINDOWS)}
LINE_TOL = 0.05
MIN_COUNTS = 1000
FORMAT = "photon-alloy-fluorescence.v1"
#: Geant4 EM constructors; the gated reference is empenelope, others diagnose it
PHYSICS = ("empenelope", "emlivermore")


def name(energy_keV: float) -> str:
    return f"gaas_{energy_keV:g}kev".replace(".", "p")


def composition():
    """Ga and As atoms per cubic Angstrom (1 cm^3 = 1e24 A^3)."""
    units = DENSITY_G_CM3 * AVOGADRO / sum(ATOMIC_WEIGHT.values()) / 1e24
    return [("Ga", units), ("As", units)]


def macro(energy_keV: float, events: int, seed: int, physics: str = "empenelope") -> str:
    if physics not in PHYSICS:
        raise ValueError(f"unknown Geant4 physics {physics!r}")
    return f"""/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds {seed} 341
/testem/det/setAbsMat {MATERIAL}
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


def tally(event, energy, position, direction, direct, events):
    """Raw spectra in ``BLOCKS`` equal history blocks, keeping zero-hit blocks."""
    if events < BLOCKS or events % BLOCKS:
        raise ValueError(f"events must be a positive multiple of {BLOCKS}")
    event = np.asarray(event)
    if np.any(event < 0) or np.any(event >= events):
        raise ValueError("photon event id outside the declared history range")
    block = event * BLOCKS // events
    index = np.searchsorted(EDGES_EV, energy, side="right") - 1
    inside = (index >= 0) & (index < EDGES_EV.size - 1)
    out = {}
    for label, aperture in apertures(THICKNESS_MM).items():
        hit = hits(aperture, position, direction)
        out[label] = {}
        for kind, mask in {"total": hit, "direct": hit & direct, "redist": hit & ~direct}.items():
            counts = np.zeros((BLOCKS, EDGES_EV.size - 1), dtype=np.int64)
            selected = mask & inside
            np.add.at(counts, (block[selected], index[selected]), 1)
            out[label][kind] = counts.tolist()
    return dict(block_sizes=[events // BLOCKS] * BLOCKS, edges_eV=EDGES_EV.tolist(), apertures=out)


def indirect_ga_k_fraction(interactions, layer_elements) -> dict:
    """Ga photoabsorptions above the Ga K edge, split by primary/later flights.

    Later flights (generation > 0) are fluorescence or scattered photons; above
    the Ga K edge they are dominated by As K fluorescence once As K is excited.
    """
    from pyrite.montecarlo.transport.photon_navigation import PHOTON_CHANNEL_ORDER

    photo = PHOTON_CHANNEL_ORDER.index("photoelectric")
    ga = layer_elements.index("Ga")
    mask = (
        (np.asarray(interactions["element"]) == ga)
        & (np.asarray(interactions["channel"]) == photo)
        & (np.asarray(interactions["energy_eV"]) > GA_K_EDGE_EV)
    )
    later = mask & (np.asarray(interactions["generation"]) > 0)
    return dict(ga_above_k_edge=int(mask.sum()), indirect=int(later.sum()))


def run_pyrite(energy_keV: float, events: int, seed: int, first_history: int) -> dict:
    from pyrite.montecarlo.transport.photon_histories import track_photon_histories

    layer = composition()
    exits = {k: [] for k in ("history_id", "energy_eV", "position_ang", "direction", "generation")}
    counts = dict(ga_above_k_edge=0, indirect=0)
    for start in range(0, events, 1000):
        n = min(1000, events - start)
        ids = first_history + np.arange(start, start + n)
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
        for key in exits:
            exits[key].append(result.exits[key])
        diag = indirect_ga_k_fraction(result.interactions, [s for s, _ in layer])
        for key in counts:
            counts[key] += diag[key]
    rows = {k: np.concatenate(v) for k, v in exits.items()}
    record = tally(
        rows["history_id"] - first_history,
        rows["energy_eV"],
        rows["position_ang"],
        rows["direction"],
        rows["generation"] == 0,
        events,
    )
    record.update(
        engine="pyrite-cpu-photon-only",
        seed=seed,
        first_history=first_history,
        ga_k_excitation=counts,
    )
    return record


def verify_geant4_log(text: str, energy_keV: float, events: int) -> None:
    """Reject truncated runs, wrong primaries and a different target material."""
    match = re.search(r"The run was\s+(\d+) gamma of\s+([\d.eE+-]+) (\w+)", text)
    if match is None:
        raise ValueError("missing completed Geant4 run summary")
    count, energy, unit = match.groups()
    scale = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}.get(unit)
    if int(count) != events or scale is None or not np.isclose(float(energy) * scale, energy_keV):
        raise ValueError("Geant4 run summary does not match requested photons")
    if MATERIAL not in text:
        raise ValueError(f"Geant4 log does not name {MATERIAL}")


def reduce_geant4(log: Path, energy_keV: float, events: int, physics: str = "empenelope") -> dict:
    with gzip.open(log, "rt") as stream:
        verify_geant4_log(stream.read(), energy_keV, events)
    event, energy, position, direction, _, direct = _parse(
        [log], THICKNESS_MM, events, source_creator="primary"
    )
    record = tally(event, energy, position, direction, direct, events)
    record.update(
        engine=f"geant4-testem5-{physics}-cascade",
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    )
    return record


def merge(records: list[dict]) -> dict:
    """Concatenate independent shards' history blocks; metadata must agree."""
    if not records:
        raise ValueError("no records to merge")
    keys = ("format", "engine", "energy_keV", "edges_eV")
    for record in records[1:]:
        if any(record[k] != records[0][k] for k in keys):
            raise ValueError("shards disagree on format, engine, energy or edges")
    identities = [
        (r.get("seed"), r.get("first_history"), r.get("log_sha256"), json.dumps(r["apertures"]))
        for r in records
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate shard records")
    merged = {k: records[0][k] for k in keys}
    merged["block_sizes"] = [n for r in records for n in r["block_sizes"]]
    merged["apertures"] = {
        a: {
            kind: [row for r in records for row in r["apertures"][a][kind]]
            for kind in records[0]["apertures"][a]
        }
        for a in records[0]["apertures"]
    }
    merged["shards"] = [
        {k: r[k] for k in ("seed", "first_history", "log_sha256", "ga_k_excitation") if k in r}
        for r in records
    ]
    return merged


def window_counts(record: dict, aperture: str, kind: str, window: str) -> np.ndarray:
    edges = np.asarray(record["edges_eV"])
    lo, hi = WINDOWS[window]
    columns = (edges[:-1] >= lo) & (edges[1:] <= hi)
    if not columns.any():
        raise ValueError(f"no histogram bins inside {window}")
    return np.asarray(record["apertures"][aperture][kind])[:, columns].sum(axis=1)


def compare(pyrite: dict, geant4: dict) -> dict:
    """Gate every declared line window with enough Geant4 counts."""
    energy = float(geant4["energy_keV"])
    if float(pyrite["energy_keV"]) != energy or pyrite["edges_eV"] != geant4["edges_eV"]:
        raise ValueError("records differ in incident energy or histogram edges")
    report = {}
    for aperture in geant4["apertures"]:
        report[aperture] = {}
        for window in WINDOWS:
            g_counts = window_counts(geant4, aperture, "total", window)
            g = _mean_se(g_counts, geant4["block_sizes"])
            p = _mean_se(window_counts(pyrite, aperture, "total", window), pyrite["block_sizes"])
            gated = window in GATED[energy] and int(g_counts.sum()) >= MIN_COUNTS
            entry = _gate(p, g, LINE_TOL, gated=gated)
            entry["geant4_counts"] = int(g_counts.sum())
            report[aperture][window] = entry
    excitation = [s["ga_k_excitation"] for s in pyrite.get("shards", []) if "ga_k_excitation" in s]
    if excitation:
        total = sum(e["ga_above_k_edge"] for e in excitation)
        report["indirect_ga_k_fraction"] = (
            sum(e["indirect"] for e in excitation) / total if total else None
        )
    return report


def failures(report: dict) -> list[str]:
    return [
        f"{aperture}/{window}"
        for aperture, entries in report.items()
        if isinstance(entries, dict)
        for window, entry in entries.items()
        if entry["gated"] and not entry["pass_"]
    ]


def _load(path: Path) -> dict:
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def _write(path: Path, record: dict) -> None:
    text = json.dumps(record, sort_keys=True) + "\n"
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    m = sub.add_parser("macros", help="write Geant4 shard macros")
    m.add_argument("output", type=Path)
    m.add_argument("--events", type=int, default=1_000_000)
    m.add_argument("--shards", type=int, default=4)
    m.add_argument("--seed", type=int, default=34120)
    m.add_argument("--physics", choices=PHYSICS, default="empenelope")
    p = sub.add_parser("pyrite", help="run one PyRITE shard")
    p.add_argument("output", type=Path)
    p.add_argument("--energy-keV", type=float, choices=ENERGIES_KEV, required=True)
    p.add_argument("--events", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--seed", type=int, default=34121)
    g = sub.add_parser("reduce-geant4", help="reduce one Geant4 shard log")
    g.add_argument("output", type=Path)
    g.add_argument("--energy-keV", type=float, choices=ENERGIES_KEV, required=True)
    g.add_argument("--events", type=int, required=True)
    g.add_argument("--log", type=Path, required=True)
    g.add_argument("--physics", choices=PHYSICS, default="empenelope")
    j = sub.add_parser("merge", help="merge shard records")
    j.add_argument("output", type=Path)
    j.add_argument("inputs", type=Path, nargs="+")
    c = sub.add_parser("compare", help="gate PyRITE against Geant4 per energy")
    c.add_argument("directory", type=Path, help="holds <case>_{pyrite,geant4}.json.gz")
    c.add_argument("--output", type=Path)
    c.add_argument("--geant4", choices=PHYSICS, default="empenelope", help="reference arm")
    args = parser.parse_args()

    if args.action == "macros":
        args.output.mkdir(parents=True, exist_ok=True)
        for i, energy in enumerate(ENERGIES_KEV):
            for shard in range(args.shards):
                seed = args.seed + 100 * i + shard
                text = macro(energy, args.events, seed, args.physics)
                suffix = "" if args.physics == "empenelope" else f"_{args.physics}"
                (args.output / f"{name(energy)}_s{shard:02d}{suffix}.mac").write_text(text)
        return 0
    if args.action == "pyrite":
        record = run_pyrite(args.energy_keV, args.events, args.seed, args.shard * args.events)
    elif args.action == "reduce-geant4":
        record = reduce_geant4(args.log, args.energy_keV, args.events, args.physics)
    elif args.action == "merge":
        _write(args.output, merge([_load(path) for path in args.inputs]))
        return 0
    else:
        report = {}
        arm = "" if args.geant4 == "empenelope" else f"_{args.geant4}"
        for energy in ENERGIES_KEV:
            stem = args.directory / name(energy)
            report[name(energy)] = compare(
                _load(Path(f"{stem}_pyrite.json.gz")), _load(Path(f"{stem}_geant4{arm}.json.gz"))
            )
        failed = [f"{case}/{f}" for case, r in report.items() for f in failures(r)]
        report["failures"] = failed
        report["gated"] = sum(
            e["gated"]
            for r in report.values()
            if isinstance(r, dict)
            for entries in r.values()
            if isinstance(entries, dict)
            for e in entries.values()
        )
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return 1 if failed else 0
    record.update(format=FORMAT, energy_keV=args.energy_keV)
    _write(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
