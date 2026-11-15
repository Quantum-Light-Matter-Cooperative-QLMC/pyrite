"""511 keV photon escape from thick W and Si slabs (issue #341).

Annihilation photons (m_e c^2 = 510.99895 keV) born in a target reach a
detector through photon transport: uncollided escape, Compton build-up,
Rayleigh redirection and, in W, K fluorescence after photoabsorption of
degraded photons. This check isolates that photon chain from positron
transport. Monoenergetic 511 keV photons are used in two source geometries on
the thick-target slabs of ``cases.CASES`` (W 4 mm, Si 25 mm; 5 cm footprint;
both ideal apertures):

- ``ext``: a normal-incidence pencil beam from 1 nm upstream of the entrance face;
- ``int``: a pencil source at mid-depth inside the slab, along +z, standing in
  for a photon born at an annihilation point.

Both engines kill charged secondaries at creation (PyRITE photon-only;
TestEm5 ``killSecondaries 3``). Geant4 uses complete vacancy cascades.

Gates were fixed before any PyRITE or Geant4 spectrum was inspected. The
pre-registered reference is Geant4 ``empenelope``; an ``emlivermore`` arm is
report-only. For each configuration and aperture, an observable holding at
least ``MIN_COUNTS`` reference counts passes when
``|P - G| <= max(tol * G, 3 sigma)``. Observables are uncollided (direct)
counts in 504-520 keV (tol 2%); redistributed counts in 10-100, 100-300
and 300-504 keV (tol 5%); and, in W, raw total counts in the W K-alpha
window 57-60 keV (tol 10%). ``sigma`` combines both engines' independent
history-block errors.

Validation: photon-annihilation-escape
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from cases import DENSITY, LATERAL_MM, MM, apertures, hits
from compare import _gate, _mean_se, _parse

AVOGADRO = 6.02214076e23
ENERGY_KEV = 510.99895
#: config -> (element, TestEm5 material, thickness mm, source)
CONFIGS = {
    "w_ext": ("W", "Tungsten", 4.0, "ext"),
    "w_int": ("W", "Tungsten", 4.0, "int"),
    "si_ext": ("Si", "Silicon", 25.0, "ext"),
    "si_int": ("Si", "Silicon", 25.0, "int"),
}
PHYSICS = ("empenelope", "emlivermore")
BLOCKS = 100
EDGES_EV = np.concatenate(
    [np.arange(1.0e3, 100.0e3, 500.0), np.arange(100.0e3, 520.0e3 + 1.0, 2.0e3)]
)
#: (name, label, low eV, high eV, tolerance, elements or None for all)
OBSERVABLES = (
    ("direct_511", "direct", 504.0e3, 520.0e3, 0.02, None),
    ("redist_10-100keV", "redist", 10.0e3, 100.0e3, 0.05, None),
    ("redist_100-300keV", "redist", 100.0e3, 300.0e3, 0.05, None),
    ("redist_300-504keV", "redist", 300.0e3, 504.0e3, 0.05, None),
    ("W_Ka", "total", 57.0e3, 60.0e3, 0.10, ("W",)),
)
MIN_COUNTS = 1000
FORMAT = "photon-annihilation-escape.v1"


def source_mm(config: str) -> float:
    """Source depth in PyRITE's frame (slab from z = 0 to z = T)."""
    _, _, thickness, kind = CONFIGS[config]
    return -1e-6 if kind == "ext" else 0.5 * thickness


def composition(element: str):
    rho, atomic_weight = DENSITY[element]
    return [(element, rho * AVOGADRO / atomic_weight / 1e24)]


def macro(config: str, events: int, seed: int, physics: str = "empenelope") -> str:
    if physics not in PHYSICS:
        raise ValueError(f"unknown Geant4 physics {physics!r}")
    _, material, thickness, _ = CONFIGS[config]
    x_mm = source_mm(config) - 0.5 * thickness  # TestEm5 absorber spans -T/2..T/2 on x
    return f"""/control/verbose 0
/run/verbose 0
/run/numberOfThreads 1
/random/setSeeds {seed} 341
/testem/det/setAbsMat {material}
/testem/det/setAbsThick {thickness:g} mm
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
/gun/energy {ENERGY_KEV} keV
/gun/position {x_mm:.7f} 0 0 mm
/gun/direction 1 0 0
/testem/stack/killSecondaries 3
/run/printProgress 0
/run/beamOn {events}
"""


def tally(config, event, energy, position, direction, direct, events):
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
    for name, aperture in apertures(CONFIGS[config][2]).items():
        hit = hits(aperture, position, direction)
        out[name] = {}
        for kind, mask in {"total": hit, "direct": hit & direct, "redist": hit & ~direct}.items():
            counts = np.zeros((BLOCKS, EDGES_EV.size - 1), dtype=np.int64)
            selected = mask & inside
            np.add.at(counts, (block[selected], index[selected]), 1)
            out[name][kind] = counts.tolist()
    return dict(
        config=config,
        block_sizes=[events // BLOCKS] * BLOCKS,
        edges_eV=EDGES_EV.tolist(),
        apertures=out,
    )


def run_pyrite(config: str, events: int, seed: int, first_history: int) -> dict:
    from pyrite.montecarlo.transport.photon_histories import track_photon_histories

    element, _, thickness, _ = CONFIGS[config]
    layer = composition(element)
    exits = {k: [] for k in ("history_id", "energy_eV", "position_ang", "direction", "generation")}
    for start in range(0, events, 1000):
        n = min(1000, events - start)
        ids = first_history + np.arange(start, start + n)
        result = track_photon_histories(
            np.tile([0.0, 0.0, source_mm(config) * MM], (n, 1)),
            np.tile([0.0, 0.0, 1.0], (n, 1)),
            np.full(n, ENERGY_KEV * 1000.0),
            np.ones(n),
            ids,
            np.zeros(n, dtype=np.int64),
            [(0.0, thickness * MM, layer)],
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
    rows = {k: np.concatenate(v) for k, v in exits.items()}
    record = tally(
        config,
        rows["history_id"] - first_history,
        rows["energy_eV"],
        rows["position_ang"],
        rows["direction"],
        rows["generation"] == 0,
        events,
    )
    record.update(engine="pyrite-cpu-photon-only", seed=seed, first_history=first_history)
    return record


def verify_geant4_log(text: str, config: str, events: int) -> None:
    """Reject truncated runs, wrong primaries and a different target material."""
    match = re.search(r"The run was\s+(\d+) gamma of\s+([\d.eE+-]+) (\w+)", text)
    if match is None:
        raise ValueError("missing completed Geant4 run summary")
    count, energy, unit = match.groups()
    scale = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}.get(unit)
    if (
        int(count) != events
        or scale is None
        or not np.isclose(float(energy) * scale, ENERGY_KEV, rtol=1e-5)
    ):
        raise ValueError("Geant4 run summary does not match requested photons")
    if f" of {CONFIGS[config][1]}" not in text:
        raise ValueError(f"Geant4 log does not name {CONFIGS[config][1]}")


def reduce_geant4(log: Path, config: str, events: int, physics: str = "empenelope") -> dict:
    with gzip.open(log, "rt") as stream:
        verify_geant4_log(stream.read(), config, events)
    event, energy, position, direction, _, direct = _parse(
        [log], CONFIGS[config][2], events, source_creator="primary"
    )
    record = tally(config, event, energy, position, direction, direct, events)
    record.update(
        engine=f"geant4-testem5-{physics}-cascade",
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    )
    return record


def merge(records: list[dict]) -> dict:
    """Concatenate independent shards' history blocks; metadata must agree."""
    if not records:
        raise ValueError("no records to merge")
    keys = ("format", "engine", "config", "edges_eV")
    for record in records[1:]:
        if any(record[k] != records[0][k] for k in keys):
            raise ValueError("shards disagree on format, engine, config or edges")
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
        {k: r[k] for k in ("seed", "first_history", "log_sha256") if k in r} for r in records
    ]
    return merged


def window_counts(record, aperture, label, lo, hi) -> np.ndarray:
    edges = np.asarray(record["edges_eV"])
    columns = (edges[:-1] >= lo) & (edges[1:] <= hi)
    if not columns.any() or not np.isclose(edges[:-1][columns].min(), lo):
        raise ValueError(f"window [{lo}, {hi}) does not align with histogram edges")
    if not np.isclose(edges[1:][columns].max(), hi):
        raise ValueError(f"window [{lo}, {hi}) does not align with histogram edges")
    return np.asarray(record["apertures"][aperture][label])[:, columns].sum(axis=1)


def compare(pyrite: dict, geant4: dict) -> dict:
    """Gate every declared observable with enough reference counts."""
    config = geant4["config"]
    if pyrite["config"] != config or pyrite["edges_eV"] != geant4["edges_eV"]:
        raise ValueError("records differ in configuration or histogram edges")
    element = CONFIGS[config][0]
    report = {}
    for aperture in geant4["apertures"]:
        report[aperture] = {}
        for name, label, lo, hi, tol, elements in OBSERVABLES:
            if elements is not None and element not in elements:
                continue
            g_counts = window_counts(geant4, aperture, label, lo, hi)
            g = _mean_se(g_counts, geant4["block_sizes"])
            p = _mean_se(window_counts(pyrite, aperture, label, lo, hi), pyrite["block_sizes"])
            entry = _gate(p, g, tol, gated=int(g_counts.sum()) >= MIN_COUNTS)
            entry["geant4_counts"] = int(g_counts.sum())
            report[aperture][name] = entry
    return report


def failures(report: dict) -> list[str]:
    return [
        f"{aperture}/{name}"
        for aperture, entries in report.items()
        for name, entry in entries.items()
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
    m.add_argument("--events", type=int, default=100_000)
    m.add_argument("--shards", type=int, default=10)
    m.add_argument("--seed", type=int, default=34140)
    m.add_argument("--physics", choices=PHYSICS, default="empenelope")
    p = sub.add_parser("pyrite", help="run one PyRITE shard")
    p.add_argument("output", type=Path)
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--events", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--seed", type=int, default=34141)
    g = sub.add_parser("reduce-geant4", help="reduce one Geant4 shard log")
    g.add_argument("output", type=Path)
    g.add_argument("--config", choices=CONFIGS, required=True)
    g.add_argument("--events", type=int, required=True)
    g.add_argument("--log", type=Path, required=True)
    g.add_argument("--physics", choices=PHYSICS, default="empenelope")
    j = sub.add_parser("merge", help="merge shard records")
    j.add_argument("output", type=Path)
    j.add_argument("inputs", type=Path, nargs="+")
    c = sub.add_parser("compare", help="gate PyRITE against a Geant4 arm")
    c.add_argument("directory", type=Path, help="holds <config>_{pyrite,geant4}.json.gz")
    c.add_argument("--output", type=Path)
    c.add_argument("--geant4", choices=PHYSICS, default="empenelope", help="reference arm")
    args = parser.parse_args()

    if args.action == "macros":
        args.output.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.physics == "empenelope" else f"_{args.physics}"
        for i, config in enumerate(CONFIGS):
            for shard in range(args.shards):
                text = macro(config, args.events, args.seed + 100 * i + shard, args.physics)
                (args.output / f"{config}_s{shard:02d}{suffix}.mac").write_text(text)
        return 0
    if args.action == "pyrite":
        record = run_pyrite(args.config, args.events, args.seed, args.shard * args.events)
    elif args.action == "reduce-geant4":
        record = reduce_geant4(args.log, args.config, args.events, args.physics)
    elif args.action == "merge":
        _write(args.output, merge([_load(path) for path in args.inputs]))
        return 0
    else:
        arm = "" if args.geant4 == "empenelope" else f"_{args.geant4}"
        report = {
            config: compare(
                _load(args.directory / f"{config}_pyrite.json.gz"),
                _load(args.directory / f"{config}_geant4{arm}.json.gz"),
            )
            for config in CONFIGS
        }
        failed = [f"{config}/{f}" for config, r in report.items() for f in failures(r)]
        gated = sum(e["gated"] for r in report.values() for a in r.values() for e in a.values())
        report = dict(configs=report, failures=failed, gated=gated, reference=args.geant4)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return 1 if failed else 0
    record.update(format=FORMAT)
    _write(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
