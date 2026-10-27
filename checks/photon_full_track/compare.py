"""Reduce and compare the #341 thick-target photon full-track check.

Usage:
  compare.py reduce-geant4 CASE LOG.gz [LOG.gz ...] [--events N_PER_LOG]
  compare.py reduce-geant4-cascade CASE OUTPUT.json LOG.gz [...] [--events N_PER_LOG]
  compare.py compare-cascade CASE REFERENCE.json[.gz] [--output REPORT.json]
  compare.py reduce-pyrite CASE SHARD.npz [SHARD.npz ...]
  compare.py compact RECORD.json OUTPUT.json.gz
  compare.py [--output REPORT.json]

Geant4 logs hold one ``GAM`` line per photon leaving the vacuum world
(``patch_testem5.py``); each is mapped into PyRITE's frame and tested against
both apertures of ``cases.apertures``. Counts are binned on ``cases.edges_eV``
per incident electron and split into direct (no interaction) and
redistributed parts. Statistical errors come from 100 equal event blocks
(Geant4) or from batch-to-batch scatter (PyRITE), so photons correlated within
one history are not treated as independent. Reduced records go to
``reference/<case>.json`` and ``results/<case>.json``.

Gates and their post-inspection revision are documented in README. Per case
and aperture, an observable passes when ``|P - G| <= max(tol * G, 3 sigma)``.
"""

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from cases import CASES, MM, apertures, edges_eV

HERE = Path(__file__).resolve().parent
BLOCKS = 100
#: (name, low eV, high eV or None for E0, relative tolerance)
BANDS = (
    ("1-10keV", 1.0e3, 10.0e3, 0.10),
    ("10-30keV", 10.0e3, 30.0e3, 0.10),
    ("30-100keV", 30.0e3, 100.0e3, 0.07),
    ("100-300keV", 100.0e3, 300.0e3, 0.07),
    ("0.3-1MeV", 300.0e3, 1.0e6, 0.07),
    ("1-3MeV", 1.0e6, 3.0e6, 0.07),
    ("3-10MeV", 3.0e6, 10.0e6, 0.07),
)
#: W K-shell lines (eV): Ka2/Ka1 window and Kb window, net of a linear continuum
W_K_LINES = (("W_Ka", 57.0e3, 60.0e3), ("W_Kb", 66.5e3, 69.5e3))
REDIST_ABS_TOL = 0.02
REDIST_TOL = 0.10
MIN_COUNTS = 1000


def _parse(paths, thickness_mm, events_per_log, *, source_creator="eBrem"):
    """Shard ``k`` of ``paths`` holds events ``[k, k + 1) * events_per_log``."""
    rows = []
    for k, path in enumerate(paths):
        offset = k * events_per_log
        with gzip.open(path, "rt") as f:
            for line in f:
                i = line.find("GAM ")
                if i < 0:
                    continue
                p = line[i:].split()
                rows.append(
                    (offset + int(p[1]), float(p[2]), *map(float, p[3:9]), p[9], int(p[10]))
                )
    if not rows:
        raise SystemExit("no GAM records")
    event = np.array([r[0] for r in rows])
    energy = np.array([r[1] for r in rows])
    g4 = np.array([r[2:8] for r in rows])
    # TestEm5 (x, y, z) -> PyRITE (y, z, x + T/2); positions in Angstrom
    position = np.stack([g4[:, 1], g4[:, 2], g4[:, 0] + 0.5 * thickness_mm], axis=1) * MM
    direction = g4[:, [4, 5, 3]]
    creator = np.array([r[8] for r in rows])
    # PyRITE's "direct" is an uncollided *source* photon; fluorescence and other
    # secondary photons are redistributed even when they escape uncollided.
    direct = np.array([r[9] for r in rows], dtype=bool) & (creator == source_creator)
    return event, energy, position, direction, creator, direct


def reduce_geant4(case, paths, events_per_log):
    from cases import hits

    _, _, thickness_mm, energy_MeV = CASES[case]
    edges = edges_eV(energy_MeV)
    paths = sorted(paths)
    events = events_per_log * len(paths)
    event, energy, position, direction, creator, direct = _parse(
        paths, thickness_mm, events_per_log
    )
    if event.max() >= events:
        raise SystemExit(f"event id {event.max()} exceeds {events} events")
    block = event * BLOCKS // events
    record = dict(
        case=case,
        engine="geant4-11.4.2-testem5-empenelope",
        events=events,
        logs=[Path(p).name for p in paths],
        edges_eV=edges.tolist(),
        escaped_per_primary=energy.size / events,
        creators={c: int(n) for c, n in zip(*np.unique(creator, return_counts=True), strict=True)},
        apertures={},
    )
    for name, aperture in apertures(thickness_mm).items():
        hit = hits(aperture, position, direction)
        entry = dict(creators={})
        for label, mask in (("total", hit), ("direct", hit & direct), ("redist", hit & ~direct)):
            counts = np.zeros((BLOCKS, edges.size - 1))
            np.add.at(
                counts,
                (block[mask], np.clip(np.searchsorted(edges, energy[mask], "right") - 1, 0, None)),
                (energy[mask] >= edges[0]) & (energy[mask] < edges[-1]),
            )
            entry[label] = counts.tolist()  # per block, raw counts
        entry["creators"] = {
            c: int(n) for c, n in zip(*np.unique(creator[hit], return_counts=True), strict=True)
        }
        entry["detected_per_primary"] = float(hit.sum() / events)
        record["apertures"][name] = entry
    return record


def reduce_pyrite(case, paths):
    shards = [np.load(p) for p in paths]
    meta = [json.loads(str(s["meta"])) for s in shards]
    starts = [m["first_batch"] for m in meta]
    if len(set(starts)) != len(starts):
        raise SystemExit("duplicate shards")
    size = np.concatenate([s["batch_size"] for s in shards])
    edges = shards[0]["edges_eV"]
    widths = np.diff(edges)
    record = dict(
        case=case,
        engine="pyrite-photon-only-weighted-bremslib",
        events=int(size.sum()),
        edges_eV=edges.tolist(),
        meta=meta[0]
        | dict(
            first_batches=sorted(starts),
            transport_s=sum(m["transport_s"] for m in meta),
            photon_s=sum(m["photon_s"] for m in meta),
        ),
        apertures={},
    )
    for name in ("T", "R"):
        entry = {}
        for label, key in (("total", "total"), ("direct", "direct"), ("redist", "redist")):
            # photons/(primary eV) per batch -> expected counts per batch
            per = np.concatenate([s[f"{name}_{key}"] for s in shards]) * widths * size[:, None]
            entry[label] = per.tolist()
        entry["batch_size"] = size.tolist()
        entry["detected_per_primary"] = float(
            np.sum(np.concatenate([s[f"{name}_det"] for s in shards]) * size) / size.sum()
        )
        record["apertures"][name] = entry
    return record


def _counts(entry, label, lo, hi, edges):
    """Retain whole-bin selection, including in compact window records."""
    counts = np.asarray(entry[label])
    if "windows_eV" in entry:
        return counts[:, entry["windows_eV"].index([lo, hi])]
    sel = (edges[:-1] >= lo) & (edges[1:] <= hi)
    return counts[:, sel].sum(axis=1)


def _mean_se(per_block, sizes):
    """Ratio estimate over independent blocks; preserve within-block covariance."""
    n = np.asarray(sizes, float)
    mean = per_block.sum() / n.sum()
    rate = per_block / n
    # ratio estimator over unequal blocks
    var = np.sum(n**2 * (rate - mean) ** 2) / (n.sum() ** 2) * len(n) / max(len(n) - 1, 1)
    return mean, float(np.sqrt(var))


def _estimate(entry, label, sizes, lo, hi, edges):
    """Per-primary mean and standard error of counts in [lo, hi) from blocks."""
    return _mean_se(_counts(entry, label, lo, hi, edges), sizes)


def _fraction(entry, sizes, lo, hi, edges):
    """Delta-method SE of R/T, with paired numerator and denominator blocks."""
    redist = _counts(entry, "redist", lo, hi, edges)
    total = _counts(entry, "total", lo, hi, edges)
    fraction = redist.sum() / total.sum()
    _, residual_se = _mean_se(redist - fraction * total, sizes)
    return fraction, residual_se / (total.sum() / np.sum(sizes))


def _line(entry, label, sizes, edges, lo, hi):
    """Net line counts: window minus a linear continuum from equal side windows."""
    width = hi - lo
    peak = _counts(entry, label, lo, hi, edges)
    left = _counts(entry, label, lo - width / 2, lo, edges)
    right = _counts(entry, label, hi, hi + width / 2, edges)
    return _mean_se(peak - left - right, sizes)


def compact_record(record):
    """Lossless block projections for every reported band and line window.

    Fine spectral shape is discarded; all means, SEs and cross-window
    covariance used by this comparison are preserved. Source metadata stays.
    """
    top = CASES[record["case"]][3] * 1e6
    windows = [(lo, min(hi, top)) for _, lo, hi, _ in BANDS if lo < top]
    if CASES[record["case"]][0] == "W":
        for _, lo, hi in W_K_LINES:
            width = hi - lo
            windows.extend(((lo, hi), (lo - width / 2, lo), (hi, hi + width / 2)))
    edges = np.asarray(record["edges_eV"])
    entries = {}
    for name, entry in record["apertures"].items():
        entries[name] = entry | {
            label: np.column_stack(
                [_counts(entry, label, lo, hi, edges) for lo, hi in windows]
            ).tolist()
            for label in ("total", "direct", "redist")
        }
        entries[name]["windows_eV"] = [list(window) for window in windows]
    return record | dict(format="photon-full-track-windows.v1", apertures=entries)


def load_record(path):
    """Read an original reduction or its committed gzip window projection."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def _gate(p, g, tol, gated=True, abs_tol=None):
    """``|P - G| <= max(tol * G, abs_tol, 3 sigma)``; ``gated=False`` only reports."""
    sigma = float(np.hypot(p[1], g[1]))
    limit = max(tol * abs(g[0]), abs_tol or 0.0, 3 * sigma)
    return dict(
        pyrite=p,
        geant4=g,
        ratio=p[0] / g[0] if g[0] else None,
        sigma=sigma,
        limit=limit,
        gated=bool(gated),
        pass_=bool(abs(p[0] - g[0]) <= limit),
    )


def compare_case(case, *, reference=None):
    g = load_record(HERE / "reference" / f"{case}.json") if reference is None else reference
    if g["case"] != case:
        raise ValueError("reference case does not match requested comparison")
    p = load_record(HERE / "results" / f"{case}.json")
    edges = np.asarray(g["edges_eV"])
    if not np.array_equal(edges, p["edges_eV"]):
        raise SystemExit(f"{case}: histogram edges differ")
    top = CASES[case][3] * 1e6
    g_sizes = [g["events"] / BLOCKS] * BLOCKS
    report = {}
    for name in ("T", "R"):
        ge, pe = g["apertures"][name], p["apertures"][name]
        rows = {}
        for band, lo, hi, tol in BANDS:
            if lo >= top:
                continue
            hi = min(hi, top)
            est = {
                label: (
                    _estimate(pe, label, pe["batch_size"], lo, hi, edges),
                    _estimate(ge, label, g_sizes, lo, hi, edges),
                )
                for label in ("total", "direct", "redist")
            }
            (pt, gt), (pr, gr) = est["total"], est["redist"]
            if gt[0] * g["events"] < MIN_COUNTS:
                rows[band] = dict(skipped=f"fewer than {MIN_COUNTS} Geant4 counts")
                continue
            rows[band] = dict(
                # Totals/direct carry the BremsLib-vs-PenBrem source: reported only.
                total=_gate(pt, gt, tol, gated=False),
                direct=_gate(*est["direct"], tol, gated=False),
                redistributed=_gate(pr, gr, REDIST_TOL, gated=gr[0] * g["events"] >= MIN_COUNTS),
                redistributed_fraction=_gate(
                    _fraction(pe, pe["batch_size"], lo, hi, edges),
                    _fraction(ge, g_sizes, lo, hi, edges),
                    0.0,
                    abs_tol=REDIST_ABS_TOL,
                ),
            )
        if CASES[case][0] == "W" and top > 70e3:
            for line, lo, hi in W_K_LINES:
                gl = _line(ge, "total", g_sizes, edges, lo, hi)
                if gl[0] * g["events"] < MIN_COUNTS:
                    continue
                pl = _line(pe, "total", pe["batch_size"], edges, lo, hi)
                rows[line] = dict(net=_gate(pl, gl, 0.10))
        report[name] = rows
    return report


def verify_cascade_logs(case, paths, events_per_log):
    """Require completed primary histories and both cascade settings in every shard."""
    if events_per_log <= 0 or not paths or len(set(map(str, paths))) != len(paths):
        raise ValueError("expected nonduplicate logs and a positive event count")
    expected_energy = CASES[case][3] * 1e6
    for path in paths:
        summary, auger, ignore = None, False, False
        with gzip.open(path, "rt") as stream:
            for line in stream:
                match = re.search(r"The run was\s+(\d+) e- of\s+([\d.eE+-]+) (\w+)", line)
                if match:
                    summary = match.groups()
                auger |= bool(re.search(r"Auger electron cascade enabled\s+1\s*$", line))
                ignore |= bool(re.search(r"Ignore cuts flag:\s+1\s*$", line))
        if summary is None:
            raise ValueError(f"{path}: missing completed electron run summary")
        count, energy, unit = summary
        energy_eV = float(energy) * {"eV": 1, "keV": 1000, "MeV": 1e6}.get(unit, 0)
        if int(count) != events_per_log or energy_eV != expected_energy:
            raise ValueError(f"{path}: wrong primary count or energy")
        if not (auger and ignore):
            raise ValueError(f"{path}: complete vacancy cascades not enabled")


def _summary(report):
    for c, ap in report.items():
        for a, rows in ap.items():
            for b, row in rows.items():
                if "skipped" in row:
                    continue
                cells = []
                for key, v in row.items():
                    flag = ("ok" if v["pass_"] else "FAIL") if v["gated"] else "rep"
                    value = (
                        f"{v['pyrite'][0]:.3f}/{v['geant4'][0]:.3f}"
                        if key == "redistributed_fraction"
                        else f"{v['ratio']:.3f}"
                    )
                    cells.append(f"{key[:8]} {value} {flag}")
                print(f"{c:10s} {a} {b:11s} " + "  ".join(cells))


def main(argv):
    if argv and argv[0] == "compare-cascade":
        case, path = argv[1:3]
        reference = load_record(Path(path))
        if reference.get("relaxation") != "complete-cascade":
            raise ValueError("expected a verified complete-cascade reference")
        matched = compare_case(case, reference=reference)
        historical = compare_case(case)

        def gated(report):
            return {
                (a, b, k)
                for a, rows in report.items()
                for b, row in rows.items()
                for k, v in row.items()
                if isinstance(v, dict) and v["gated"]
            }

        missing = sorted(gated(historical) - gated(matched))
        failures = [
            (a, b, k)
            for a, rows in matched.items()
            for b, row in rows.items()
            for k, v in row.items()
            if isinstance(v, dict) and v["gated"] and not v["pass_"]
        ]
        record = dict(
            scope=case,
            historical=historical,
            matched=matched,
            failures=failures,
            gated_count=len(gated(matched)),
            missing_gates=missing,
        )
        if "--output" in argv:
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps(record, indent=2))
        _summary({case: matched})
        print(
            f"{len(failures)} failing, {len(missing)} missing gates (only {case}; other cases not rerun)"
        )
        return bool(failures or missing or not gated(matched))
    if argv and argv[0] == "compact":
        record = compact_record(json.loads(Path(argv[1]).read_text()))
        with gzip.GzipFile(argv[2], "wb", mtime=0) as stream:
            stream.write(json.dumps(record, separators=(",", ":")).encode())
        return 0
    if argv and argv[0] in ("reduce-geant4", "reduce-geant4-cascade"):
        events = None
        if "--events" in argv:
            i = argv.index("--events")
            events = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]
        cascade = argv[0] == "reduce-geant4-cascade"
        case = argv[1]
        paths = argv[3:] if cascade else argv[2:]
        from make_macros import EVENTS

        if cascade:
            verify_cascade_logs(case, paths, events or EVENTS)
        record = reduce_geant4(case, paths, events or EVENTS)
        out = Path(argv[2]) if cascade else HERE / "reference" / f"{case}.json"
        if cascade:
            digests = {}
            for path in paths:
                with Path(path).open("rb") as stream:
                    digests[Path(path).name] = hashlib.file_digest(stream, "sha256").hexdigest()
            record.update(relaxation="complete-cascade", log_sha256=digests)
    elif argv and argv[0] == "reduce-pyrite":
        record = reduce_pyrite(argv[1], argv[2:])
        out = HERE / "results" / f"{argv[1]}.json"
    else:
        output = argv[argv.index("--output") + 1] if "--output" in argv else None
        # Missing cases must fail rather than silently certify a partial run.
        report = {c: compare_case(c) for c in CASES}
        failures = [
            (c, a, b, k)
            for c, ap in report.items()
            for a, rows in ap.items()
            for b, row in rows.items()
            for k, v in row.items()
            if isinstance(v, dict) and v["gated"] and not v["pass_"]
        ]
        _summary(report)
        print(f"{len(failures)} failing gates")
        if output:
            Path(output).write_text(json.dumps(dict(report=report, failures=failures), indent=1))
        return 1 if failures else 0
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
