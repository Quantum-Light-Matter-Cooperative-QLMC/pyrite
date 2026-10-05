"""Issue #317 acceptance: primary T/R vs Geant4 DPWA single scattering within 3 sigma.

Reads the PyRITE batch records written by ``run_issue317.sbatch`` (``prod_bench.py``
JSON, one per 10,000-primary batch; archived in ``results/issue317_records.json.gz``)
and the Geant4 TestEm5 DPWA shard logs in ``reference/<case>_ssdpwa.log.gz``, sums both to 100,000 primaries per case, and
prints transmitted/backscattered primary fractions with
``z = (p_PyRITE - p_G4) / sqrt(p1 q1 / N1 + p2 q2 / N2)``.

Usage:
  uv run python checks/full_track_bremslib/compare_issue317.py [RECORDS] [--output JSON]

Validation: inelastic-angular-deflection
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("si_300kev", "si_800kev", "w_300kev", "w_800kev")
_T = re.compile(r"primary particle transmitted = ([\d.]+) %")
_R = re.compile(r"primary particle reflected = ([\d.]+) %")


def geant4_dpwa(reference=HERE / "reference"):
    """Per case ``(N, n_transmitted, n_reflected)`` over the concatenated DPWA shard logs."""
    out = {}
    for case in CASES:
        text = gzip.open(reference / f"{case}_ssdpwa.log.gz", "rt").read()
        t, r = _T.findall(text), _R.findall(text)
        assert len(t) == len(r) == 10, case
        n = 10_000  # run_ssdpwa.sbatch: /run/beamOn 10000 per shard
        out[case] = (
            n * len(t),
            sum(round(float(x) * n / 100.0) for x in t),
            sum(round(float(x) * n / 100.0) for x in r),
        )
    return out


def load_records(source):
    """Batch records from a directory of ``prod_bench.py`` JSONs or a ``.json.gz`` list."""
    source = Path(source)
    if source.is_dir():
        return [
            json.loads(p.read_text())
            for p in sorted(source.glob("*.json"))
            if p.name != "summary.json"
        ]
    return json.loads(gzip.open(source, "rt").read())


def pyrite_records(records, inelastic="continuous", atomic="kawrakow"):
    """Per case ``(N, n_transmitted, n_backscattered, batches)``."""
    out = defaultdict(lambda: [0, 0, 0, []])
    for rec in records:
        if rec["inelastic"] != inelastic or rec.get("atomic", "none") != atomic:
            continue
        row = out[rec["case"]]
        row[0] += rec["Ne"]
        row[1] += rec["n_transmitted"]
        row[2] += rec["n_backscattered"]
        row[3].append(rec["first_batch"])
    return {case: tuple(row) for case, row in out.items()}


def z_score(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    return (p1 - p2) / ((p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "records",
        type=Path,
        nargs="?",
        default=HERE / "results" / "issue317_records.json.gz",
        help="directory of batch JSONs, or the archived .json.gz list (default)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    g4 = geant4_dpwa()
    records = load_records(args.records)
    summary = {}
    ok = True
    for label, inelastic in (("continuous", "continuous"), ("shell", "shell")):
        py = pyrite_records(records, inelastic)
        for case in CASES:
            if case not in py:
                continue
            n, t, r, batches = py[case]
            gn, gt, gr = g4[case]
            zt, zr = z_score(t, n, gt, gn), z_score(r, n, gr, gn)
            row = dict(
                N=n,
                batches=sorted(batches),
                T=t / n,
                R=r / n,
                g4_N=gn,
                g4_T=gt / gn,
                g4_R=gr / gn,
                z_T=zt,
                z_R=zr,
            )
            summary[f"{case}/{label}"] = row
            print(
                f"{case:10s} {label:10s} PyRITE T/R {t / n:.4f}/{r / n:.4f}  "
                f"G4 DPWA {gt / gn:.4f}/{gr / gn:.4f}  z {zt:+.1f}/{zr:+.1f}"
            )
            if label == "continuous":
                ok &= n == 100_000 and abs(zt) < 3.0 and abs(zr) < 3.0
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("ACCEPTANCE (continuous, 100k, |z| < 3):", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
