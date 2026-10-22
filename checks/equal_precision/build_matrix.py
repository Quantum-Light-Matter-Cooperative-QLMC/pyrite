"""Expand the frozen #510 parameter matrix; never estimates a missing range."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from pyrite.perf.benchmark_matrix import (
    RANGE_COLUMNS,
    coverage_table,
    expand_matrix,
    parse_range_table,
)

HERE = Path(__file__).parent
REPO = HERE.parents[1]


def estar_to_csv(text):
    """Keep ESTAR text-output columns 1 (MeV) and 5 (CSDA range, g/cm^2).

    Only the full seven-column table is accepted: omitting a stopping-power
    column would silently shift CSDA range out of column 5.
    """
    if "CSDA" not in text:
        raise ValueError("ESTAR text has no CSDA column header")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(RANGE_COLUMNS)
    for line in text.splitlines():
        fields = line.split()
        try:
            [float(v) for v in fields]
        except ValueError:
            continue
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError(f"ESTAR row needs all seven columns, space-delimited: {line!r}")
        writer.writerow([fields[0], fields[4]])
    return out.getvalue()


def pinned_estar_ranges(spec, spec_dir):
    """Return ``MATERIAL=PATH`` items for the spec's committed ESTAR files."""
    files = spec["range_source"].get("estar_files", {})
    return [f"{material}={spec_dir / path}" for material, path in files.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=HERE / "matrix-spec.json")
    parser.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="MATERIAL=PATH",
        help="pinned CSDA-range CSV, or raw ESTAR text with --estar; "
        "default: the spec's committed ESTAR files",
    )
    parser.add_argument("--estar", action="store_true", help="range files are ESTAR text output")
    parser.add_argument("--output", type=Path, default=HERE / "matrix.json")
    args = parser.parse_args()
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    estar = args.estar or not args.range
    items = args.range or pinned_estar_ranges(spec, args.spec.parent)
    ranges, sources = {}, {}
    for item in items:
        material, _, path = item.partition("=")
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        table = parse_range_table(estar_to_csv(text) if estar else text)
        ranges[material] = table
        sources[material] = {
            "path": Path(path).resolve().relative_to(REPO).as_posix(),
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_csv_sha256": table["sha256"],
        }
    cases = expand_matrix(spec, ranges)
    ids = {case["id"] for case in cases}
    missing = [i for i in spec["interaction_checks"]["angle_x_thickness"] if i not in ids]
    if missing:
        raise SystemExit(f"interaction checks name unknown cases: {missing}")
    matrix = {
        "schema": "pyrite.equal-precision.matrix.v1",
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "range_sources": sources,
        "detector_frame": spec["detector_frame"],
        "n_cases": len(cases),
        "status_counts": {
            s: sum(c["status"] == s for c in cases) for s in sorted({c["status"] for c in cases})
        },
        "coverage": coverage_table(cases),
        "cases": cases,
    }
    args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases -> {args.output} {matrix['status_counts']}")


if __name__ == "__main__":
    main()
