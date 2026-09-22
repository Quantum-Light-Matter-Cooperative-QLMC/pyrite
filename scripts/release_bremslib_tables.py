"""Build the BremsLib-derived table release and, optionally, pin it in the wheel.

Maintainer-only: needs a BremsLib checkout (the 810 MB library). Writes
``bremslib-tables.zip`` and ``bremslib-tables.json`` into ``--out``. Publish
the zip at ``--url`` and commit the JSON as the shipped index -- ``--pin``
copies it into ``src/pyrite/data/xsgen/`` -- so that
``pyrite tables fetch bremslib`` installs exactly these tables.

Usage::

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python scripts/release_bremslib_tables.py \\
        --source ../BremsLib_v2.0.8 --url https://.../bremslib-tables.zip --pin

Run it when the catalogue admits a new transport element, when the BremsLib
deposit versions, or when the release transform
(:data:`pyrite.xsgen.bremslib.release.VARIANT`) changes. Each changes the
index, and the changed index is what makes existing installs fetch again.
"""

import argparse
import shutil
import sys
from pathlib import Path

from pyrite.xsgen.bremslib.read import COMPLETE_T1_MAX_MEV
from pyrite.xsgen.bremslib.release import build_release, release_index_path

_REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=None, help="BremsLib checkout")
    parser.add_argument(
        "--out", type=Path, default=_REPO / "build" / "xsgen-release", help="output directory"
    )
    parser.add_argument("--url", default=None, help="where the archive will be published")
    parser.add_argument(
        "--t1-max", type=float, default=COMPLETE_T1_MAX_MEV, help="energy bound in MeV"
    )
    parser.add_argument(
        "--element", type=int, action="append", default=None, help="restrict to Z (repeatable)"
    )
    parser.add_argument("--pin", action="store_true", help="copy the index into the package")
    args = parser.parse_args(argv)

    archive, index = build_release(
        args.out,
        source_path=args.source,
        elements=args.element,
        t1_max_MeV=args.t1_max,
        url=args.url,
    )
    print(f"archive: {archive} ({index.archive_bytes / 1e6:.1f} MB)")
    print(f"sha256:  {index.archive_sha256}")
    print(f"tables:  {len(index.tables)} (Z = {', '.join(str(e.z) for e in index.tables)})")
    if args.pin:
        pinned = release_index_path()
        if not pinned.resolve().is_relative_to(_REPO):
            print(f"refusing to pin outside the checkout: {pinned}", file=sys.stderr)
            return 1
        shutil.copyfile(archive.with_suffix(".json"), pinned)
        print(f"pinned:  {pinned}")
    if index.url is None:
        print("no --url given: the index pins the digest only; install with --archive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
