"""Build the ELSEPA elastic table release and, optionally, pin it in the wheel.

Maintainer-only: needs gfortran for any table not already in the user table
store. Writes ``elsepa-tables.zip`` and ``elsepa-tables.json`` into
``--out``. Publish the zip at ``--url`` and commit the JSON as the shipped
index -- ``--pin`` copies it into ``src/pyrite/data/xsgen/`` -- so that
``pyrite tables fetch elsepa`` installs exactly these tables.

Usage::

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python scripts/release_elsepa_tables.py \\
        --generate --url https://.../elsepa-tables.zip --pin

Run it when the catalogue admits a new transport element or elementary
crystal, or when the vendored ELSEPA source, the deck, or the production
energy grid changes. Each re-keys tables, and the changed index is what makes
existing installs fetch again.
"""

import argparse
import shutil
import sys
from pathlib import Path

from pyrite.xsgen.elsepa.release import build_release, release_index_path

_REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=_REPO / "build" / "xsgen-release", help="output directory"
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="where the archive will be published (repeatable, in fetch order)",
    )
    parser.add_argument("--generate", action="store_true", help="generate absent tables")
    parser.add_argument("--pin", action="store_true", help="copy the index into the package")
    args = parser.parse_args(argv)

    archive, index = build_release(args.out, urls=args.url, generate=args.generate)
    print(f"archive: {archive} ({index.archive_bytes / 1e6:.1f} MB)")
    print(f"sha256:  {index.archive_sha256}")
    print(f"tables:  {len(index.tables)} ({', '.join(e.label for e in index.tables)})")
    if args.pin:
        pinned = release_index_path()
        if not pinned.resolve().is_relative_to(_REPO):
            print(f"refusing to pin outside the checkout: {pinned}", file=sys.stderr)
            return 1
        shutil.copyfile(archive.with_suffix(".json"), pinned)
        print(f"pinned:  {pinned}")
    if not index.urls:
        print("no --url given: the index pins the digest only; install with --archive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
