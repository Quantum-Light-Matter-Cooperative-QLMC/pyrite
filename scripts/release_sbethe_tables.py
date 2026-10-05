"""Build the SBETHE catalogue table release and, optionally, pin it in the wheel.

With ``--projectile positron``, the archive and index names include
``-positron`` and ``--pin`` writes the separate positron index.

Maintainer-only: needs gfortran and the fetched SBETHE ``sdbase`` tree for any
table not already in the user table store. Writes ``sbethe-tables.zip`` and
``sbethe-tables.json`` into ``--out``. Publish the zip at ``--url`` and commit
the JSON as the shipped index -- ``--pin`` copies it into
``src/pyrite/data/xsgen/`` -- so that ``pyrite tables fetch sbethe-tables``
installs exactly these tables.

Usage::

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python scripts/release_sbethe_tables.py \\
        --generate --url https://.../sbethe-tables.zip --pin

Run it when the catalogue gains a material or medium, or when a composition,
the vendored SBETHE source, or the deck changes. Each re-keys tables, and the
changed index is what makes existing installs fetch again.
"""

import argparse
import shutil
import sys
from pathlib import Path

from pyrite.xsgen.sbethe.release import build_release, release_index_path

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
    parser.add_argument(
        "--projectile",
        choices=("electron", "positron"),
        default="electron",
        help="table species (default: electron); positrons use separate archive/index names",
    )
    parser.add_argument("--generate", action="store_true", help="generate absent tables")
    parser.add_argument(
        "--material", action="append", default=None, help="restrict to catalogue key (repeatable)"
    )
    parser.add_argument("--pin", action="store_true", help="copy the index into the package")
    args = parser.parse_args(argv)

    archive, index = build_release(
        args.out,
        urls=args.url,
        generate=args.generate,
        keys=args.material,
        projectile=args.projectile,
    )
    print(f"archive: {archive} ({index.archive_bytes / 1e6:.1f} MB)")
    print(f"sha256:  {index.archive_sha256}")
    print(f"tables:  {len(index.tables)}")
    if args.pin:
        pinned = release_index_path(projectile=args.projectile)
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
