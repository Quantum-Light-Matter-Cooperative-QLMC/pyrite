"""Stage PyRITE's Zenodo table record, then pin its published file URLs.

The record (concept DOI 10.5281/zenodo.23220627) holds one zip and one
``sha256sum`` file per upstream code; see :mod:`pyrite.xsgen.bundles`.

Usage::

    # 1. Install the pinned datasets, then stage the upload set.
    pixi run pyrite tables fetch
    pixi run python scripts/release_table_bundles.py \\
        build --version 2 --archives build/xsgen-release

    # 2. In Zenodo, choose "New version" on the table record, upload every
    #    file in build/zenodo-tables/ except table-bundles.json and
    #    zenodo-metadata.json, apply the metadata, and publish.

    # 3. Pin the published record and check its bytes.
    pixi run python scripts/release_table_bundles.py \\
        pin --record 12345678

``build`` reuses local copies of the pinned table archives from
``--archives`` (published names) and downloads the rest, checking each pinned
digest. ``pin`` writes ``src/pyrite/data/xsgen/table-bundles.json`` and
re-pins the ELSEPA, SBETHE and BremsLib indexes to their bundles, after
downloading every bundle back from Zenodo and checking it.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from pyrite.xsgen.bundles import (
    ZENODO_FILE_URL,
    build_bundles,
    deposit_metadata,
    pin_bundles,
    render_readme,
    verify_published,
)

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "build" / "zenodo-tables"


def _creators() -> list[dict]:
    """Creators from ``.zenodo.json``: the PyRITE authors who prepared the set."""
    return json.loads((_REPO / ".zenodo.json").read_text(encoding="utf-8"))["creators"]


def _build(args: argparse.Namespace) -> int:
    index = build_bundles(args.out, archives=args.archives)
    shutil.copyfile(_REPO / "THIRD-PARTY-NOTICES.md", args.out / "THIRD-PARTY-NOTICES.md")
    (args.out / "README.md").write_text(render_readme(index, args.version), encoding="utf-8")
    (args.out / "zenodo-metadata.json").write_text(
        json.dumps(deposit_metadata(args.version, _creators()), indent=2) + "\n",
        encoding="utf-8",
    )
    for name, entry in index["bundles"].items():
        print(f"{name:9s} {entry['bytes'] / 1e6:7.1f} MB  {entry['sha256']}")
    print(f"staged:   {args.out}")
    return 0


def _pin(args: argparse.Namespace) -> int:
    staged = json.loads((args.out / "table-bundles.json").read_text(encoding="utf-8"))
    probe = json.loads(json.dumps(staged))
    for entry in probe["bundles"].values():
        entry["urls"] = [ZENODO_FILE_URL.format(record=args.record, name=entry["file"])]
    if not args.skip_download_check:
        with tempfile.TemporaryDirectory() as work:
            problems = verify_published(probe, work)
        if problems:
            print("published bundles do not verify:", *problems, sep="\n- ", file=sys.stderr)
            return 1
    pin_bundles(staged, args.record)
    print(f"pinned record {args.record}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=_OUT, help="staging directory")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the upload set")
    build.add_argument("--version", required=True, help="table set version label")
    build.add_argument("--archives", type=Path, help="local copies of the pinned table archives")
    pin = sub.add_parser("pin", help="pin the published record's URLs")
    pin.add_argument("--record", type=int, required=True, help="published Zenodo record id")
    pin.add_argument(
        "--skip-download-check", action="store_true", help="pin without downloading back"
    )
    args = parser.parse_args(argv)
    return _build(args) if args.command == "build" else _pin(args)


if __name__ == "__main__":
    raise SystemExit(main())
