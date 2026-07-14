"""``cxr`` command-line entry point.

A single console script with subcommands, wired in pyproject.toml as
``cxr = "cxr_mc.cli:main"``:

    cxr scan <material> [--quick] [--workers N]   # run a sweep -> checkpoint
    cxr export [stem]                             # analysis app -> results/<stem>.html
    cxr analyze [-d/--default] [<material>] [--watch] [--edit]  # launch the analysis app
    cxr check [--watch] [--edit]                  # launch the validation app
    cxr check-config [manifest]                    # validate a full material catalog
    cxr slim <checkpoint> [--grid] [--drop-wide-brem] [--downcast]  # shrink a pkl for transfer
    cxr archive <stem> [label]                    # copy active checkpoint to the shelf
    cxr restore <label> [--as <stem>]             # copy a shelved checkpoint back
    cxr archives                                  # list the shelf
    cxr union <stem> <label>                      # merge a shelved checkpoint into active
    cxr remote <subcommand> ...                   # [dev, optional] run sweeps on a remote GPU box
"""

import argparse
import sys

from . import __version__


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="cxr",
        description="Coherent X-ray radiation (PXR + coherent bremsstrahlung) toolkit.",
    )
    ap.add_argument("--version", action="version", version=f"cxr-mc {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    if raw_argv[:1] == ["check-config"]:
        from . import check_config

        check_config.add_subparser(sub)
    else:
        from . import analyze, archive, check, check_config, export, remote, scan, slim

        scan.add_subparser(sub)
        export.add_subparser(sub)
        analyze.add_subparser(sub)
        slim.add_subparser(sub)
        archive.add_subparser(sub)
        remote.add_subparser(sub)
        check.add_subparser(sub)
        check_config.add_subparser(sub)

    args = ap.parse_args(raw_argv)
    return args.func(args)


if __name__ == "__main__":
    main()
