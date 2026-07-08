"""``cxr archive`` / ``restore`` / ``archives`` -- a durable local shelf for
checkpoints (checkpoint lifecycle, component 2).

The GPU box is scratch compute; the laptop is the durable store. A grid-filtered
pull lands in the ACTIVE slot ``checkpoints/<stem>.pkl`` (exactly what
``analysis.ipynb`` loads). These commands add a two-tier model on top of that:

    checkpoints/<stem>.pkl            active slot (viz loads this)
    checkpoints/archive/<label>.pkl   long-term shelf (named snapshots you keep)

    cxr archive hopg                  # -> archive/hopg-20260704.pkl
    cxr archive hopg good-thickness   # -> archive/good-thickness.pkl
    cxr restore hopg-20260704         # -> checkpoints/hopg.pkl (stem inferred)
    cxr restore good-thickness --as hopg
    cxr archives                      # list the shelf

All operations are pure local file copies (atomic temp+replace), so this lives on
the ``cxr`` console script next to ``slim`` -- library-side and unit-testable --
not in ``dev/remote.py``. The shelf stores whatever the active slot holds at the
time, typically the grid-filtered view.
"""

import argparse
import datetime
import os
import re
import shutil
from pathlib import Path

from . import _checkpoint_io

# Anchored to the repo root (src/cxr_mc/archive.py -> parents[2] = repo root), the
# same dir run.load_checkpoint reads, so `cxr archive` works from any cwd. Computed
# here rather than imported from run to keep the montecarlo import chain (and its
# GPU-detection banner) out of a plain archive command.
DEFAULT_ROOT = str(Path(__file__).resolve().parents[2] / "checkpoints")
ARCHIVE_SUBDIR = "archive"
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")  # a trailing -YYYYMMDD default-label stamp


def _archive_dir(root):
    return os.path.join(root, ARCHIVE_SUBDIR)


def _default_label(stem):
    """Default archive label for a stem: ``<stem>-<YYYYMMDD>`` (today)."""
    return f"{stem}-{datetime.date.today():%Y%m%d}"


def _stem_from_label(label):
    """Infer the active-slot stem a label restores to: strip a trailing
    ``-<YYYYMMDD>`` default-label stamp, else use the label verbatim."""
    return _DATE_SUFFIX_RE.sub("", label)


def _atomic_copy(src, dst):
    """Copy ``src`` -> ``dst`` via a temp + ``os.replace``, so a reader never sees
    a half-written pickle and an interrupted copy can't corrupt an existing file."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def archive_checkpoint(stem, label=None, *, force=False, root=DEFAULT_ROOT):
    """Copy the active slot ``<root>/<stem>.pkl`` to ``<root>/archive/<label>.pkl``.
    ``label`` defaults to ``<stem>-<YYYYMMDD>``. Refuses to overwrite an existing
    label without ``force``. Returns the archive path."""
    src = os.path.join(root, f"{stem}.pkl")
    if not os.path.isfile(src):
        raise SystemExit(f"no such active checkpoint: {src}")
    label = label or _default_label(stem)
    dst = os.path.join(_archive_dir(root), f"{label}.pkl")
    if os.path.exists(dst) and not force:
        raise SystemExit(f"archive already exists: {dst} (pass --force to overwrite)")
    _atomic_copy(src, dst)
    print(f"archived checkpoints/{stem}.pkl -> {ARCHIVE_SUBDIR}/{label}.pkl")
    return dst


def restore_checkpoint(label, stem=None, *, force=False, root=DEFAULT_ROOT):
    """Copy ``<root>/archive/<label>.pkl`` back to the active slot
    ``<root>/<stem>.pkl``. ``stem`` defaults to the label with a trailing
    ``-<YYYYMMDD>`` stripped. Refuses to overwrite an existing active slot without
    ``force``. Returns the active-slot path."""
    src = os.path.join(_archive_dir(root), f"{label}.pkl")
    if not os.path.isfile(src):
        raise SystemExit(f"no such archive: {src}")
    stem = stem or _stem_from_label(label)
    dst = os.path.join(root, f"{stem}.pkl")
    if os.path.exists(dst) and not force:
        raise SystemExit(f"active checkpoint already exists: {dst} (pass --force to overwrite)")
    _atomic_copy(src, dst)
    print(f"restored {ARCHIVE_SUBDIR}/{label}.pkl -> checkpoints/{stem}.pkl")
    return dst


def _record_count(path):
    """Total records in an archived checkpoint (``sum(len(by_E) ...)``), or None
    if it can't be read as a results store. Reads via ``_checkpoint_io.load``
    (TODO P2 #8), which transparently handles both gzip-compressed and legacy
    plain-pickle archives -- the shelf can hold either, since ``archive``/
    ``restore`` just copy whatever bytes the active slot already has."""
    try:
        results = _checkpoint_io.load(path)
        return sum(len(v) for v in results.values())
    except Exception:
        return None


def list_archives(root=DEFAULT_ROOT):
    """Print the shelf: each label with its size (MB) and record count, sorted by
    label. Returns the list of labels."""
    adir = _archive_dir(root)
    if not os.path.isdir(adir):
        print("(no archives)")
        return []
    labels = sorted(f[:-4] for f in os.listdir(adir) if f.endswith(".pkl"))
    if not labels:
        print("(no archives)")
        return []
    for label in labels:
        path = os.path.join(adir, f"{label}.pkl")
        mb = os.path.getsize(path) / 1e6
        n = _record_count(path)
        n_str = "?" if n is None else str(n)
        print(f"  {label:<32}  {mb:7.1f} MB  {n_str:>6} records")
    return labels


def _cli_archive(args):
    archive_checkpoint(args.stem, args.label, force=args.force)


def _cli_restore(args):
    restore_checkpoint(args.label, args.stem, force=args.force)


def _cli_archives(args):
    list_archives()


def add_subparser(sub):
    """Register the ``archive`` / ``restore`` / ``archives`` subcommands on an
    argparse subparsers object."""
    ap = sub.add_parser("archive", help="copy an active checkpoint to the long-term shelf")
    ap.add_argument("stem", help="active checkpoint stem, e.g. hopg")
    ap.add_argument("label", nargs="?", default=None, help="archive label (default: <stem>-<date>)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing archive label")
    ap.set_defaults(func=_cli_archive)

    rp = sub.add_parser("restore", help="copy an archived checkpoint back to the active slot")
    rp.add_argument("label", help="archive label to restore")
    rp.add_argument("--as", dest="stem", default=None, help="active stem (default: inferred)")
    rp.add_argument("--force", action="store_true", help="overwrite an existing active checkpoint")
    rp.set_defaults(func=_cli_restore)

    lp = sub.add_parser("archives", help="list the long-term shelf")
    lp.set_defaults(func=_cli_archives)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxr-archive", description="local checkpoint archive shelf")
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
