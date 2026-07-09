"""``cxr archive`` / ``restore`` / ``archives`` / ``union`` -- a durable local
shelf for checkpoints (checkpoint lifecycle, component 2).

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
    cxr union hopg good-thickness     # merge archive/good-thickness.pkl into
                                       # checkpoints/hopg.pkl (TODO P2 #8)

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


def _material_of(results):
    """The crystal/material shared by every record in a results store, read off
    the first record's ``case["crystal"]`` (same field ``run._crystal_of`` keys
    the per-material checkpoint split on). ``None`` for an empty store -- nothing
    to check the caller's material claim against."""
    for recs in results.values():
        for r in recs.values():
            return r["case"]["crystal"]
    return None


def _union_results(live, archived):
    """Merge ``archived`` into ``live`` at (config name, E0) granularity, LIVE
    winning on any (name, E0) collision.

    Records carry no run-id/timestamp (see
    ``docs/superpowers/specs/2026-07-04-checkpoint-lifecycle-design.md``, "Out of
    scope"), so there is no principled way to prefer one side over the other on
    overlap by recency -- the live checkpoint is what's currently in front of you
    (and, by default, the one just re-derived from the box), so it wins ties.
    Returns a NEW dict; does not mutate either input."""
    merged = {name: dict(recs) for name, recs in live.items()}
    for name, recs in archived.items():
        merged.setdefault(name, {})
        for E0, rec in recs.items():
            merged[name].setdefault(E0, rec)  # no-op if E0 already came from live
    return merged


def union_checkpoint(
    stem, label, *, pre_archive=True, delete_archive=False, force=False, root=DEFAULT_ROOT
):
    """Merge the archived checkpoint ``<root>/archive/<label>.pkl`` into the
    active slot ``<root>/<stem>.pkl``, in place. Live wins on any overlapping
    (config name, E0) point (see :func:`_union_results`).

    Refuses to union checkpoints for different materials (compared by each
    store's ``case["crystal"]``, see :func:`_material_of`) -- unioning e.g. hopg
    into mos2 would silently pollute a checkpoint with another crystal's records.

    By default archives the live checkpoint FIRST (via :func:`archive_checkpoint`,
    same default label), so the union is undoable via ``cxr restore``; pass
    ``pre_archive=False`` to skip it. ``force`` is forwarded to that pre-archive
    step's overwrite guard. The source archive is left intact by default; pass
    ``delete_archive=True`` to remove it once the union has landed. Returns the
    active-slot path.

    If the pre-archive step's default label (``<stem>-<today>``) would collide
    with the archive being unioned in, refuses outright -- even with ``force``
    -- rather than letting the pre-union backup silently overwrite the very
    archive the union is reading from (that archive is meant to be left intact
    by default). Pass ``pre_archive=False`` to union same-day round-trips like
    this, or archive the live checkpoint under an explicit label first.
    """
    live_path = os.path.join(root, f"{stem}.pkl")
    if not os.path.isfile(live_path):
        raise SystemExit(f"no such active checkpoint: {live_path}")
    archive_path = os.path.join(_archive_dir(root), f"{label}.pkl")
    if not os.path.isfile(archive_path):
        raise SystemExit(f"no such archive: {archive_path}")

    if pre_archive:
        pre_archive_dst = os.path.join(_archive_dir(root), f"{_default_label(stem)}.pkl")
        if os.path.abspath(pre_archive_dst) == os.path.abspath(archive_path):
            raise SystemExit(
                f"refusing to union: the pre-union backup would write to "
                f"{ARCHIVE_SUBDIR}/{label}.pkl, which is the same archive being "
                f"unioned in -- this would destroy it even with --force. Pass "
                f"--no-archive to skip the pre-union backup, or archive the live "
                f"checkpoint under a different label first"
            )

    live = _checkpoint_io.load(live_path)
    archived = _checkpoint_io.load(archive_path)
    live_material = _material_of(live)
    archived_material = _material_of(archived)
    if (
        live_material is not None
        and archived_material is not None
        and live_material != archived_material
    ):
        raise SystemExit(
            f"material mismatch: checkpoints/{stem}.pkl is {live_material!r}, "
            f"{ARCHIVE_SUBDIR}/{label}.pkl is {archived_material!r} -- refusing to union"
        )

    if pre_archive:
        archive_checkpoint(stem, force=force, root=root)

    merged = _union_results(live, archived)
    tmp = live_path + ".tmp"
    _checkpoint_io.dump(merged, tmp)
    os.replace(tmp, live_path)

    if delete_archive:
        os.remove(archive_path)

    n_before = sum(len(v) for v in live.values())
    n_after = sum(len(v) for v in merged.values())
    print(
        f"unioned {ARCHIVE_SUBDIR}/{label}.pkl into checkpoints/{stem}.pkl "
        f"({n_before} -> {n_after} records)"
    )
    return live_path


def _cli_archive(args):
    archive_checkpoint(args.stem, args.label, force=args.force)


def _cli_restore(args):
    restore_checkpoint(args.label, args.stem, force=args.force)


def _cli_archives(args):
    list_archives()


def _cli_union(args):
    union_checkpoint(
        args.stem,
        args.label,
        pre_archive=not args.no_archive,
        delete_archive=args.delete_archive,
        force=args.force,
    )


def add_subparser(sub):
    """Register the ``archive`` / ``restore`` / ``archives`` / ``union`` subcommands
    on an argparse subparsers object."""
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

    up = sub.add_parser("union", help="merge an archived checkpoint into the active slot")
    up.add_argument("stem", help="active checkpoint stem, e.g. hopg")
    up.add_argument("label", help="archive label to union in")
    up.add_argument(
        "--no-archive",
        action="store_true",
        help="skip archiving the live checkpoint before mutating it (default: archive first)",
    )
    up.add_argument(
        "--delete-archive",
        action="store_true",
        help="delete the source archive after a successful union (default: leave it intact)",
    )
    up.add_argument(
        "--force", action="store_true", help="overwrite an existing pre-union archive label"
    )
    up.set_defaults(func=_cli_union)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxr-archive", description="local checkpoint archive shelf")
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
