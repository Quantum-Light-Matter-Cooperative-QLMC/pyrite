"""``cxr archive`` / ``restore`` / ``archives`` / ``union`` -- a durable local
shelf for checkpoints (checkpoint lifecycle, component 2).

The GPU box is scratch compute; the laptop is the durable store. A grid-filtered
pull lands in the ACTIVE slot ``checkpoints/<stem>.pkl`` (exactly what
``cxr app analysis`` loads). These commands add a two-tier model on top of that:

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
not in ``cxr_mc.remote``. The shelf stores whatever the active slot holds at the
time, typically the grid-filtered view.
"""

import datetime
import json
import os
import re
import shutil
from pathlib import Path

import click

from ..cli import _completion as _cli_completion
from ..cli import _core as _cli_core
from ..cli import json as cli_json
from ..paths import workspace_root
from . import _checkpoint_io, _checkpoint_store

DEFAULT_ROOT = str(workspace_root() / "checkpoints")
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


def _active_paths(stem, root):
    directory = Path(root) / stem
    if (directory / "line.pkl").is_file():
        return directory, True
    legacy = Path(root) / f"{stem}.pkl"
    return legacy, False


def _archive_paths(label, root):
    directory = Path(_archive_dir(root)) / label
    if (directory / "line.pkl").is_file():
        return directory, True
    legacy = Path(_archive_dir(root)) / f"{label}.pkl"
    return legacy, False


def _atomic_copytree(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = f"{dst}.tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def archive_checkpoint(stem, label=None, *, force=False, root=DEFAULT_ROOT):
    """Copy the active slot ``<root>/<stem>.pkl`` to ``<root>/archive/<label>.pkl``.
    ``label`` defaults to ``<stem>-<YYYYMMDD>``. Refuses to overwrite an existing
    label without ``force``. Returns the archive path."""
    src, is_directory = _active_paths(stem, root)
    if not src.exists():
        raise SystemExit(f"no such active checkpoint: {src}")
    label = label or _default_label(stem)
    dst = Path(_archive_dir(root)) / (label if is_directory else f"{label}.pkl")
    if os.path.exists(dst) and not force:
        raise SystemExit(f"archive already exists: {dst} (pass --force to overwrite)")
    if is_directory:
        _atomic_copytree(str(src), str(dst))
    else:
        _atomic_copy(str(src), str(dst))
    suffix = "" if is_directory else ".pkl"
    print(f"archived checkpoints/{stem}{suffix} -> {ARCHIVE_SUBDIR}/{label}{suffix}")
    return str(dst)


def restore_checkpoint(label, stem=None, *, force=False, root=DEFAULT_ROOT):
    """Copy ``<root>/archive/<label>.pkl`` back to the active slot
    ``<root>/<stem>.pkl``. ``stem`` defaults to the label with a trailing
    ``-<YYYYMMDD>`` stripped. Refuses to overwrite an existing active slot without
    ``force``. Returns the active-slot path."""
    src, is_directory = _archive_paths(label, root)
    if not src.exists():
        raise SystemExit(f"no such archive: {src}")
    stem = stem or _stem_from_label(label)
    dst = Path(root) / (stem if is_directory else f"{stem}.pkl")
    if os.path.exists(dst) and not force:
        raise SystemExit(f"active checkpoint already exists: {dst} (pass --force to overwrite)")
    if is_directory:
        _atomic_copytree(str(src), str(dst))
    else:
        _atomic_copy(str(src), str(dst))
    suffix = "" if is_directory else ".pkl"
    print(f"restored {ARCHIVE_SUBDIR}/{label}{suffix} -> checkpoints/{stem}{suffix}")
    return str(dst)


def _record_count(path):
    """Total records in an archived checkpoint (``sum(len(by_E) ...)``), or None
    if it can't be read as a results store. Reads via ``_checkpoint_io.load``
    (TODO P2 #8), which transparently handles both gzip-compressed and legacy
    plain-pickle archives -- the shelf can hold either, since ``archive``/
    ``restore`` just copy whatever bytes the active slot already has."""
    try:
        path = Path(path)
        results = (
            _checkpoint_store.load(path.name, path.parent)
            if path.is_dir()
            else _checkpoint_io.load(str(path))
        )
        return sum(len(v) for v in results.values())
    except Exception:
        return None


def _dataset_identity(path, is_directory):
    """Read dataset identity from a component checkpoint manifest, if present."""
    if not is_directory:
        return None
    manifest = Path(path) / "meta.json"
    if not manifest.is_file():
        return None
    try:
        with manifest.open() as handle:
            return json.load(handle).get("dataset_identity")
    except (OSError, ValueError, TypeError):
        return None


def list_archives(root=DEFAULT_ROOT):
    """Print the shelf: each label with its size (MB) and record count, sorted by
    label. Returns the list of labels."""
    adir = _archive_dir(root)
    if not os.path.isdir(adir):
        print("(no archives)")
        return []
    labels = sorted(
        {
            entry.name
            for entry in Path(adir).iterdir()
            if entry.is_dir() and (entry / "line.pkl").is_file()
        }
        | {entry.stem for entry in Path(adir).glob("*.pkl")}
    )
    if not labels:
        print("(no archives)")
        return []
    for label in labels:
        path, is_directory = _archive_paths(label, root)
        mb = (
            sum(item.stat().st_size for item in path.glob("*.pkl")) / 1e6
            if is_directory
            else path.stat().st_size / 1e6
        )
        n = _record_count(path)
        n_str = "?" if n is None else str(n)
        identity = _dataset_identity(path, is_directory)
        profile = "" if identity is None else f"  {identity['profile']}"
        print(f"  {label:<32}  {mb:7.1f} MB  {n_str:>6} records{profile}")
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
    live_path, live_is_directory = _active_paths(stem, root)
    if not live_path.exists():
        raise SystemExit(f"no such active checkpoint: {live_path}")
    archive_path, archive_is_directory = _archive_paths(label, root)
    if not archive_path.exists():
        raise SystemExit(f"no such archive: {archive_path}")

    if pre_archive:
        pre_archive_dst = Path(_archive_dir(root)) / (
            _default_label(stem) if live_is_directory else f"{_default_label(stem)}.pkl"
        )
        if os.path.abspath(pre_archive_dst) == os.path.abspath(archive_path):
            raise SystemExit(
                f"refusing to union: the pre-union backup would write to "
                f"{ARCHIVE_SUBDIR}/{label}.pkl, which is the same archive being "
                f"unioned in -- this would destroy it even with --force. Pass "
                f"--no-archive to skip the pre-union backup, or archive the live "
                f"checkpoint under a different label first"
            )

    live = (
        _checkpoint_store.load(live_path.name, live_path.parent)
        if live_is_directory
        else _checkpoint_io.load(str(live_path))
    )
    archived = (
        _checkpoint_store.load(archive_path.name, archive_path.parent)
        if archive_is_directory
        else _checkpoint_io.load(str(archive_path))
    )
    live_identity = _dataset_identity(live_path, live_is_directory)
    archived_identity = _dataset_identity(archive_path, archive_is_directory)
    if (
        live_identity is not None
        and archived_identity is not None
        and live_identity.get("parameter_sha256") != archived_identity.get("parameter_sha256")
    ):
        raise SystemExit(
            "dataset identity mismatch: "
            f"active {live_identity.get('profile')}:{live_identity.get('parameter_sha256', '')[:12]} "
            f"!= archive {archived_identity.get('profile')}:"
            f"{archived_identity.get('parameter_sha256', '')[:12]} -- refusing to union"
        )
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
    if live_is_directory:
        _checkpoint_store.save(stem, root, merged)
    else:
        tmp = f"{live_path}.tmp"
        _checkpoint_io.dump(merged, tmp)
        os.replace(tmp, live_path)

    if delete_archive:
        shutil.rmtree(archive_path) if archive_is_directory else os.remove(archive_path)

    n_before = sum(len(v) for v in live.values())
    n_after = sum(len(v) for v in merged.values())
    print(
        f"unioned {ARCHIVE_SUBDIR}/{label}.pkl into checkpoints/{stem}.pkl "
        f"({n_before} -> {n_after} records)"
    )
    return str(live_path)


def _cli_archive(args):
    archive_checkpoint(args.stem, args.label, force=args.force)


def _cli_restore(args):
    restore_checkpoint(args.label, args.stem, force=args.force)


def _cli_archives(args):
    if getattr(args, "json_output", False):

        def _loader(path):
            path = Path(path)
            return (
                _checkpoint_store.load(path.name, path.parent)
                if path.is_dir()
                else _checkpoint_io.load(str(path))
            )

        result = cli_json.archives(DEFAULT_ROOT, loader=_loader)
        _cli_core.emit_json_result(result)
        return
    list_archives()


def _cli_union(args):
    union_checkpoint(
        args.stem,
        args.label,
        pre_archive=not args.no_archive,
        delete_archive=args.delete_archive,
        force=args.force,
    )


@click.command(
    "archive",
    help=(
        "Copy an active checkpoint to long-term shelf.\n\n"
        "LABEL defaults to a date-stamped label inferred from STEM. Existing "
        "labels are preserved unless --force."
    ),
)
@click.argument("stem", shell_complete=_cli_completion.complete_archive_stem)
@click.argument("label", required=False)
@click.option("--force", is_flag=True, help="Overwrite existing archive label.")
def archive_command(stem, label, force):
    return _cli_core.invoke_legacy(_cli_archive, stem=stem, label=label, force=force)


@click.command(
    "restore",
    help=(
        "Copy an archived checkpoint back to active slot.\n\n"
        "Active stem is inferred from LABEL unless --as is supplied. Existing "
        "active checkpoints are preserved unless --force."
    ),
)
@click.argument("label", shell_complete=_cli_completion.complete_archive_label)
@click.option("--as", "stem", default=None, help="Active stem (default: inferred).")
@click.option("--force", is_flag=True, help="Overwrite existing active checkpoint.")
def restore_command(label, stem, force):
    return _cli_core.invoke_legacy(_cli_restore, label=label, stem=stem, force=force)


@click.command("archives", help="List long-term checkpoint shelf.")
@_cli_core.output_option
def archives_command(json_output):
    if json_output:
        return _cli_core.invoke_legacy(_cli_archives, json_output=True)
    return _cli_core.invoke_legacy(_cli_archives)


@click.command(
    "union",
    help=(
        "Merge an archived checkpoint into active slot.\n\n"
        "Requires matching materials. Live records win overlaps; source archive "
        "is retained and live checkpoint is backed up by default."
    ),
)
@click.argument("stem", shell_complete=_cli_completion.complete_archive_stem)
@click.argument("label", shell_complete=_cli_completion.complete_archive_label)
@click.option(
    "--no-archive",
    is_flag=True,
    help="Skip pre-union backup of live checkpoint.",
)
@click.option(
    "--delete-archive",
    is_flag=True,
    help="Delete source archive after successful union.",
)
@click.option("--force", is_flag=True, help="Overwrite existing pre-union archive label.")
def union_command(stem, label, no_archive, delete_archive, force):
    return _cli_core.invoke_legacy(
        _cli_union,
        stem=stem,
        label=label,
        no_archive=no_archive,
        delete_archive=delete_archive,
        force=force,
    )


@click.group("cxr-archive")
def standalone_command():
    """Manage local checkpoint archive shelf."""


for _command in (archive_command, restore_command, archives_command, union_command):
    standalone_command.add_command(_command)


def main(argv=None):
    return _cli_core.run(standalone_command, argv, prog_name="cxr-archive")


if __name__ == "__main__":
    raise SystemExit(main())
