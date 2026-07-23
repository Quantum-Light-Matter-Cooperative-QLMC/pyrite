"""Argparse wiring, CLI handlers, and scan/check compat orchestrators."""

import argparse
import sys

from ..scan import load_all_materials
from . import config, lifecycle, scripts, state, transport, viewer


def remote_scan(material, quick=False, workers=None):
    """Submit one material through SLURM and follow it to completion.

    This compatibility helper deliberately does not pull: callers that need a
    checkpoint can apply their own grid/trim policy after it returns.
    """
    transport._check_materials([material])
    jobid = lifecycle.start_queue([material], quick=quick, workers=workers, chunk_minutes=0)
    viewer.attach(jobid)
    return jobid


def remote_check(
    ne=20_000,
    ne_brem=200,
    ne_supp=200,
    tmd_azimuth=0.0,
    refresh=False,
    no_sync=False,
):
    """Submit the Zhai reproduction through SLURM, follow it, then pull cache."""
    jobid = lifecycle.start_zhai_queue(
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_azimuth=tmd_azimuth,
        refresh=refresh,
        no_sync=no_sync,
    )
    if not viewer.attach(jobid):
        print("Zhai job is still running or its viewer disconnected; skipping automatic cache pull")
        return
    if not state._job_succeeded(jobid):
        job_state = state._job_state(jobid) or "no terminal state recorded"
        raise SystemExit(
            f"Zhai SLURM job {jobid} did not complete successfully ({job_state}); cache not pulled"
        )
    lifecycle.pull_zhai_cache()


def _ensure_utf8_stdio():
    """The box's output is UTF-8 (job logs embed tqdm block-glyph progress bars
    like `████▌`). On Windows stdout defaults to cp1252 -- and when it is,
    printing that text raises UnicodeEncodeError, so `status`/`jobs`/`logs`/
    `attach` would crash on the glyphs. Force UTF-8 with replacement so they
    never do. Scoped to remote subcommands (called from their CLI wrappers, not
    at import time) so it doesn't change stdio encoding for unrelated ``cxr``
    commands."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass


# ---- CLI wiring ----------------------------------------------------------------
def _dispatch(handler):
    def _run_cli(args):
        _ensure_utf8_stdio()
        return handler(args)

    return _run_cli


def _selected_materials(args, attribute):
    """Resolve explicit material arguments or the shared ``--all`` manifest."""
    explicit = getattr(args, attribute)
    if args.all:
        if explicit:
            raise SystemExit(f"{args.remote_command} --all does not take material names")
        return load_all_materials(config.MATS_FILE)
    if explicit:
        return explicit if isinstance(explicit, list) else [explicit]
    raise SystemExit(f"{args.remote_command} needs material name(s), or use --all")


def _cli_scan(args):
    if args.quick and args.grid:
        raise SystemExit(
            "scan --quick --grid: quick checkpoints aren't grid-filterable "
            "(their grid isn't reproducible from material_sweep), so the "
            "trailing pull would fail after the whole sweep ran. Drop --grid."
        )
    materials = _selected_materials(args, "material")
    # `start_queue` validates material tokens, refuses checkpoint collisions,
    # syncs once, and submits one bounded-concurrency scheduler job for the materials.
    jobid = lifecycle.start_queue(
        materials,
        quick=args.quick,
        workers=args.workers,
        parallel_materials=getattr(args, "parallel_materials", None),
        chunk_minutes=getattr(args, "chunk_minutes", 10.0),
        no_sync=args.no_sync,
    )
    if not viewer.attach(jobid):
        print("scan is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        print("warning: the SLURM scan produced no successful checkpoints; nothing to pull")
        return
    stems = scripts._stems(completed, args.quick)
    # Code is already synced by the queue, so a grid pull skips its own sync;
    # preserve the requested grid/trim policy.
    lifecycle.pull(
        stems,
        grid=args.grid,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        no_sync=True,
    )
    for stem in stems:
        print(
            f"\ndone. checkpoints/{stem}.pkl is local; run `cxr analyze {stem}` "
            f"(or run `cxr export`) -- visualization and static-HTML export "
            "stay local."
        )


def _cli_pull(args):
    lifecycle.pull(
        _selected_materials(args, "material"),
        grid=not args.full,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        level9=args.level9,
        no_sync=args.no_sync,
    )


def _cli_start(args):
    materials = _selected_materials(args, "materials")
    jobid = lifecycle.start_queue(
        materials,
        quick=args.quick,
        workers=args.workers,
        parallel_materials=args.parallel_materials,
        chunk_minutes=args.chunk_minutes,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
    )
    if args.follow and not args.dry_run:
        viewer.attach(jobid)


def _cli_attach(args):
    viewer.attach(args.jobid, args.verbose)


def _cli_jobs(args):
    viewer.list_jobs()


def _cli_status(args):
    viewer.job_status(args.jobid, args.verbose)


def _cli_logs(args):
    viewer.tail_logs(args.jobid, args.follow)


def _cli_stop(args):
    lifecycle.stop_jobs(args.materials, args.all)


def _cli_reap(args):
    lifecycle.reap_reservations(min_age_minutes=args.min_age_minutes, yes=args.yes)


def _cli_clear(args):
    if args.all_checkpoints:
        if args.materials:
            args._clear_parser.error("clear --all takes no material argument")
        lifecycle.clear_all_remote(args.yes)
        return
    if not args.materials:
        args._clear_parser.error("clear needs material(s), or --all")
    lifecycle.clear_remote(args.materials, args.yes)


def _cli_sync(args):
    transport.sync_code()


def _cli_check(args):
    if args.follow and not args.detached:
        args._check_parser.error("--follow requires --detached")
    if args.pull:
        lifecycle.pull_zhai_cache()
        return
    if args.detached:
        jobid = lifecycle.start_zhai_queue(
            ne=args.ne,
            ne_brem=args.ne_brem,
            ne_supp=args.ne_supp,
            tmd_azimuth=args.tmd_azimuth,
            refresh=args.refresh,
            no_sync=args.no_sync,
        )
        if args.follow:
            viewer.attach(jobid)
        return
    remote_check(
        ne=args.ne,
        ne_brem=args.ne_brem,
        ne_supp=args.ne_supp,
        tmd_azimuth=args.tmd_azimuth,
        refresh=args.refresh,
        no_sync=args.no_sync,
    )


def _build_remote_parser(ap):
    """Add every ``remote`` subcommand (scan/pull/start/attach/jobs/status/logs/
    stop/clear/sync) to ``ap``'s own subparsers. Nested one level under a
    ``remote`` group -- rather than flat alongside ``cxr scan`` etc -- because
    several of these names (``scan`` in particular) collide with top-level cxr
    subcommands that mean something different (a local sweep vs this
    sync+run-on-the-box+pull)."""
    sub = ap.add_subparsers(dest="remote_command", required=True)

    s = sub.add_parser("scan", help="sync code, submit sweep(s), wait, and pull checkpoints")
    s.add_argument("material", nargs="?")
    s.add_argument(
        "-a", "--all", action="store_true", help="run every material in mats_to_sim.toml"
    )
    s.add_argument("--quick", action="store_true")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument(
        "--parallel-materials",
        type=int,
        choices=range(1, config.MAX_PARALLEL_MATERIALS + 1),
        default=None,
        metavar="N",
        help="simultaneous material scans in one GPU allocation; only with "
        "--chunk-minutes 0, which defaults it to 2 (max: 4)",
    )
    s.add_argument(
        "--chunk-minutes",
        type=float,
        default=10.0,
        help="length of each self-resubmitting SLURM slice in minutes; "
        "0 = one whole-box monolithic run (default: 10.0)",
    )
    s.add_argument("--no-sync", action="store_true", help="skip the code upload")
    s.add_argument(
        "--grid", action="store_true", help="grid-filter the checkpoint on the box before pulling"
    )
    s.add_argument("--drop-wide-brem", action="store_true", help="with --grid: drop wide-brem too")
    s.add_argument("--downcast", action="store_true", help="with --grid: downcast to float32 too")
    s.set_defaults(func=_dispatch(_cli_scan))

    st = sub.add_parser(
        "start",
        help="sync code and submit a SLURM queue of materials (survives disconnect)",
    )
    st.add_argument("materials", nargs="*", help="one or more crystal keys")
    st.add_argument(
        "-a", "--all", action="store_true", help="queue every material in mats_to_sim.toml"
    )
    st.add_argument("--quick", action="store_true")
    st.add_argument("--workers", type=int, default=None)
    st.add_argument(
        "--parallel-materials",
        type=int,
        choices=range(1, config.MAX_PARALLEL_MATERIALS + 1),
        default=None,
        metavar="N",
        help="simultaneous material scans in one GPU allocation; only with "
        "--chunk-minutes 0, which defaults it to 2 (max: 4)",
    )
    st.add_argument(
        "--chunk-minutes",
        type=float,
        default=10.0,
        help="length of each self-resubmitting SLURM slice in minutes; "
        "0 = one whole-box monolithic run (default: 10.0)",
    )
    st.add_argument("--no-sync", action="store_true", help="skip the code upload")
    st.add_argument(
        "--dry-run",
        action="store_true",
        help="print the SLURM batch script + submission command, don't ssh",
    )
    st.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="track the job live after launching (Ctrl-C disconnects; job keeps running)",
    )
    st.set_defaults(func=_dispatch(_cli_start))

    at = sub.add_parser(
        "attach",
        help="live-track a job: status report, refreshed in place (Ctrl-C disconnects; default: latest)",
    )
    at.add_argument("jobid", nargs="?", default=None)
    at.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="add SLURM allocation detail; repeat for the recent log tail (mirrors status)",
    )
    at.set_defaults(func=_dispatch(_cli_attach))

    jb = sub.add_parser("jobs", help="list jobs with SLURM IDs, materials, and last events")
    jb.set_defaults(func=_dispatch(_cli_jobs))

    js = sub.add_parser(
        "status", help="show one job; -v allocation, -vv case progress and recent log"
    )
    js.add_argument("jobid", nargs="?", default=None)
    js.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="add SLURM allocation detail; repeat for case progress and recent log",
    )
    js.set_defaults(func=_dispatch(_cli_status))

    lg = sub.add_parser("logs", help="show a job's diagnostic log (default: latest)")
    lg.add_argument("jobid", nargs="?", default=None)
    lg.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="stream live; Ctrl-C disconnects without stopping the job",
    )
    lg.set_defaults(func=_dispatch(_cli_logs))

    sp = sub.add_parser(
        "stop",
        help="cancel active SLURM job(s) by material, or every live job",
        description="cancel active SLURM job(s) by material, or every live job",
    )
    sp.add_argument("materials", nargs="*", help="material name(s) owned by live jobs")
    sp.add_argument("-a", "--all", action="store_true", help="stop every live job")
    sp.set_defaults(func=_dispatch(_cli_stop))

    rp = sub.add_parser(
        "reap",
        help="release checkpoint reservations orphaned by hard-killed jobs (dry preview unless --yes)",
        description="release checkpoint reservations orphaned by hard-killed jobs "
        "whose owning SLURM allocation is gone (dry preview unless --yes)",
    )
    rp.add_argument(
        "--min-age-minutes",
        type=float,
        default=5.0,
        help="only reap reservations whose newest lock is older than this, guarding "
        "the reserve-before-submit window (default: 5.0)",
    )
    rp.add_argument(
        "--yes", action="store_true", help="actually release (default: dry preview only)"
    )
    rp.set_defaults(func=_dispatch(_cli_reap))

    p = sub.add_parser("pull", help="fetch one or more existing checkpoints from the box")
    p.add_argument("material", nargs="*", help="checkpoint stem(s), e.g. mose2 mose2_quick")
    p.add_argument(
        "-a", "--all", action="store_true", help="pull every material in mats_to_sim.toml"
    )
    p.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="pull the full, un-filtered checkpoint instead of the default grid-filtered pull",
    )
    p.add_argument(
        "--drop-wide-brem", action="store_true", help="with grid pull: drop wide-brem too"
    )
    p.add_argument(
        "--downcast", action="store_true", help="with grid pull: downcast to float32 too"
    )
    p.add_argument(
        "--level9",
        action="store_true",
        help="recompress on the box at gzip level 9 before transfer (lossless, "
        "just smaller/slower than the level-6 default a live sweep writes at)",
    )
    p.add_argument(
        "--no-sync", action="store_true", help="with grid pull: skip the pre-pull code sync"
    )
    p.set_defaults(func=_dispatch(_cli_pull))

    c = sub.add_parser(
        "clear", help="delete a material's accumulated checkpoints on the box, or --all"
    )
    c.add_argument(
        "materials",
        nargs="*",
        metavar="material",
        help="crystal key(s); clears both <material>.pkl and <material>_quick.pkl for each",
    )
    c.add_argument(
        "--all",
        dest="all_checkpoints",
        action="store_true",
        help="empty the entire checkpoints/ directory (mutually exclusive with a material)",
    )
    c.add_argument("--yes", action="store_true", help="actually delete (default: dry preview only)")
    c.set_defaults(func=_dispatch(_cli_clear), _clear_parser=c)

    sy = sub.add_parser("sync", help="push the current code to the box only")
    sy.set_defaults(func=_dispatch(_cli_sync))

    ck = sub.add_parser(
        "check",
        help="run the Zhai reproduction + supplementary MC on the box, pull caches back",
    )
    ck.add_argument(
        "--ne", type=int, default=20_000, help="Fig.1c anchor line electrons per energy"
    )
    ck.add_argument(
        "--ne-brem", default=200, type=int, help="Fig.1c anchor bremsstrahlung electrons per energy"
    )
    ck.add_argument(
        "--ne-supp", type=int, default=200, help="supplementary electrons per polar-tilt spectrum"
    )
    ck.add_argument(
        "--tmd-azimuth",
        type=float,
        default=0.0,
        help="exploratory azimuth for TMD studies whose azimuth is unreported",
    )
    ck.add_argument(
        "--refresh", action="store_true", help="recompute even if a matching cache exists"
    )
    ck.add_argument("--no-sync", action="store_true", help="skip the code upload")
    mode = ck.add_mutually_exclusive_group()
    mode.add_argument(
        "--detached",
        "-d",
        action="store_true",
        help="launch as a DETACHED job (survives disconnect)",
    )
    ck.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="with --detached: track the job live after launching",
    )
    mode.add_argument(
        "--pull", action="store_true", help="skip the run; just fetch existing zhai cache files"
    )
    ck.set_defaults(func=_dispatch(_cli_check), _check_parser=ck)

    return ap


def add_subparser(sub):
    """Register the ``remote`` subcommand group on an argparse subparsers
    object. Everything under it (``cxr remote scan|pull|start|attach|jobs|
    status|logs|stop|clear|sync``) is optional -- it only works with an ssh host
    configured to run sweeps on (default 'qlmc', see CXR_REMOTE_HOST)."""
    ap = sub.add_parser(
        "remote", help="[dev] push code + run/manage MC sweeps on a remote GPU box over ssh"
    )
    return _build_remote_parser(ap)


def main(argv=None):
    ap = _build_remote_parser(
        argparse.ArgumentParser(
            prog="cxr-remote", description="push code + run/manage MC sweeps on a remote GPU box"
        )
    )
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
