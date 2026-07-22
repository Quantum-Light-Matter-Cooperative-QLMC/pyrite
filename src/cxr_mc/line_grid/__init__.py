"""`cxr line-grid` command group.

Job verbs (``status``/``attach``/``logs``/``stop``) delegate to ``cxr_mc.remote``
for output byte-identical to ``cxr remote``; ``derive``/``submit``/``apply``/
``set``/``set-brem``/``defaults``/``show``/``regen-golden`` call the package
modules. Heavy modules (``derive``, ``golden``) import lazily inside handlers so
``cxr`` startup stays cheap.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from cxr_mc import remote
from cxr_mc.line_grid import apply, defaults, job


def _floats(s):
    return [float(x) for x in s.split(",")] if s else None


# --- local derive / remote submit ------------------------------------------


def _derive_argv(args):
    argv = []
    for flag in ("materials", "energies", "tilts", "azimuths", "thickness"):
        val = getattr(args, flag)
        if val:
            argv += [f"--{flag}", val]
    if args.set_default:
        argv.append("--set-default")
    return argv


def _cli_derive(args):
    from cxr_mc.line_grid import derive

    return derive.main(_derive_argv(args))


def _cli_submit(args):
    job.start(
        materials=args.materials or job.DEFAULT_MATERIALS,
        energies=args.energies or job.DEFAULT_ENERGIES,
        slice_minutes=args.slice_minutes,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
    )
    return 0


# --- job management (delegated to remote for parity) -----------------------


def _cli_status(args):
    remote.job_status(args.jobid, detail=args.verbose)
    return 0


def _cli_attach(args):
    remote.attach(args.jobid)
    return 0


def _cli_logs(args):
    remote.tail_logs(args.jobid, args.follow)
    return 0


def _cli_stop(args):
    jobid = args.jobid or remote._latest_jobid()
    if not jobid:
        raise SystemExit("no jobs to stop")
    remote._stop_jobid(jobid)
    return 0


# --- apply / set / defaults / show / regen ---------------------------------


def _pull_combined(json_name=None):
    """scp the combined derivation JSON back from the remote box; return local path."""
    name = json_name or job.DEFAULT_JSON_OUT
    local = Path(name).name
    remote._run(["scp", f"{remote.HOST}:{remote.REMOTE_DIR}/{name}", local])
    return local


def _cli_apply(args):
    path = _pull_combined() if args.pull else args.json
    if not path:
        raise SystemExit("no JSON: pass a path or --pull")
    apply.apply_file(
        path,
        materials=args.materials,
        force=args.force,
        dry_run=args.dry_run,
        date=str(date.today()),
        regen_golden=args.regen_golden,
    )
    if args.regen_golden and not args.dry_run:
        from cxr_mc.line_grid import golden

        golden.regen()
    return 0


def _cli_set(args):
    apply.set_line_grid(
        args.material, args.energy, args.stop, num=args.num, start_eV=args.start, note=args.note
    )
    return 0


def _cli_set_brem(args):
    apply.set_brem_grid(args.material, args.stop, step_eV=args.step, note=args.note)
    return 0


def _cli_defaults(args):
    if args.set:
        defaults.update_defaults(
            tilts=_floats(args.tilts),
            azimuths=_floats(args.azimuths),
            thickness_ang=_floats(args.thickness),
            brem_step_ev=args.brem_step,
        )
    for key, val in defaults.load_defaults().items():
        print(f"{key} = {val}")
    return 0


def _cli_show(args):
    print(apply.show(args.material))
    return 0


def _cli_regen_golden(args):
    from cxr_mc.line_grid import golden

    return golden.regen(check=args.check)


# --- parser wiring ----------------------------------------------------------


def add_subparser(sub):
    ap = sub.add_parser(
        "line-grid", help="derive/apply per-material line-grid bounds (local or on qlmc)"
    )
    g = ap.add_subparsers(dest="lg_command", required=True)

    for name in ("derive", "submit"):
        p = g.add_parser(name)
        p.add_argument("--materials", default=None)
        p.add_argument("--energies", default=None)
        p.add_argument("--tilts", default=None)
        p.add_argument("--azimuths", default=None)
        p.add_argument("--thickness", default=None)
        p.add_argument("--set-default", action="store_true")
        if name == "submit":
            p.add_argument("--slice-minutes", type=float, default=job.DEFAULT_SLICE_MINUTES)
            p.add_argument("--no-sync", action="store_true")
            p.add_argument("--dry-run", action="store_true")
            p.set_defaults(func=_cli_submit)
        else:
            p.set_defaults(func=_cli_derive)

    st = g.add_parser("status")
    st.add_argument("jobid", nargs="?", default=None)
    st.add_argument("-v", "--verbose", action="count", default=0)
    st.set_defaults(func=_cli_status)

    at = g.add_parser("attach")
    at.add_argument("jobid", nargs="?", default=None)
    at.set_defaults(func=_cli_attach)

    lg = g.add_parser("logs")
    lg.add_argument("jobid", nargs="?", default=None)
    lg.add_argument("-f", "--follow", action="store_true")
    lg.set_defaults(func=_cli_logs)

    sp = g.add_parser("stop")
    sp.add_argument("jobid", nargs="?", default=None)
    sp.set_defaults(func=_cli_stop)

    ap_apply = g.add_parser("apply")
    ap_apply.add_argument("json", nargs="?", default=None)
    ap_apply.add_argument("--materials", default=None)
    ap_apply.add_argument("--pull", action="store_true")
    ap_apply.add_argument("--force", action="store_true")
    ap_apply.add_argument("--regen-golden", action="store_true")
    ap_apply.add_argument("--dry-run", action="store_true")
    ap_apply.set_defaults(func=_cli_apply)

    se = g.add_parser("set")
    se.add_argument("material")
    se.add_argument("--energy", type=float, required=True)
    se.add_argument("--stop", type=float, required=True)
    se.add_argument("--num", type=int, default=None)
    se.add_argument("--start", type=float, default=None)
    se.add_argument("--note", default=None)
    se.set_defaults(func=_cli_set)

    sb = g.add_parser("set-brem")
    sb.add_argument("material")
    sb.add_argument("--stop", type=float, required=True)
    sb.add_argument("--step", type=float, default=None)
    sb.add_argument("--note", default=None)
    sb.set_defaults(func=_cli_set_brem)

    df = g.add_parser("defaults")
    df.add_argument("--set", action="store_true")
    df.add_argument("--tilts", default=None)
    df.add_argument("--azimuths", default=None)
    df.add_argument("--thickness", default=None)
    df.add_argument("--brem-step", type=float, default=None)
    df.set_defaults(func=_cli_defaults)

    sh = g.add_parser("show")
    sh.add_argument("material", nargs="?", default=None)
    sh.set_defaults(func=_cli_show)

    rg = g.add_parser("regen-golden")
    rg.add_argument("--check", action="store_true")
    rg.set_defaults(func=_cli_regen_golden)

    return ap
