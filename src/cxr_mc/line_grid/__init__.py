"""`cxr line-grid` command group.

Job verbs (``status``/``attach``/``logs``/``stop``) delegate to ``cxr_mc.remote``
for output byte-identical to ``cxr remote``; ``derive``/``submit``/``apply``/
``set``/``set-brem``/``defaults``/``show``/``regen-golden`` call the package
modules. Heavy modules (``derive``, ``golden``) import lazily inside handlers so
``cxr`` startup stays cheap.
"""

from __future__ import annotations

import argparse
import math
from datetime import date

from cxr_mc import remote
from cxr_mc.line_grid import apply, defaults, job


def _floats(s):
    return [float(x) for x in s.split(",")] if s else None


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _domain_csv(name, lower, upper, *, upper_inclusive):
    def parse(value):
        try:
            values = _floats(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
        if values is None:
            return []
        for item in values:
            upper_ok = item <= upper if upper_inclusive else item < upper
            if not math.isfinite(item) or item < lower or not upper_ok:
                relation = "<=" if upper_inclusive else "<"
                raise argparse.ArgumentTypeError(
                    f"{name} values must satisfy {lower:g} <= value {relation} {upper:g}"
                )
        return values

    return parse


def _positive_csv(name):
    def parse(value):
        try:
            values = _floats(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
        if values is None:
            raise argparse.ArgumentTypeError(f"{name} requires at least one value")
        if any(not math.isfinite(item) or item <= 0 for item in values):
            raise argparse.ArgumentTypeError(f"{name} values must be finite and positive")
        return values

    return parse


# --- local derive / remote submit ------------------------------------------


def _derive_argv(args):
    argv = []
    for flag in ("materials", "energies", "tilts", "azimuths", "thickness"):
        val = getattr(args, flag)
        if val:
            argv += [f"--{flag}", val]
    if getattr(args, "brem_step", None) is not None:
        argv += ["--brem-step", str(args.brem_step)]
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
        tilts=args.tilts,
        azimuths=args.azimuths,
        thickness=args.thickness,
        set_default=args.set_default,
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
    return remote.tail_logs(args.jobid, args.follow)


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
    job._validate_remote_output_name(name)
    local = name
    remote_path = remote.remote_path(name)
    remote._run(["scp", remote.scp_remote_path(remote_path), local])
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
    supplied = [
        flag
        for flag, value in (
            ("--tilts", args.tilts),
            ("--azimuths", args.azimuths),
            ("--thickness", args.thickness),
            ("--brem-step", args.brem_step),
        )
        if value is not None
    ]
    if supplied and not args.set:
        args.parser.error(f"{', '.join(supplied)} require --set")
    if args.set:
        defaults.update_defaults(
            tilts=args.tilts,
            azimuths=args.azimuths,
            thickness_ang=args.thickness,
            brem_step_ev=args.brem_step,
        )
    for key, val in defaults.load_defaults().items():
        print(f"{key} = {val}")
    return 0


def _cli_show(args):
    try:
        result = apply.show(args.material)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(result)
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
            p.add_argument("--brem-step", type=_positive_float, default=None)
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
    se.add_argument("--energy", type=_positive_float, required=True)
    se.add_argument("--stop", type=_positive_float, required=True)
    se.add_argument("--num", type=_positive_int, default=None)
    se.add_argument("--start", type=_positive_float, default=None)
    se.add_argument("--note", default=None)
    se.set_defaults(func=_cli_set)

    sb = g.add_parser("set-brem")
    sb.add_argument("material")
    sb.add_argument("--stop", type=_positive_float, required=True)
    sb.add_argument("--step", type=_positive_float, default=None)
    sb.add_argument("--note", default=None)
    sb.set_defaults(func=_cli_set_brem)

    df = g.add_parser("defaults")
    df.add_argument("--set", action="store_true")
    df.add_argument("--tilts", type=_domain_csv("tilt", 0, 90, upper_inclusive=False))
    df.add_argument("--azimuths", type=_domain_csv("azimuth", 0, 360, upper_inclusive=True))
    df.add_argument("--thickness", type=_positive_csv("thickness"))
    df.add_argument("--brem-step", type=_positive_float, default=None)
    df.set_defaults(func=_cli_defaults, parser=df)

    sh = g.add_parser("show")
    sh.add_argument("material", nargs="?", default=None)
    sh.set_defaults(func=_cli_show)

    rg = g.add_parser("regen-golden")
    rg.add_argument("--check", action="store_true")
    rg.set_defaults(func=_cli_regen_golden)

    return ap
