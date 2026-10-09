"""Detector-cone variance reduction (#203): analog baseline and splitting prototype.

Remote only (GPU box via ``pyrite remote``); the 5 MeV h-BN case is heavy.

``baseline`` times the analog 5 MeV h-BN 1 mm tilt/azimuth 10/100 case: CUDA
transport, the per-electron line-mass/brem monitor and the full spectrum phase,
then the per-electron CPU core on a smaller count. It records per-electron line
mass so the analog relative standard error (RSE) can be reported.

``compare`` runs the per-electron CPU core analog and with detector-cone
splitting on identical seeds, scores weighted per-history line mass,
bremsstrahlung band integral and characteristic yield, and reports means,
RSEs, wall times and the figure of merit ``FOM = 1/(RSE^2 T)``.

    python checks/detector_cone_variance_reduction.py baseline --out B.json
    python checks/detector_cone_variance_reduction.py compare --out C.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

MATERIAL = "hbn"


def build(energy_keV, thickness_ang, tilt, azimuth, ne, seed):
    from pyrite.energy_grid.bandwidth_check import build_case

    config = {
        "thickness_ang": float(thickness_ang),
        "tilt_deg": float(tilt),
        "tilt_azim_deg": float(azimuth),
        "n_electrons": int(ne),
    }
    return build_case(MATERIAL, float(energy_keV), config, seed=int(seed), resolution="local")


def _settings(ne):
    from pyrite._precision import Precision

    return Precision(
        target_rse=0.1, min_electrons=ne, max_electrons=ne, block_electrons=ne,
        observables=("line", "brem"),
    )  # fmt: skip


def _stats(values):
    values = np.asarray(values, dtype=float)
    n = values.size
    total = float(values.sum())
    mean = total / n if n else 0.0
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    rse = sd / (np.sqrt(n) * abs(mean)) if n > 1 and mean else None
    ranked = np.sort(values)[::-1]
    return {
        "n": n,
        "mean": mean,
        "rse": rse,
        "max_share": float(ranked[0] / total) if total else None,
        "top10_share": float(ranked[:10].sum() / total) if total else None,
        "ess": float(total**2 / np.sum(values**2)) if total else 0.0,
    }


def _host(value):
    from pyrite._backend import _to_cpu

    return np.asarray(_to_cpu(value))


def baseline(args):
    from pyrite._backend import BACKEND, REAL
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.adaptive import case_measure

    report = {"args": vars(args), "real": np.dtype(REAL).name, "device": BACKEND.device.name}
    for label, core, ne in (("cuda", "auto", args.ne), ("cpu_per_electron", "per-electron", args.ne_cpu)):
        if ne <= 0:
            continue
        case = build(args.energy, args.thickness, args.tilt, args.azimuth, ne, args.seed)
        row = {"core": core, "ne": ne}
        started = time.perf_counter()
        tp = runner._transport_case(case, transport_core=core)
        row["transport_s"] = time.perf_counter() - started
        row["n_segments"] = int(_host(tp["segs"]["L_ang"]).size)
        started = time.perf_counter()
        values = case_measure(case, _settings(ne))(0, ne, tp["segs"])
        row["measure_s"] = time.perf_counter() - started
        row["line"] = _stats(_host(values["line"]))
        row["brem"] = _stats(_host(values["brem"]))
        row["line_values"] = _host(values["line"]).tolist()
        if label == "cuda" and not args.skip_spectrum:
            started = time.perf_counter()
            out = runner._spectrum_case(case, tp)
            row["spectrum_s"] = time.perf_counter() - started
            row["line_grid_nodes"] = int(np.asarray(out["E_grid"]).size)
            row["line_grid_stop_eV"] = float(np.asarray(out["E_grid"])[-1])
            row["line_grid_resolved"] = out.get("line_grid_resolved")
        report[label] = {k: v for k, v in row.items()}
        print(json.dumps({k: v for k, v in row.items() if k != "line_values"}, default=str), flush=True)
        del tp
        BACKEND.release_memory()
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))


JOB_KIND = "detector-cone-vr"


def submit(args):
    """Submit ``python checks/<this> <rest...>`` as one remote SLURM job."""
    import shlex

    from pyrite import remote
    from pyrite.energy_grid.convergence_job import _precision_payload
    from pyrite.remote._queue_scripts import _uv_sync_block

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    jobid = remote._new_jobid()
    jobdir = remote.remote_path(remote.JOBS_SUBDIR, jobid)
    env = "PYRITE_FP64=1 " if args.fp64 else "env -u PYRITE_FP64 "
    # The job runs its own copy of this script, embedded below, so a later
    # code sync from another checkout cannot remove or change it mid-queue.
    command = (
        f'{env}PYRITE_MC_BACKEND=cuda {remote.shell_remote_uv()} run --no-sync python "$JOBDIR/check.py" '
        + " ".join(shlex.quote(token) for token in rest)
    )
    source = Path(__file__).read_text()
    if "\nPYRITE_CHECK_EOF\n" in source:
        raise SystemExit("script contains the heredoc terminator")
    payload = _precision_payload(jobdir, [command], remote, _uv_sync_block(), label=JOB_KIND)
    embed = f"cat > \"$JOBDIR/check.py\" <<'PYRITE_CHECK_EOF'\n{source}PYRITE_CHECK_EOF\n"
    marker = "\n[ -f \"$JOBDIR/STOP\" ]"
    payload = payload.replace(marker, "\n" + embed.rstrip("\n") + marker, 1)
    script = remote._slurm_batch_script(
        jobid,
        payload,
        job_name=JOB_KIND,
        time_limit=str(int(args.time_limit_minutes)),
    )
    metadata = f"job: {jobid}\nkind: {JOB_KIND}\nprogress_dashboard: False\n"
    if args.dry_run:
        print(script)
        return
    remote.sync_code()
    remote._stage_job_script(jobid, [], remote._write_job_script_command(jobdir, metadata), script)
    print(f"submitted SLURM job {remote._submit_staged_job(jobid, [], nice=True)} as {jobid}")


def pull(args):
    from pyrite import remote

    for name in args.names:
        remote._run(
            remote.config.scp_argv(
                remote.scp_remote_path(remote.remote_path(name)), str(Path(args.dest) / name)
            )
        )
        print(f"pulled {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub_submit = sub.add_parser("submit", help="run the remaining arguments remotely")
    sub_submit.add_argument("--time-limit-minutes", type=int, default=120)
    sub_submit.add_argument("--fp64", action="store_true")
    sub_submit.add_argument("--dry-run", action="store_true")
    sub_submit.add_argument("rest", nargs=argparse.REMAINDER)
    sub_pull = sub.add_parser("pull", help="copy remote output files (relative to the checkout)")
    sub_pull.add_argument("names", nargs="+")
    sub_pull.add_argument("--dest", default=".")
    base = sub.add_parser("baseline")
    for p in (base,):
        p.add_argument("--energy", type=float, default=5000.0)
        p.add_argument("--thickness", type=float, default=1.0e7)
        p.add_argument("--tilt", type=float, default=10.0)
        p.add_argument("--azimuth", type=float, default=100.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
    base.add_argument("--ne", type=int, default=2000)
    base.add_argument("--ne-cpu", type=int, default=100)
    base.add_argument("--skip-spectrum", action="store_true")
    args = parser.parse_args()
    {"baseline": baseline, "submit": submit, "pull": pull}[args.command](args)


if __name__ == "__main__":
    main()
