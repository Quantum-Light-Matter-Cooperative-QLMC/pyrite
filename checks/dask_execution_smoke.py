"""Small real execution comparison; run under a bounded SLURM allocation.

Uses the existing physics kernels and canonical content keys. This is a
connectivity/correctness smoke check, not a scaling or physics benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np


def smoke_cases():
    import pyrite as pr

    scene = pr.Scene(
        beam=pr.Beam(
            energy_keV=30, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()
        ),
        target=pr.geometry.Slab("hopg", thickness_ang=100, tilt_deg=30, footprint=None),
        numerics=pr.Numerics(
            sampling=pr.config.FixedSampling(4, 2),
            energy_bins=pr.detectors.EnergyBins(
                cxr=np.linspace(100, 1_000, 19), brem=np.linspace(100, 1_000, 19)
            ),
            cxr_reflections=pr.config.CxrReflections(n_families=1, max_reflections=2),
        ),
    )
    (case,) = pr.Sweep(base=scene).cases()
    return [replace(case, name=f"dask-smoke-{i}", seed=41 + i) for i in range(3)]


def worker_environment(marker):
    """Read/write the shared path from a worker, without client-side inference."""
    from importlib.metadata import version

    path = Path(marker)
    host = socket.gethostname()
    assert path.read_text() == "pyrite-dask-shared-path"
    path.with_name(f"worker-{host}.txt").write_text(host)
    result = {
        "host": host,
        "pid": os.getpid(),
        "python": sys.executable,
        "pyrite": version("pyrite-mc"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    if os.environ.get("PYRITE_MC_BACKEND") == "cuda":
        import cupy as cp

        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert visibility and visibility not in ("-1", "NoDevFiles")
        assert cp.cuda.runtime.getDeviceCount() == 1
        properties = cp.cuda.runtime.getDeviceProperties(0)
        result["cuda"] = {
            "visibility": visibility,
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "device_name": properties["name"].decode(),
            "memory_bytes": properties["totalGlobalMem"],
            "array_check": float(cp.sum(cp.asarray([1.0, 2.0])).get()),
            "fp64": os.environ.get("PYRITE_FP64"),
        }
    return result


def check_outputs(reference, actual):
    keys = ("E_grid", "E_grid_brem", "spec", "brem", "brem_wide", "spec_characteristic")
    compared = []
    for expected, observed in zip(reference, actual, strict=True):
        fields = []
        for key in keys:
            if key not in expected:
                continue
            np.testing.assert_allclose(observed[key], expected[key], rtol=1e-10, atol=1e-18)
            assert np.isfinite(observed[key]).all()
            fields.append(key)
        assert "spec" in fields and "brem_wide" in fields
        assert np.any(expected["brem_wide"] > 0), "smoke must score nonzero radiation"
        compared.append(fields)
    return compared


def open_client(args, stack):
    from distributed import Client, LocalCluster

    if args.mode == "local":
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=True,
            memory_limit="4GiB",
            dashboard_address=None,
        )
    else:
        from dask_jobqueue import SLURMCluster

        backend = "cuda" if args.gpu else "cpu"
        exports = [
            "export PATH=$HOME/.local/bin:$PATH",
            f"export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYRITE_MC_BACKEND={backend}",
            "export PYRITE_FP64=" + shlex.quote(os.environ.get("PYRITE_FP64", "0")),
        ]
        for key in ("PYRITE_HOME", "PYRITE_CATALOG", "PYRITE_USER_CATALOG"):
            exports.append(f"export {key}={shlex.quote(os.environ[key])}")
        exports.append(f"cd {shlex.quote(str(Path.cwd()))}")
        directives = [f"--nodelist={args.worker_node}"]
        worker_options = []
        if args.gpu:
            directives.append(f"--gres={args.gres}")
            worker_options = ["--resources", "GPU=1"]
        cluster = SLURMCluster(
            queue=args.partition,
            cores=1,
            processes=1,
            memory="4GiB",
            walltime="00:05:00",
            python=sys.executable,
            scheduler_options={"host": args.scheduler_host, "dashboard_address": None},
            job_extra_directives=directives,
            worker_extra_args=worker_options,
            job_script_prologue=exports,
            log_directory=str(args.output),
            local_directory="/tmp/pyrite-dask-smoke",
        )
        (args.output / "worker.sbatch").write_text(cluster.job_script())
    stack.enter_context(cluster)
    client = stack.enter_context(Client(cluster))
    if args.mode == "slurm":
        cluster.scale(1)
    print("scheduler", cluster.scheduler_address, flush=True)
    client.wait_for_workers(1, timeout=60)
    return client


def no_recomputation(*_args, **_kwargs):
    raise AssertionError("committed worker artifacts/checkpoints must be recovered")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "slurm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--scheduler-host")
    parser.add_argument("--worker-node")
    parser.add_argument("--native-slurm", action="store_true")
    parser.add_argument(
        "--gpu", action="store_true", help="Use one site-configured SLURM CUDA worker"
    )
    parser.add_argument("--gres", default="gpu:1")
    args = parser.parse_args()
    if args.mode == "slurm" and not (args.scheduler_host and args.worker_node):
        parser.error("SLURM mode requires --scheduler-host and --worker-node")
    if args.gpu and (args.mode != "slurm" or os.environ.get("PYRITE_FP64") != "1"):
        parser.error("GPU comparison requires SLURM mode and PYRITE_FP64=1")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    from pyrite.api import source_content_key_fn
    from pyrite.execution import (
        CaseArtifactStore,
        DaskConfig,
        DaskExecutor,
        LocalExecutor,
        ResourceRequest,
    )
    from pyrite.montecarlo import run_cases
    from pyrite.runs.run import run_sweep

    cases = smoke_cases()
    key_fn = source_content_key_fn()
    report = {"coordinator": socket.gethostname(), "case_keys": [key_fn(c) for c in cases]}
    stamp = Path.cwd() / ".pyrite-sync"
    report["code_identity"] = stamp.read_text() if stamp.exists() else "local working tree"
    report["gpu_requested"] = args.gpu
    report["fp64"] = os.environ.get("PYRITE_FP64")
    started = perf_counter()
    reference = run_cases(cases, engine="cpu", max_workers=0, progress=False)
    report["local_seconds"] = perf_counter() - started
    store = CaseArtifactStore(args.output / "artifacts")
    marker = args.output / "shared-marker.txt"
    marker.write_text("pyrite-dask-shared-path")
    resources = ResourceRequest(
        cpu_cores=1,
        memory_bytes=4 * 1024**3,
        gpu_count=int(args.gpu),
        accelerator="cuda" if args.gpu else "cpu",
    )
    with ExitStack() as stack:
        client = open_client(args, stack)
        report["workers"] = client.run(worker_environment, str(marker))
        print(json.dumps(report["workers"], indent=2), flush=True)
        executor = DaskExecutor(
            store,
            key_fn,
            config=DaskConfig(
                cluster="existing", scheduler_address=client.scheduler.address, max_in_flight=1
            ),
            client=client,
            resources=resources,
        )
        started = perf_counter()
        refs = executor.execute(cases, progress=False)
        report["dask_seconds"] = perf_counter() - started
        report["compared_fields"] = check_outputs(reference, [store.load(ref) for ref in refs])
        report["dask_run_dir"] = str(executor.last_run_dir)

    # Simulate a coordinator restart after worker publication but before its
    # authoritative checkpoint commit. No new scheduler or computation allowed.
    checkpoint = args.output / "checkpoint" / "hopg"
    for _ in range(2):
        assert run_sweep(
            cases,
            {},
            checkpoint_path=str(checkpoint),
            progress=False,
            content_key_fn=key_fn,
            artifact_store=store,
            executor=LocalExecutor(no_recomputation),
        )
    assert (checkpoint / "line.h5").exists()
    report["checkpoint_recovery_and_resume"] = True
    if args.native_slurm:
        from pyrite.execution import SlurmArrayConfig, SlurmArrayExecutor

        native_store = CaseArtifactStore(args.output / "native-artifacts")
        native = SlurmArrayExecutor(
            native_store,
            key_fn,
            resources=resources,
            config=SlurmArrayConfig(
                partition=args.partition,
                walltime="00:05:00",
                max_concurrent=1,
                poll_seconds=1,
                gres=args.gres if args.gpu else None,
            ),
        )
        started = perf_counter()
        refs = native.execute(cases, progress=False)
        report["native_slurm_seconds"] = perf_counter() - started
        report["native_compared_fields"] = check_outputs(
            reference, [native_store.load(ref) for ref in refs]
        )
        report["native_job_ids"] = native.submitted_job_ids
    report["status"] = "passed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
