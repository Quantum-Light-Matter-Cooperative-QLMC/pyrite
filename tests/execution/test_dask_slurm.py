"""SLURM-provisioned Dask workers: packing, site pass-through, lifetime and GPU jobs.

Job scripts come from the real dask-jobqueue classes; nothing is submitted.
"""

from __future__ import annotations

import re
import sys
import types

import pytest

from pyrite.execution import (
    CaseArtifactStore,
    DaskConfig,
    DaskExecutor,
    ResourceRequest,
    inspect_execution,
)
from pyrite.execution._dask_slurm import lifetime_args, walltime_seconds
from pyrite.execution.profiles import execution_profile_from_dict

dask = pytest.importorskip("dask")
pytest.importorskip("dask_jobqueue")

SITE = {
    "account": "pi_lab",
    "job-extra-directives": ["--qos=normal", "--constraint=avx512"],
    "job_script_prologue": ["module load python", "source /shared/venv/bin/activate"],
    "python": "/shared/venv/bin/python",
    "interface": "ib0",
}


def _executor(tmp_path, dask_table, resources=None):
    profile = execution_profile_from_dict(
        "site",
        {
            "backend": "dask",
            "resources": resources or {"cpu_cores": 6, "memory_bytes": "24GiB"},
            "dask": dask_table,
        },
    )
    return DaskExecutor(
        CaseArtifactStore(tmp_path),
        lambda case: case["key"],
        resources=profile.resources,
        config=profile.dask,
    )


def _job_script(executor):
    from dask_jobqueue.slurm import SLURMJob

    return SLURMJob(
        scheduler="tcp://10.0.0.1:8786", **executor._slurm_cluster_kwargs()
    ).job_script()


def _assert_site(script):
    assert "#SBATCH -A pi_lab" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --constraint=avx512" in script
    assert "module load python\nsource /shared/venv/bin/activate" in script
    assert "/shared/venv/bin/python -m distributed.cli.dask_worker" in script
    assert "--interface ib0" in script


def test_cpu_job_packs_one_single_threaded_worker_per_core(tmp_path):
    executor = _executor(tmp_path, {"cluster": "slurm", "workers": 2, "jobqueue": SITE})
    script = _job_script(executor)
    assert "#SBATCH --cpus-per-task=6" in script
    assert "--nworkers 6" in script and "--nthreads 1" in script
    assert "#SBATCH --mem=24G" in script
    _assert_site(script)
    assert executor.topology()["max_worker_processes"] == 12
    plan = executor.runtime_plan([{}] * 40)
    assert plan["effective_workers"] == 12
    assert plan["worker_processes_per_allocation"] == 6


def test_site_settings_from_named_dask_config_section(tmp_path):
    section = dict(dask.config.get("jobqueue.slurm"))
    section.update({key.replace("_", "-"): value for key, value in SITE.items()})
    section.update(queue="site-default", walltime="02:00:00")
    with dask.config.set({"jobqueue.mysite": section}):
        executor = _executor(tmp_path, {"cluster": "slurm", "config_name": "mysite"})
        script = _job_script(executor)
        _assert_site(script)
        assert "#SBATCH -p site-default" in script
        assert "#SBATCH -t 02:00:00" in script
        # PyRITE fields override the site section.
        override = _executor(
            tmp_path,
            {
                "cluster": "slurm",
                "config_name": "mysite",
                "partition": "fast",
                "walltime": "00:20:00",
            },
        )
        script = _job_script(override)
        assert "#SBATCH -p fast" in script and "#SBATCH -t 00:20:00" in script


def test_profile_fields_override_jobqueue_table(tmp_path):
    executor = _executor(
        tmp_path,
        {
            "cluster": "slurm",
            "partition": "fast",
            "walltime": "03:00:00",
            "jobqueue": {"queue": "slow", "walltime": "00:10:00", "memory": "1GiB"},
        },
    )
    kwargs = executor._slurm_cluster_kwargs()
    assert (kwargs["queue"], kwargs["walltime"], kwargs["memory"]) == (
        "fast",
        "03:00:00",
        "24576MiB",
    )


def test_lifetime_defaults_from_walltime_unless_user_sets_one(tmp_path):
    executor = _executor(tmp_path, {"cluster": "slurm", "walltime": "1-00:00:00"})
    assert executor._slurm_cluster_kwargs()["worker_extra_args"] == [
        "--lifetime",
        "85200s",
        "--lifetime-stagger",
        "300s",
    ]
    own = _executor(
        tmp_path,
        {
            "cluster": "slurm",
            "walltime": "01:00:00",
            "jobqueue": {"worker_extra_args": ["--lifetime=50m"]},
        },
    )
    assert own._slurm_cluster_kwargs()["worker_extra_args"] == ["--lifetime=50m"]


@pytest.mark.parametrize(
    "text, seconds",
    [("30", 1800), ("10:30", 630), ("02:00:00", 7200), ("1-02", 93600), ("x", None), (None, None)],
)
def test_walltime_formats(text, seconds):
    assert walltime_seconds(text) == seconds


def test_short_walltime_gets_no_lifetime():
    assert lifetime_args("00:01:30") == []
    lifetime, stagger = (int(re.sub("s$", "", v)) for v in lifetime_args("00:30:00")[1::2])
    assert lifetime + stagger < 1800 - 60


GPU = {"cpu_cores": 8, "gpu_count": 1, "accelerator": "cuda"}


def test_gpu_job_with_site_gres_runs_one_labelled_worker(tmp_path):
    executor = _executor(
        tmp_path,
        {"cluster": "slurm", "jobqueue": {"job_extra_directives": ["--gres=gpu:a100:1"]}},
        resources=GPU,
    )
    executor.validate_run_options({})
    script = _job_script(executor)
    assert "#SBATCH --gres=gpu:a100:1" in script
    assert "--nworkers" not in script and "--nthreads 8" in script
    assert "--resources GPU=1" in script
    assert executor.topology()["worker_processes_per_allocation"] == 1


def test_gpu_job_without_site_directive_fails_before_staging(tmp_path):
    executor = _executor(tmp_path, {"cluster": "slurm"}, resources=GPU)
    with pytest.raises(ValueError, match="device isolation"):
        executor.validate_run_options({})
    with pytest.raises(ValueError, match="device isolation"):
        executor.execute([{"crystal": "hopg", "key": "a" * 64}])
    assert not (tmp_path / ".execution-jobs").exists()


@pytest.mark.parametrize(
    "table, error",
    [
        ({"jobqueue": {"cores": 4}}, "set by PyRITE"),
        ({"jobqueue": {"no_such_option": 1}}, "not accepted"),
        ({"jobqueue": {"job_script_prologue": ["a\nb"]}}, "control"),
        ({"config_name": "missing-site"}, "no jobqueue.missing-site"),
    ],
)
def test_bad_site_configuration_rejected_before_staging(tmp_path, table, error):
    with pytest.raises(ValueError, match=error):
        _executor(tmp_path, {"cluster": "slurm", **table}).validate_run_options({})


def test_jobqueue_only_for_slurm_and_existing_needs_one_scheduler():
    with pytest.raises(ValueError, match="only to cluster='slurm'"):
        DaskConfig(cluster="local", jobqueue={"account": "x"})
    with pytest.raises(ValueError, match="exactly one"):
        DaskConfig(cluster="existing", scheduler_address="tcp://a:1", scheduler_file="/s.json")
    with pytest.raises(ValueError, match="apply only to existing"):
        DaskConfig(cluster="slurm", scheduler_file="/s.json")


def test_existing_scheduler_file_and_slurm_adapt(tmp_path, monkeypatch):
    calls = {}

    class Client:
        def __init__(self, address=None, *, scheduler_file=None):
            calls["client"] = (address, scheduler_file)

    class SLURMCluster:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def adapt(self, **kwargs):
            calls["adapt"] = kwargs

    monkeypatch.setitem(
        sys.modules, "distributed", types.SimpleNamespace(Client=Client, LocalCluster=None)
    )
    existing = _executor(tmp_path, {"cluster": "existing", "scheduler_file": "/shared/s.json"})
    existing._open_client()
    assert calls["client"] == (None, "/shared/s.json")
    import dask_jobqueue

    monkeypatch.setattr(dask_jobqueue, "SLURMCluster", SLURMCluster)
    fixed = _executor(tmp_path, {"cluster": "slurm", "workers": 3})
    fixed._open_client()
    assert calls["adapt"] == {"minimum_jobs": 3, "maximum_jobs": 3}
    assert calls["kwargs"]["processes"] == 6
    elastic = _executor(
        tmp_path, {"cluster": "slurm", "workers": 1, "max_workers": 4, "adaptive": True}
    )
    elastic._open_client()
    assert calls["adapt"] == {"minimum_jobs": 1, "maximum_jobs": 4}


def test_inspection_counts_allocations_and_packed_units():
    profile = execution_profile_from_dict(
        "site",
        {
            "backend": "dask",
            "resources": {"cpu_cores": 4, "memory_bytes": "8GiB"},
            "dask": {"cluster": "slurm", "workers": 2},
        },
    )
    plan = inspect_execution(profile, 20)
    assert plan["max_simultaneous_units_requested"] == 8
    assert plan["worker_processes_per_allocation"] == 4
    assert plan["max_cpus_requested"] == 8
    assert plan["max_memory_bytes_requested"] == 16 * 1024**3


def test_existing_profiles_still_parse():
    profile = execution_profile_from_dict(
        "cpu-dask",
        {
            "backend": "dask",
            "max_attempts": 2,
            "resources": {"cpu_cores": 8, "memory_bytes": "32GiB", "accelerator": "cpu"},
            "dask": {
                "cluster": "slurm",
                "partition": "cpu",
                "walltime": "01:00:00",
                "workers": 2,
                "max_workers": 6,
                "adaptive": True,
                "cases_per_task": 2,
                "max_in_flight": 12,
            },
        },
    )
    assert profile.dask.walltime == "01:00:00" and profile.dask.jobqueue == {}
    assert ResourceRequest(cpu_cores=8) != profile.resources
