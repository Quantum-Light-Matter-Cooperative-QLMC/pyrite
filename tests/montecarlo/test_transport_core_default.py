"""Which transport core a run gets, and where it is allowed to run.

``transport_core="auto"`` is the shipped default, so these pin the policy that
decides for a caller: the electron count, the groove, whether this process has a
CUDA device, and the ``CXR_MC_TRANSPORT_CORE`` pin. They also pin the routing
that follows from it -- device transport stays in the driver process, and every
worker pool is nailed to the CPU core, because the one thing the single-context
design cannot survive is a pool of CUDA contexts.

No GPU required: the device probe is patched, since what is under test is the
decision, not the kernel (``test_transport_per_electron.py`` owns the kernel).
"""

import pytest

from cxr_mc.montecarlo import runner
from cxr_mc.montecarlo.transport import (
    CUDA_TRANSPORT_MIN_ELECTRONS,
    resolve_transport_core,
)


@pytest.fixture
def cuda(monkeypatch):
    """Answer the device probe without a device."""

    def _set(available=True):
        monkeypatch.setattr(
            "cxr_mc.montecarlo.transport._cuda_transport_available",
            lambda: available,
        )

    return _set


def _case(Ne=4000, Ne_brem=None, **extra):
    return dict(Ne=Ne, Ne_brem=Ne if Ne_brem is None else Ne_brem, **extra)


# ---- the policy ------------------------------------------------------------


def test_auto_takes_the_device_above_the_threshold(cuda):
    cuda(True)
    assert resolve_transport_core("auto", CUDA_TRANSPORT_MIN_ELECTRONS + 1) == "cuda"


@pytest.mark.parametrize("Ne", [1, 450, CUDA_TRANSPORT_MIN_ELECTRONS])
def test_auto_keeps_the_cpu_core_at_or_below_the_threshold(cuda, Ne):
    # Strictly above: at the crossover the launch overhead has not been paid off,
    # so the boundary electron count stays on the core it has always used.
    cuda(True)
    assert resolve_transport_core("auto", Ne) == "lockstep"


def test_auto_keeps_the_cpu_core_without_a_device(cuda):
    cuda(False)
    assert resolve_transport_core("auto", 100_000) == "lockstep"


def test_auto_keeps_the_cpu_core_for_a_grooved_run(cuda):
    # The new cores are ungrooved-only. "auto" degrades rather than raising --
    # a groove is a physics choice, not a request for a particular core.
    cuda(True)
    assert resolve_transport_core("auto", 100_000, groove=object()) == "lockstep"


@pytest.mark.parametrize("core", ["lockstep", "per-electron", "cuda"])
def test_an_explicit_core_is_never_substituted(cuda, core):
    # Including "cuda" with no device: an explicit request must fail loudly
    # downstream, not fall back into a different sampler behind the caller.
    cuda(False)
    assert resolve_transport_core(core, 10) == core


def test_an_unknown_core_is_rejected():
    with pytest.raises(ValueError, match="transport_core"):
        resolve_transport_core("gpu", 10)


# ---- the process-wide pin --------------------------------------------------


def test_env_pin_forces_the_cpu_core_above_the_threshold(cuda, monkeypatch):
    cuda(True)
    monkeypatch.setenv("CXR_MC_TRANSPORT_CORE", "lockstep")
    assert resolve_transport_core("auto", 100_000) == "lockstep"


def test_env_pin_overrides_an_explicit_request(cuda, monkeypatch):
    # The pin exists to reproduce a run without editing call sites, so it has to
    # win over the call site too.
    cuda(True)
    monkeypatch.setenv("CXR_MC_TRANSPORT_CORE", "cuda")
    assert resolve_transport_core("lockstep", 10) == "cuda"


def test_env_pin_is_validated(monkeypatch):
    monkeypatch.setenv("CXR_MC_TRANSPORT_CORE", "gpu")
    with pytest.raises(ValueError, match="CXR_MC_TRANSPORT_CORE"):
        resolve_transport_core("auto", 10)


def test_empty_env_pin_is_ignored(cuda, monkeypatch):
    cuda(True)
    monkeypatch.setenv("CXR_MC_TRANSPORT_CORE", "")
    assert resolve_transport_core("auto", 100_000) == "cuda"


# ---- what a case resolves to ----------------------------------------------


def test_a_case_counts_both_electron_populations(cuda):
    # One transport serves lines and brem, so what matters is the wider of the
    # two -- a case can cross the threshold on its brem count alone.
    cuda(True)
    assert runner._case_transport_core(_case(Ne=10, Ne_brem=4000)) == "cuda"
    assert runner._case_transport_core(_case(Ne=4000, Ne_brem=10)) == "cuda"
    assert runner._case_transport_core(_case(Ne=10, Ne_brem=10)) == "lockstep"


def test_a_grooved_case_stays_on_the_cpu_core(cuda):
    cuda(True)
    assert runner._case_transport_core(_case(groove_spacing_ang=1.0e4)) == "lockstep"


def test_a_case_honors_an_explicit_request(cuda):
    cuda(True)
    assert runner._case_transport_core(_case(), "lockstep") == "lockstep"


def test_an_empty_case_resolves(cuda):
    # runtime_plan reports on cases[0] and is called with partial dicts.
    cuda(True)
    assert runner._case_transport_core({}) == "lockstep"


def test_a_run_moves_to_the_device_only_when_every_case_does(cuda):
    # Mixed runs keep the pipeline: taking it away would strand the CPU-core
    # cases in the driver with nothing overlapping them.
    cuda(True)
    assert runner._cuda_transport_run([_case(), _case()])
    assert not runner._cuda_transport_run([_case(), _case(Ne=10)])
    assert not runner._cuda_transport_run([])


# ---- routing ---------------------------------------------------------------


def test_runtime_plan_reports_a_serial_device_transport(cuda, monkeypatch):
    cuda(True)
    monkeypatch.setattr(runner, "_GPU", True)
    plan = runner.runtime_plan([_case(), _case()], max_workers=8)
    assert plan["transport_core"] == "cuda"
    assert plan["engine"] == "serial"
    assert plan["effective_workers"] == 1


def test_runtime_plan_keeps_the_pipeline_for_cpu_transport(cuda, monkeypatch):
    cuda(True)
    monkeypatch.setattr(runner, "_GPU", True)
    plan = runner.runtime_plan([_case(Ne=10), _case(Ne=10)], max_workers=8)
    assert plan["transport_core"] == "lockstep"
    assert plan["engine"] == "gpu-pipeline"


def test_device_transport_runs_in_this_process_with_resident_segments(cuda, monkeypatch):
    # The pipeline would need a second CUDA context per worker and could not
    # return device arrays through a pickle anyway.
    cuda(True)
    monkeypatch.setattr(runner, "_GPU", True)
    monkeypatch.setattr(runner, "case_runtime_plan", lambda case: {})
    calls = []

    def fake_run_case(case, record_timing=False, keep_segments_on_device=False, **kwargs):
        calls.append(keep_segments_on_device)
        return {"case": case}

    monkeypatch.setattr(runner, "run_case", fake_run_case)

    def no_pool(*args, **kwargs):  # pragma: no cover - a pass means it is unused
        raise AssertionError("device transport must not start a worker pool")

    monkeypatch.setattr(runner, "_gpu_pipeline_workers", no_pool)

    out = runner.run_cases([_case(), _case()], max_workers=8, progress=False)
    assert calls == [True, True]
    assert len(out) == 2


def test_transport_only_device_run_does_not_ask_for_residency(cuda, monkeypatch):
    # Nothing consumes the segments, so keeping them on the card is pure cost.
    cuda(True)
    monkeypatch.setattr(runner, "_GPU", True)
    seen = []

    def fake_transport_case(case, record_timing=False, **kwargs):
        seen.append(kwargs.get("keep_segments_on_device", False))
        return {}

    monkeypatch.setattr(runner, "_transport_case", fake_transport_case)
    runner.run_cases([_case()], max_workers=8, progress=False, transport_only=True)
    assert seen == [False]
