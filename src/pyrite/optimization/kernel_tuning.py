"""Optuna tuning of CUDA execution controls for one fixed Scene (#776).

``tune_kernels`` times a complete in-memory :func:`pyrite.simulate` of the
scene under candidate :class:`~pyrite.config.KernelTuning` values, each in a
fresh worker process, and accepts a candidate only when its spectra match the
baseline's. The controls are execution choices, not physical parameters, so
this is separate from :class:`OptimizationStudy`. Results are specific to the
GPU, driver, precision and workload that produced them. Use objective="transport"
or objective="spectrum" to minimize the corresponding synchronized runner
phase, while still validating full spectra at every candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

_SCHEMA = "pyrite-kernel-tuning-v1"
_FAMILIES = ("cxr.incoherent", "cxr.coherent", "characteristic", "channeling", "transition", "brem")
_DIMENSIONS = ("electron_batch", "cxr_chunk", "brem_chunk", "schedule", "continuation_rows")


@dataclass(frozen=True)
class KernelLattice:
    """Candidate values per execution control; an empty tuple keeps the scene's value.

    ``electron_batch``, ``schedule`` and ``continuation_rows`` set
    :class:`~pyrite.config.TransportKernelTuning`; ``cxr_chunk`` and
    ``brem_chunk`` set the spectrum segment chunks. ``None`` in a tuple selects
    the process default. Combinations the config rejects (a row quantum with
    full-history) are skipped without running.
    """

    electron_batch: tuple[int | None, ...] = ()
    cxr_chunk: tuple[int | None, ...] = ()
    brem_chunk: tuple[int | None, ...] = ()
    schedule: tuple[Literal["full-history", "continuation", "auto"] | None, ...] = ()
    continuation_rows: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        for name in _DIMENSIONS:
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(set(values)) != len(values):
                raise ValueError(f"{name} must be a tuple of distinct values")
        if not any(getattr(self, name) for name in _DIMENSIONS):
            raise ValueError("KernelLattice needs at least one tuned dimension")

    def candidate(self, base: Any, choice: dict[str, Any]) -> Any:
        """``base`` KernelTuning with ``choice`` applied; raises if inconsistent."""
        transport = replace(
            base.transport,
            **{
                k: choice[k]
                for k in ("electron_batch", "schedule", "continuation_rows")
                if k in choice
            },
        )
        spectral = {k: choice[k] for k in ("cxr_chunk", "brem_chunk") if k in choice}
        return replace(base, transport=transport, **spectral)


@dataclass(frozen=True)
class KernelTrial:
    """One measured candidate; ``median_s`` is None when it was rejected.

    ``repeat_of`` names the earlier trial whose outcome a resampled, identical
    candidate reused instead of being timed again.
    """

    number: int
    tuning: Any
    state: Literal["complete", "rejected"]
    median_s: float | None = None
    samples_s: tuple[float, ...] = ()
    reason: str | None = None
    repeat_of: int | None = None


@dataclass(frozen=True)
class KernelTuningResult:
    """Baseline, every trial, and the configuration to apply.

    ``chosen`` is the fastest numerically equivalent trial, or the baseline
    when none beat it, and ``best`` its tuning; apply it with
    ``numerics.with_kernel_tuning(best)`` in a fresh process. ``environment`` records the device, driver, CUDA/CuPy
    versions and precision: the timings do not transfer to other hardware.
    """

    baseline: KernelTrial
    trials: tuple[KernelTrial, ...]
    chosen: KernelTrial
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Any:
        """The ``KernelTuning`` to apply."""
        return self.chosen.tuning

    @property
    def improved(self) -> bool:
        """Whether a candidate beat the baseline median."""
        return self.chosen is not self.baseline

    @property
    def speedup(self) -> float:
        """Baseline median over the chosen configuration's median."""
        return _median(self.baseline) / _median(self.chosen)


def _median(trial: KernelTrial) -> float:
    if trial.median_s is None:
        raise ValueError(f"trial {trial.number} was rejected and has no timing")
    return float(trial.median_s)


def _tuning_dict(tuning: Any) -> dict[str, Any]:
    return asdict(tuning)


def _tuning_from_dict(data: dict[str, Any]) -> Any:
    from .._numerics_groups import KernelTuning, TransportKernelTuning

    transport: dict[str, Any] = data["transport"]
    return KernelTuning(
        transport_core=data["transport_core"],
        cxr_chunk=data["cxr_chunk"],
        brem_chunk=data["brem_chunk"],
        transport=TransportKernelTuning(**transport),
    )


def _environment() -> dict[str, Any]:
    from .. import __version__
    from .._backend import BACKEND

    env: dict[str, Any] = {
        "backend": BACKEND.name,
        "device": getattr(BACKEND.device, "name", str(BACKEND.device)),
        "precision": "fp64" if os.environ.get("PYRITE_FP64", "0") not in ("", "0") else "fp32",
        "pyrite_version": __version__,
    }
    try:
        import cupy

        env |= {
            "cupy": cupy.__version__,
            "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cupy.cuda.runtime.driverGetVersion(),
        }
    except Exception:  # noqa: BLE001 - provenance only; the CUDA check is separate
        pass
    return env


def _family(result: Any, path: str) -> Any:
    value = result
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _worker(payload_path: Path, output_path: Path, result_path: Path) -> None:
    """Fresh process: time ``simulate`` under one tuning, save its spectra."""
    import pyrite

    from .._backend import BACKEND

    payload = pickle.loads(payload_path.read_bytes())
    if BACKEND.name != "cuda":
        raise RuntimeError(f"CUDA required, resolved backend={BACKEND.name!r}")
    scene = payload["scene"]
    numerics = scene.numerics.with_kernel_tuning(_tuning_from_dict(payload["tuning"]))
    objective = payload.get("objective", "total")
    samples, result = [], None
    for index in range(payload["warmups"] + payload["repeats"]):
        BACKEND.synchronize()
        start = perf_counter()
        stage_seconds = {}
        if objective == "total":
            result = pyrite.simulate(scene, numerics=numerics)
        else:
            # The public simulate() path still builds the same case and the
            # complete result. Time only the selected internal phase, with
            # explicit synchronization before/after its work. The stage wrappers
            # are local to this single worker and restored after the call.
            from unittest.mock import patch
            from ..montecarlo import runner

            transport = runner._transport_case
            spectrum = runner._spectrum_with_recovery

            def timed(stage, function):
                def wrapped(*args, **kwargs):
                    BACKEND.synchronize()
                    started = perf_counter()
                    value = function(*args, **kwargs)
                    BACKEND.synchronize()
                    stage_seconds[stage] = stage_seconds.get(stage, 0.0) + perf_counter() - started
                    return value
                return wrapped

            with (
                patch.object(runner, "_transport_case", timed("transport", transport)),
                patch.object(runner, "_spectrum_with_recovery", timed("spectrum", spectrum)),
            ):
                result = pyrite.simulate(scene, numerics=numerics)
        BACKEND.synchronize()
        if index >= payload["warmups"]:
            if objective != "total" and objective not in stage_seconds:
                raise RuntimeError(f"runner did not report {objective} phase")
            samples.append(
                perf_counter() - start if objective == "total" else stage_seconds[objective]
            )
    arrays = {}
    for path in _FAMILIES:
        density = _family(result, path)
        if density is not None:
            arrays[f"{path}:grid"] = np.asarray(density.photon_energy_eV)
            arrays[f"{path}:density"] = np.asarray(density.density)
    np.savez(output_path, allow_pickle=False, **arrays)
    result_path.write_text(json.dumps({"samples_s": samples, "environment": _environment()}))


def _equivalent(candidate: Path, reference: Path, *, rtol: float, atol: float) -> str | None:
    """None when spectra agree: grids exactly, densities within tolerance."""
    with np.load(reference) as ref, np.load(candidate) as cur:
        if set(ref.files) != set(cur.files):
            return "candidate produced different spectral families"
        for key in ref.files:
            a, b = ref[key], cur[key]
            if a.shape != b.shape:
                return f"{key} shape differs"
            exact = key.endswith(":grid")
            if exact and not np.array_equal(a, b):
                return f"{key} differs"
            if not exact and not np.allclose(b, a, rtol=rtol, atol=atol):
                return f"{key} differs beyond rtol={rtol}, atol={atol}"
    return None


def _measure(scene, tuning, *, warmups, repeats, timeout, workdir: Path, label: str,
             objective: str = "total"):
    payload, output, record = (workdir / f"{label}.{ext}" for ext in ("pkl", "npz", "json"))
    payload.write_bytes(
        pickle.dumps(
            {"scene": scene, "tuning": _tuning_dict(tuning), "warmups": warmups,
             "repeats": repeats, "objective": objective}
        )
    )
    env = os.environ | {"PYRITE_MC_BACKEND": "cuda"}  # no hidden CPU fallback in timings
    command = [sys.executable, "-m", __name__, str(payload), str(output), str(record)]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[-2500:]
        raise RuntimeError(f"worker failed ({done.returncode}): {detail}")
    return output, json.loads(record.read_text())


def _study_identity(scene, base, lattice, *, warmups, repeats, rtol, atol,
                    environment, objective="total") -> str:
    from ..campaign.lowering import build_case
    from ..campaign.profiles import case_content_key

    payload = {
        "schema": _SCHEMA,
        "case": case_content_key(build_case(scene)),
        "baseline": _tuning_dict(base),
        "lattice": asdict(lattice),
        "protocol": [warmups, repeats, rtol, atol, objective],
        "environment": environment,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def tune_kernels(
    scene: Any,
    lattice: KernelLattice,
    *,
    trials: int = 16,
    warmups: int = 1,
    repeats: int = 3,
    rtol: float = 1e-4,
    atol: float = 1e-12,
    storage: str | None = None,
    study_name: str = "kernel-tuning",
    timeout: float = 1800.0,
    seed: int = 42,
    objective: Literal["total", "transport", "spectrum"] = "total",
    on_progress: Callable[[int, int, str], None] | None = None,
) -> KernelTuningResult:
    """Tune ``scene``'s CUDA execution controls over ``lattice`` with Optuna.

    The baseline is the scene's own ``numerics.kernel_tuning`` with
    ``transport_core="cuda"``; every candidate changes only ``lattice``
    dimensions. Each run is a fresh process with an explicit CUDA backend:
    ``warmups`` untimed calls absorb compilation, then the median of
    ``repeats`` synchronized ``simulate`` wall times is the default objective.
    ``objective="transport"`` or ``"spectrum"`` instead times just the named
    runner phase with synchronization. The other phase is still evaluated so
    spectra are compared against the baseline. ``on_progress(done, total, state)``
    runs in the coordinator after baseline and each finished/pruned trial; no
    updates occur while one blocking worker is still computing.
    Candidates whose spectral grids differ, or whose densities differ beyond
    ``rtol``/``atol`` from the baseline, are rejected rather than timed
    favorably; the tolerance is a tuning guard, not scientific validation.
    ``trials`` is the total across resumes of a persistent ``storage`` study,
    which refuses to resume a different workload, lattice or device.

    Requires the ``optimization`` extra and a CUDA device.
    """
    from ..campaign.model import Scene

    if not isinstance(scene, Scene):
        raise TypeError("scene must be a pyrite.Scene")
    if not isinstance(lattice, KernelLattice):
        raise TypeError("lattice must be a KernelLattice")
    if min(trials, warmups + 1, repeats) < 1 or timeout <= 0:
        raise ValueError("trials and repeats must be positive, warmups non-negative")
    if objective not in ("total", "transport", "spectrum"):
        raise ValueError("objective must be total, transport, or spectrum")
    if objective == "transport" and (lattice.cxr_chunk or lattice.brem_chunk):
        raise ValueError("transport objective accepts only transport dimensions")
    if objective == "spectrum" and (
        lattice.electron_batch or lattice.schedule or lattice.continuation_rows
    ):
        raise ValueError("spectrum objective accepts only spectral chunk dimensions")
    if rtol < 0 or atol < 0 or not np.isfinite([rtol, atol]).all():
        raise ValueError("rtol and atol must be finite and non-negative")
    base = scene.numerics.kernel_tuning
    if base.transport_core not in ("auto", "cuda"):
        raise ValueError("tune_kernels times the CUDA transport core; use transport_core='auto'")
    base = replace(base, transport_core="cuda")
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("install pyrite-mc[optimization] / uv sync --extra optimization") from exc

    measure = dict(warmups=warmups, repeats=repeats, timeout=timeout, objective=objective)
    with tempfile.TemporaryDirectory(prefix="pyrite-kernel-tuning-") as tmp:
        workdir = Path(tmp)
        reference, record = _measure(scene, base, workdir=workdir, label="baseline", **measure)
        environment = record["environment"]
        baseline = KernelTrial(
            -1, base, "complete", statistics.median(record["samples_s"]), tuple(record["samples_s"])
        )
        if on_progress is not None:
            on_progress(1, trials + 1, "baseline")
        identity = _study_identity(
            scene,
            base,
            lattice,
            warmups=warmups,
            repeats=repeats,
            rtol=rtol,
            atol=atol,
            environment=environment,
            objective=objective,
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            storage=storage,
            study_name=study_name,
            load_if_exists=storage is not None,
        )
        stored = study.user_attrs.get(_SCHEMA)
        if stored not in (None, identity) or (stored is None and study.trials):
            raise ValueError("stored study differs in workload, lattice, protocol or device")
        study.set_user_attr(_SCHEMA, identity)

        def objective_fn(trial: Any) -> float:
            choice = {
                name: trial.suggest_categorical(name, list(getattr(lattice, name)))
                for name in _DIMENSIONS
                if getattr(lattice, name)
            }
            try:
                tuning = lattice.candidate(base, choice)
            except ValueError as exc:
                return _reject(trial, None, f"invalid combination: {exc}")
            trial.set_user_attr("tuning", _tuning_dict(tuning))
            earlier = _earlier_trial(study, trial, optuna)
            if earlier is not None:
                trial.set_user_attr("repeat_of", earlier.number)
                if earlier.state != optuna.trial.TrialState.COMPLETE:
                    return _reject(trial, tuning, earlier.user_attrs["rejection_reason"])
                trial.set_user_attr("samples_s", earlier.user_attrs["samples_s"])
                return float(earlier.value)
            try:
                output, measured = _measure(
                    scene, tuning, workdir=workdir, label=f"trial-{trial.number}", **measure
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                return _reject(trial, tuning, str(exc))
            if measured["environment"] != environment:
                return _reject(trial, tuning, "worker device or precision changed")
            mismatch = _equivalent(output, reference, rtol=rtol, atol=atol)
            if mismatch is not None:
                return _reject(trial, tuning, mismatch)
            trial.set_user_attr("samples_s", measured["samples_s"])
            return float(statistics.median(measured["samples_s"]))

        def _reject(trial, tuning, reason):
            trial.set_user_attr("rejection_reason", reason)
            raise optuna.TrialPruned(reason)

        remaining = max(0, trials - len(study.trials))
        if remaining:
            def report_progress(study, trial):
                if on_progress is not None:
                    on_progress(1 + len(study.trials), trials + 1, trial.state.name.lower())

            study.optimize(objective_fn, n_trials=remaining, callbacks=[report_progress])
    return _summarize(study, baseline, environment, optuna)


def _earlier_trial(study, trial, optuna):
    """The first finished trial that measured the same resolved tuning."""
    tuning = trial.user_attrs["tuning"]
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    for other in study.get_trials(deepcopy=False, states=finished):
        if other.number != trial.number and other.user_attrs.get("tuning") == tuning:
            if "repeat_of" not in other.user_attrs:
                return other
    return None


def _summarize(study, baseline: KernelTrial, environment, optuna) -> KernelTuningResult:
    records = []
    for trial in study.trials:
        tuning = trial.user_attrs.get("tuning")
        tuning = None if tuning is None else _tuning_from_dict(tuning)
        repeat = trial.user_attrs.get("repeat_of")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            samples = tuple(trial.user_attrs.get("samples_s", ()))
            records.append(
                KernelTrial(trial.number, tuning, "complete", trial.value, samples, None, repeat)
            )
        else:
            reason = trial.user_attrs.get("rejection_reason", str(trial.state))
            records.append(
                KernelTrial(trial.number, tuning, "rejected", reason=reason, repeat_of=repeat)
            )
    complete = [t for t in records if t.state == "complete"]
    winner = min(complete, key=_median, default=None)
    chosen = winner if winner is not None and _median(winner) < _median(baseline) else baseline
    return KernelTuningResult(baseline, tuple(records), chosen, environment)


if __name__ == "__main__":
    _worker(*(Path(arg) for arg in sys.argv[1:4]))
