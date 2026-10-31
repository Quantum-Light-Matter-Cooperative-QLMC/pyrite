"""Optuna tuning of CUDA execution controls for one fixed Scene (#776).

``tune_kernels`` tunes transport and spectrum controls as two independent
studies by default. Transport candidates run only electron transport
(:func:`pyrite.phases.simulate_transport`) and must reproduce the baseline's
transported arrays bit for bit. Spectrum candidates rescore one kept
transport (:func:`pyrite.phases.score_spectra`) and must reproduce its spectra
within tolerance. The controls are execution choices, not physical
parameters, so this is separate from :class:`OptimizationStudy`. Results are
specific to the GPU, driver, precision and workload that produced them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._simulation_progress import SimulationProgress
from ._kernel_worker import tuning_from_dict

_SCHEMA = "pyrite-kernel-tuning-v2"
_PHASE_DIMENSIONS = {
    "transport": (
        "electron_batch",
        "schedule",
        "continuation_rows",
        "scratch_mib",
        "kernel_threads",
    ),
    "spectrum": ("cxr_chunk", "brem_chunk"),
}
_DIMENSIONS = _PHASE_DIMENSIONS["transport"] + _PHASE_DIMENSIONS["spectrum"]
Phase = Literal["transport", "spectrum", "total"]


@dataclass(frozen=True)
class KernelLattice:
    """Candidate values per execution control; an empty tuple keeps the scene's value.

    ``electron_batch``, ``schedule``, ``continuation_rows``, ``scratch_mib``
    and ``kernel_threads`` set :class:`~pyrite.config.TransportKernelTuning`
    (the transport phase);
    ``cxr_chunk`` and ``brem_chunk`` set the spectrum segment chunks (the
    spectrum phase). ``None`` in a tuple selects the process default.
    Combinations the config rejects (a row quantum with full-history) are
    never run.
    """

    electron_batch: tuple[int | None, ...] = ()
    cxr_chunk: tuple[int | None, ...] = ()
    brem_chunk: tuple[int | None, ...] = ()
    schedule: tuple[Literal["full-history", "continuation", "auto"] | None, ...] = ()
    continuation_rows: tuple[int | None, ...] = ()
    scratch_mib: tuple[int | None, ...] = ()
    kernel_threads: tuple[int | None, ...] = ()

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
            base.transport, **{k: choice[k] for k in _PHASE_DIMENSIONS["transport"] if k in choice}
        )
        spectral = {k: choice[k] for k in _PHASE_DIMENSIONS["spectrum"] if k in choice}
        return replace(base, transport=transport, **spectral)

    def dimensions(self, phase: Phase) -> tuple[str, ...]:
        """Tuned dimensions of ``phase`` (``"total"``: all of them)."""
        names = _DIMENSIONS if phase == "total" else _PHASE_DIMENSIONS[phase]
        return tuple(name for name in names if getattr(self, name))

    def choices(self, base: Any, phase: Phase) -> list[dict[str, Any]]:
        """Every valid combination of ``phase``'s dimensions, in lattice order."""
        names = self.dimensions(phase)
        valid = []
        for values in itertools.product(*(getattr(self, name) for name in names)):
            choice = dict(zip(names, values, strict=True))
            try:
                self.candidate(base, choice)
            except ValueError:
                continue
            valid.append(choice)
        return valid


@dataclass(frozen=True)
class KernelTrial:
    """One measured candidate; ``median_s`` is None when it was rejected.

    ``repeat_of`` names the earlier trial whose outcome a resampled, identical
    candidate reused instead of being timed again; ``-1`` marks a candidate
    equal to the baseline, which reuses the baseline's timings.
    """

    number: int
    tuning: Any
    state: Literal["complete", "rejected"]
    median_s: float | None = None
    samples_s: tuple[float, ...] = ()
    reason: str | None = None
    repeat_of: int | None = None


@dataclass(frozen=True)
class KernelPhaseResult:
    """One phase's study: baseline, every trial, and the configuration kept.

    ``chosen`` is the fastest equivalent trial when an interleaved re-timing
    against the baseline (``confirmation_s``: baseline samples, candidate
    samples) confirmed at least the requested speedup; otherwise the baseline.
    ``median_s`` times only this phase (``"total"``: a whole simulation).
    """

    phase: Phase
    baseline: KernelTrial
    trials: tuple[KernelTrial, ...]
    chosen: KernelTrial
    confirmation_s: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    @property
    def best(self) -> Any:
        """The ``KernelTuning`` this phase keeps."""
        return self.chosen.tuning

    @property
    def improved(self) -> bool:
        """Whether a confirmed candidate replaced the baseline."""
        return self.chosen is not self.baseline

    @property
    def speedup(self) -> float:
        """Baseline over chosen median; from the confirmation timings when run."""
        if self.confirmation_s is not None and self.improved:
            base, best = self.confirmation_s
            return statistics.median(base) / statistics.median(best)
        return _median(self.baseline) / _median(self.chosen)


@dataclass(frozen=True)
class KernelTuningResult:
    """Every phase study and the combined configuration to apply.

    ``best`` joins the transport phase's transport controls with the spectrum
    phase's chunks; apply it with ``numerics.with_kernel_tuning(best)``.
    ``environment`` records the device, driver, CUDA/CuPy versions and
    precision: the timings do not transfer to other hardware. ``baseline``,
    ``trials``, ``chosen``, ``improved`` and ``speedup`` read the single
    phase of a one-phase tuning.
    """

    phases: tuple[KernelPhaseResult, ...]
    best: Any
    environment: dict[str, Any] = field(default_factory=dict)

    def phase(self, name: Phase) -> KernelPhaseResult | None:
        """The study of phase ``name``, or None when it was not tuned."""
        return next((p for p in self.phases if p.phase == name), None)

    @property
    def transport(self) -> KernelPhaseResult | None:
        return self.phase("transport")

    @property
    def spectrum(self) -> KernelPhaseResult | None:
        return self.phase("spectrum")

    def _only(self) -> KernelPhaseResult:
        if len(self.phases) != 1:
            raise AttributeError("several phases were tuned; read result.transport/.spectrum")
        return self.phases[0]

    baseline = property(lambda self: self._only().baseline, doc="The single phase's baseline.")
    trials = property(lambda self: self._only().trials, doc="The single phase's trials.")
    chosen = property(lambda self: self._only().chosen, doc="The single phase's chosen trial.")
    improved = property(lambda self: any(p.improved for p in self.phases))
    speedup = property(lambda self: self._only().speedup, doc="The single phase's speedup.")


@dataclass(frozen=True)
class TuningProgress:
    """One ``tune_kernels`` progress update.

    ``trial`` counts this phase's runs (0 is the baseline) out of ``trials``;
    ``state`` is ``"running"``, ``"baseline"``, ``"complete"``,
    ``"rejected"`` or ``"confirmed"``/``"unconfirmed"``. While a candidate
    runs, ``simulation`` carries its batch/stage progress and ``run`` counts
    its warm-up and timed runs out of ``runs``.
    """

    phase: Phase
    trial: int
    trials: int
    state: str
    run: int = 0
    runs: int = 0
    simulation: SimulationProgress | None = None


def _median(trial: KernelTrial) -> float:
    if trial.median_s is None:
        raise ValueError(f"trial {trial.number} was rejected and has no timing")
    return float(trial.median_s)


class _WorkerError(RuntimeError):
    pass


class _Worker:
    """One persistent measurement process; see :mod:`._kernel_worker`."""

    def __init__(self, payload: Path, workdir: Path, timeout: float, on_event) -> None:
        self._payload, self._workdir, self._timeout = payload, workdir, timeout
        self._on_event = on_event
        self._process: subprocess.Popen[str] | None = None
        self.environment: dict[str, Any] | None = None

    def _start(self) -> None:
        env = os.environ | {"PYRITE_MC_BACKEND": "cuda"}  # no hidden CPU fallback in timings
        self._stderr = tempfile.TemporaryFile("w+", dir=self._workdir)
        command = [sys.executable, "-m", "pyrite.optimization._kernel_worker", str(self._payload)]
        self._process = subprocess.Popen(
            command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, text=True, bufsize=1,
        )  # fmt: skip
        self._lines: queue.Queue[str | None] = queue.Queue()
        stdout = self._process.stdout
        assert stdout is not None

        def pump() -> None:
            for line in stdout:
                self._lines.put(line)
            self._lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        ready = self._next()
        if ready.get("event") != "ready":
            raise _WorkerError(f"worker did not start: {ready}")
        self.environment = ready["environment"]

    def _next(self) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=self._timeout)
        except queue.Empty:
            self.close()
            raise _WorkerError(f"no worker reply within {self._timeout:g} s") from None
        if line is None:
            self._stderr.seek(0)
            detail = self._stderr.read().strip()[-2500:]
            self.close()
            raise _WorkerError(f"worker exited: {detail}")
        return json.loads(line)

    def measure(self, request: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        if self._process is None:
            self._start()
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        while True:
            message = self._next()
            if message["event"] == "progress":
                self._on_event(message)
            elif message["event"] == "error":
                raise _WorkerError(message["message"])
            else:
                return message["samples_s"], message["check"]

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait()
            self._process = None


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


def _mismatch(check: dict[str, Any], reference: dict[str, Any], rtol: float, atol: float):
    if "recovery" in check:
        return f"needed allocation recovery: {check['recovery']}"
    if "digest" in reference:
        return None if check["digest"] == reference["digest"] else "transport arrays differ"
    return _equivalent(Path(check["spectra"]), Path(reference["spectra"]), rtol=rtol, atol=atol)


def _study_identity(scene, base, lattice, phase, protocol, environment) -> str:
    from ..campaign.lowering import build_case
    from ..campaign.profiles import case_content_key

    payload = {
        "schema": _SCHEMA,
        "phase": phase,
        "case": case_content_key(build_case(scene)),
        "baseline": asdict(base),
        "lattice": {name: getattr(lattice, name) for name in lattice.dimensions(phase)},
        "protocol": protocol,
        "environment": environment,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class _PhaseRun:
    """State one phase study shares between its Optuna objective and summary."""

    phase: Phase
    lattice: KernelLattice
    base: Any
    measure: Callable[..., tuple[list[float], dict[str, Any]]]
    report: Callable[..., None]
    rtol: float
    atol: float
    reference: dict[str, Any] = field(default_factory=dict)
    baseline_s: tuple[float, ...] = ()

    def objective(self, trial: Any, optuna: Any, study: Any, total: int) -> float:
        choice = {
            name: trial.suggest_categorical(name, list(getattr(self.lattice, name)))
            for name in self.lattice.dimensions(self.phase)
        }
        try:
            tuning = self.lattice.candidate(self.base, choice)
        except ValueError as exc:
            return self._reject(trial, optuna, f"invalid combination: {exc}")
        trial.set_user_attr("tuning", asdict(tuning))
        if tuning == self.base:  # the baseline itself: reuse its measurement
            trial.set_user_attr("repeat_of", -1)
            trial.set_user_attr("samples_s", list(self.baseline_s))
            return float(statistics.median(self.baseline_s))
        earlier = _earlier_trial(study, trial, optuna)
        if earlier is not None:
            trial.set_user_attr("repeat_of", earlier.number)
            if earlier.state != optuna.trial.TrialState.COMPLETE:
                return self._reject(trial, optuna, earlier.user_attrs["rejection_reason"])
            trial.set_user_attr("samples_s", earlier.user_attrs["samples_s"])
            return float(earlier.value)
        label = f"{self.phase}-{trial.number}"
        number = min(1 + sum(t.state.is_finished() for t in study.trials), total)
        self.report(self.phase, number, total, "running")
        try:
            samples, check = self.measure(self.phase, tuning, label, number, total)
        except _WorkerError as exc:
            return self._reject(trial, optuna, str(exc))
        mismatch = _mismatch(check, self.reference, self.rtol, self.atol)
        if mismatch is not None:
            return self._reject(trial, optuna, mismatch)
        trial.set_user_attr("samples_s", samples)
        return float(statistics.median(samples))

    @staticmethod
    def _reject(trial, optuna, reason):
        trial.set_user_attr("rejection_reason", reason)
        raise optuna.TrialPruned(reason)


def tune_kernels(
    scene: Any,
    lattice: KernelLattice,
    *,
    trials: int | None = None,
    warmups: int = 1,
    repeats: int = 3,
    rtol: float = 1e-4,
    atol: float = 1e-12,
    min_speedup: float = 1.02,
    storage: str | None = None,
    study_name: str = "kernel-tuning",
    timeout: float = 1800.0,
    seed: int = 42,
    objective: Literal["phases", "total"] = "phases",
    isolate: Literal["phase", "candidate"] = "phase",
    on_progress: Callable[[TuningProgress], None] | None = None,
) -> KernelTuningResult:
    """Tune ``scene``'s CUDA execution controls over ``lattice`` with Optuna.

    The baseline is the scene's own ``numerics.kernel_tuning`` with
    ``transport_core="cuda"``; every candidate changes only ``lattice``
    dimensions.

    With ``objective="phases"`` (default) the transport and spectrum
    dimensions are independent studies: transport candidates time
    :func:`~pyrite.phases.simulate_transport` alone and must reproduce the
    baseline transport bit for bit; spectrum candidates time
    :func:`~pyrite.phases.score_spectra` on one kept baseline transport, and
    their spectra must match the baseline's within ``rtol``/``atol``. Each
    phase searches only its own dimensions, so the work is the sum of the two
    lattices, not their product, and ``best`` joins both winners.
    ``objective="total"`` instead times whole :func:`pyrite.simulate` calls
    over every dimension jointly.

    ``trials`` per phase defaults to every valid combination; a smaller count
    searches with TPE. Each candidate gets ``warmups`` untimed then ``repeats``
    synchronized runs; the fastest is then re-timed against the baseline in
    ``repeats`` interleaved pairs and kept only when that confirms at least
    ``min_speedup``. Candidates that need allocation-retry chunk halving are
    rejected rather than timed favorably; the tolerance is a tuning guard,
    not scientific validation.

    Measurements run in a CUDA worker process separate from the caller.
    ``isolate="phase"`` (default) reuses one worker per phase, so imports,
    tables and kernel compilation are paid once and the memory pool is
    released before each candidate; ``"candidate"`` starts a fresh worker per
    candidate, as cold as a new run but much slower. ``on_progress`` receives
    :class:`TuningProgress` per candidate and, while one runs, per transport
    batch and spectrum stage. ``trials`` counts across resumes of a persistent
    ``storage`` study, which refuses a different workload, lattice or device.

    Requires the ``optimization`` extra and a CUDA device.
    """
    base = _checked_baseline(
        scene, lattice, trials, warmups, repeats, timeout, objective, isolate, rtol, atol,
        min_speedup,
    )  # fmt: skip
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("install pyrite-mc[optimization] / uv sync --extra optimization") from exc

    phases: tuple[Phase, ...] = (
        ("total",)
        if objective == "total"
        else tuple(p for p in ("transport", "spectrum") if lattice.dimensions(p))
    )
    protocol = [warmups, repeats, rtol, atol, min_speedup, isolate]
    with tempfile.TemporaryDirectory(prefix="pyrite-kernel-tuning-") as tmp:
        workdir = Path(tmp)
        payload = workdir / "payload.pkl"
        payload.write_bytes(
            pickle.dumps(
                {"scene": scene, "base": asdict(base), "warmups": warmups,
                 "repeats": repeats, "workdir": str(workdir)}
            )
        )  # fmt: skip
        tuner = _Tuner(workdir, payload, timeout, isolate, on_progress)
        try:
            results = tuple(
                tuner.run_phase(
                    phase,
                    scene,
                    lattice,
                    base,
                    optuna,
                    trials=trials,
                    seed=seed,
                    rtol=rtol,
                    atol=atol,
                    repeats=repeats,
                    min_speedup=min_speedup,
                    storage=storage,
                    study_name=f"{study_name}:{phase}",
                    protocol=protocol,
                )  # fmt: skip
                for phase in phases
            )
        finally:
            tuner.close()
    best = base
    for result in results:
        best = _join(best, result.best, result.phase)
    return KernelTuningResult(results, best, tuner.environment or {})


def _checked_baseline(
    scene, lattice, trials, warmups, repeats, timeout, objective, isolate, rtol, atol, min_speedup
) -> Any:
    """Validate ``tune_kernels`` arguments; the scene's tuning on the CUDA core."""
    from ..campaign.model import Scene

    if not isinstance(scene, Scene):
        raise TypeError("scene must be a pyrite.Scene")
    if not isinstance(lattice, KernelLattice):
        raise TypeError("lattice must be a KernelLattice")
    if (trials is not None and trials < 1) or warmups < 0 or repeats < 1 or timeout <= 0:
        raise ValueError("trials and repeats must be positive, warmups non-negative")
    if objective not in ("phases", "total") or isolate not in ("phase", "candidate"):
        raise ValueError("objective must be 'phases' or 'total'; isolate 'phase' or 'candidate'")
    if rtol < 0 or atol < 0 or not np.isfinite([rtol, atol]).all() or min_speedup < 1:
        raise ValueError("rtol and atol must be finite and non-negative, min_speedup >= 1")
    base = scene.numerics.kernel_tuning
    if base.transport_core not in ("auto", "cuda"):
        raise ValueError("tune_kernels times the CUDA transport core; use transport_core='auto'")
    return replace(base, transport_core="cuda")


def _join(best: Any, chosen: Any, phase: Phase) -> Any:
    if phase == "transport":
        return replace(best, transport=chosen.transport)
    if phase == "spectrum":
        return replace(best, cxr_chunk=chosen.cxr_chunk, brem_chunk=chosen.brem_chunk)
    return chosen


class _Tuner:
    """Worker lifecycle, environment pinning and progress for one tuning."""

    def __init__(self, workdir, payload, timeout, isolate, on_progress) -> None:
        self.workdir, self.isolate, self.on_progress = workdir, isolate, on_progress
        self.environment: dict[str, Any] | None = None
        self._context: tuple[Phase, int, int] = ("total", 0, 0)
        self._worker = _Worker(payload, workdir, timeout, self._forward)

    def report(self, phase, trial, trials, state, **extra) -> None:
        self._context = (phase, trial, trials)
        if self.on_progress is not None:
            self.on_progress(TuningProgress(phase, trial, trials, state, **extra))

    def _forward(self, message: dict[str, Any]) -> None:
        phase, trial, trials = self._context
        event = SimulationProgress(
            message["phase"], message["stage"], message["done"], message["total"]
        )
        self.report(
            phase, trial, trials, "running",
            run=message["run"], runs=message["runs"], simulation=event,
        )  # fmt: skip

    def measure(self, phase, tuning, label, trial, trials, **counts):
        self._context = (phase, trial, trials)
        request = {"phase": phase, "tuning": asdict(tuning), "label": label, **counts}
        try:
            samples, check = self._worker.measure(request)
        finally:
            if self.isolate == "candidate":
                self._worker.close()
        env = self._worker.environment
        if self.environment is None:
            self.environment = env
        elif env != self.environment:
            raise _WorkerError("worker device or precision changed")
        return samples, check

    def close(self) -> None:
        self._worker.close()

    def run_phase(
        self, phase, scene, lattice, base, optuna, *, trials, seed, rtol, atol, repeats,
        min_speedup, storage, study_name, protocol,
    ) -> KernelPhaseResult:  # fmt: skip
        valid = lattice.choices(base, phase)
        exhaustive = trials is None or trials >= len(valid)
        total = len(valid) if exhaustive else int(trials)
        self.report(phase, 0, total, "running")
        samples, reference = self.measure(phase, base, f"{phase}-baseline", 0, total)
        baseline = KernelTrial(-1, base, "complete", statistics.median(samples), tuple(samples))
        self.report(phase, 0, total, "baseline")
        identity = _study_identity(scene, base, lattice, phase, protocol, self.environment)
        # Exhaustive studies enqueue every valid combination; TPE otherwise.
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=storage,
            study_name=study_name,
            load_if_exists=storage is not None,
        )
        stored = study.user_attrs.get(_SCHEMA)
        if stored not in (None, identity) or (stored is None and study.trials):
            raise ValueError("stored study differs in workload, lattice, protocol or device")
        study.set_user_attr(_SCHEMA, identity)
        if exhaustive:
            done = {json.dumps(t.params, sort_keys=True) for t in study.trials}
            pending = [c for c in valid if json.dumps(c, sort_keys=True) not in done]
            for choice in pending:
                study.enqueue_trial(choice)
            remaining = len(pending)
        else:
            remaining = max(0, total - len(study.trials))
        run = _PhaseRun(
            phase,
            lattice,
            base,
            self.measure,
            self.report,
            rtol,
            atol,
            reference,
            baseline.samples_s,
        )

        def finished(study, trial):
            state = "complete" if trial.state == optuna.trial.TrialState.COMPLETE else "rejected"
            self.report(phase, min(len(study.trials), total), total, state)

        if remaining:
            study.optimize(
                lambda trial: run.objective(trial, optuna, study, total),
                n_trials=remaining,
                callbacks=[finished],
            )
        records = _records(study, optuna)
        return self._confirm(phase, baseline, records, repeats, min_speedup, total)

    def _confirm(self, phase, baseline, records, repeats, min_speedup, total):
        """Re-time the fastest candidate against the baseline in interleaved pairs."""
        complete = [t for t in records if t.state == "complete" and t.repeat_of is None]
        winner = min(complete, key=_median, default=None)
        if winner is None or _median(winner) >= _median(baseline):
            return KernelPhaseResult(phase, baseline, records, baseline)
        pairs: tuple[list[float], list[float]] = ([], [])
        try:
            for _ in range(repeats):
                for side, tuning in enumerate((baseline.tuning, winner.tuning)):
                    samples, _check = self.measure(
                        phase, tuning, f"{phase}-confirm", total, total, warmups=0, repeats=1
                    )
                    pairs[side].extend(samples)
        except _WorkerError:
            return KernelPhaseResult(phase, baseline, records, baseline)
        confirmation = (tuple(pairs[0]), tuple(pairs[1]))
        confirmed = statistics.median(pairs[0]) >= min_speedup * statistics.median(pairs[1])
        self.report(phase, total, total, "confirmed" if confirmed else "unconfirmed")
        chosen = winner if confirmed else baseline
        return KernelPhaseResult(phase, baseline, records, chosen, confirmation)


def _earlier_trial(study, trial, optuna):
    """The first finished trial that measured the same resolved tuning."""
    tuning = trial.user_attrs["tuning"]
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    for other in study.get_trials(deepcopy=False, states=finished):
        if other.number != trial.number and other.user_attrs.get("tuning") == tuning:
            if "repeat_of" not in other.user_attrs:
                return other
    return None


def _records(study, optuna) -> tuple[KernelTrial, ...]:
    records = []
    for trial in study.trials:
        tuning = trial.user_attrs.get("tuning")
        tuning = None if tuning is None else tuning_from_dict(tuning)
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
    return tuple(records)
