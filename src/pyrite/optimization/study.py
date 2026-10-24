"""Backend-independent optimization study lifecycle and evaluation adapter.

No optimizer dependency or Monte Carlo engine is imported at module import time.
The execution backend remains owned by ``runs.run_sweep`` (#681).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, Literal, Protocol

from .space import ParameterSpace, apply_scene_parameters

Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True, slots=True)
class Objective:
    """Named scalar observable with an explicit unit and optimization direction."""

    name: str
    direction: Direction
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("objective name must be nonempty")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("objective direction must be minimize or maximize")
        if not isinstance(self.unit, str):
            raise TypeError("objective unit must be a string (empty means dimensionless)")


@dataclass(frozen=True, slots=True)
class Constraint:
    """Physical feasibility bound evaluated on the same observation dictionary."""

    name: str
    relation: Literal["<=", ">="]
    bound: float
    unit: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name must be nonempty")
        if self.relation not in ("<=", ">="):
            raise ValueError("constraint relation must be <= or >=")
        if not isfinite(self.bound):
            raise ValueError("constraint bound must be finite")
        if not isinstance(self.unit, str):
            raise TypeError("constraint unit must be a string")

    def violation(self, value: float) -> float:
        """Nonpositive is feasible; positive is an actual constraint violation."""
        return value - self.bound if self.relation == "<=" else self.bound - value


@dataclass(frozen=True)
class Proposal:
    """Optimizer proposal with opaque backend token (never a physics-cache ID)."""

    parameters: Mapping[str, Any]
    token: Any = field(repr=False, compare=False)


class Optimizer(Protocol):
    """Ask/tell contract allowing later batched/distributed scheduling."""

    def ask(self) -> Proposal: ...

    def tell(
        self,
        proposal: Proposal,
        values: tuple[float, ...] | None,
        *,
        constraint_violations: tuple[float, ...] = (),
        failed: bool = False,
    ) -> None: ...


class TrialState(StrEnum):
    COMPLETED = "completed"
    INFEASIBLE = "infeasible"
    FAILED = "failed"


@dataclass(frozen=True)
class OptimizationTrial:
    number: int
    parameters: Mapping[str, Any]
    state: TrialState
    objectives: Mapping[str, float] = field(default_factory=dict)
    constraints: Mapping[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class OptimizationResult:
    objectives: tuple[Objective, ...]
    trials: tuple[OptimizationTrial, ...]

    @property
    def feasible_trials(self) -> tuple[OptimizationTrial, ...]:
        return tuple(t for t in self.trials if t.state is TrialState.COMPLETED)

    @property
    def pareto_front(self) -> tuple[OptimizationTrial, ...]:
        """All nondominated *feasible* trials; deterministic proposal order."""
        good = self.feasible_trials

        def dominates(a: OptimizationTrial, b: OptimizationTrial) -> bool:
            comparisons = [
                (a.objectives[o.name] >= b.objectives[o.name])
                if o.direction == "maximize" else
                (a.objectives[o.name] <= b.objectives[o.name])
                for o in self.objectives
            ]
            strict = any(a.objectives[o.name] != b.objectives[o.name] for o in self.objectives)
            return all(comparisons) and strict

        return tuple(t for t in good if not any(dominates(other, t) for other in good))

    @property
    def best_trial(self) -> OptimizationTrial | None:
        if len(self.objectives) != 1:
            raise ValueError(
                "best_trial is undefined for multi-objective studies; use pareto_front"
            )
        feasible = self.feasible_trials
        if not feasible:
            return None
        obj = self.objectives[0]
        return (max if obj.direction == "maximize" else min)(
            feasible, key=lambda trial: trial.objectives[obj.name]
        )


@dataclass(frozen=True)
class OptimizationStudy:
    """Scientific study over an immutable PyRITE scene.

    The evaluator receives ``(resolved_scene, trial_number)`` and returns
    scalar observables in the units declared by objective/constraint definitions.
    The trial number lets an evaluator choose independent Monte Carlo seeds;
    it is *not* a simulation content key.

    Execution remains serial in this initial slice; the ask/tell boundary makes
    batch evaluation possible later without replacing the executor (#681).
    """

    base: Any
    parameters: ParameterSpace
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not self.objectives:
            raise ValueError("at least one objective is required")
        if len({o.name for o in self.objectives}) != len(self.objectives):
            raise ValueError("duplicate objective names")
        if len({c.name for c in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate constraint names")
        # Validate paths with their *existing* leaf values: the lower bounds
        # may jointly violate physical constraints even for a feasible study.
        from ..campaign.model import Scene, _descend, _parts, _replace_path

        if not isinstance(self.base, Scene):
            raise TypeError("base must be a PyRITE Scene")
        for path in self.parameters.parameters:
            if path.split(".", 1)[0].split("[", 1)[0] not in (
                "beam", "target", "detector", "filters", "acquisition", "physics", "quantum"
            ):
                raise ValueError(f"{path!r} is not a physical Scene parameter path")
            parsed = _parts(path)
            current = self.base
            for name, indexes in parsed:
                current = _descend(current, name, indexes, path)
            _replace_path(self.base, parsed, current, path)

    def scene(self, values: Mapping[str, Any]) -> Any:
        return apply_scene_parameters(self.base, self.parameters, values)

    def run(
        self,
        evaluator: Callable[[Any, int], Mapping[str, float]],
        optimizer: Optimizer,
        *,
        max_trials: int,
        on_trial: Callable[[OptimizationTrial], None] | None = None,
    ) -> OptimizationResult:
        if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
            raise ValueError("max_trials must be a positive integer")
        # Optuna (and other configured adapters) must not silently optimize
        # different objectives, constraint bounds, or parameter distributions.
        for attr in ("parameters", "objectives", "constraints"):
            configured = getattr(optimizer, attr, None)
            if configured is not None and configured != getattr(self, attr):
                raise ValueError(f"optimizer {attr} do not match the study")
        history: list[OptimizationTrial] = []
        for number in range(max_trials):
            proposal = optimizer.ask()
            try:
                scene = self.scene(proposal.parameters)
                measured = evaluator(scene, number)
                needed = {o.name for o in self.objectives} | {c.name for c in self.constraints}
                missing = needed - measured.keys()
                if missing:
                    raise ValueError(f"evaluator missing observables: {sorted(missing)}")
                objective_values = {o.name: float(measured[o.name]) for o in self.objectives}
                constraint_values = {c.name: float(measured[c.name]) for c in self.constraints}
                observed_values = (*objective_values.values(), *constraint_values.values())
                if any(not isfinite(x) for x in observed_values):
                    raise ValueError("non-finite objective or constraint value")
                violations = tuple(c.violation(constraint_values[c.name]) for c in self.constraints)
            except Exception as exc:
                # Only candidate/physics/scoring failures are recoverable. A
                # failed backend tell is an optimizer-state error and propagates.
                optimizer.tell(proposal, None, failed=True)
                trial = OptimizationTrial(number, dict(proposal.parameters), TrialState.FAILED,
                                          error=f"{type(exc).__name__}: {exc}")
            else:
                optimizer.tell(
                    proposal, tuple(objective_values.values()), constraint_violations=violations
                )
                state = (
                    TrialState.INFEASIBLE if any(v > 0 for v in violations)
                    else TrialState.COMPLETED
                )
                trial = OptimizationTrial(number, dict(proposal.parameters), state,
                                          objective_values, constraint_values)
            history.append(trial)
            if on_trial is not None:
                on_trial(trial)
        return OptimizationResult(self.objectives, tuple(history))


@dataclass(frozen=True)
class PyriteEvaluator:
    """Use existing scene lowering, sweep checkpoint/CAS and executor contracts.

    ``score`` reduces normal ``run_sweep`` records to scalar metrics. The score
    and analysis configuration, including energy window and beam-current
    normalization, are the caller's responsibility: never use sampled peak
    heights as a default across nonidentical energy grids.

    A distinct seed per trial prevents accidental reuse of physical results for
    independent Monte Carlo replicates, assuming ``case_content_key`` includes
    the case seed. Do not provide a custom key function that omits that seed.
    """

    score: Callable[[dict], Mapping[str, float]]
    executor: Any = None
    run_options: Mapping[str, Any] = field(default_factory=dict)
    base_seed: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError("base_seed must be a nonnegative integer")
        reserved = {"executor", "on_case", "on_progress"}
        if reserved & self.run_options.keys():
            raise ValueError(f"run_options may not override {sorted(reserved)}")

    def __call__(self, scene: Any, trial_number: int) -> Mapping[str, float]:
        from dataclasses import replace

        from ..campaign.model import Sweep
        from ..runs.run import run_sweep

        case, = Sweep(base=scene).cases()
        # The name is the checkpoint-record key, whereas content keys exclude
        # the name and include the seed. Keep replicate records distinct too.
        case = replace(
            case, seed=self.base_seed + trial_number,
            name=f"{case.name} [optimization-trial-{trial_number:08d}]",
        )
        results: dict = {}
        options = dict(self.run_options)
        options.setdefault("progress", False)
        # Per-study durable resume needs its own compatibility-checked identity.
        # Do not blindly reuse checkpoints from earlier unrelated studies.
        options.setdefault("resume", False)
        complete = run_sweep([case], results, executor=self.executor, **options)
        if not complete:
            raise RuntimeError("sweep stopped before trial completed")
        return self.score(results)
