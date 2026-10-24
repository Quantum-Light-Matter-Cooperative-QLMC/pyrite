"""Optional Optuna ask/tell sampler and persistent scientific study adapter.

Optuna's RDB store is the authoritative record for optimizer state and trial
metadata; no parallel JSON ledger that can diverge from it is maintained.
"""

from __future__ import annotations

from typing import Any

from .space import Categorical, Continuous, Integer, ParameterSpace
from .study import Constraint, Objective, OptimizationTrial, Proposal, TrialState

_CONSTRAINT_KEY = "pyrite_constraint_violations"
_TRIAL_KEY = "pyrite_optimization_trial_v1"
_DEFINITION_KEY = "pyrite_optimization_definition_sha256_v1"


class OptunaOptimizer:
    """Optuna generator with crash recovery and definition-checked SQLite resume.

    ``storage`` accepts an Optuna RDB URL (e.g. ``sqlite:///study.db``).
    Persisted studies MUST have a stable ``study_name``. Only one active
    coordinator may operate on a persisted study at a time. Starting a second
    writer concurrently or manually editing Optuna's tables is unsupported.

    The study definition and evaluator identity are bound on the first call to
    ``OptimizationStudy.run``, before asking any trial. Completed trial scores,
    constraint outcomes, and failures live in Optuna's trial user attributes.
    """

    def __init__(
        self,
        parameters: ParameterSpace,
        objectives: tuple[Objective, ...],
        *,
        constraints: tuple[Constraint, ...] = (),
        seed: int | None = None,
        storage: str | None = None,
        study_name: str | None = None,
    ) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna optimizer requires the optional dependency: "
                "uv sync --extra optimization"
            ) from exc
        if not objectives:
            raise ValueError("optimizer requires at least one objective")
        if storage is None and study_name is not None:
            raise ValueError("study_name requires storage")
        if storage is not None and (not isinstance(study_name, str) or not study_name):
            raise ValueError("persistent Optuna optimizer requires a nonempty study_name")
        self._optuna = optuna
        self.parameters = parameters
        self.objectives = tuple(objectives)
        self.constraints = tuple(constraints)
        self.persistent = storage is not None
        sampler_options: dict[str, Any] = {"seed": seed}
        if constraints:
            sampler_options["constraints_func"] = lambda trial: trial.user_attrs.get(
                _CONSTRAINT_KEY, (float("inf"),) * len(constraints)
            )
        self.study = optuna.create_study(
            directions=[obj.direction for obj in objectives],
            sampler=optuna.samplers.TPESampler(**sampler_options),
            storage=storage,
            study_name=study_name,
            load_if_exists=self.persistent,
        )
        if [direction.name.lower() for direction in self.study.directions] != [
            obj.direction for obj in objectives
        ]:
            raise ValueError("existing Optuna study has different objective directions")

    def prepare_resume(self, definition_digest: str | None) -> tuple[OptimizationTrial, ...]:
        """Validate the definition and recover interrupted Optuna trial transitions.

        A trial interrupted before observation metadata was durably recorded is
        marked FAILED, never rewarded and never silently re-evaluated with an
        unspecified seed. A completed observation whose metadata was recorded
        before the crash is finalized without re-running the expensive physics.
        """
        prior = self.study.get_trials(deepcopy=False)
        if self.persistent:
            if not definition_digest:
                raise ValueError("persistent optimization requires definition digest")
            old = self.study.user_attrs.get(_DEFINITION_KEY)
            if old is None:
                if prior:
                    raise ValueError("existing Optuna study lacks PyRITE definition identity")
                self.study.set_user_attr(_DEFINITION_KEY, definition_digest)
            elif old != definition_digest:
                raise ValueError("optimization study definition mismatch on resume")

        for frozen in prior:
            if frozen.state == self._optuna.trial.TrialState.RUNNING:
                stored = frozen.user_attrs.get(_TRIAL_KEY)
                if stored is None or stored.get("state") == TrialState.FAILED.value:
                    self.study.tell(frozen.number, state=self._optuna.trial.TrialState.FAIL)
                else:
                    vals = tuple(stored["objectives"][obj.name] for obj in self.objectives)
                    self.study.tell(frozen.number, vals if len(vals) > 1 else vals[0])
            elif frozen.state not in (
                self._optuna.trial.TrialState.COMPLETE,
                self._optuna.trial.TrialState.FAIL,
            ):
                raise ValueError(f"unsupported stored Optuna trial state: {frozen.state}")
        return tuple(self._load_trial(frozen) for frozen in self.study.get_trials(deepcopy=True))

    def _load_trial(self, frozen: Any) -> OptimizationTrial:
        saved = frozen.user_attrs.get(_TRIAL_KEY)
        if saved is None:
            if frozen.state != self._optuna.trial.TrialState.FAIL:
                raise ValueError(f"trial {frozen.number} lacks durable PyRITE metadata")
            # Abrupt termination before completion metadata was persisted.
            return OptimizationTrial(
                frozen.number, dict(frozen.params), TrialState.FAILED,
                error="Interrupted before trial observables were committed",
            )
        if frozen.params != saved["parameters"]:
            raise ValueError(f"trial {frozen.number} parameter metadata mismatch")
        state = TrialState(saved["state"])
        if state is TrialState.FAILED:
            if frozen.state != self._optuna.trial.TrialState.FAIL:
                raise ValueError("stored trial state disagrees with Optuna failure")
        elif frozen.state != self._optuna.trial.TrialState.COMPLETE:
            raise ValueError("stored trial state disagrees with Optuna completion")
        return OptimizationTrial(
            frozen.number, dict(saved["parameters"]), state,
            dict(saved["objectives"]), dict(saved["constraints"]), saved["error"],
        )

    def ask(self) -> Proposal:
        trial = self.study.ask()
        values: dict[str, Any] = {}
        for path, param in self.parameters.parameters.items():
            if isinstance(param, Continuous):
                values[path] = trial.suggest_float(path, param.low, param.high, log=param.log)
            elif isinstance(param, Integer):
                values[path] = trial.suggest_int(
                    path, param.low, param.high, step=param.step, log=param.log
                )
            elif isinstance(param, Categorical):
                values[path] = trial.suggest_categorical(path, param.choices)
            else:
                raise TypeError(f"unsupported optimization parameter {path!r}")
        return Proposal(values, trial)

    def record_trial(self, proposal: Proposal, result: OptimizationTrial) -> None:
        """Durably write observations BEFORE the Optuna trial transition.

        If a process dies between this write and ``tell``, ``prepare_resume``
        replays the transition using the recorded values (without a new run).
        """
        if result.parameters != proposal.parameters:
            raise ValueError("trial result parameters do not match proposal")
        if result.number != proposal.token.number:
            raise ValueError("trial result number does not match proposal")
        if result.state is not TrialState.FAILED and self.constraints:
            violations = tuple(
                constraint.violation(result.constraints[constraint.name])
                for constraint in self.constraints
            )
            proposal.token.set_user_attr(_CONSTRAINT_KEY, violations)
        proposal.token.set_user_attr(_TRIAL_KEY, {
            "parameters": dict(result.parameters),
            "state": result.state.value,
            "objectives": dict(result.objectives),
            "constraints": dict(result.constraints),
            "error": result.error,
        })

    def tell(
        self,
        proposal: Proposal,
        values: tuple[float, ...] | None,
        *,
        constraint_violations: tuple[float, ...] = (),
        failed: bool = False,
    ) -> None:
        if failed:
            self.study.tell(proposal.token, state=self._optuna.trial.TrialState.FAIL)
            return
        if values is None or len(values) != len(self.objectives):
            raise ValueError("optimizer feedback must match objective count")
        if len(constraint_violations) != len(self.constraints):
            raise ValueError("optimizer feedback must match constraint count")
        if self.constraints:
            proposal.token.set_user_attr(_CONSTRAINT_KEY, constraint_violations)
        self.study.tell(proposal.token, values if len(values) > 1 else values[0])
