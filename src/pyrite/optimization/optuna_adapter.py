"""Optional Optuna ask/tell sampler; no dependency on PyRITE execution backends."""

from __future__ import annotations

from typing import Any

from .space import Categorical, Continuous, Integer, ParameterSpace
from .study import Constraint, Objective, Proposal

_CONSTRAINT_KEY = "pyrite_constraint_violations"


class OptunaOptimizer:
    """Optuna candidate generator supporting mixed spaces and Pareto studies.

    This initial adapter uses in-memory studies. Persistent storage requires
    durable, checked scene/objective identities and is a separate slice of #682.
    Trials are reported through ``tell`` only after their evaluation completes.
    """

    def __init__(
        self,
        parameters: ParameterSpace,
        objectives: tuple[Objective, ...],
        *,
        constraints: tuple[Constraint, ...] = (),
        seed: int | None = None,
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
        self._optuna = optuna
        self.parameters = parameters
        self.objectives = objectives
        self.constraints = constraints
        sampler_options: dict[str, Any] = {"seed": seed}
        if constraints:
            sampler_options["constraints_func"] = lambda trial: trial.user_attrs.get(
                _CONSTRAINT_KEY, (float("inf"),) * len(constraints)
            )
        self.study = optuna.create_study(
            directions=[obj.direction for obj in objectives],
            sampler=optuna.samplers.TPESampler(**sampler_options),
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
