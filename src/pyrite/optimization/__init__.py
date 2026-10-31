"""Optional scientific optimization above Scene, Sweep and #681 executors.

The public API here does not import Optuna or launch a numerical engine until
explicitly requested. No change to existing sweeping or execution defaults.
"""

from .kernel_tuning import (
    KernelPhaseResult,
    KernelSearchSpace,
    KernelTrial,
    KernelTuningResult,
    TuningProgress,
    tune_kernels,
)
from .optuna_adapter import OptunaOptimizer
from .scoring import FixedWindowYield
from .space import Categorical, Continuous, Integer, ParameterSpace, apply_scene_parameters
from .study import (
    Constraint,
    Objective,
    OptimizationResult,
    OptimizationStudy,
    OptimizationTrial,
    Optimizer,
    Proposal,
    PyriteEvaluator,
    TrialState,
)

__all__ = [
    "Categorical",
    "Continuous",
    "Integer",
    "ParameterSpace",
    "FixedWindowYield",
    "KernelSearchSpace",
    "KernelPhaseResult",
    "KernelTrial",
    "KernelTuningResult",
    "apply_scene_parameters",
    "Constraint",
    "Objective",
    "OptimizationResult",
    "OptimizationStudy",
    "OptimizationTrial",
    "Optimizer",
    "OptunaOptimizer",
    "Proposal",
    "PyriteEvaluator",
    "TrialState",
    "TuningProgress",
    "tune_kernels",
]


def __getattr__(name: str):
    # ``KernelLattice`` was renamed to ``KernelSearchSpace`` (#825); experimental
    # API, kept as a warning alias for migration.
    if name == "KernelLattice":
        import warnings

        warnings.warn(
            "pyrite.optimization.KernelLattice is renamed KernelSearchSpace",
            DeprecationWarning,
            stacklevel=2,
        )
        return KernelSearchSpace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
