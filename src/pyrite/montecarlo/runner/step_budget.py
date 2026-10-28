"""Per-electron transport step budget and its retry after a step-limit abort."""

from ..._env import env_value
from ..transport import TransportStepLimitError

#: First per-electron transport step budget, and the largest the runner doubles
#: it to after a :class:`TransportStepLimitError` (#192).
TRANSPORT_MAX_STEPS = 20_000
TRANSPORT_MAX_STEPS_CEILING = 8 * TRANSPORT_MAX_STEPS


def initial_step_budget() -> int:
    """Initial transport capacity, not a cutoff on electron trajectories.

    The standard allocation remains 20,000 rows per electron. A smaller
    starting value is useful for thin-target tutorials: an exhausted step
    budget triggers a deterministic retry with more capacity.
    """
    name = "PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS"
    raw = env_value(name, str(TRANSPORT_MAX_STEPS))
    try:
        steps = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an integer between 1 and {TRANSPORT_MAX_STEPS_CEILING}"
        ) from exc
    if not 1 <= steps <= TRANSPORT_MAX_STEPS_CEILING:
        raise ValueError(f"{name} must be an integer between 1 and {TRANSPORT_MAX_STEPS_CEILING}")
    return steps


def retry_step_budget(simulate, *args, **kwargs):
    """Run transport with a retrying step-storage capacity.

    Start at :data:`TRANSPORT_MAX_STEPS` by default, double the allocation
    on :class:`TransportStepLimitError` and stop at
    :data:`TRANSPORT_MAX_STEPS_CEILING`, where the error propagates.
    The initial capacity can be reduced using
    ``PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS`` without truncating tracks.

    Thick MeV cases outlive the default step budget (5 MeV h-BN, 10 mm:
    ~23-32k steps). The trajectories depend on the seed alone, so a rerun at a
    doubled budget is exactly the run that budget gives from the start.
    """
    max_steps = initial_step_budget()
    while True:
        try:
            return simulate(*args, max_steps=max_steps, **kwargs)
        except TransportStepLimitError:
            if max_steps >= TRANSPORT_MAX_STEPS_CEILING:
                raise
            max_steps = min(2 * max_steps, TRANSPORT_MAX_STEPS_CEILING)
