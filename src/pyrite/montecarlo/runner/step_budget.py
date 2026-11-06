"""Per-electron transport step budget and its retry after a step-limit abort."""

from ..._env import env_value
from ..transport import TransportStepLimitError

#: Default per-electron transport step limit. A smaller initial budget from
#: ``PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS`` doubles up to the ceiling after a
#: :class:`TransportStepLimitError` (#192).
TRANSPORT_MAX_STEPS = 400_000
TRANSPORT_MAX_STEPS_CEILING = TRANSPORT_MAX_STEPS


def initial_step_budget() -> int:
    """Initial step budget, not a cutoff below the transport step limit.

    A smaller starting value reduces fixed lockstep buffers: an exhausted step
    budget triggers a deterministic retry with a larger one.
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

    Start at :data:`TRANSPORT_MAX_STEPS` by default, where the error propagates.
    ``PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS`` starts lower without truncating
    tracks: the budget doubles on :class:`TransportStepLimitError` and stops at
    :data:`TRANSPORT_MAX_STEPS_CEILING`.

    The trajectories depend on the seed alone, so a rerun at a doubled budget
    is exactly the run that budget gives from the start.
    """
    max_steps = initial_step_budget()
    while True:
        try:
            return simulate(*args, max_steps=max_steps, **kwargs)
        except TransportStepLimitError:
            if max_steps >= TRANSPORT_MAX_STEPS_CEILING:
                raise
            max_steps = min(2 * max_steps, TRANSPORT_MAX_STEPS_CEILING)
