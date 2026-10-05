"""Per-electron transport step budget and its retry after a step-limit abort."""

from ..transport import TransportStepLimitError

#: First per-electron transport step budget, and the largest the runner doubles
#: it to after a :class:`TransportStepLimitError` (#192).
TRANSPORT_MAX_STEPS = 20_000
TRANSPORT_MAX_STEPS_CEILING = 8 * TRANSPORT_MAX_STEPS


def retry_step_budget(simulate, *args, **kwargs):
    """Return ``simulate(*args, max_steps=..., **kwargs)``, doubling ``max_steps`` from
    :data:`TRANSPORT_MAX_STEPS` after each :class:`TransportStepLimitError`
    until :data:`TRANSPORT_MAX_STEPS_CEILING`, where the error propagates.

    Thick MeV cases outlive the default step budget (5 MeV h-BN, 10 mm:
    ~23-32k steps). The trajectories depend on the seed alone, so a rerun at a
    doubled budget is exactly the run that budget gives from the start.
    """
    max_steps = TRANSPORT_MAX_STEPS
    while True:
        try:
            return simulate(*args, max_steps=max_steps, **kwargs)
        except TransportStepLimitError:
            if max_steps >= TRANSPORT_MAX_STEPS_CEILING:
                raise
            max_steps *= 2
