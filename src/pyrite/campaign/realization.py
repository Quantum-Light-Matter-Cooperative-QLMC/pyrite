"""Bind an executor's admitted numerical realization before case identity."""


def prepare_executor_case(executor, case):
    """Preserve ordinary cases; bind an opt-in core before any key is read."""
    realization = getattr(executor, "transport_realization", None)
    if realization is None:
        return case
    from ..montecarlo.realization import bind_realization

    return bind_realization(case, realization)


def prepare_executor_cases(executor, cases):
    """Preserve the ordinary batch; bind opt-in cases before resume/admission."""
    if getattr(executor, "transport_realization", None) is None:
        return cases
    return [prepare_executor_case(executor, case) for case in cases]
