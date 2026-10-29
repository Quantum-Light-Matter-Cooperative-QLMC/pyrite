"""Versioned identity for the default coupled transport approximation (#788)."""

RADIATIVE_MOMENT_MODEL = "radiative-moments-v1/positive-linear-log-energy-v1/rtol2e-4"


def radiative_moment_mode(model: str) -> str:
    """Resolve a persisted model marker; refuse unknown numerical policies."""
    if model == "exact":
        return "exact"
    if model == RADIATIVE_MOMENT_MODEL:
        return "table"
    raise ValueError(f"unsupported radiative_moment_model: {model}")
