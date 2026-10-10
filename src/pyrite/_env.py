"""Small environment helpers shared by PyRITE runtime entry points."""

import os
from typing import overload

#: Set by PyRITE on argv it generates itself (remote job scripts, the local
#: ``--nsys`` re-exec). A deprecated CLI option carried there repeats a choice the
#: user was already warned about -- or a default PyRITE filled in -- so the child
#: stays quiet rather than warning a second time into a job log.
GENERATED_INVOCATION_ENV = "PYRITE_GENERATED_INVOCATION"

CXR_ENV_ALIASES = {
    "PYRITE_MC_CXR_CHUNK": "PYRITE_MC_SPEC_CHUNK",
    "PYRITE_MC_CXR_BUDGET_MB": "PYRITE_MC_SPEC_BUDGET_MB",
}


@overload
def env_value(name: str, default: str) -> str: ...


@overload
def env_value(name: str, default: None = None) -> str | None: ...


def env_value(name: str, default: str | None = None) -> str | None:
    """Return one PyRITE environment value or its default."""
    canonical = next((new for new, old in CXR_ENV_ALIASES.items() if old == name), name)
    legacy = CXR_ENV_ALIASES.get(canonical)
    if canonical in os.environ:
        return os.environ[canonical]
    if legacy is not None and legacy in os.environ:
        from ._api_deprecations import warn_renamed

        warn_renamed(legacy, canonical)
        return os.environ[legacy]
    return os.environ.get(name, default)


def set_canonical_env(name: str, value: str) -> None:
    """Set one PyRITE environment value for a generated process."""
    name = next((new for new, old in CXR_ENV_ALIASES.items() if old == name), name)
    legacy = CXR_ENV_ALIASES.get(name)
    if legacy is not None:
        os.environ.pop(legacy, None)
    os.environ[name] = value
