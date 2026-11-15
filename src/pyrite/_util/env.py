"""Small environment helpers shared by PyRITE runtime entry points."""

import os
from typing import overload

#: Set by PyRITE on argv it generates itself (remote job scripts, the local
#: ``--nsys`` re-exec). A deprecated CLI option carried there repeats a choice the
#: user was already warned about -- or a default PyRITE filled in -- so the child
#: stays quiet rather than warning a second time into a job log.
GENERATED_INVOCATION_ENV = "PYRITE_GENERATED_INVOCATION"


@overload
def env_value(name: str, default: str) -> str: ...


@overload
def env_value(name: str, default: None = None) -> str | None: ...


def env_value(name: str, default: str | None = None) -> str | None:
    """Return one PyRITE environment value or its default."""
    return os.environ.get(name, default)


def set_canonical_env(name: str, value: str) -> None:
    """Set one PyRITE environment value for a generated process."""
    os.environ[name] = value
