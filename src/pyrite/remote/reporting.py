"""Invocation-scoped presentation policy for CLI and silent library callers."""

import builtins
from contextlib import contextmanager
from contextvars import ContextVar

_QUIET = ContextVar("pyrite_remote_quiet", default=False)


def is_quiet() -> bool:
    return _QUIET.get()


def report(*args, **kwargs):
    if not is_quiet():
        builtins.print(*args, **kwargs)


@contextmanager
def quiet():
    token = _QUIET.set(True)
    try:
        yield
    finally:
        _QUIET.reset(token)
