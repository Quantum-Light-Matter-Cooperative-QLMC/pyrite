"""Warning helpers for the 0.11 spectral vocabulary migration (issue #442)."""

import warnings
from functools import wraps

from ._module_deprecations import _window

DEPRECATED_IN = "0.11.0"
REMOVE_IN = _window(DEPRECATED_IN)


def warn_renamed(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Name the canonical replacement and the two-minor removal target."""
    warnings.warn(
        f"{old} is deprecated and will be removed in {REMOVE_IN}; use {new} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def renamed_fields(**aliases: str):
    """Preserve old dataclass keywords/attributes while storing canonical fields."""

    def decorate(cls):
        original = cls.__init__
        cls._field_aliases = dict(aliases)

        @wraps(original)
        def initialize(self, *args, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"cannot specify both {old} and {new}")
                    warn_renamed(f"{cls.__name__}.{old}", f"{cls.__name__}.{new}", stacklevel=4)
                    kwargs[new] = kwargs.pop(old)
            original(self, *args, **kwargs)

        cls.__init__ = initialize
        for old, new in aliases.items():

            def read(self, old=old, new=new):
                warn_renamed(f"{cls.__name__}.{old}", f"{cls.__name__}.{new}")
                return getattr(self, new)

            def write(self, value, old=old, new=new):
                warn_renamed(f"{cls.__name__}.{old}", f"{cls.__name__}.{new}")
                setattr(self, new, value)

            setattr(cls, old, property(read, write, doc=f"Deprecated alias for {new}."))
        return cls

    return decorate
