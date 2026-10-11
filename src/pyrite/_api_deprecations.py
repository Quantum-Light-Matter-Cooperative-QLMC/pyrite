"""Deprecation warnings and registries for renamed or now-required Python API spellings."""

import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ApiDeprecation:
    """One deprecated Python API spelling and its canonical replacement."""

    spelling: str
    replacement: str
    deprecated_in: str
    remove_in: str


def _api_entry(spelling: str, replacement: str, *, since: str) -> ApiDeprecation:
    return ApiDeprecation(spelling, replacement, since, _window(since))


#: Keyed by the deprecated spelling as a user writes it (issue #451). Each row is
#: held to the shipping version by ``tests/test_deprecation_schedule.py``.
API_DEPRECATIONS: dict[str, ApiDeprecation] = {
    entry.spelling: entry
    for entry in (
        _api_entry(
            "Beam() without transverse=",
            "Beam(transverse=Spot(fwhm_x_mm=1.0)) or Beam(transverse=PointSpot())",
            since="0.12.0",
        ),
        _api_entry(
            "Beam() without longitudinal=",
            "Beam(longitudinal=PointBunch()) or Beam(longitudinal=BunchLength(rms_fs=...))",
            since="0.12.0",
        ),
        _api_entry(
            "Slab() without footprint=",
            "Slab(footprint=Footprint(5.0, 5.0)) or Slab(footprint=None) for an infinite slab",
            since="0.12.0",
        ),
        _api_entry(
            "Stack() without footprint=",
            "Stack(footprint=Footprint(5.0, 5.0)) or Stack(footprint=None) for infinite extent",
            since="0.12.0",
        ),
        _api_entry("pyrite.Point", "pyrite.PointSpot", since="0.12.0"),
        _api_entry("simulate(pixel_scorer=)", "PlanarDetector(scorer=...)", since="0.12.0"),
        _api_entry("Scene(pixel_scorer=)", "PlanarDetector(scorer=...)", since="0.12.0"),
        _api_entry(
            "Numerics(backend=)",
            "PYRITE_MC_BACKEND set before importing pyrite (or `pyrite config setup`)",
            since="0.12.0",
        ),
        _api_entry("pyrite.campaign.beam.Point", "pyrite.campaign.beam.PointSpot", since="0.12.0"),
    )
}


def warn_api(spelling: str, *, stacklevel: int = 3) -> None:
    """Warn that registered *spelling* is deprecated, naming its replacement."""
    entry = API_DEPRECATIONS[spelling]
    # An omitted argument becomes an error at the target; a renamed name is removed.
    outcome = "an error" if " without " in entry.spelling else "removed"
    warnings.warn(
        f"{entry.spelling} is deprecated and will be {outcome} in {entry.remove_in}; "
        f"use {entry.replacement} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
