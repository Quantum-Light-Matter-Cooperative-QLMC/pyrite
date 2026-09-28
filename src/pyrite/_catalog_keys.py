"""Cheap packaged-catalog key reads, without the scientific material modules.

Shell completion, `runs.scan`'s material validation and checkpoint identity
resolution all need the catalog's top-level key sets, and all three run where
importing :mod:`pyrite.materials` would be far too expensive -- a Tab press, an
argument check, a stem lookup. They list the packaged catalog's object files instead.

This lives at the package root because the callers span `cli/`, `runs/` and
`checkpoints/`; sourcing it from `cli._completion` is what made two domain
packages import the CLI (issue #64, finding 2).
"""

import re
import tomllib
from functools import lru_cache

from ._catalog_layout import bundled_catalog, object_keys, selected_catalog

# ``@`` is allowed so ``<material>@<label>-<digest>`` checkpoint @-stems surface
# in local checkpoint-stem completion alongside legacy ``--<fidelity>-`` stems.
SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*")


def _section_keys(section: str) -> tuple[str, ...]:
    try:
        from .console.config import resolve

        path = bundled_catalog() if resolve("catalog.path").source == "built-in default" else selected_catalog()
        keys = object_keys(path, section)
    except OSError, tomllib.TOMLDecodeError:
        return ()
    return tuple(key for key in keys if SAFE_TOKEN_RE.fullmatch(key))


@lru_cache(maxsize=1)
def material_keys() -> tuple[str, ...]:
    """Return catalog ``[materials.*]`` keys."""
    return _section_keys("materials")


@lru_cache(maxsize=1)
def profile_keys() -> tuple[str, ...]:
    """Return catalog ``[profiles.*]`` keys."""
    return _section_keys("profiles")


@lru_cache(maxsize=1)
def beam_keys() -> tuple[str, ...]:
    """Return catalog ``[beams.*]`` keys."""
    return _section_keys("beams")


@lru_cache(maxsize=1)
def detector_keys() -> tuple[str, ...]:
    """Return catalog ``[detectors.*]`` keys."""
    return _section_keys("detectors")
