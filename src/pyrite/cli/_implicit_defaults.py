"""Resolve a run's profile and warn about implicitly chosen example defaults.

Stage 1 of issue #214: a run that names no profile falls back to ``standard``,
and a user-catalog profile that names no beam or detector falls back to the
bundled examples. Both still work; both warn with the removal target from
`_deprecations.IMPLICIT_DEFAULTS`. Bundled catalog profiles are examples
themselves and stay silent.
"""

from ..console import config as _cli_config
from ._deprecations import warn_implicit_default


def warn_implicit_profile(profile: _cli_config.ResolvedValue) -> None:
    """Warn when a run's *profile* came from the built-in ``standard`` fallback."""
    if profile.source == "built-in default":
        warn_implicit_default("profile", "this run")


def warn_implicit_instrument(profile: str) -> None:
    """Warn when *profile* in a user-selected catalog names no beam or detector."""
    from .._catalog_layout import bundled_catalog, selected_catalog
    from ..materials import CATALOG

    if selected_catalog().resolve() == bundled_catalog().resolve():
        return
    if profile not in CATALOG.profile_names:
        return
    subject = f"profile '{profile}'"
    if CATALOG.profile_beam(profile) is None:
        warn_implicit_default("beam", subject)
    if profile not in CATALOG.profile_detectors:
        warn_implicit_default("detector", subject)
