"""Reject runs that leave their profile, beam, or detector to an example default.

Issue #214 deprecated the implicit fallbacks in 0.4.0; issue #387 made them
errors in 0.6.0. A run that names no profile no longer falls back to
``standard``, and a user-catalog profile that names no beam or detector no
longer falls back to the bundled examples. Bundled catalog profiles are
examples themselves and stay exempt.
"""

import click

from ..console import config as _cli_config

PROFILE_REQUIRED = (
    "this run names no profile; pass PROFILE, set PYRITE_PROFILE, or run "
    "'pyrite config set profile.current NAME' (the implicit 'standard' fallback "
    "was removed in 0.6.0); try the bundled demo: pyrite run quickstart"
)


def implicit_profile_error(profile: _cli_config.ResolvedValue) -> str | None:
    """The usage error for *profile* if it came from the built-in fallback."""
    if profile.source == "built-in default":
        return PROFILE_REQUIRED
    return None


def require_explicit_profile(profile: _cli_config.ResolvedValue) -> None:
    """Raise a usage error when *profile* came from the built-in fallback."""
    if (error := implicit_profile_error(profile)) is not None:
        raise click.UsageError(error)


def require_explicit_instrument(profile: str) -> None:
    """Raise a usage error when user-catalog *profile* names no beam or detector."""
    from .._catalog_layout import bundled_catalog, selected_catalog
    from ..materials import CATALOG

    # Profiles read with the bundled catalog -- demos and user-layer profiles
    # alike -- stay exempt: the layer holds campaigns that moved out of the
    # bundle in #403, and attaching an instrument would change their identity.
    if selected_catalog().resolve() == bundled_catalog().resolve():
        return
    if profile not in CATALOG.profile_names:
        return
    missing: dict[str, str] = {}
    if CATALOG.profile_beam(profile) is None:
        missing["beam"] = f"set its beam with 'pyrite profile set {profile} --beam BEAM'"
    detectors = CATALOG.profile_detector_set(profile)
    if (
        profile not in CATALOG.profile_detectors
        and profile not in CATALOG.profile_physical_detectors
        and tuple(detectors) == ("default",)
        and not detectors["default"]
    ):
        missing["detector"] = (
            f"declare '[profiles.{profile}.detectors.ID]' or set the legacy 'detector' reference"
        )
    if missing:
        raise click.UsageError(
            f"profile '{profile}' names no {' or '.join(missing)}; {'; '.join(missing.values())} "
            "(implicit example beam/detector selection was removed in 0.6.0)"
        )
