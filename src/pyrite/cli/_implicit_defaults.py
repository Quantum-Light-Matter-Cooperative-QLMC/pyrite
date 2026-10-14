"""Reject runs that leave their profile, beam, or detector to an example default.

Issue #214 deprecated the implicit fallbacks in 0.4.0; issue #387 made them
errors in 0.6.0. A run that names no profile no longer falls back to
``standard``, and a user-selected catalog profile must name a beam.
Implicit scalar detector acceptance was removed in 0.11.0;
use the bundled ``reference`` object to preserve its Timepix3 aperture.
"""

from collections.abc import Mapping

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
    """Enforce the beam requirement and scalar detector deprecation window."""
    from .._catalog_layout import bundled_catalog, selected_catalog
    from ..materials import CATALOG

    if profile not in CATALOG.profile_names:
        return
    if (
        selected_catalog().resolve() != bundled_catalog().resolve()
        and CATALOG.profile_beam(profile) is None
    ):
        raise click.UsageError(
            f"profile '{profile}' names no beam; set its beam with "
            f"'pyrite profile set {profile} --beam BEAM' "
            "(implicit example beam selection was removed in 0.6.0)"
        )
    require_explicit_detector(profile, CATALOG.profile_detector_set(profile))


def require_explicit_detector(profile: str, detectors: Mapping[str, Mapping[str, object]]) -> None:
    """Reject scalar detectors that inherit unnamed acceptance."""
    implicit = [
        name
        for name, spec in detectors.items()
        if "distance_mm" not in spec
        and any(
            key not in spec
            for key in ("observation_angle_deg", "polar_acceptance_deg", "solid_angle_sr")
        )
    ]
    if not implicit:
        return
    replacement = f"pyrite profile set {profile} --detector reference"
    if len(detectors) > 1 or tuple(detectors) != ("default",):
        replacement = "; ".join(
            f'set profiles.{profile}.detectors.{name} = "reference"' for name in implicit
        )
    # A partial custom angle/aperture must be completed without changing it.
    from ..campaign.sweep import TIMEPIX3_DOMEGA_SR, TIMEPIX3_DTHETA_OBS_DEG

    reference = {
        "observation_angle_deg": 90.0,
        "polar_acceptance_deg": TIMEPIX3_DTHETA_OBS_DEG,
        "solid_angle_sr": TIMEPIX3_DOMEGA_SR,
    }
    if any(
        value != reference[key]
        for name in implicit
        for key, value in detectors[name].items()
        if key in reference
    ):
        replacement = (
            "keep the configured geometry and fill its missing scalar fields explicitly "
            "(see pyrite detector show reference for inherited Timepix3 acceptance)"
        )
    detail = (
        f"profile '{profile}' inherits implicit detector acceptance "
        f"({', '.join(implicit)}); use '{replacement}'"
    )
    raise click.UsageError(f"{detail} (implicit detector setup was removed in 0.11.0)")
