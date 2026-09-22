"""Lightweight structural check for the ``sweep-profiles.md`` guide's TOML block.

The block is an illustrative catalog fragment -- a named ``[beams.*]`` table
plus one ``[profiles.*]`` table that references it -- not a complete catalog:
it has no ``[materials.*]`` section, so it cannot be fed to the full catalog
loader (``pyrite.materials.catalog``) without building a synthetic catalog
around it. That is new schema plumbing the task explicitly allows dropping
(see the "verify documented code blocks" task doc, checklist step 6).

This checks the two things derivable from the fragment alone: it parses as
valid TOML, and any named beam or detector reference under ``[profiles.*]``
names a corresponding top-level object present in the same fragment.
"""

import tomlkit
from tomlkit.exceptions import TOMLKitError

from pyrite.devtools.doc_blocks import FencedBlock


def check_toml_block(block: FencedBlock) -> list[str]:
    """Return error strings for one ``toml`` fenced block; empty means clean."""
    if block.skip_reason is not None:
        return []

    try:
        document = tomlkit.parse(block.body)
    except TOMLKitError as exc:
        return [f"{block.location}: invalid TOML: {exc}"]

    beams = document.get("beams", {})
    detectors = document.get("detectors", {})
    profiles = document.get("profiles", {})
    errors: list[str] = []
    for profile_name, profile in profiles.items():
        beam_name = profile.get("beam") if hasattr(profile, "get") else None
        if beam_name is not None and beam_name not in beams:
            errors.append(
                f"{block.location}: profile {profile_name!r} references "
                f"undefined beam {beam_name!r}"
            )
        detector_name = profile.get("detector") if hasattr(profile, "get") else None
        if isinstance(detector_name, str) and detector_name not in detectors:
            errors.append(
                f"{block.location}: profile {profile_name!r} references "
                f"undefined detector {detector_name!r}"
            )
    return errors
