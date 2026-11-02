"""Value-returning profile selection shared by CLI and remote Python API."""


def select_materials(profile: str, materials=None) -> list[str]:
    from ..materials import CATALOG

    if profile not in CATALOG.profile_names:
        available = ", ".join(sorted(CATALOG.profile_names)) or "(none defined)"
        raise ValueError(f"unknown profile {profile!r}; available: {available}")
    membership = CATALOG.profile_materials(profile)
    selected = (
        list(materials)
        if materials is not None
        else list(membership if membership is not None else CATALOG.material_keys)
    )
    if not selected:
        raise ValueError("materials must contain at least one material")
    unknown = [m for m in selected if m not in CATALOG.material_keys]
    if unknown:
        raise ValueError(f"unknown material(s): {', '.join(unknown)}")
    non_members = [m for m in selected if membership is not None and m not in membership]
    if non_members:
        raise ValueError(
            f"profile {profile!r} does not include {', '.join(non_members)} (members: {', '.join(membership or [])})"
        )
    return selected
