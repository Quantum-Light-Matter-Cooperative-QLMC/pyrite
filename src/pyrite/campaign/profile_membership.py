"""Profile material membership validation and mutation."""

from ._profile_document import existing_profile, material_rows


def membership_target(document, name):
    """Return the profile's explicit ``materials`` list or raise with guidance."""
    target = existing_profile(document, name)
    materials = target.get("materials")
    if materials is None:
        raise ValueError(
            f"profile {name!r} has implicit all-catalog-materials membership; "
            "it already includes every material. To restrict it, use: "
            f"pyrite profile set {name} --material MATERIAL,..."
        )
    if not isinstance(materials, list):
        raise ValueError(f"profiles.{name}.materials must be an array of material keys")
    return target, materials


def csv_materials(material_csv):
    requested = [key.strip() for key in material_csv.split(",") if key.strip()]
    if not requested:
        raise ValueError("--material requires at least one material key")
    return requested


def validate_materials(document, requested):
    requested = set(requested)
    known = material_rows(document)
    unknown = sorted(requested - set(known))
    if unknown:
        raise ValueError(f"unknown material: {', '.join(unknown)}")
    return [key for key in known if key in requested]


def group_materials(document, requested, *, allow_unknown=False):
    """Validate requested membership and return catalog-ordered keys."""
    requested = list(requested)
    if not requested:
        raise ValueError("provide MATERIAL keys")
    if allow_unknown:
        known = material_rows(document)
        requested = list(dict.fromkeys(requested))
        return [key for key in known if key in requested] + [
            key for key in requested if key not in known
        ]
    return validate_materials(document, requested)


def add_membership(document, name, requested):
    target, membership = membership_target(document, name)
    requested = validate_materials(document, requested)
    added = [key for key in requested if key not in membership]
    target["materials"] = validate_materials(document, [*membership, *requested])
    return added, sorted(set(requested) - set(added))


def remove_membership(document, name, requested):
    target, membership = membership_target(document, name)
    requested = list(dict.fromkeys(requested))
    removed = [key for key in requested if key in membership]
    target["materials"] = [key for key in membership if key not in removed]
    return removed, sorted(set(requested) - set(removed))
