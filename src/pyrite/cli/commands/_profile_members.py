"""Material-membership helpers shared by the catalog profile commands."""

from __future__ import annotations

from pyrite.campaign import profile_edit as _profile_edit


def csv_materials(material_csv):
    return _profile_edit.csv_materials(material_csv)


def group_materials(document, requested, *, allow_unknown=False):
    return _profile_edit.group_materials(
        document,
        requested,
        allow_unknown=allow_unknown,
    )


def validate_materials(document, requested):
    return _profile_edit.validate_materials(document, requested)


def add_membership(document, name, requested):
    return _profile_edit.add_membership(document, name, requested)


def remove_membership(document, name, requested):
    return _profile_edit.remove_membership(document, name, requested)

