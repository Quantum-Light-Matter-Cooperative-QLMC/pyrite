"""Profile authoring services preserve representations and catalog transactions."""

import pytest
import tomlkit

from pyrite._catalog_layout import bundled_catalog, read_raw
from pyrite.campaign import profile_lifecycle
from pyrite.campaign.profile_summary import profile_payload
from pyrite.catalog_edit import CatalogDraft
from pyrite.cli import command
from tests.helpers.cli import assert_clean_result, invoke

SCALAR = {"observation_angle_deg": 92.0, "polar_acceptance_deg": 2.0, "solid_angle_sr": 0.01}


@pytest.mark.parametrize(
    "declaration, identifier, reference",
    [
        ({"detector": SCALAR}, "default", None),
        ({"detector": "named"}, "default", "named"),
        ({"detectors": {"sensor": SCALAR}}, "sensor", None),
        ({"detectors": {"sensor": "named"}}, "sensor", "named"),
    ],
)
def test_detector_summaries_preserve_named_inline_and_legacy_storage(
    declaration, identifier, reference
):
    document = tomlkit.document()
    document.add(tomlkit.comment("keep catalog comments and declarations"))
    document.update({"detectors": {"named": SCALAR}, "profiles": {"sample": declaration}})
    original = tomlkit.dumps(document)
    payload = profile_payload(document, "sample")
    assert payload["detectors"] == {
        identifier: {
            "kind": "scalar",
            "reference": reference,
            "settings": SCALAR,
            "acceptance": SCALAR,
        }
    }
    assert tomlkit.dumps(document) == original


def test_detector_summary_collection_precedence_keeps_legacy_fields_and_order():
    document = tomlkit.parse("""
# Legacy declarations are retained until explicit migration.
[profiles.sample]
detector = "named"
[profiles.sample.detectors]
zeta = "named"
alpha = { observation_angle_deg = 91.0 }
[detectors.named]
observation_angle_deg = 92.0
polar_acceptance_deg = 2.0
solid_angle_sr = 0.01
""")
    original = tomlkit.dumps(document)
    payload = profile_payload(document, "sample")
    assert list(payload["detectors"]) == ["zeta", "alpha"]
    assert payload["detector_ref"] == "named"
    assert payload["detector_present"] is True
    assert payload["detector"] == SCALAR
    assert payload["detectors"]["zeta"]["acceptance"] == SCALAR
    assert payload["detectors"]["alpha"]["acceptance"]["observation_angle_deg"] == 91.0
    assert tomlkit.dumps(document) == original


def create(document, name):
    profile_lifecycle.create_profile(
        document,
        name,
        "quickstart",
        updates={"energy": [30.0]},
        beam_name=None,
        detector_name="reference",
        transport_updates={},
        materials="hopg",
    )


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_profile_service_and_cli_resolve_identically_without_implicit_save(_isolate_user_catalog):
    selected = bundled_catalog()
    before = snapshot(_isolate_user_catalog)
    draft = CatalogDraft.open(selected)
    create(draft.document, "service")
    proposed = draft.validate(profile="service")
    assert snapshot(_isolate_user_catalog) == before
    draft.save()
    assert_clean_result(
        invoke(
            command,
            [
                "profile",
                "create",
                "batch",
                "--from",
                "quickstart",
                "--energy",
                "30",
                "--detector",
                "reference",
                "--material",
                "hopg",
            ],
        )
    )
    rows = read_raw(selected)["profiles"]
    assert rows["service"] == rows["batch"]
    actual = CatalogDraft.open(selected).validate(profile="batch")
    assert actual.profile_materials("batch") == proposed.profile_materials("service")
    assert actual.profile_detector("batch") == proposed.profile_detector("service")
    assert actual.profile_numerics("batch") == proposed.profile_numerics("service")


@pytest.mark.parametrize("failure", ["invalid", "readonly", "stale"])
def test_rejected_profile_drafts_leave_persistent_state_unchanged(_isolate_user_catalog, failure):
    selected = bundled_catalog()
    draft = CatalogDraft.open(selected)
    if failure == "readonly":
        profile_lifecycle.rename_profile(draft.document, "quickstart", "renamed")
        message = "read-only"
    else:
        create(draft.document, "service")
        if failure == "invalid":
            draft.document["profiles"]["service"]["materials"] = ["unknown"]
            message = "unknown"
        else:
            newer = CatalogDraft.open(selected)
            create(newer.document, "newer")
            newer.save()
            message = "changed while editing"
    before = snapshot(_isolate_user_catalog)
    with pytest.raises(ValueError, match=message):
        draft.save()
    assert snapshot(_isolate_user_catalog) == before
