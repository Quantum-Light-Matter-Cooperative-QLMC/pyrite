"""Remote resolve tests."""

import pytest

from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _fake_remote_catalog,
    lifecycle_pull,
)


def test_resolve_profile_stems_resolves_one_stem_per_detector(monkeypatch, multi_detector_standard):
    seen = []

    def resolve(material, profile, *, detector_id=None, **_kwargs):
        seen.append(detector_id)
        return f"{material}-{detector_id}"

    monkeypatch.setattr(lifecycle_pull, "resolve_profile_stem", resolve)

    stems = lifecycle.resolve_profile_stems("hopg", "standard")

    assert len(stems) == 2
    assert len(set(stems)) == 2
    assert len(seen) == 2


def test_resolve_survey_stems_discovers_matching_survey_dir(monkeypatch, capsys):

    survey_stem = f"hopg@survey-{'d' * 12}"
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: f"{survey_stem}\nhopg\nunrelated--full-{'b' * 12}\n",
    )

    resolved = lifecycle._resolve_survey_stems(["hopg"])

    assert resolved == ["hopg", survey_stem]
    assert "survey checkpoint" in capsys.readouterr().out


def test_resolve_survey_stems_ignores_other_materials_and_profiles(monkeypatch):
    other_material_survey = f"wse2--survey-{'c' * 12}"
    same_material_full_variant = f"hopg--full-{'d' * 12}"
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: f"{other_material_survey}\n{same_material_full_variant}\n",
    )

    assert lifecycle._resolve_survey_stems(["hopg"]) == ["hopg"]


def test_resolve_survey_stems_does_not_duplicate_an_explicitly_requested_stem(monkeypatch):

    survey_stem = f"hopg@survey-{'d' * 12}"
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: f"{survey_stem}\n")

    assert lifecycle._resolve_survey_stems(["hopg", survey_stem]) == ["hopg", survey_stem]


def test_resolve_survey_stems_skips_quick_and_already_qualified_stems(monkeypatch):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda *_a: pytest.fail("must not list checkpoints/")
    )

    assert lifecycle._resolve_survey_stems(["hopg_quick"]) == ["hopg_quick"]
    qualified = f"hopg--survey-{'a' * 12}"
    assert lifecycle._resolve_survey_stems([qualified]) == [qualified]


def test_resolve_profile_stem_picks_newest_and_reports_alternates(monkeypatch, capsys):
    a = f"hopg--full-{'a' * 12}"
    b = f"hopg--full-{'b' * 12}"
    meta = {
        "hopg": (
            50,
            {"dataset_identity": {"catalog_profile": "standard", "parameter_sha256": "c" * 64}},
        ),
        a: (
            100,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "a" * 64}},
        ),
        b: (
            200,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "b" * 64}},
        ),
    }
    monkeypatch.setattr(transport, "_ssh_capture", _fake_remote_catalog(f"hopg\n{a}\n{b}\n", meta))

    stem = lifecycle.resolve_profile_stem("hopg", "sub_100keV")

    assert stem == b  # newest mtime (200) wins
    out = capsys.readouterr().out
    assert "2 hashes found" in out
    assert ("a" * 12) in out


def test_resolve_profile_stem_hash_prefix_pins_one(monkeypatch):
    a = f"hopg--full-{'a' * 12}"
    b = f"hopg--full-{'b' * 12}"
    meta = {
        a: (
            100,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "a" * 64}},
        ),
        b: (
            200,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "b" * 64}},
        ),
    }
    monkeypatch.setattr(transport, "_ssh_capture", _fake_remote_catalog(f"{a}\n{b}\n", meta))

    assert lifecycle.resolve_profile_stem("hopg", "sub_100keV", hash_prefix="a" * 12) == a


def test_resolve_profile_stem_skips_retired_survey_tier(monkeypatch):
    full = f"hopg--full-{'a' * 12}"
    survey = f"hopg--survey-{'b' * 12}"
    meta = {
        full: (
            100,
            {
                "dataset_identity": {
                    "catalog_profile": "sub_100keV",
                    "fidelity": "full",
                    "parameter_sha256": "a" * 64,
                }
            },
        ),
        survey: (
            200,
            {
                "dataset_identity": {
                    "catalog_profile": "sub_100keV",
                    "fidelity": "survey",
                    "parameter_sha256": "b" * 64,
                }
            },
        ),
    }
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        _fake_remote_catalog(f"{full}\n{survey}\n", meta),
    )

    assert lifecycle.resolve_profile_stem("hopg", "sub_100keV") == full


def test_resolve_profile_stem_raises_with_no_match(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", _fake_remote_catalog("hopg\n", {}))

    with pytest.raises(SystemExit, match=r"no remote checkpoint matches hopg@sub_100keV"):
        lifecycle.resolve_profile_stem("hopg", "sub_100keV")
