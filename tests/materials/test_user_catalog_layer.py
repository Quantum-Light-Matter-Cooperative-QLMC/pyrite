"""User catalog layer: user profiles read over the bundled catalog (#403)."""

import shutil

import pytest

from pyrite import _catalog_layout as layout

_DEMO = 'materials = ["hopg"]\n'


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    """A stand-in bundled catalog with one demo profile."""
    root = tmp_path / "bundled"
    (root / "profiles").mkdir(parents=True)
    (root / "catalog.toml").write_text("schema_version = 1\n")
    (root / "profiles" / "demo.toml").write_text(_DEMO)
    monkeypatch.setattr(layout, "bundled_catalog", lambda: root)
    return root


@pytest.fixture
def user(_isolate_user_catalog):
    shutil.rmtree(_isolate_user_catalog / "profiles", ignore_errors=True)
    (_isolate_user_catalog / "profiles").mkdir()
    return _isolate_user_catalog


def test_bundled_catalog_reads_user_profiles(bundled, user):
    (user / "profiles" / "mine.toml").write_text('materials = ["hbn"]\n')

    raw = layout.read_raw(bundled)

    assert raw["profiles"] == {"demo": {"materials": ["hopg"]}, "mine": {"materials": ["hbn"]}}
    assert layout.object_keys(bundled, "profiles") == ("demo", "mine")
    assert layout.catalog_root(bundled) == user


def test_other_catalogs_ignore_the_user_layer(bundled, user, tmp_path):
    (user / "profiles" / "mine.toml").write_text('materials = ["hbn"]\n')
    external = tmp_path / "external"
    shutil.copytree(bundled, external)

    assert set(layout.read_raw(external)["profiles"]) == {"demo"}
    assert layout.catalog_root(external) == external


def test_user_profile_may_not_reuse_a_bundled_name(bundled, user):
    (user / "profiles" / "Demo.toml").write_text(_DEMO)

    with pytest.raises(layout.CatalogLayoutError, match="reserved by bundled demo"):
        layout.read_sources(bundled)


def test_new_profiles_go_to_the_user_layer(bundled, user):
    text = layout.read_text(bundled) + '\n[profiles.mine]\nmaterials = ["hbn"]\n'

    layout.write_text(bundled, text)

    assert (user / "profiles" / "mine.toml").read_text() == 'materials = ["hbn"]\n'
    assert not (bundled / "profiles" / "mine.toml").exists()
    # Editing and deleting a user profile stays in the layer.
    layout.write_text(bundled, layout.read_text(bundled).replace('"hbn"', '"mos2"'))
    assert (user / "profiles" / "mine.toml").read_text() == 'materials = ["mos2"]\n'


@pytest.mark.parametrize(
    "edit",
    [
        lambda text: text.replace('"hopg"', '"hbn"'),
        lambda text: text.replace("[profiles.demo]", "[profiles.renamed]"),
    ],
    ids=["edit", "rename"],
)
def test_bundled_demo_profiles_are_read_only(bundled, user, edit):
    with pytest.raises(layout.BundledProfileError, match="pyrite profile create NAME --from demo"):
        layout.write_text(bundled, edit(layout.read_text(bundled)))

    assert (bundled / "profiles" / "demo.toml").read_text() == _DEMO
    assert not any((user / "profiles").iterdir())


def test_developer_writes_target_bundled_profiles(bundled, user):
    text = layout.read_text(bundled).replace('"hopg"', '"hbn"') + "\n[profiles.new_demo]\n"

    with layout.bundled_profile_writes():
        layout.write_text(bundled, text)

    assert (bundled / "profiles" / "demo.toml").read_text() == 'materials = ["hbn"]\n'
    assert (bundled / "profiles" / "new_demo.toml").exists()
    assert not any((user / "profiles").iterdir())


def test_profile_sources_name_each_layer(bundled, user, tmp_path):
    (user / "profiles" / "mine.toml").write_text('materials = ["hbn"]\n')
    external = tmp_path / "external"
    shutil.copytree(bundled, external)

    assert layout.profile_sources(bundled) == {"demo": "bundled", "mine": "user"}
    assert layout.profile_sources(external) == {"demo": "catalog"}
    with layout.bundled_profile_writes():
        assert layout.profile_sources(bundled) == {"demo": "bundled"}
