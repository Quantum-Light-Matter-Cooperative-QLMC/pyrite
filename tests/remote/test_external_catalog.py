"""Selected catalogs join remote payload identity and batch environment."""

import pytest

from pyrite.remote import config, scripts, transport
from tests.helpers.material_authoring import preserve_catalog_registry  # noqa: F401


def test_external_catalog_is_staged_and_exported(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "catalog.toml").write_text("schema_version = 1\n")
    (catalog / "cifs").mkdir()
    (catalog / "cifs" / "sample.cif").write_text("data_sample\n")
    (catalog / ".git").mkdir()
    (catalog / ".git" / "private").write_text("never stage this\n")
    monkeypatch.setenv("PYRITE_CATALOG", str(catalog))
    monkeypatch.setattr(config, "SYNC_PATHS", [])

    entries = transport._sync_entries()
    digest = transport._payload_digest(entries)
    script = scripts._slurm_batch_script("job1", "echo ok", job_name="catalog-test")

    assert {arc for arc, _ in entries} == {
        "external-catalog/catalog.toml",
        "external-catalog/cifs/sample.cif",
    }
    assert (
        f"export PYRITE_CATALOG={config.shell_word(config.remote_path('external-catalog'))}"
        in script
    )
    assert f"export PYRITE_HOME={config.shell_word(config.remote_dir())}" in script
    (catalog / "cifs" / "sample.cif").write_text("data_changed\n")
    assert transport._payload_digest(transport._sync_entries()) != digest

    outside = tmp_path / "outside.cif"
    outside.write_text("data_outside\n")
    (catalog / "cifs" / "leak.cif").symlink_to(outside)
    with pytest.raises(SystemExit, match="unsafe symlink"):
        transport._sync_entries()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.usefixtures("empty_user_catalog")
def test_bundled_catalog_is_exported_explicitly(explicit, monkeypatch):
    """Regression (#290): selecting the bundled catalog must override any
    catalog configured or staged on the box, not fall back to it."""
    from pyrite._catalog_layout import bundled_catalog

    if explicit:
        monkeypatch.setenv("PYRITE_CATALOG", str(bundled_catalog()))
    else:
        monkeypatch.delenv("PYRITE_CATALOG", raising=False)
        # built-in default, regardless of the developer's config store
        monkeypatch.setattr("pyrite._catalog_layout.selected_catalog", bundled_catalog)
    monkeypatch.setattr(config, "SYNC_PATHS", [])
    bundled = config.shell_word(config.remote_path("src/pyrite/data/catalog"))

    assert not config.external_catalog_selected()
    assert transport._sync_entries() == []
    script = scripts._slurm_batch_script("job1", "echo ok", job_name="catalog-test")
    assert f"export PYRITE_CATALOG={bundled}" in script
    assert f"PYRITE_CATALOG={bundled}" in config.remote_runtime_env()
    user = config.shell_word(config.remote_path("user-catalog"))
    assert f"export PYRITE_USER_CATALOG={user}" in script
    assert f"PYRITE_USER_CATALOG={user}" in config.remote_runtime_env()


def test_user_profiles_ship_beside_the_bundled_catalog(monkeypatch, empty_user_catalog):
    """#403: user-layer profiles and artifacts ride the sync as user-catalog/."""
    from pyrite._catalog_layout import bundled_catalog

    layer = empty_user_catalog
    (layer / "profiles").mkdir()
    (layer / "profiles" / "mine.toml").write_text('materials = ["hopg"]\n')
    (layer / "energy-grid-artifacts" / "ab").mkdir(parents=True)
    (layer / "energy-grid-artifacts" / "ab" / "abcd.json").write_text("{}\n")
    (layer / "config.toml").write_text("[remote]\n")  # never shipped
    monkeypatch.setattr("pyrite._catalog_layout.selected_catalog", bundled_catalog)
    monkeypatch.setattr(config, "SYNC_PATHS", [])

    assert [arc for arc, _ in transport._sync_entries()] == [
        "user-catalog/energy-grid-artifacts/ab/abcd.json",
        "user-catalog/profiles/mine.toml",
    ]


def test_user_instruments_resolve_after_remote_transfer(
    tmp_path, monkeypatch, _isolate_user_catalog
):
    from pyrite import _catalog_layout as layout
    from pyrite.campaign.profiles import named_profile_identity

    bundle = layout.bundled_catalog()
    layer = _isolate_user_catalog
    bundled = named_profile_identity("hopg", "survey", catalog_profile="quickstart")
    existing_user = named_profile_identity("hopg", "survey", catalog_profile="coh_test")
    for table, source in (("beams", "example"), ("detectors", "reference")):
        (layer / table).mkdir()
        (layer / table / "mine.toml").write_bytes((bundle / table / f"{source}.toml").read_bytes())
    (layer / "profiles").mkdir(exist_ok=True)
    profile = (bundle / "profiles" / "quickstart.toml").read_text()
    (layer / "profiles" / "mine.toml").write_text(
        profile.replace('beam = "example"', 'beam = "mine"').replace(
            'detector = "reference"', 'detector = "mine"'
        )
    )
    monkeypatch.setattr(config, "SYNC_PATHS", [])
    local = named_profile_identity("hopg", "survey", catalog_profile="mine")
    assert named_profile_identity("hopg", "survey", catalog_profile="quickstart") == bundled
    assert named_profile_identity("hopg", "survey", catalog_profile="coh_test") == existing_user
    entries = transport._sync_entries()
    assert {
        "user-catalog/beams/mine.toml",
        "user-catalog/detectors/mine.toml",
        "user-catalog/profiles/mine.toml",
    } <= {arc for arc, _ in entries}
    remote = tmp_path / "remote-layer"
    for arc, file in entries:
        target = remote / arc.removeprefix("user-catalog/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    monkeypatch.setenv("PYRITE_USER_CATALOG", str(remote))
    assert named_profile_identity("hopg", "survey", catalog_profile="mine") == local
    assert named_profile_identity("hopg", "survey", catalog_profile="quickstart") == bundled


def test_user_materials_cifs_round_trip_with_same_identity(
    tmp_path, monkeypatch, empty_user_catalog
):
    from pyrite import _catalog_layout as layout
    from pyrite.materials import build_crystal, build_medium, load_material_catalog, save_material

    source = layout.bundled_catalog().parent / "cifs" / "silicon.cif"
    definition = build_crystal(
        "remote-si",
        cif=source,
        formula="Si",
        phase="diamond",
        surface_hkl=(1, 1, 1),
        B_ang2=0.46,
    )
    local = save_material(definition)
    save_material(build_medium("remote-silica", formula="SiO2", density_g_cm3=2.2))
    monkeypatch.setattr(config, "SYNC_PATHS", [])
    entries = transport._sync_entries()
    arcs = {arc for arc, _ in entries}
    assert "user-catalog/materials/remote-si.toml" in arcs
    assert "user-catalog/crystals/remote-si.toml" in arcs
    assert "user-catalog/media/remote-silica.toml" in arcs
    assert any(arc.startswith("user-catalog/cifs/") for arc in arcs)
    remote = tmp_path / "remote-layer"
    for arc, file in entries:
        target = remote / arc.removeprefix("user-catalog/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    monkeypatch.setenv("PYRITE_USER_CATALOG", str(remote))
    loaded = load_material_catalog()
    assert (
        loaded.crystals[definition.name].composition == local.crystals[definition.name].composition
    )
    assert build_crystal("alias", crystal=definition.name).identity == definition.identity
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (remote / "cifs" / "unsafe.cif").symlink_to(outside)
    with pytest.raises(SystemExit, match="unsafe symlink"):
        transport._sync_entries()
