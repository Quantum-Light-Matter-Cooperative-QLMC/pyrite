"""Hash-pinned fetched datasets: location, verification, and install rules."""

import hashlib
from dataclasses import replace

import pytest

from pyrite import datasets
from pyrite.xsgen import DataFetchError

VETTED = b"record one\r\nrecord two\r\n"
PUBLISHED = VETTED + b"\r\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """A small EEDL-shaped pin whose published bytes carry one extra CRLF."""
    fake = replace(
        datasets.EEDL,
        sha256=_sha(VETTED),
        download_sha256=_sha(PUBLISHED),
        strip_suffix=b"\r\n",
    )
    monkeypatch.setitem(datasets.DATASETS, "eedl", fake)
    monkeypatch.setattr(datasets, "datasets_dir", lambda: tmp_path / "datasets")
    return tmp_path / "datasets" / "eedl" / "EEDL.endf"


def test_real_pins_match_the_model_identity_constants():
    from pyrite.montecarlo.eadl_relaxation import EADL_SHA256
    from pyrite.montecarlo.eedl_ionization import EEDL_SHA256

    assert datasets.EEDL.sha256 == EEDL_SHA256
    assert datasets.EADL.sha256 == EADL_SHA256


def test_the_epdl_pin_matches_the_photon_table_constant():
    from pyrite.materials.photon_cross_sections import EPDL_TABLE_SHA256

    assert datasets.EPDL.sha256 == EPDL_TABLE_SHA256


def test_mirrors_follow_upstream_and_derived_data_has_only_pyrite_releases():
    release = "https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/releases/download/"
    for dataset in (datasets.EEDL, datasets.EADL):
        assert dataset.urls[0].startswith("https://nuclear.llnl.gov/")
        assert dataset.urls[1].startswith(release)
    assert datasets.EPDL.urls == (f"{release}tables-epdl-1/epdl2025_mf23.npz",)
    assert datasets.EEDL.url.startswith("https://nuclear.llnl.gov/EPICS/ENDF2025/")
    assert datasets.EADL.download_sha256 is None


def test_datasets_are_shared_across_workspaces(monkeypatch, tmp_path):
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "data"))
    shared = tmp_path / "data" / "datasets/eadl/EADL2025.ALL"
    for home in ("a", "b"):
        monkeypatch.setenv("PYRITE_HOME", str(tmp_path / home))
        assert datasets.dataset_path("eadl") == shared


def test_legacy_workspace_dataset_is_read_through(monkeypatch, tmp_path):
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "ws"))
    legacy = tmp_path / "ws" / "datasets/eadl/EADL2025.ALL"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"x")
    with pytest.warns(RuntimeWarning, match="legacy workspace dataset"):
        assert datasets.dataset_path("eadl") == legacy


def test_missing_dataset_names_the_fetch_command(pinned):
    with pytest.raises(datasets.DatasetNotFoundError, match="pyrite tables fetch eedl"):
        datasets.require_dataset("eedl")
    assert datasets.verify_dataset("eedl") == datasets.MISSING


def test_published_bytes_install_as_the_vetted_form(pinned, tmp_path, monkeypatch):
    def fake_download(url, destination, label):
        destination.write_bytes(PUBLISHED)
        return _sha(PUBLISHED)

    monkeypatch.setattr("pyrite.xsgen.fetch._download", fake_download)

    result = datasets.fetch_dataset("eedl")

    assert result.installed and result.source == datasets.EEDL.url
    assert pinned.read_bytes() == VETTED
    assert datasets.verify_dataset("eedl") == datasets.OK
    assert datasets.require_verified("eedl") == pinned


def test_an_unreachable_upstream_falls_back_to_the_mirror(pinned, monkeypatch):
    mirrored = replace(
        datasets.DATASETS["eedl"],
        urls=("https://upstream.invalid/EEDL", "https://mirror.invalid/EEDL"),
    )
    monkeypatch.setitem(datasets.DATASETS, "eedl", mirrored)
    tried = []

    def fake_download(url, destination, label):
        tried.append(url)
        if "upstream" in url:
            raise DataFetchError(f"could not download {label}: unreachable")
        destination.write_bytes(VETTED)
        return _sha(VETTED)

    monkeypatch.setattr("pyrite.xsgen.fetch._download", fake_download)

    result = datasets.fetch_dataset("eedl")

    assert tried == list(mirrored.urls)
    assert result.installed and result.source == "https://mirror.invalid/EEDL"
    assert pinned.read_bytes() == VETTED


def test_a_tampered_mirror_is_rejected_and_every_attempt_is_reported(pinned, monkeypatch):
    mirrored = replace(
        datasets.DATASETS["eedl"],
        urls=("https://upstream.invalid/EEDL", "https://mirror.invalid/EEDL"),
    )
    monkeypatch.setitem(datasets.DATASETS, "eedl", mirrored)

    def fake_download(url, destination, label):
        if "upstream" in url:
            raise DataFetchError(f"could not download {label}: unreachable")
        destination.write_bytes(b"tampered")
        return _sha(b"tampered")

    monkeypatch.setattr("pyrite.xsgen.fetch._download", fake_download)

    with pytest.raises(DataFetchError, match="every download location") as caught:
        datasets.fetch_dataset("eedl")
    assert "unreachable" in str(caught.value)
    assert "SHA-256 mismatch" in str(caught.value)
    assert not pinned.exists()


@pytest.mark.parametrize("body", [VETTED, PUBLISHED])
def test_a_local_copy_in_either_form_installs(pinned, tmp_path, body):
    source = tmp_path / "copy"
    source.write_bytes(body)

    result = datasets.fetch_dataset("eedl", source)

    assert result.installed
    assert pinned.read_bytes() == VETTED
    assert datasets.fetch_dataset("eedl", source).installed is False


def test_a_wrong_file_installs_nothing(pinned, tmp_path):
    source = tmp_path / "copy"
    source.write_bytes(b"not the pinned bytes\r\n")

    with pytest.raises(DataFetchError, match="nothing was installed"):
        datasets.fetch_dataset("eedl", source)
    assert not pinned.exists()
    assert list(pinned.parent.iterdir()) == []


def test_an_existing_different_file_is_reported_not_replaced(pinned, tmp_path):
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"user data")
    source = tmp_path / "copy"
    source.write_bytes(VETTED)

    with pytest.raises(DataFetchError, match="move or remove it"):
        datasets.fetch_dataset("eedl", source)
    assert pinned.read_bytes() == b"user data"
    assert datasets.verify_dataset("eedl") == datasets.MISMATCH
    with pytest.raises(datasets.DatasetMismatchError):
        datasets.require_verified("eedl")


def test_unknown_dataset_is_a_key_error():
    with pytest.raises(KeyError, match="choose from eedl, eadl"):
        datasets.get("penelope")


def test_the_eedl_loader_fails_naming_the_fetch_command(monkeypatch, tmp_path):
    from pyrite.montecarlo.eedl_ionization import load_eedl_shell_ionization

    monkeypatch.setattr(datasets, "datasets_dir", lambda: tmp_path)
    with pytest.raises(datasets.DatasetNotFoundError, match="pyrite tables fetch eedl"):
        load_eedl_shell_ionization("C")


def _bundle(tmp_path, members):
    import zipfile

    path = tmp_path / "epics.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    return path


def test_the_zenodo_bundle_is_the_last_fallback(pinned, tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, {"eedl/EEDL.endf": VETTED})
    pin = datasets.BundlePin("epics", "epics.zip", _sha(bundle.read_bytes()), ("https://z/epics",))
    monkeypatch.setattr(datasets, "bundle_pin", lambda name: pin if name == "epics" else None)
    tried = []

    def fake_download(url, destination, label):
        tried.append(url)
        if url != "https://z/epics":
            raise DataFetchError(f"could not download {label}: unreachable")
        destination.write_bytes(bundle.read_bytes())
        return pin.sha256

    monkeypatch.setattr("pyrite.xsgen.fetch._download", fake_download)

    result = datasets.fetch_dataset("eedl")

    assert tried == [*datasets.DATASETS["eedl"].urls, "https://z/epics"]
    assert result.source == "https://z/epics"
    assert pinned.read_bytes() == VETTED
    assert sorted(p.name for p in pinned.parent.iterdir()) == ["EEDL.endf"]


def test_a_local_bundle_zip_installs_its_member(pinned, tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, {"eedl/EEDL.endf": VETTED, "eadl/EADL2025.ALL": b"other"})
    pin = datasets.BundlePin("epics", "epics.zip", _sha(bundle.read_bytes()), ())
    monkeypatch.setattr(datasets, "bundle_pin", lambda name: pin)

    assert datasets.fetch_dataset("eedl", bundle).installed
    assert pinned.read_bytes() == VETTED


def test_a_bundle_with_the_wrong_member_installs_nothing(pinned, tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, {"eedl/EEDL.endf": b"tampered"})
    pin = datasets.BundlePin("epics", "epics.zip", _sha(bundle.read_bytes()), ())
    monkeypatch.setattr(datasets, "bundle_pin", lambda name: pin)

    with pytest.raises(DataFetchError, match="nothing was installed"):
        datasets.fetch_dataset("eedl", bundle)
    assert not pinned.exists()


def test_every_fetched_dataset_is_pinned_in_a_shipped_bundle():
    for dataset in datasets.DATASETS.values():
        assert dataset.bundle is not None, dataset.name
        pin = datasets.bundle_pin(dataset.bundle)
        assert pin is not None and pin.file == f"{dataset.bundle}.zip"
