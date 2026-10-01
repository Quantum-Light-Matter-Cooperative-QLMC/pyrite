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
    assert datasets.EEDL.url.startswith("https://nuclear.llnl.gov/EPICS/ENDF2025/")
    assert datasets.EADL.download_sha256 is None


def test_datasets_follow_the_selected_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "ws"))
    assert datasets.dataset_path("eadl") == (tmp_path / "ws").resolve() / (
        "datasets/eadl/EADL2025.ALL"
    )


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
