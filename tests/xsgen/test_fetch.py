"""Pinned SBETHE reference-data download and selective extraction."""

import hashlib
import io
import json
import zipfile

import pytest

from pyrite.xsgen import DataFetchError
from pyrite.xsgen import fetch as fetch_module


def _archive() -> bytes:
    """A miniature of the pinned archive, which nests every member under ``sbethe/``."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("sbethe/sdbase/pdatconf.p14", "atoms\n")
        bundle.writestr("sbethe/sdbase/pdcompos.pen", "materials\n")
        bundle.writestr("sbethe/sdbase/atparams.tab", "atomic parameters\n")
        bundle.writestr("sbethe/sdbase/exp-param.tab", "extrapolation parameters\n")
        bundle.writestr("sbethe/sdbase/shparams.tab", "shell parameters\n")
        bundle.writestr("sbethe/sdbase/oos01.tab", "oscillators\n")
        bundle.writestr("sbethe/docs/manual.pdf", b"not extracted")
        bundle.writestr("sbethe/sbethe.exe", b"not extracted")
        bundle.writestr("sbethe/sbethe.f", b"vendored separately")
    return stream.getvalue()


@pytest.fixture(autouse=True)
def _no_github_token(monkeypatch):
    """The mirror is a GitHub Release URL; keep these tests on the public path."""
    for name in fetch_module.GITHUB_TOKEN_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fetch_module.shutil, "which", lambda name: None)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_fetch_verifies_then_extracts_only_sdbase(monkeypatch, tmp_path):
    payload = _archive()
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_module, "urlopen", lambda *args, **kwargs: _Response(payload))
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: tmp_path / code / name)

    result = fetch_module.fetch_sbethe()

    assert result.installed is True
    assert result.file_count == 6
    assert (result.path / "oos01.tab").read_text() == "oscillators\n"
    assert not (result.path.parent / "docs").exists()
    assert not (result.path.parent / "sbethe.exe").exists()
    marker = json.loads((result.path / ".pyrite-fetch.json").read_text())
    assert marker["source_url"] == fetch_module.SBETHE_MIRROR_URL
    assert marker["source_archive"] is None


def test_the_mirror_is_tried_first_and_mendeley_is_kept_as_fallback():
    assert fetch_module.SBETHE_ARCHIVE_URLS == (
        fetch_module.SBETHE_MIRROR_URL,
        fetch_module.SBETHE_ARCHIVE_URL,
    )
    assert fetch_module._RELEASE_ASSET_URL.fullmatch(fetch_module.SBETHE_MIRROR_URL)
    assert fetch_module.SBETHE_ARCHIVE_URL.startswith("https://data.mendeley.com/")
    # The digest is the data's identity; a mirror never re-pins it (#516).
    assert fetch_module.SBETHE_ARCHIVE_SHA256 == (
        "d5d4879c2073ada3bd799fe0727054549cd6ec65cc699acbd0ff25fd5c3c4144"
    )


# What Mendeley served on 2026-10-15: HTTP 200 with a JSON error body.
_ERROR_BODY = b'{"message":"invalid json response body ... <!DOCTYPE html>"}'


def test_a_bad_mirror_body_falls_back_to_mendeley(monkeypatch, tmp_path):
    payload = _archive()
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())

    def open_url(request, **kwargs):
        if request.full_url == fetch_module.SBETHE_MIRROR_URL:
            return _Response(_ERROR_BODY)
        return _Response(payload)

    monkeypatch.setattr(fetch_module, "urlopen", open_url)
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: tmp_path / code / name)

    result = fetch_module.fetch_sbethe()

    assert result.installed is True
    marker = json.loads((result.path / ".pyrite-fetch.json").read_text())
    assert marker["source_url"] == fetch_module.SBETHE_ARCHIVE_URL


def test_mendeley_down_installs_from_the_mirror(monkeypatch, tmp_path):
    payload = _archive()
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())

    def open_url(request, **kwargs):
        if "mendeley" in request.full_url:
            pytest.fail("Mendeley contacted although the mirror served the pinned bytes")
        return _Response(payload)

    monkeypatch.setattr(fetch_module, "urlopen", open_url)
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: tmp_path / code / name)

    assert fetch_module.fetch_sbethe().installed is True


def test_every_bad_location_is_listed_and_its_body_described(monkeypatch, tmp_path):
    destination = tmp_path / "sbethe" / "sdbase"
    monkeypatch.setattr(fetch_module, "urlopen", lambda *a, **k: _Response(_ERROR_BODY))
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: destination)

    with pytest.raises(DataFetchError, match="every download location") as caught:
        fetch_module.fetch_sbethe()

    message = str(caught.value)
    assert fetch_module.SBETHE_MIRROR_URL in message
    assert fetch_module.SBETHE_ARCHIVE_URL in message
    assert f"{len(_ERROR_BODY)} bytes, not a zip archive" in message
    assert not destination.exists()


def test_a_local_archive_is_recorded_as_the_source(monkeypatch, tmp_path):
    payload = _archive()
    archive = tmp_path / "sbethe.zip"
    archive.write_bytes(payload)
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_module, "urlopen", lambda *a, **k: pytest.fail("network used"))
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: tmp_path / code / name)

    result = fetch_module.fetch_sbethe(archive)

    marker = json.loads((result.path / ".pyrite-fetch.json").read_text())
    assert marker["source_url"] is None
    assert marker["source_archive"] == str(archive.resolve())


def test_fetch_identifies_itself_to_the_upstream_host(monkeypatch, tmp_path):
    payload = _archive()
    seen = []
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())

    def open_request(request, **kwargs):
        seen.append(request)
        return _Response(payload)

    monkeypatch.setattr(fetch_module, "urlopen", open_request)
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: tmp_path / code / name)

    fetch_module.fetch_sbethe()

    assert seen[0].get_header("User-agent") == fetch_module._USER_AGENT


def test_complete_install_is_idempotent_without_network(monkeypatch, tmp_path):
    destination = tmp_path / "sbethe" / "sdbase"
    destination.mkdir(parents=True)
    for name in fetch_module.REQUIRED_SBETHE_FILES:
        (destination / name).write_text("present\n")
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: destination)
    monkeypatch.setattr(
        fetch_module,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("an installed database must not access the network"),
    )

    result = fetch_module.fetch_sbethe()

    assert result.installed is False
    assert result.path == destination


def test_hash_mismatch_installs_nothing(monkeypatch, tmp_path):
    payload = _archive()
    destination = tmp_path / "sbethe" / "sdbase"
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(fetch_module, "urlopen", lambda *args, **kwargs: _Response(payload))
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: destination)

    with pytest.raises(DataFetchError, match="SHA-256 mismatch"):
        fetch_module.fetch_sbethe()

    assert not destination.exists()


def test_archive_member_cannot_escape_sdbase(monkeypatch, tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("sdbase/../../escape", "no\n")
    payload = stream.getvalue()
    destination = tmp_path / "sbethe" / "sdbase"
    monkeypatch.setattr(fetch_module, "SBETHE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_module, "urlopen", lambda *args, **kwargs: _Response(payload))
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: destination)

    with pytest.raises(DataFetchError, match="unsafe path"):
        fetch_module.fetch_sbethe()

    assert not (tmp_path / "escape").exists()


def test_a_deposit_v1_sdbase_is_not_accepted_as_installed(monkeypatch, tmp_path):
    """Deposit version 1 ships neither `atparams.tab` nor `exp-param.tab`.

    `sbethe.f` OPENs both. Treating such a directory as a complete install would
    defer the failure to the middle of a Fortran run.
    """
    destination = tmp_path / "sbethe" / "sdbase"
    destination.mkdir(parents=True)
    for name in ("pdatconf.p14", "pdcompos.pen", "oos01.tab"):
        (destination / name).write_text("v1\n")
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: destination)

    assert fetch_module._installed_sbethe(destination) is False
    with pytest.raises(DataFetchError, match="incomplete"):
        fetch_module.fetch_sbethe()


def test_incomplete_existing_directory_is_not_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "sbethe" / "sdbase"
    destination.mkdir(parents=True)
    (destination / "user-note.txt").write_text("keep\n")
    monkeypatch.setattr(fetch_module, "fetched_data_dir", lambda code, name: destination)

    with pytest.raises(DataFetchError, match="incomplete"):
        fetch_module.fetch_sbethe()

    assert (destination / "user-note.txt").read_text() == "keep\n"
