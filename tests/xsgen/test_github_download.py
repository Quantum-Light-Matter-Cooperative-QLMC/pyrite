"""Authenticated download of PyRITE's own table archives from GitHub Releases."""

import hashlib
import io
import json
import subprocess
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler

import pytest

from pyrite.xsgen import DataFetchError
from pyrite.xsgen import fetch as fetch_module
from pyrite.xsgen.elsepa.release import load_release_index

REPO = "Quantum-Light-Matter-Cooperative-QLMC/pyrite"
URL = f"https://github.com/{REPO}/releases/download/tables-elsepa-1/elsepa-tables.zip"
ASSET_API_URL = f"https://api.github.com/repos/{REPO}/releases/assets/606233700"
TOKEN = "ghp_secret_never_shown"
PAYLOAD = b"archive bytes"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _release_body(name: str = "elsepa-tables.zip") -> bytes:
    return json.dumps({"assets": [{"name": name, "url": ASSET_API_URL}]}).encode()


@pytest.fixture
def no_token(monkeypatch):
    for name in fetch_module.GITHUB_TOKEN_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fetch_module.shutil, "which", lambda name: None)


def test_the_shipped_elsepa_index_pins_a_github_release_url():
    index = load_release_index()

    assert index is not None and index.urls
    assert fetch_module._RELEASE_ASSET_URL.fullmatch(index.urls[0])


def test_token_order_is_pyrite_then_github_then_gh(no_token, monkeypatch):
    assert fetch_module._github_token() is None

    monkeypatch.setattr(fetch_module.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        fetch_module.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="gho_cli\n"),
    )
    assert fetch_module._github_token() == ("gho_cli", "`gh auth token`")

    monkeypatch.setenv("GITHUB_TOKEN", "ghs_actions")
    assert fetch_module._github_token() == ("ghs_actions", "GITHUB_TOKEN")

    monkeypatch.setenv("PYRITE_GITHUB_TOKEN", "ghp_user")
    assert fetch_module._github_token() == ("ghp_user", "PYRITE_GITHUB_TOKEN")


def test_a_logged_out_gh_yields_no_token(no_token, monkeypatch):
    monkeypatch.setattr(fetch_module.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        fetch_module.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout=""),
    )

    assert fetch_module._github_token() is None


def test_a_token_downloads_the_asset_through_the_api(no_token, monkeypatch, tmp_path):
    monkeypatch.setenv("PYRITE_GITHUB_TOKEN", TOKEN)
    seen = []

    def open_url(request, **kwargs):
        seen.append(request)
        if request.full_url.endswith("/releases/tags/tables-elsepa-1"):
            return _Response(_release_body())
        return _Response(PAYLOAD)

    monkeypatch.setattr(fetch_module, "urlopen", open_url)

    digest = fetch_module._download(URL, tmp_path / "a.zip", "ELSEPA tables")

    assert digest == hashlib.sha256(PAYLOAD).hexdigest()
    assert (tmp_path / "a.zip").read_bytes() == PAYLOAD
    tags, asset = seen
    assert tags.full_url == f"https://api.github.com/repos/{REPO}/releases/tags/tables-elsepa-1"
    assert asset.full_url == ASSET_API_URL
    assert asset.get_header("Accept") == "application/octet-stream"
    for request in seen:
        # Unredirected only: a redirect to the asset store must not carry it.
        assert request.unredirected_hdrs["Authorization"] == f"Bearer {TOKEN}"
        assert "Authorization" not in request.headers


def test_the_token_is_dropped_on_redirect():
    request = fetch_module._authorized(ASSET_API_URL, TOKEN, "application/octet-stream")
    store = "https://objects.githubusercontent.com/presigned"

    redirected = HTTPRedirectHandler().redirect_request(
        request, None, 302, "Found", {"Location": store}, store
    )

    assert redirected is not None
    assert not redirected.has_header("Authorization")
    assert TOKEN not in repr(redirected.header_items())


@pytest.mark.parametrize(
    ("code", "reason"),
    [(401, "PYRITE_GITHUB_TOKEN was rejected"), (404, "cannot read the repository")],
)
def test_api_failures_name_the_token_source_but_not_the_token(
    no_token, monkeypatch, tmp_path, code, reason
):
    monkeypatch.setenv("PYRITE_GITHUB_TOKEN", TOKEN)

    def open_url(request, **kwargs):
        raise HTTPError(request.full_url, code, "denied", {}, None)

    monkeypatch.setattr(fetch_module, "urlopen", open_url)

    with pytest.raises(DataFetchError, match=reason) as caught:
        fetch_module._download(URL, tmp_path / "a.zip", "ELSEPA tables")
    assert TOKEN not in str(caught.value)
    assert "--archive PATH" in str(caught.value)


def test_a_release_without_the_asset_says_so(no_token, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(
        fetch_module, "urlopen", lambda request, **kw: _Response(_release_body("other.zip"))
    )

    with pytest.raises(DataFetchError, match="has no elsepa-tables.zip"):
        fetch_module._download(URL, tmp_path / "a.zip", "ELSEPA tables")


def test_without_a_token_the_plain_url_is_tried_and_failure_names_the_fixes(
    no_token, monkeypatch, tmp_path
):
    seen = []

    def open_url(request, **kwargs):
        seen.append(request)
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(fetch_module, "urlopen", open_url)

    with pytest.raises(DataFetchError) as caught:
        fetch_module._download(URL, tmp_path / "a.zip", "ELSEPA tables")

    message = str(caught.value)
    for fix in ("PYRITE_GITHUB_TOKEN", "GITHUB_TOKEN", "--archive PATH"):
        assert fix in message
    (request,) = seen
    assert request.full_url == URL
    assert not request.has_header("Authorization")


def test_other_urls_never_look_up_a_token(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fetch_module, "_github_token", lambda: pytest.fail("token looked up for a non-GitHub URL")
    )
    monkeypatch.setattr(fetch_module, "urlopen", lambda request, **kw: _Response(PAYLOAD))

    digest = fetch_module._download("https://example.invalid/a.zip", tmp_path / "a.zip", "x")

    assert digest == hashlib.sha256(PAYLOAD).hexdigest()


def test_a_tampered_release_asset_fails_digest_verification(no_token, monkeypatch, tmp_path):
    monkeypatch.setenv("PYRITE_GITHUB_TOKEN", TOKEN)

    def open_url(request, **kwargs):
        if "/releases/tags/" in request.full_url:
            return _Response(_release_body())
        return _Response(b"tampered")

    monkeypatch.setattr(fetch_module, "urlopen", open_url)

    with pytest.raises(DataFetchError, match="SHA-256 mismatch") as caught:
        fetch_module._obtain(None, (URL,), tmp_path, "ELSEPA tables", "0" * 64)
    assert TOKEN not in str(caught.value)


def test_release_indexes_read_the_legacy_single_url():
    from pyrite.xsgen._urls import archive_urls

    assert archive_urls({"url": None}) == ()
    assert archive_urls({"url": URL}) == (URL,)
    assert archive_urls({"urls": [URL, "https://mirror.invalid/a.zip"]}) == (
        URL,
        "https://mirror.invalid/a.zip",
    )
    with pytest.raises(ValueError, match="list"):
        archive_urls({"urls": URL})


def test_archive_download_falls_through_to_the_next_location(no_token, monkeypatch, tmp_path):
    def open_url(request, **kwargs):
        if "first.invalid" in request.full_url:
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)
        return _Response(PAYLOAD)

    monkeypatch.setattr(fetch_module, "urlopen", open_url)
    urls = ("https://first.invalid/a.zip", "https://second.invalid/a.zip")

    path = fetch_module._obtain(None, urls, tmp_path, "tables", hashlib.sha256(PAYLOAD).hexdigest())

    assert path.read_bytes() == PAYLOAD


def test_a_transient_bad_body_is_retried_and_still_digest_checked(no_token, monkeypatch, tmp_path):
    bodies = [b"interstitial", PAYLOAD]

    monkeypatch.setattr(fetch_module, "urlopen", lambda request, **kwargs: _Response(bodies.pop(0)))

    path = fetch_module._obtain(
        None, ("https://a.invalid/a.zip",), tmp_path, "tables", hashlib.sha256(PAYLOAD).hexdigest()
    )

    assert path.read_bytes() == PAYLOAD
    assert bodies == []
