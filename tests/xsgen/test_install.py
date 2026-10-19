"""The Python fetch entry point mirrors ``pyrite tables fetch``."""

import pytest

from pyrite.xsgen import install
from pyrite.xsgen._errors import DataFetchError


def test_bare_fetch_runs_every_step_and_reports_failures(monkeypatch):
    seen = []

    def fake(code, archive, *, projectile="electron"):
        seen.append((code, projectile))
        if code == "bremslib":
            raise DataFetchError("offline")
        return {"code": code, "installed": True}

    monkeypatch.setattr(install, "fetch_one", fake)
    with pytest.raises(DataFetchError, match=r"1 of 10 failed:\nbremslib: offline"):
        install.fetch_tables()
    assert seen == list(install.BARE_FETCH_STEPS)


def test_single_fetch_tags_positron_and_validates(monkeypatch):
    monkeypatch.setattr(
        install,
        "fetch_one",
        lambda code, archive, *, projectile: {"code": code, "installed": False},
    )
    assert install.fetch_tables("ELSEPA", projectile="positron") == [
        {"code": "elsepa", "installed": False, "projectile": "positron"}
    ]
    with pytest.raises(ValueError, match="unknown code"):
        install.fetch_tables("nope")
    with pytest.raises(ValueError, match="positron"):
        install.fetch_tables("eedl", projectile="positron")
    with pytest.raises(ValueError, match="archive"):
        install.fetch_tables(archive="x.zip")
