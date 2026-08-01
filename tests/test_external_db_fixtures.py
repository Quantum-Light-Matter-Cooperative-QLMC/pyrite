from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from external_db_fixtures import (
    MPQueryError,
    fetch_external,
    fetch_mp_lattice,
    resolve_mp_api_key,
)


def test_resolve_mp_api_key_reads_local_dotenv(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# local only\nexport MP_API_KEY='dotenv-key'\n")

    assert resolve_mp_api_key(environ={}, dotenv_path=dotenv) == "dotenv-key"


def test_resolve_mp_api_key_prefers_exported_value(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("MP_API_KEY=dotenv-key\n")

    assert resolve_mp_api_key(environ={"MP_API_KEY": "exported-key"}, dotenv_path=dotenv) == "exported-key"


def test_mp_only_fetch_without_key_skips(monkeypatch) -> None:
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr("external_db_fixtures._REPO_ROOT", Path("/nonexistent"))

    assert fetch_external(None, "mp-149") is None


def test_fetch_mp_lattice_uses_supported_client_contract() -> None:
    calls: list[str] = []

    class Lattice:
        abc = (5.43, 5.43, 5.43)
        angles = (90.0, 90.0, 90.0)

    class Structure:
        lattice = Lattice()

    class Rester:
        def __init__(self, api_key: str) -> None:
            calls.append(api_key)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get_structure_by_material_id(self, material_id: str) -> Structure:
            assert material_id == "mp-149"
            return Structure()

    assert fetch_mp_lattice("mp-149", "test-key", rester_factory=Rester) == (
        5.43,
        5.43,
        5.43,
        90.0,
        90.0,
        90.0,
    )
    assert calls == ["test-key"]


def test_fetch_mp_lattice_surfaces_configured_key_failure_without_secret() -> None:
    class FailingRester:
        def __init__(self, _api_key: str) -> None:
            pass

        def __enter__(self):
            raise PermissionError("denied")

        def __exit__(self, *_args) -> None:
            return None

    with pytest.raises(MPQueryError, match=r"mp-149 .*PermissionError") as error:
        fetch_mp_lattice("mp-149", "secret-value", rester_factory=FailingRester)
    assert "secret-value" not in str(error.value)


def test_refresh_retains_cached_entry_after_configured_mp_failure(monkeypatch) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "refresh_external_cif.py"
    spec = importlib.util.spec_from_file_location("refresh_external_cif_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cached = {"hfte2": {"source": "mp:mp-32887"}}
    written: list[dict[str, dict[str, object]]] = []

    monkeypatch.setattr(module, "iter_specs_sorted", lambda: iter([("hfte2", None, "mp-32887")]))
    monkeypatch.setattr(module, "load_cached_lattices", lambda: cached)
    monkeypatch.setattr(module, "write_cached_lattices", lambda entries: written.append(entries.copy()))

    def fail_fetch(_cod_id, _mp_id):
        raise MPQueryError("Materials Project query failed for mp-32887 (HTTPError)")

    monkeypatch.setattr(module, "fetch_external", fail_fetch)

    assert module.main() == 1
    assert written == [cached]
