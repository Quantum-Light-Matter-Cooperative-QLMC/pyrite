"""Target/sequence CLI lifecycle over real single-file and directory catalogs."""

import json

import pytest
import tomlkit

from pyrite._catalog_layout import bundled_catalog, read_raw, read_text, write_text
from pyrite.cli import _catalog_io
from pyrite.cli.commands import filter as filter_cli
from pyrite.cli.commands import profile as profile_cli
from pyrite.cli.commands import target as target_cli
from pyrite.geometry import Slab, Stack, load_filters, load_target
from pyrite.materials.catalog import load_material_catalog
from tests.helpers.cli import invoke
from tests.helpers.user_catalog import copy_full_catalog

SLAB_ARGS = ["--material", "silicon", "--thickness-ang", "20000", "--tilt-deg", "30"]
PLATE_ARGS = ["--material", "silicon", "--thickness-mm", "0.1", "--size-mm", "20", "10"]


@pytest.fixture(params=["file", "directory"])
def path(request, tmp_path, monkeypatch):
    if request.param == "file":
        path = tmp_path / "catalog.toml"
        path.write_text(read_text(bundled_catalog()))
    else:
        path = copy_full_catalog(tmp_path / "catalog")
    document = tomlkit.parse(read_text(path))
    document["profiles"]["standard"]["materials"] = ["silicon"]
    document["profiles"]["standard"].pop("overrides", None)
    write_text(path, tomlkit.dumps(document))
    monkeypatch.setattr(_catalog_io, "active_catalog_path", lambda: path)
    return path


def ok(command, *args):
    result = invoke(command, list(args))
    assert result.exit_code == 0, result.output
    return result


def test_target_lifecycle(path):
    ok(target_cli.command, "create", "film", *SLAB_ARGS, "--description", "thin")
    assert isinstance(load_target("film", catalog=load_material_catalog(path)), Slab)
    ok(target_cli.command, "create", "copy", "--from", "film", "--thickness-ang", "30000")
    ok(
        target_cli.command,
        "create",
        "supported",
        "--layer",
        "silicon",
        "20000",
        "--layer",
        "sio2",
        "30000",
    )
    catalog = load_material_catalog(path)
    assert isinstance(load_target("supported", catalog=catalog), Stack)
    ok(target_cli.command, "validate", "supported")

    before = read_text(path)
    preview = ok(target_cli.command, "attach", "standard", "film", "--dry-run")
    assert read_text(path) == before
    assert "geometry after" in preview.output
    ok(target_cli.command, "attach", "standard", "film")
    assert read_raw(path)["profiles"]["standard"]["targets"] == {"silicon": "film"}
    conflict = invoke(target_cli.command, ["attach", "standard", "copy"])
    assert conflict.exit_code != 0 and "detach it first" in conflict.output

    shown = json.loads(ok(target_cli.command, "show", "film", "-o", "json").stdout)
    assert shown["payload"]["referenced_by"] == [{"profile": "standard", "material": "silicon"}]
    assert "unused" in ok(profile_cli.command, "show", "standard").output

    before = read_text(path)
    refused = invoke(target_cli.command, ["delete", "film"])
    assert refused.exit_code != 0 and "cannot delete referenced" in refused.output
    assert read_text(path) == before

    ok(target_cli.command, "rename", "film", "renamed")
    assert read_raw(path)["profiles"]["standard"]["targets"] == {"silicon": "renamed"}
    ok(target_cli.command, "detach", "standard", "silicon")
    assert "targets" not in read_raw(path)["profiles"]["standard"]
    preview = ok(target_cli.command, "delete", "renamed")
    assert "re-run with -y" in preview.output and "renamed" in read_raw(path)["targets"]
    ok(target_cli.command, "delete", "renamed", "--yes")
    assert "renamed" not in read_raw(path)["targets"]


def test_invalid_target_does_not_write(path):
    before = read_text(path)
    result = invoke(target_cli.command, ["create", "bad", "--material", "silicon"])
    assert result.exit_code != 0 and "targets.bad" in result.output
    assert read_text(path) == before


def test_sequence_lifecycle_detach_preserves_resolved_values(path):
    ok(filter_cli.command, "create", "a", *PLATE_ARGS, "--distance-mm", "50")
    ok(filter_cli.command, "create", "b", *PLATE_ARGS, "--distance-mm", "60")
    ok(filter_cli.command, "create-sequence", "pair", "a", "b")
    ok(filter_cli.command, "create", "clone", "--from", "pair")
    catalog = load_material_catalog(path)
    assert len(load_filters("clone", catalog=catalog)) == 2
    nested = invoke(filter_cli.command, ["create-sequence", "nest", "pair"])
    assert nested.exit_code != 0 and "cannot nest" in nested.output
    edit = invoke(filter_cli.command, ["set", "pair", "--thickness-mm", "1"])
    assert edit.exit_code != 0 and "sequence" in edit.output

    ok(filter_cli.command, "attach", "standard", "a")
    ok(filter_cli.command, "attach", "standard", "pair")
    resolved = load_material_catalog(path).profile_filters["standard"]
    assert [row["distance_mm"] for row in resolved] == [50.0, 50.0, 60.0]
    rows = json.loads(ok(profile_cli.command, "filter", "list", "standard", "-o", "json").stdout)[
        "payload"
    ]["filters"]
    assert rows[1]["ref"] == "pair"
    assert [plate["expanded_index"] for plate in rows[1]["plates"]] == [2, 3]
    assert "sequence pair" in ok(profile_cli.command, "filter", "list", "standard").output
    assert "preset pair" in ok(profile_cli.command, "show", "standard").output

    override = invoke(profile_cli.command, ["filter", "set", "standard", "2", "--name", "x"])
    assert override.exit_code != 0 and "accept only ref" in override.output

    ok(filter_cli.command, "detach", "standard", "2")
    assert load_material_catalog(path).profile_filters["standard"] == resolved
    placements = read_raw(path)["profiles"]["standard"]["filters"]
    assert len(placements) == 3 and "ref" not in placements[1]
