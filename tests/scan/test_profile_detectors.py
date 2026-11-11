"""CLI-authored detector collections execute and resume independently."""

import json

import numpy as np
import pytest

from pyrite import materials
from pyrite.cli import command
from pyrite.montecarlo import runner
from pyrite.observations import ObservationStore
from pyrite.runs import scan
from tests.helpers.cli import invoke
from tests.helpers.user_catalog import copy_full_catalog


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["scalar", "planar"])
def test_cli_collection_scores_and_resumes_each_detector(tmp_path, monkeypatch, kind):
    catalog = copy_full_catalog(tmp_path / "catalog")

    def cli(*args, exit_code=0):
        # CliRunner shares one process across edits and runs. A fresh CLI process
        # loads its singleton from the edited catalog selected at startup.
        if args[0] == "run":
            monkeypatch.setattr(
                materials, "CATALOG", materials.load_material_catalog(catalog, profile="multi")
            )
        result = invoke(command, ["--catalog", str(catalog), *args])
        assert result.exit_code == exit_code, result.output
        return result

    cli("profile", "create", "multi", "--from", "quickstart", "--thickness", "100")
    cli("profile", "numerics", "set", "multi", "--cxr-histories", "2", "--brem-histories", "2")
    if kind == "scalar":
        cli("detector", "create", "clone", "side", "--from", "reference")
        cli("detector", "set", "side", "--observation-angle", "100", "-y")
    else:
        cli(
            "detector",
            "create",
            "planar",
            "side",
            "--distance-mm",
            "100",
            "--polar-deg",
            "100",
            "--shape",
            "2",
            "3",
            "--angular-shape",
            "1",
            "1",
            "--response",
            "ideal",
            "--exposure-s",
            "1",
            "--measured-edges-ev",
            "0,10000,30000",
        )
    cli("profile", "detector", "add", "multi", "side", "--detector", "side", "--yes")
    listed = cli("profile", "detector", "list", "multi", "-o", "json")
    assert list(json.loads(listed.stdout)["payload"]["detectors"]) == ["default", "side"]

    scan._load_runtime()
    sweep = scan.run_sweep
    paused = [True]
    executed = []
    transport = runner._transport_case

    def track_transport(case, *args, **kwargs):
        executed.append(float(case["theta_obs_rad"]))
        return transport(case, *args, **kwargs)

    def budgeted_sweep(*args, **kwargs):
        if paused[0] and kwargs["dataset_identity"]["variant"] == "detector-side":
            return False
        return sweep(*args, **kwargs)

    monkeypatch.setattr(runner, "_transport_case", track_transport)
    monkeypatch.setattr(scan, "run_sweep", budgeted_sweep)
    checkpoints = tmp_path / "checkpoints"
    argv = ("run", "multi", "--workers", "0", "--no-progress", "--checkpoint-dir", str(checkpoints))
    cli(*argv, exit_code=75)
    np.testing.assert_allclose(executed, np.deg2rad([90]))
    (default_meta,) = checkpoints.glob("*/meta.json")
    default_identity = json.loads(default_meta.read_text())["dataset_identity"]
    assert default_identity["variant"] == "detector-default"

    paused[0] = False
    cli(*argv)
    np.testing.assert_allclose(executed, np.deg2rad([90, 100]))
    manifests = list(checkpoints.glob("*/meta.json"))
    assert len(manifests) == 2
    identities = [json.loads(path.read_text())["dataset_identity"] for path in manifests]
    assert {row["variant"] for row in identities} == {"detector-default", "detector-side"}
    assert len({row["parameter_sha256"] for row in identities}) == 2
    observations = list((tmp_path / "observations").iterdir()) if kind == "planar" else []
    assert len(observations) == (1 if kind == "planar" else 0)
    if observations:
        store = ObservationStore(observations[0].name, observations[0].parent)
        assert store.verify() == ()
        (source,) = store.sources()
        (digest,) = store.digests(source)
        stored = store.load(digest)
        assert stored.provenance["detector_id"] == "side"
        assert stored.spatial.detector.pixels.shape == (2, 3)
        assert stored.acquisition.exposure_s == 1
        assert stored.spatial.detector.response.__class__.__name__ == "IdealPhotonCounter"

    executed.clear()
    cli(*argv)
    assert executed == []
    cli(
        "detector",
        "set",
        "side",
        "--observation-angle" if kind == "scalar" else "--polar-deg",
        "110",
        "-y",
    )
    cli(*argv)
    np.testing.assert_allclose(executed, np.deg2rad([110]))
    assert len(list(checkpoints.glob("*/meta.json"))) == 3
    assert json.loads(default_meta.read_text())["dataset_identity"] == default_identity
