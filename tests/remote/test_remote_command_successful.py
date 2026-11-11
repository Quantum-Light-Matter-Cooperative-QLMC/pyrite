"""Remote successful tests."""

import shutil

from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _remote_main,
    lifecycle_pull,
)


def test_successful_profile_run_resolves_remote_stem_before_real_pull(monkeypatch, tmp_path):
    import numpy as np

    from pyrite.checkpoints import _checkpoint_io, _checkpoint_store

    remote_stem = f"hopg@sub_100keV-{'a' * 12}"
    payload = {
        "cfg": {
            30.0: {
                "case": {"crystal": "hopg"},
                "E_grid": np.array([1.0]),
                "spec": np.array([2.0]),
            }
        }
    }
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))
    resolved = []
    transfers = []

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(lifecycle, "start_queue", lambda _materials, **_kwargs: "job")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, _materials: ["hopg"])
    monkeypatch.setattr(state, "_materials_needing_pull", lambda _jobid, mats: list(mats))

    def resolve(material, profile, **_kwargs):
        resolved.append((material, profile))
        return remote_stem

    monkeypatch.setattr(lifecycle_pull, "resolve_profile_stem", resolve)
    monkeypatch.setattr(
        lifecycle_pull,
        "_remote_meta_json",
        lambda _stem: (
            1.0,
            {
                "dataset_identity": {
                    "material": "hopg",
                    "catalog_profile": "sub_100keV",
                    "fidelity": "full",
                    "parameter_sha256": "a" * 64,
                }
            },
        ),
    )

    def download(command, destination, **_kwargs):
        transfers.append(command)
        assert f"/checkpoints/{remote_stem}" in command
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", download)

    _remote_main(["run", "sub_100keV", "--no-sync"])

    assert resolved == [("hopg", "sub_100keV")]
    assert len(transfers) == 1
    assert _checkpoint_store.checkpoint_exists(
        remote_stem, tmp_path / "pyrite-output" / "checkpoints"
    )


def test_successful_multi_detector_run_pulls_every_stem_of_completed_material(
    monkeypatch, multi_detector_standard
):
    """Regression (#289): a multi-detector profile yields one stem per detector
    per material; the post-wait pull keeps every detector stem of a completed
    material instead of zipping stems against materials."""
    pairs = scripts._material_stems(["hopg", "hbn"], False, catalog_profile="standard")
    assert [material for material, _ in pairs] == ["hopg", "hopg", "hbn", "hbn"]
    pulled = []

    monkeypatch.setattr(lifecycle, "start_queue", lambda _materials, **_kwargs: "job")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, _materials: ["hopg"])
    monkeypatch.setattr(state, "_materials_needing_pull", lambda _jobid, mats: list(mats))
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **_kwargs: pulled.append(list(stems)))

    _remote_main(["run", "standard", "-m", "hopg", "-m", "hbn", "--no-sync"])

    assert pulled == [[stem for material, stem in pairs if material == "hopg"]]


def test_successful_performance_run_auto_pulls_artifacts(monkeypatch):
    pulled = []
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, mats: mats)
    monkeypatch.setattr(state, "_materials_needing_pull", lambda _jobid, mats: list(mats))
    monkeypatch.setattr(scripts, "_stems", lambda mats, *_args, **_kwargs: mats)
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda profile: pulled.append(("performance", profile)),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda stems, **_kwargs: pulled.append(("checkpoint", stems)),
    )

    from pyrite.campaign.profiles import named_profile_stem

    _remote_main(["run", "standard", "-m", "hopg", "--perf", "--no-sync"])

    assert pulled == [("performance", "standard"), ("checkpoint", [named_profile_stem("hopg")])]
