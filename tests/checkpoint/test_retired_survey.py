"""Issue #1079: retired survey-tier checkpoints stay readable beside @profile ones."""

import json
import pickle

import pytest

from pyrite.campaign.profiles import named_profile_identity
from pyrite.checkpoints.persistence import load_checkpoint
from pyrite.cli import command as root_command
from pyrite.console.outputs import output_dir
from tests.helpers.cli import invoke

_RECORD = {"case": {"crystal": "hopg", "name": "cfg", "E0_keV": 30.0}, "spec": [1.0]}


def _write(directory, stem, identity):
    with open(directory / f"{stem}.pkl", "wb") as handle:
        pickle.dump({"cfg": {30.0: dict(_RECORD)}}, handle)
    (directory / f"{stem}.meta.json").write_text(json.dumps({"dataset_identity": identity}))


@pytest.fixture
def checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = output_dir("checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    profile = named_profile_identity("hopg", catalog_profile="sub_100keV")
    survey = {**named_profile_identity("hopg"), "fidelity": "survey"}
    stems = {
        "hopg--survey-0123456789ab": survey,
        "hopg@survey-0123456789ab": survey,
        f"hopg@sub_100keV-{profile['parameter_sha256'][:12]}": profile,
    }
    for stem, identity in stems.items():
        _write(directory, stem, identity)
    return directory, stems


def test_survey_and_profile_checkpoints_load_and_list(checkpoints):
    directory, stems = checkpoints

    for stem in stems:
        assert set(load_checkpoint(stem, directory)) == {"cfg"}

    listed = invoke(root_command, ["checkpoint", "list", "-o", "json"])
    assert listed.exit_code == 0, listed.output
    assert all(stem in listed.stdout for stem in stems)
