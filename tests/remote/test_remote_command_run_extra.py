"""Additional remote run tests."""

import pytest

from pyrite import remote
from pyrite.remote import config, state, transport
from tests.remote._remote_test_helpers import _remote_main


def test_completed_materials_reads_success_markers_from_the_queue_log(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "hopg\nhbn\n",
    )

    assert remote._completed_materials("j", ["hopg", "wse2", "hbn"]) == ["hopg", "hbn"]
    assert "s/^completed: //p" in commands[0]


def test_materials_needing_pull_skips_only_valid_zero_new_case_records(monkeypatch):
    monkeypatch.setattr(config, "JOBS_SUBDIR", "jobs")
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: (
            'hopg\t{"completed_new_cases": 0}\nwse2\t{"completed_new_cases": 2}\nmos2\tnot-json\n'
        ),
    )

    assert state._materials_needing_pull("job", ["hopg", "wse2", "mos2"]) == ["wse2", "mos2"]


@pytest.mark.parametrize("retired", ["scan", "submit", "start"])
def test_retired_remote_run_commands_are_removed(retired, capsys):
    assert _remote_main([retired, "--help"]) == 2
    assert f"No such command '{retired}'" in capsys.readouterr().err
