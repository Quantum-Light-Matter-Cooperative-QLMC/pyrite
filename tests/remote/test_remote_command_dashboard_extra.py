"""Additional remote dashboard tests."""

import json

import pytest

from pyrite import remote
from pyrite.console import dashboard as presentation
from pyrite.console.dashboard import render as dashboard_render
from pyrite.console.dashboard import state as dashboard_state
from pyrite.remote import transport, viewer


@pytest.mark.parametrize("done,total", [(0, 1000), (250, 1000), (1000, 1000)])
def test_progress_records_render_valid_electron_counts(done, total):
    snapshot = {
        "material": "hopg",
        "total_cases": 1,
        "cached_cases": 0,
        "completed_new_cases": 0,
        "state": "running",
        "activity": "computing",
        "transport_electrons_done": done,
        "transport_electrons_total": total,
    }
    records = presentation.parse_progress_records(json.dumps(snapshot))
    assert records["hopg"] == snapshot
    output = dashboard_render._format_case_progress(records, ["hopg"])
    assert f"electrons {done:,}/{total:,} ({round(100 * done / total)}%)" in output
    records["hopg"].update(activity="saving")
    assert "electrons" not in dashboard_render._format_case_progress(records, ["hopg"])
    records["hopg"].update(activity="computing", state="done")
    assert "electrons" not in dashboard_render._format_case_progress(records, ["hopg"])


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"transport_electrons_done": 1},
        {"transport_electrons_total": 10},
        {"transport_electrons_done": -1, "transport_electrons_total": 10},
        {"transport_electrons_done": 11, "transport_electrons_total": 10},
        {"transport_electrons_done": 0, "transport_electrons_total": 0},
        {"transport_electrons_done": True, "transport_electrons_total": 10},
        {"transport_electrons_done": 0, "transport_electrons_total": True},
        {"transport_electrons_done": 1.0, "transport_electrons_total": 10},
        {"transport_electrons_done": 1, "transport_electrons_total": "10"},
        {"transport_electrons_done": None, "transport_electrons_total": 10},
    ],
)
def test_progress_records_drop_invalid_electron_counts_and_keep_legacy_records(fields):
    legacy = {
        "material": "hopg",
        "total_cases": 1,
        "cached_cases": 0,
        "completed_new_cases": 0,
        "state": "running",
    }
    records = presentation.parse_progress_records(json.dumps({**legacy, **fields}))
    assert records == {"hopg": legacy}
    assert "electrons" not in dashboard_render._format_case_progress(records, ["hopg"])


def test_overall_progress_line_sums_cases_and_counts_done_materials():
    records = presentation.parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":4,"cached_cases":2,'
                '"completed_new_cases":2,"state":"done"}',
                '{"material":"hbn","total_cases":6,"cached_cases":0,'
                '"completed_new_cases":3,"state":"running"}',
            ]
        )
    )

    line = dashboard_render._overall_progress_line(records)

    assert line is not None
    assert "7/10 cases" in line  # 4 (hopg) + 3 (hbn)
    # material-weighted over the two started materials: (4/4 + 3/6) / 2 = 75%.
    assert " 75%" in line
    assert "materials" not in line
    assert "●" in line  # a still-running material keeps the aggregate active


def test_overall_progress_line_counts_unstarted_materials_in_the_denominator():
    # 2 of 4 rostered materials have started (one done, one half); the other two
    # have no progress record yet. The headline must weight over ALL four, not
    # read ~75% off the two that reported.
    records = presentation.parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":4,"cached_cases":4,'
                '"completed_new_cases":0,"state":"done"}',
                '{"material":"hbn","total_cases":4,"cached_cases":0,'
                '"completed_new_cases":2,"state":"running"}',
            ]
        )
    )

    line = dashboard_render._overall_progress_line(records, ["hopg", "hbn", "mos2", "wse2"])

    assert line is not None
    # (4/4 + 2/4 + 0 + 0) / 4 = 37.5% -> 38%, not ~75%.
    assert " 38%" in line
    assert "6/16 cases" in line
    assert "materials" not in line


def test_overall_progress_line_is_none_without_records():
    assert dashboard_render._overall_progress_line({}) is None


def test_overall_progress_line_use_cost_weights_by_compute_not_case_count():
    # hopg: 1 cheap case done of 1 total by count (100%), but that one case is
    # only 10% of the material's compute; hbn hasn't started. A case-count bar
    # would read 50% (1 material fully "done" of 2); the compute bar must read
    # far lower since hopg's done work is cheap relative to the whole sweep.
    records = presentation.parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":1,"cached_cases":0,'
                '"completed_new_cases":1,"state":"done",'
                '"done_cost":10.0,"total_cost":100.0}',
                '{"material":"hbn","total_cases":4,"cached_cases":0,'
                '"completed_new_cases":0,"state":"running",'
                '"done_cost":0.0,"total_cost":50.0}',
            ]
        )
    )

    cases_line = dashboard_render._overall_progress_line(records, ["hopg", "hbn"], use_cost=False)
    compute_line = dashboard_render._overall_progress_line(records, ["hopg", "hbn"], use_cost=True)

    assert " 50%" in cases_line
    assert " 5%" in compute_line  # (10/100 + 0/50) / 2 = 5%


def test_overall_progress_line_use_cost_falls_back_to_cases_without_cost_data():
    records = presentation.parse_progress_records(
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running"}'
    )
    line = dashboard_render._overall_progress_line(records, ["hopg"], use_cost=True)
    assert " 50%" in line
    assert "cases" not in line


def test_paused_overall_progress_uses_warning_state():
    records = presentation.parse_progress_records(
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"running"}'
    )

    line = dashboard_render._overall_progress_line(records, ["hopg"], state_override="paused")

    assert "Ⅱ" in line


def test_static_case_progress_uses_catalog_labels_and_one_row_per_material():
    records = presentation.parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":4,"state":"done"}',
                '{"material":"hbn","total_cases":4,"cached_cases":1,'
                '"completed_new_cases":1,"state":"running"}',
            ]
        )
    )

    output = dashboard_render._format_case_progress(records, ["hopg", "hbn"])

    assert output.count("HOPG") == 1
    assert output.count("h-BN") == 1
    assert "████████████████" in output
    assert "████████░░░░░░░░" in output
    assert "5/5" in output
    assert "2/4" in output


def test_static_case_progress_colors_tracks_only_on_tty(monkeypatch):
    record = {
        "hopg": {
            "material": "hopg",
            "total_cases": 2,
            "cached_cases": 0,
            "completed_new_cases": 1,
            "state": "running",
        }
    }
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(viewer.sys.stdout, "isatty", lambda: True)

    colored = dashboard_render._format_case_progress(record, ["hopg"])
    monkeypatch.setenv("NO_COLOR", "1")
    plain = dashboard_render._format_case_progress(record, ["hopg"])

    assert "\033[38;2;92;207;230m" in colored
    assert "\033[" not in plain


def test_case_progress_shows_current_crystal_parameters_for_running_material():
    records = presentation.parse_progress_records(
        '{"material":"hopg","total_cases":5,"cached_cases":1,'
        '"completed_new_cases":2,"state":"running","activity":"computing",'
        '"current":{"energy_keV":30,"tilt_deg":25,"azimuth_deg":110,"thickness_um":0.5}}'
    )

    output = dashboard_render._format_case_progress(records, ["hopg"])

    assert "NOW TESTING" in output
    assert "30 keV" in output
    assert "tilt 25°" in output
    assert "azim 110°" in output
    assert "0.5 µm" in output


def test_case_progress_labels_checkpoint_activity_and_hides_stale_current():
    records = presentation.parse_progress_records(
        '{"material":"hopg","total_cases":5,"cached_cases":3,'
        '"completed_new_cases":0,"state":"running","activity":"saving",'
        '"current":{"energy_keV":30,"tilt_deg":25,"azimuth_deg":110,"thickness_um":0.5}}'
    )

    output = dashboard_render._format_case_progress(records, ["hopg"])

    assert "saving checkpoint" in output
    assert "30 keV" not in output


def test_case_progress_omits_current_params_for_non_running_material():
    records = presentation.parse_progress_records(
        '{"material":"hopg","total_cases":5,"cached_cases":5,'
        '"completed_new_cases":0,"state":"done",'
        '"current":{"energy_keV":30,"tilt_deg":25,"azimuth_deg":110,"thickness_um":0.5}}'
    )

    output = dashboard_render._format_case_progress(records, ["hopg"])

    assert "keV" not in output  # done row falls back to the cached/new tally


def test_material_roster_summarizes_and_wraps_a_long_list():
    short = dashboard_render._format_material_roster(["hopg", "hbn"])
    assert short == "hopg, hbn"

    long_roster = [f"mat{i:02d}" for i in range(20)]
    rendered = dashboard_render._format_material_roster(long_roster)
    assert rendered.splitlines()[0] == "20 total"
    assert "mat00" in rendered
    assert "\n" in rendered  # wrapped, not one ragged line


def test_style_states_colors_step_warnings_as_warning_not_success(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(dashboard_render.sys.stdout, "isatty", lambda: True)
    done_color = "\033[38;2;170;217;76m"
    warning_color = "\033[38;2;255;213;128m"

    clean = dashboard_render.style_states("done [3/3] 2026-09-19T00:00:00+00:00")
    warned = dashboard_render.style_states("done with 2 warning(s) [3/3] 2026-09-19T00:00:00+00:00")

    assert clean.startswith(f"{done_color}done\033[0m")
    assert warned.startswith(f"{warning_color}done\033[0m")
    assert done_color not in warned


def test_line_grid_bounds_status_uses_diagnostic_metadata(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nkind: line-grid-bounds\nslice_minutes: 10\n"
                "energies: 200,250,300\nslurm_job_id: 227",
                "STATE": "FAILED (signal TERM) now",
                "SQUEUE": "job_id=227|state=NOT_QUEUED",
                "PROGRESS": "",
                "LOG": "cases: 40%|████      | 2/5 [17:31<26:18, 526.18s/it]\r",
            }
        ),
    )

    remote.job_status("j", detail=2)

    output = capsys.readouterr().out
    assert "Kind      line-grid-bounds" in output
    assert "Energies  200,250,300 keV" in output
    assert "Slice     10 min soft / 30 min hard" in output
    assert "Materials" not in output
    assert "Mode" not in output
    assert "CASE PROGRESS" not in output
    assert "last case batch" not in output
