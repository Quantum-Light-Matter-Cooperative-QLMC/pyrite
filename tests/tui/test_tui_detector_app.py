"""Headless keyboard-only checks of the optional Textual detector editor."""

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, OptionList, Static  # noqa: E402

from pyrite._catalog_layout import bundled_catalog, read_raw  # noqa: E402
from pyrite.tui import detector_app  # noqa: E402
from pyrite.tui.detector_app import (  # noqa: E402
    BrowseScreen,
    DetectorEditorApp,
    EditorScreen,
    ReviewScreen,
)


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def drive(scenario, size=(100, 40)):
    async def main():
        app = DetectorEditorApp(bundled_catalog())
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await scenario(app, pilot)

    asyncio.run(main())


async def type_into(pilot, widget, text):
    widget.focus()
    await pilot.pause()
    await pilot.press(*text)  # focus selects the old value; typing replaces it


def text(app, selector):
    return str(app.screen.query_one(selector, Static).render())


def test_browse_shows_kind_source_and_readonly_state(_isolate_user_catalog):
    async def scenario(app, pilot):
        assert isinstance(app.screen, BrowseScreen)
        options = app.screen.query_one(OptionList)
        prompts = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert any(
            p.startswith("pixelated_reference  (planar, bundled, read-only)") for p in prompts
        )
        await pilot.press("e")  # bundled: refused with a clone hint, no editor
        assert isinstance(app.screen, BrowseScreen)
        assert "read-only" in text(app, "#details")

    drive(scenario)


def test_keyboard_create_scalar_review_and_save(_isolate_user_catalog):
    async def scenario(app, pilot):
        await pilot.press("n")
        assert isinstance(app.screen, EditorScreen)
        await pilot.press(*"lab90")  # name has focus
        await type_into(pilot, app.screen.query_one("#f-observation_angle_deg", Input), "90")
        await type_into(pilot, app.screen.query_one("#f-solid_angle_sr", Input), "0.01")
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, ReviewScreen)
        assert "[detectors.lab90]" in text(app, "#diff Static")
        assert read_raw(bundled_catalog())["detectors"].get("lab90") is None  # review is inert
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, BrowseScreen)

    drive(scenario)
    row = read_raw(bundled_catalog())["detectors"]["lab90"]
    assert row == {"observation_angle_deg": 90.0, "solid_angle_sr": 0.01}


def test_switching_kind_preserves_hidden_values_and_shows_field_errors(_isolate_user_catalog):
    before = snapshot(_isolate_user_catalog)

    async def scenario(app, pilot):
        await pilot.press("n", *"chip")
        screen = app.screen
        await type_into(pilot, screen.query_one("#f-observation_angle_deg", Input), "45")
        screen.query_one("#f-kind").focus()
        await pilot.press("enter", "down", "enter")  # keyboard: scalar -> planar
        await pilot.pause()
        assert screen.kind == "planar"
        assert not screen.query_one("#f-observation_angle_deg").parent.display
        assert screen.query_one("#f-distance_mm").parent.display
        await type_into(pilot, screen.query_one("#f-distance_mm", Input), "-3")
        assert "error:" in text(app, "#e-distance_mm")  # live, color-independent
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, EditorScreen)
        assert "error:" in text(app, "#message")
        screen.query_one("#f-kind").focus()
        await pilot.press("enter", "up", "enter")
        await pilot.pause()
        assert screen.query_one("#f-observation_angle_deg", Input).value == "45"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, BrowseScreen)

    drive(scenario)
    assert snapshot(_isolate_user_catalog) == before


def test_clone_bundled_in_narrow_terminal_then_back_and_cancel(_isolate_user_catalog):
    before = snapshot(_isolate_user_catalog)

    async def scenario(app, pilot):
        await pilot.press("down", "c")  # second bundled entry
        assert isinstance(app.screen, EditorScreen)
        assert app.screen.query_one("#f-name", Input).value.endswith("_copy")
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, ReviewScreen)
        await pilot.press("escape")  # back to the editor, still unsaved
        assert isinstance(app.screen, EditorScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, BrowseScreen)

    drive(scenario, size=(40, 14))
    assert snapshot(_isolate_user_catalog) == before


def test_stale_draft_save_reports_error_and_keeps_newer_catalog(_isolate_user_catalog):
    async def scenario(app, pilot):
        await pilot.press("c", "ctrl+r")
        assert isinstance(app.screen, ReviewScreen)
        (_isolate_user_catalog / "detectors").mkdir(exist_ok=True)
        (_isolate_user_catalog / "detectors" / "newer.toml").write_text(
            "observation_angle_deg = 1.0\n"
        )
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        assert "changed while editing" in text(app, "#message")

    drive(scenario)
    assert set(read_raw(bundled_catalog())["detectors"]) >= {"newer"}
    assert not any(name.endswith("_copy") for name in read_raw(bundled_catalog())["detectors"])


def test_edit_writable_user_detector_patches_one_field(_isolate_user_catalog):
    from pyrite.tui.detector_form import DetectorSession

    session = DetectorSession(bundled_catalog())
    session.save(session.stage("clone", "mine", {}, source="reference"))

    async def scenario(app, pilot):
        names = [entry.name for entry in app.screen.session.entries()]
        app.screen.query_one(OptionList).highlighted = names.index("mine")
        await pilot.press("e")
        assert isinstance(app.screen, EditorScreen)
        assert app.screen.query_one("#f-name", Input).disabled
        await type_into(pilot, app.screen.query_one("#f-solid_angle_sr", Input), "0.002")
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, ReviewScreen), text(app, "#message")
        assert "detector set mine --yes --solid-angle 0.002" in text(app, "ReviewScreen Static.cli")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, BrowseScreen)

    drive(scenario)
    row = read_raw(bundled_catalog())["detectors"]["mine"]
    reference = read_raw(bundled_catalog())["detectors"]["reference"]
    assert row == {**reference, "solid_angle_sr": 0.002}


def test_help_and_quit_without_color(_isolate_user_catalog, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    async def scenario(app, pilot):
        await pilot.press("question_mark")
        assert "read-only" in str(app.screen.query_one(Static).render())
        await pilot.press("escape", "q")

    drive(scenario)


def test_run_maps_interruption_to_130():
    class Interrupted:
        def __init__(self, _path):
            pass

        def run(self):
            raise KeyboardInterrupt

    assert detector_app.run("unused", app_factory=Interrupted) == 130
