"""Preset framing and context without requiring ParaView in the test environment."""

import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyrite import DATA_DIR


@pytest.fixture
def preset(monkeypatch):
    simple = SimpleNamespace(
        ExtractBlock=Mock(return_value=SimpleNamespace()),
        Show=Mock(
            return_value=SimpleNamespace(
                BlockColors=["/Root/crystal", "1", "0", "0"],
                BlockOpacities=["/Root/crystal", "1"],
            )
        ),
        ColorBy=Mock(),
        ResetCamera=Mock(),
    )
    monkeypatch.setitem(sys.modules, "paraview", SimpleNamespace(simple=simple))
    return runpy.run_path(str(DATA_DIR / "paraview/pyrite_trajectories.py")), simple


def test_scene_context_cannot_inherit_opaque_red_block_styles(preset, tmp_path):
    script, simple = preset
    manifest = tmp_path / "case.vtm"
    manifest.write_text('<VTKFile><DataSet name="tracks"/><DataSet name="crystal"/></VTKFile>')
    script["_context"](object(), str(manifest), object())
    display = simple.Show.return_value
    assert display.Representation == "Wireframe"
    assert display.BlockColors == [] and display.BlockOpacities == []
    assert display.DiffuseColor == [0.75, 0.75, 0.75]
    assert display.Opacity < 0.3
    assert display.ColorArrayName == ["POINTS", ""]
    simple.ColorBy.assert_not_called()


def test_camera_fits_visible_history_instead_of_entire_scene(preset):
    script, simple = preset
    bounds = [1e6, 1e6 + 20, -2e6, -2e6 + 10, 0, 30]
    source = Mock()
    source.GetDataInformation.return_value.GetBounds.return_value = bounds
    view = Mock()
    script["_side_view"](view, source)
    view.ResetCamera.assert_called_once_with(bounds)
    simple.ResetCamera.assert_not_called()
    source.UpdatePipeline.assert_called_once()
