"""Focused presentation tests for issue #759 branding/header updates."""

import ast
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from PIL import Image

from scripts.pyrite_logo import round_svg_canvas, rounded_image, social_card_image


def test_rounded_raster_preserves_center_and_softens_corners():
    image = Image.new("RGB", (200, 100), "#0C1628")
    result = rounded_image(image, 0.04)
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((100, 0))[3] == 255
    assert result.getpixel((100, 50))[:3] == image.getpixel((100, 50))
    assert rounded_image(image, 0) is image


def test_social_card_has_horizontal_safe_area():
    wordmark = Image.new("RGB", (1200, 262), "#ffffff")
    card = social_card_image(wordmark, 0.86)
    assert card.size == (1200, 630)
    assert card.getpixel((50, 315)) == (12, 22, 40)  # navy side padding
    assert card.getpixel((110, 315)) == (255, 255, 255)  # scaled wordmark


def test_svg_uses_same_rounded_canvas_clip(tmp_path: Path):
    svg = tmp_path / "logo.svg"
    svg.write_text('<svg viewBox="0 0 200 100"><g id="figure_1"></g></svg>')
    round_svg_canvas(svg, 0.04)
    result = svg.read_text()
    assert 'clip-path="url(#pyrite-rounded-canvas)"' in result
    assert 'rx="4" ry="4"' in result


def test_desktop_header_has_no_duplicate_top_level_nav():
    conf = (Path(__file__).resolve().parents[2] / "docs" / "conf.py").read_text()
    assert '"navbar_center": []' in conf
    assert '"navbar_persistent": ["search-button-field"]' in conf
    assert '"navbar_end": ["theme-switcher"]' in conf


def test_logo_corner_radius_cli_default_is_seven_percent():
    script = (Path(__file__).resolve().parents[2] / "scripts/pyrite_logo.py").read_text()
    tree = ast.parse(script)
    defaults = [
        {kw.arg: kw.value for kw in node.keywords}["default"]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--corner-radius-fraction"
    ]
    assert len(defaults) == 1
    assert isinstance(defaults[0], ast.Constant)
    assert defaults[0].value == 0.07
