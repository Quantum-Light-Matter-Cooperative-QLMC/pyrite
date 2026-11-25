"""Publication LaTeX labels for figure titles and legends."""

import pytest

from pyrite.plots._labels import (
    field_label,
    field_value_text,
    latex_material_name,
    material_display_name,
    slice_title,
)


@pytest.mark.parametrize(
    ("name", "latex"),
    [
        ("HOPG", r"\mathrm{HOPG}"),
        ("h-BN", r"\mathrm{h}\text{-}\mathrm{BN}"),
        ("2H-MoS2", r"\mathrm{2H}\text{-}\mathrm{MoS_{2}}"),
        ("beta-FeTe", r"\beta\text{-}\mathrm{FeTe}"),
        ("MoS2 on SiO2/Si", r"\mathrm{MoS_{2}}\ \mathrm{on}\ \mathrm{SiO_{2}}/\mathrm{Si}"),
    ],
)
def test_latex_material_name(name, latex):
    assert latex_material_name(name) == latex


def test_material_display_name_uses_catalog_identity():
    assert material_display_name({"name": "hopg 500nm pol=50 az=180"}) == "HOPG"
    assert material_display_name({"name": "hbn 219nm pol=0 az=0"}) == "h-BN"
    assert material_display_name({"name": "unknownkey 1um"}) == "unknownkey"


def test_slice_title_omits_varying_and_differing_fields():
    cases = [
        {
            "name": "mos2 1um",
            "thickness_ang": 1e4,
            "E0_keV": 300.0,
            "tilt_deg": t,
            "tilt_azim_deg": 180.0,
        }
        for t in (40.0, 50.0)
    ]
    title = slice_title(cases, varying="tilt_deg")
    assert title == (
        r"$\mathrm{2H}\text{-}\mathrm{MoS_{2}},\ 1\,\mu\mathrm{m},\ E_0 = 300\,\mathrm{keV},"
        r"\ \phi_{\mathrm{azim}} = 180^\circ$"
    )


def test_field_labels():
    assert field_label("tilt_deg") == r"$\theta_{\mathrm{tilt}}\ (^\circ)$"
    assert field_label("E0_keV") == r"$E_0\ (\mathrm{keV})$"
    assert field_value_text("tilt_deg", 47.5) == "47.5"
    assert field_value_text("E0_keV", 300.0) == "300"
    assert field_label("label") == "label"
