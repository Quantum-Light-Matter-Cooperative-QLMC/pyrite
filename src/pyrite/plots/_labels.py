"""Publication-style LaTeX labels for figure titles and legends.

Renderer-neutral: every label is a ``$...$`` math string that Plotly (MathJax)
and matplotlib (mathtext) both typeset. Plotly renders math only when the whole
string is wrapped in ``$``, so a title is one math expression, never prose
mixed with math.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# A chemical-formula fragment: element symbols with optional counts ("MoS2").
_FORMULA = re.compile(r"(?:[A-Z][a-z]?\d*)+")
_GREEK = frozenset({"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"})

#: Math symbol of each sweepable case field.
FIELD_SYMBOLS = {
    "E0_keV": r"E_0",
    "tilt_deg": r"\theta_{\mathrm{tilt}}",
    "tilt_azim_deg": r"\phi_{\mathrm{tilt}}",
}
_FIELD_UNITS = {"E0_keV": r"\,\mathrm{keV}", "tilt_deg": r"^\circ", "tilt_azim_deg": r"^\circ"}
_LEGEND_UNITS = {"E0_keV": r"\mathrm{keV}", "tilt_deg": r"^\circ", "tilt_azim_deg": r"^\circ"}

#: Spectrum axis titles (Plotly/MathJax and matplotlib mathtext both typeset these).
PHOTON_ENERGY_TITLE = r"$\large{\mathrm{Photon\ energy}\ (\mathrm{eV})}$"
SPECTRAL_INTENSITY_TITLE = r"$\large{\mathrm{Intensity}\ (\mathrm{ph\,s^{-1}\,eV^{-1}\,\mu A^{-1}})}$"


def _latex_word(word: str) -> str:
    """One space-free name word: formula digits become subscripts."""
    out = []
    for part in re.split(r"([-/])", word):
        if not part:
            continue
        if part == "-":
            out.append(r"\text{-}")
        elif part == "/":
            out.append("/")
        elif part.lower() in _GREEK:
            out.append("\\" + part.lower())
        elif _FORMULA.fullmatch(part):
            out.append(r"\mathrm{" + re.sub(r"(\d+)", r"_{\1}", part) + "}")
        else:
            out.append(r"\mathrm{" + part + "}")
    return "".join(out)


def latex_material_name(name: str) -> str:
    """Math-mode body for a display name: ``"2H-MoS2"`` -> ``\\mathrm{2H}\\text{-}\\mathrm{MoS_{2}}``."""
    return r"\ ".join(_latex_word(word) for word in name.split())


def material_display_name(case: Mapping) -> str:
    """Catalog display name (``"HOPG"``, ``"h-BN"``, ``"2H-MoS2"``) for ``case``.

    The case name leads with the catalog material key; an unknown key falls
    back to the crystal key's material, then to the raw key.
    """
    key = str(case.get("name", "")).split()[0] if case.get("name") else case.get("crystal", "")
    from ..materials import CATALOG

    materials = CATALOG.materials
    spec = materials.get(key) or materials.get(case.get("crystal", ""))
    return spec.identity.name if spec is not None else str(key)


def latex_thickness(thickness_ang: float) -> str:
    """Thickness in nm below 1 um, else um, as a math body."""
    if thickness_ang < 1e4:
        return rf"{thickness_ang / 10:g}\,\mathrm{{nm}}"
    return rf"{thickness_ang / 1e4:g}\,\mu\mathrm{{m}}"


def field_label(field: str) -> str:
    """Legend title with unit, e.g. ``$\\theta_{\\mathrm{tilt}}\\ (^\\circ)$``."""
    symbol = FIELD_SYMBOLS.get(field)
    return rf"${symbol}\ ({_LEGEND_UNITS[field]})$" if symbol else field


def field_value_text(field: str, value) -> str:
    """Legend entry for one field value, unit-free (the title carries it): ``"47.5"``."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _common_value(cases: list[Mapping], key: str):
    values = {case.get(key) for case in cases}
    return values.pop() if len(values) == 1 else None


def slice_title(cases: Iterable[Mapping], *, varying: str | None = None) -> str:
    """Math title naming the material, thickness and every pinned sweep field.

    Fields that differ across ``cases`` (including ``varying``) are left out,
    so a title never claims one value for a figure that shows several.
    Thickness uses the records not on a thickness fallback when they agree.
    """
    cases = list(cases)
    if not cases:
        return ""
    parts = [latex_material_name(material_display_name(cases[0]))]
    pinned = [case for case in cases if "thickness_fallback" not in case] or cases
    thickness = _common_value(pinned, "thickness_ang")
    if thickness is not None:
        parts.append(latex_thickness(float(thickness)))
    for field, symbol in FIELD_SYMBOLS.items():
        if field == varying:
            continue
        value = _common_value(cases, field)
        if value is not None:
            parts.append(rf"{symbol} = {float(value):g}{_FIELD_UNITS[field]}")
    return r"$\Large{" + r",\ ".join(parts) + "}$"


def append_to_title(title: str | None, text: str) -> str | None:
    """Append plain ``text`` to ``title``, inside the math when it is math."""
    if not title:
        return title
    if title.startswith("$") and title.endswith("$") and len(title) > 1:
        return title[:-1] + r"\ (\mathrm{" + text.replace(" ", r"\ ") + "})$"
    return f"{title} ({text})"
