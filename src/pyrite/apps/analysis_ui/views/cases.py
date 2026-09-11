from __future__ import annotations

from pyrite.plots.altair.spectra import multi_case_spectrum_chart

from ..controls import axes_panel
from .common import axis_warning_block, themed_chart

CASE_BASKET_CAP = 12


def render_case_comparison(
    mo,
    *,
    basket,
    picker,
    add_button,
    remove_selector,
    remove_button,
    clear_button,
    controls,
    values,
    axes,
    settings,
    theme,
):
    picker_block = mo.vstack(
        [
            mo.md(
                "Pick cases from the checkpoint currently loaded above, then add them "
                "to the comparison basket. Switch material/face and add more to compare "
                "across materials — the basket persists."
            ),
            picker,
            add_button,
            mo.hstack([remove_selector, remove_button, clear_button], wrap=True),
        ]
    )

    if not basket:
        return mo.vstack([picker_block, mo.md("*Add 2+ cases from the picker above to compare.*")])

    cases = [(entry, entry["case"]["label"]) for entry in basket]
    narrow = multi_case_spectrum_chart(
        cases,
        settings,
        include_brem=values["brem"],
        include_line=values["line"],
        include_characteristic=values["characteristic"],
        x_domain=axes.narrow.x_domain,
        x_type=axes.narrow.x_type,
        y_type=axes.narrow.y_type,
        band="narrow",
    )
    broad = multi_case_spectrum_chart(
        cases,
        settings,
        include_brem=values["brem"],
        include_line=values["line"],
        include_characteristic=values["characteristic"],
        x_domain=axes.broad.x_domain,
        x_type=axes.broad.x_type,
        y_type=axes.broad.y_type,
        band="broad",
    )
    narrow = themed_chart(narrow, theme)
    broad = themed_chart(broad, theme)

    rows = [
        {"label": entry["case"]["label"], "material": entry["case"]["material"]} for entry in basket
    ]
    parts = [picker_block]
    if len(basket) >= CASE_BASKET_CAP:
        parts.append(
            mo.md(f"*Basket capped at {CASE_BASKET_CAP} entries — oldest drop as you add more.*")
        )
    parts.extend(
        [
            mo.md("**Basket contents**"),
            mo.ui.table(rows, selection=None),
            mo.hstack(
                [controls["line"], controls["brem"], controls["characteristic"]],
                wrap=True,
            ),
            axes_panel(mo, controls["axes"]),
        ]
    )
    warning = axis_warning_block(mo, axes)
    if warning is not None:
        parts.append(warning)
    parts.extend(
        [
            mo.md("**Narrowband**"),
            narrow if narrow is not None else mo.md("*No narrowband spectra in the basket.*"),
            mo.md("**Broadband**"),
            broad if broad is not None else mo.md("*No broadband spectra in the basket.*"),
        ]
    )
    return mo.vstack(parts)
