from pyrite.plots.altair.spectra import multi_case_spectrum_chart

from ..controls import axes_panel
from .common import themed_chart

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
    components = set(values["components"])
    narrow = multi_case_spectrum_chart(
        cases,
        settings,
        include_brem="Bremsstrahlung" in components,
        include_line="Line" in components,
        include_characteristic="Characteristic" in components,
        x_type=axes.narrow.x_type,
        y_type=axes.narrow.y_type,
        band="narrow",
    )
    broad = multi_case_spectrum_chart(
        cases,
        settings,
        include_brem="Bremsstrahlung" in components,
        include_line="Line" in components,
        include_characteristic="Characteristic" in components,
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
            controls["components"],
            axes_panel(mo, controls["axes"]),
        ]
    )
    parts.extend(
        [
            mo.md("**Narrowband**"),
            narrow if narrow is not None else mo.md("*No narrowband spectra in the basket.*"),
            mo.md("**Broadband**"),
            broad if broad is not None else mo.md("*No broadband spectra in the basket.*"),
        ]
    )
    return mo.vstack(parts)
