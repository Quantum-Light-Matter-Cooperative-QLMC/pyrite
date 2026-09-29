from pyrite._formatting import fmt_thickness
from pyrite.apps._design import apply_altair_theme


def themed_chart(chart, theme):
    return apply_altair_theme(chart, theme) if chart is not None else None


def optional_selector(mo, widget, values, formatter):
    if len(values) > 1:
        return widget
    value = widget.value
    return mo.md(formatter(value) if value is not None else "")


def thickness_selector(mo, widget, values, *, label: str = "crystal thickness"):
    return optional_selector(
        mo,
        widget,
        values,
        lambda value: f"{label}: {fmt_thickness(value)}",
    )
