from pyrite._formatting import fmt_thickness
from pyrite.apps._design import apply_altair_theme, apply_plotly_theme


def themed_chart(chart, theme):
    if chart is None:
        return None
    return (
        apply_plotly_theme(chart, theme)
        if callable(getattr(chart, "update_layout", None))
        else apply_altair_theme(chart, theme)
    )


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
