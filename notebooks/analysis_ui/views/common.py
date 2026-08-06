from __future__ import annotations

from cxr_mc.sweep import fmt_thickness


def axis_warning_block(mo, axes):
    warnings = [*axes.narrow.warnings, *axes.broad.warnings]
    if not warnings:
        return None
    return mo.callout(
        mo.md("\n".join(f"- {warning}" for warning in warnings)),
        kind="warn",
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
