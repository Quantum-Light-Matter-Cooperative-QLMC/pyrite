"""Shared presentation primitives for PyRITE Marimo applications."""

from collections.abc import Iterable, Mapping
from html import escape
from importlib.resources import files
from string import Template

THEMES = {
    "dark": {
        "bg": "#12202B",
        "surface": "#1D303D",
        "text": "#DCEEF2",
        "muted": "#91AEB7",
        "subtle": "#B9D0D6",
        "rule": "#41606F",
        "focus": "#72C7D5",
        "compute": "#D8A657",
        "failure": "#D8787E",
        "done": "#A6E4EF",
    },
    "light": {
        "bg": "#F4F7F8",
        "surface": "#FFFFFF",
        "text": "#172832",
        "muted": "#526B76",
        "subtle": "#405A65",
        "rule": "#C6D2D8",
        "focus": "#187E91",
        "compute": "#9A6516",
        "failure": "#B44750",
        "done": "#3A98AA",
    },
}

# Backward-compatible named palette for code that imports COLORS directly.
COLORS = {
    "beamline_navy": THEMES["dark"]["bg"],
    "instrument_slate": THEMES["dark"]["surface"],
    "xray_ice": THEMES["dark"]["text"],
    "detector_cyan": THEMES["dark"]["focus"],
    "tungsten_amber": THEMES["dark"]["compute"],
    "validation_rose": THEMES["dark"]["failure"],
}

SPACING = {"xs": ".35rem", "sm": ".65rem", "md": "1rem", "lg": "1.5rem"}
TYPOGRAPHY = {
    "display": '"IBM Plex Sans Condensed", "Arial Narrow", sans-serif',
    "body": '"IBM Plex Sans", "Aptos", "Segoe UI", sans-serif',
    "data": '"IBM Plex Mono", monospace',
}
BORDERS = {"width": "1px", "radius": ".35rem"}
FOCUS = {"width": "3px", "offset": "3px"}
WIDTHS = {"prose": "72ch", "breakpoint": "768px"}

STATUS_KINDS = {
    "Ready": "ready",
    "Running": "running",
    "Cached": "cached",
    "Passed": "passed",
    "Failed": "failed",
    "Completed—interpret": "interpret",
    "Skipped": "skipped",
}


def notebook_css() -> str:
    """Return shared light/dark CSS; the resolved mode lives on body[data-theme]."""
    dark = THEMES["dark"]
    light = THEMES["light"]
    tokens = {
        "BORDERS_RADIUS": BORDERS["radius"],
        "BORDERS_WIDTH": BORDERS["width"],
        "FOCUS_OFFSET": FOCUS["offset"],
        "FOCUS_WIDTH": FOCUS["width"],
        "TYPOGRAPHY_BODY": TYPOGRAPHY["body"],
        "TYPOGRAPHY_DATA": TYPOGRAPHY["data"],
        "TYPOGRAPHY_DISPLAY": TYPOGRAPHY["display"],
        "WIDTHS_BREAKPOINT": WIDTHS["breakpoint"],
        "WIDTHS_PROSE": WIDTHS["prose"],
        "DARK_BG": dark["bg"],
        "DARK_COMPUTE": dark["compute"],
        "DARK_DONE": dark["done"],
        "DARK_FAILURE": dark["failure"],
        "DARK_FOCUS": dark["focus"],
        "DARK_MUTED": dark["muted"],
        "DARK_RULE": dark["rule"],
        "DARK_SUBTLE": dark["subtle"],
        "DARK_SURFACE": dark["surface"],
        "DARK_TEXT": dark["text"],
        "LIGHT_BG": light["bg"],
        "LIGHT_COMPUTE": light["compute"],
        "LIGHT_DONE": light["done"],
        "LIGHT_FAILURE": light["failure"],
        "LIGHT_FOCUS": light["focus"],
        "LIGHT_MUTED": light["muted"],
        "LIGHT_RULE": light["rule"],
        "LIGHT_SUBTLE": light["subtle"],
        "LIGHT_SURFACE": light["surface"],
        "LIGHT_TEXT": light["text"],
    }
    template = files("pyrite.apps").joinpath("_design.css").read_text()
    return Template(template).substitute(tokens).strip()


def style_sheet(mo):
    # The style tag applies globally; hide its otherwise empty layout item.
    return mo.Html(notebook_css()).style({"display": "none"})


def theme_switch(mo):
    """Return the shared appearance selector used by every PyRITE app."""
    from pyrite.apps._widgets import ThemeSelect

    return mo.ui.anywidget(ThemeSelect())


def resolved_theme(theme_ui, fallback: str = "light") -> str:
    """Resolved ``light``/``dark`` value from :func:`theme_switch`."""
    state = theme_ui.value or {}
    value = state.get("resolved") if isinstance(state, Mapping) else None
    return value if value in THEMES else fallback


def configure_matplotlib_theme(theme: str) -> str:
    """Set Matplotlib rcParams to the resolved PyRITE appearance.

    marimo intentionally does not restore Matplotlib defaults when switching
    from dark to light, so the app-level switch must do that explicitly.
    """
    mode = theme if theme in THEMES else "light"
    try:
        import matplotlib.style

        matplotlib.style.use("dark_background" if mode == "dark" else "default")
    except ImportError:
        pass
    return mode


def apply_altair_theme(chart, theme: str):
    """Apply explicit Vega-Lite text/surface colors to an Altair chart.

    marimo applies its own frontend Vega theme from the configured display
    theme. PyRITE's runtime switch is independent of that config, so these
    explicit chart settings keep labels, legends, and titles synchronized with
    the selected PyRITE appearance. Existing chart typography is retained while
    theme colors are merged into each config section. Explicit chart titles are
    rewritten too, because a title-level color beats ``configure_title`` in
    Vega-Lite.
    """
    if chart is None:
        return None
    mode = theme if theme in THEMES else "light"
    palette = THEMES[mode]

    try:
        raw_config = chart.to_dict(validate=False).get("config", {})
        config = raw_config if isinstance(raw_config, dict) else {}
    except TypeError, ValueError:
        config = {}

    def config_section(name: str) -> dict:
        section = config.get(name, {})
        return dict(section) if isinstance(section, dict) else {}

    axis_config = config_section("axis") | {
        "labelColor": palette["text"],
        "titleColor": palette["text"],
        "gridColor": palette["rule"],
        "domainColor": palette["muted"],
        "tickColor": palette["muted"],
    }
    legend_config = config_section("legend") | {
        "labelColor": palette["text"],
        "titleColor": palette["text"],
    }
    header_config = config_section("header") | {
        "labelColor": palette["text"],
        "titleColor": palette["text"],
    }
    title_config = config_section("title") | {
        "color": palette["text"],
        "subtitleColor": palette["muted"],
    }
    text_config = config_section("text") | {"color": palette["text"]}
    view_config = config_section("view") | {"stroke": palette["rule"]}

    chart = (
        chart.configure(background=palette["surface"])
        .configure_axis(**axis_config)
        .configure_legend(**legend_config)
        .configure_header(**header_config)
        .configure_title(**title_config)
        .configure_text(**text_config)
        .configure_view(**view_config)
    )

    # ``configure_title`` does not override a color embedded directly in a
    # chart's TitleParams. Spectrum charts use explicit titles, so normalize
    # the top-level title while preserving its text and other properties.
    try:
        import altair as alt

        spec = chart.to_dict(validate=False)
        title = spec.get("title")
        if isinstance(title, str):
            chart = chart.properties(title=alt.TitleParams(text=title, color=palette["text"]))
        elif isinstance(title, dict):
            title = dict(title)
            title["color"] = palette["text"]
            title["subtitleColor"] = palette["muted"]
            chart = chart.properties(title=alt.TitleParams(**title))
    except ImportError, TypeError, ValueError:
        # The configure_* rules above still cover charts without a serializable
        # top-level title (or environments where Altair is optional).
        pass

    return chart


def static_altair_chart(mo, chart):
    """Render a non-selectable chart through marimo's schema-safe path.

    Marimo's Arrow data transformer extends Vega-Lite with
    ``format.type = "arrow"``. Altair's raw MIME renderer validates that
    extension against the upstream schema and rejects it; ``mo.ui.altair_chart``
    deliberately serializes Marimo transformers without that validation.
    """
    if chart is None:
        return None
    return mo.ui.altair_chart(
        chart,
        chart_selection=False,
        legend_selection=False,
    )


# A white canvas makes the yellow end of many scientific sequential maps hard
# to distinguish.  This clipped, high-contrast sequential map keeps the full
# energy trajectory visible without changing the underlying scalar values.
_LIGHT_ENERGY_COLORSCALE = [
    [0.00, "#352A86"],
    [0.20, "#245DA8"],
    [0.40, "#1687A7"],
    [0.60, "#169878"],
    [0.80, "#9A7412"],
    [1.00, "#B23A32"],
]


def _colorbar_title_text(colorbar) -> str:
    title = getattr(colorbar, "title", None)
    text = getattr(title, "text", None) if title is not None else None
    return "" if text is None else str(text)


def _theme_plotly_colorbar(colorbar, *, text: str, surface: str, rule: str, mode: str):
    """Theme a Plotly colorbar and keep its title on one readable line."""
    raw_title = _colorbar_title_text(colorbar)
    is_energy = "energy" in raw_title.lower() or "kev" in raw_title.lower()

    title_text = raw_title
    if is_energy:
        # Plotly will happily wrap narrow colorbar titles into a pile of glyphs.
        # Normalize common explicit breaks and use a short horizontal label.
        title_text = (
            raw_title.replace("<br />", " ")
            .replace("<br/>", " ")
            .replace("<br>", " ")
            .replace("\n", " ")
        )
        title_text = " ".join(title_text.split()) or "Energy (keV)"
        if len(title_text) > 24:
            title_text = "Energy (keV)"

    update: dict[str, object] = dict(
        tickfont=dict(color=text, size=11),
        outlinecolor=rule,
        outlinewidth=1,
        bgcolor="rgba(255,255,255,0.88)" if mode == "light" else "rgba(18,32,43,0.82)",
    )
    if raw_title:
        update["title"] = dict(
            text=title_text,
            side="top",
            font=dict(color=text, size=12),
        )
    if is_energy:
        update.update(thickness=18, len=0.72, xpad=8, ypad=6)
    colorbar.update(**update)
    return is_energy


def apply_plotly_theme(fig, theme: str):
    """Apply the shared palette to Plotly, including 2D/3D text and colorbars."""
    mode = theme if theme in THEMES else "light"
    palette = THEMES[mode]
    text = palette["text"]
    surface = palette["surface"]
    rule = palette["rule"]
    muted = palette["muted"]

    axis_2d = dict(
        color=text,
        tickfont=dict(color=text),
        title=dict(font=dict(color=text)),
        gridcolor=rule,
        zerolinecolor=muted,
        linecolor=muted,
    )
    axis_3d = dict(
        backgroundcolor=surface,
        gridcolor=rule,
        zerolinecolor=muted,
        color=text,
        tickfont=dict(color=text),
        title=dict(font=dict(color=text)),
    )

    fig.update_layout(
        template="plotly_white" if mode == "light" else "plotly_dark",
        paper_bgcolor=surface,
        plot_bgcolor=surface,
        font=dict(color=text),
        title=dict(font=dict(color=text)),
        legend=dict(
            font=dict(color=text),
            title=dict(font=dict(color=text)),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=axis_2d,
        yaxis=axis_2d,
        scene=dict(
            bgcolor=surface,
            xaxis=axis_3d,
            yaxis=axis_3d,
            zaxis=axis_3d,
        ),
    )

    # Some PyRITE traces own their own colorbars; layout.font does not always
    # override those nested fonts. Set them explicitly and, for the energy
    # trajectories, use a light-canvas-safe scale and a slightly stronger line.
    for trace in fig.data:
        energy_owner = None
        for owner_name in ("marker", "line"):
            owner = getattr(trace, owner_name, None)
            colorbar = getattr(owner, "colorbar", None) if owner is not None else None
            if colorbar is not None:
                if _theme_plotly_colorbar(
                    colorbar, text=text, surface=surface, rule=rule, mode=mode
                ):
                    energy_owner = owner

        colorbar = getattr(trace, "colorbar", None)
        if colorbar is not None:
            _theme_plotly_colorbar(colorbar, text=text, surface=surface, rule=rule, mode=mode)

        if energy_owner is not None and mode == "light":
            # Scatter3d energy trajectories carry the scalar mapping on
            # ``line``; marker-based plots are handled equivalently.
            try:
                energy_owner.colorscale = _LIGHT_ENERGY_COLORSCALE
            except AttributeError, ValueError:
                pass
            if hasattr(energy_owner, "width"):
                try:
                    width = energy_owner.width
                    energy_owner.width = max(4, float(width or 0))
                except TypeError, ValueError:
                    pass

    if getattr(fig.layout, "annotations", None):
        for annotation in fig.layout.annotations:
            annotation.font.color = text
    if getattr(fig.layout.scene, "annotations", None):
        for annotation in fig.layout.scene.annotations:
            annotation.font.color = text

    return fig


def page_title(mo, title: str, intro: str, *, eyebrow: str = "PyRITE"):
    return mo.Html(
        '<header class="pyrite-shell pyrite-title">'
        f'<div class="pyrite-title__eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(intro)}</p></header>"
    )


def context_rail(mo, values: Mapping[str, object] | Iterable[tuple[str, object]]):
    items = values.items() if isinstance(values, Mapping) else values
    body = "".join(
        '<div class="pyrite-rail__item">'
        f"<dt>{escape(str(label))}</dt><dd>{escape(_display(value))}</dd></div>"
        for label, value in items
    )
    return mo.Html(f'<dl class="pyrite-shell pyrite-rail">{body}</dl>')


def status_badge(mo, state: str):
    kind = STATUS_KINDS.get(state, "skipped")
    return mo.Html(
        f'<span class="pyrite-shell pyrite-badge pyrite-badge--{kind}">{escape(state)}</span>'
    )


def control_group(mo, label: str, controls):
    return mo.vstack(
        [mo.Html(f'<div class="pyrite-shell pyrite-group__label">{escape(label)}</div>'), controls]
    ).style({"border-left": "2px solid var(--pyrite-rule)", "padding-left": "1rem"})


def directional_state(mo, title: str, detail: str, action: str, *, kind: str = "info"):
    return mo.callout(
        mo.md(f"**{title}**\n\n{detail}\n\nNext: `{action}`"),
        kind=kind,
    )


#: Cell fill segments, drawn bottom-to-top, mapped to the palette in CSS
#: (``.pyrite-grid__seg--<state>``). Order is the reading order of the stack.
SCAN_GRID_STATES = ("done", "cached", "running", "excluded")
_SCAN_GRID_LEGEND = {
    "done": ("This session", "var(--pyrite-done)"),
    "cached": ("Cached", "var(--pyrite-focus)"),
    "running": ("Running", "var(--pyrite-compute)"),
    "excluded": ("Penetration-excluded", "var(--pyrite-failure)"),
    "remaining": ("Remaining", "var(--pyrite-surface)"),
}


def _scan_cell(cell: Mapping[str, float] | None, max_weight: float) -> str:
    """One grid cell: a cost-sized marker whose vertical fill stacks the state
    fractions (see :data:`SCAN_GRID_STATES`). ``cell`` is ``None`` for an absent
    (unrequested) case, or a mapping of state -> fraction in [0, 1] plus
    ``weight`` (relative compute cost). The marker side scales with
    ``sqrt(weight / max_weight)`` so the heavy high-energy corner reads as heavy,
    with a floor so light cells stay legible."""
    if cell is None:
        return '<div class="pyrite-grid__cell" style="opacity:.28"></div>'
    weight = float(cell.get("weight", 0.0))
    side = 100.0 if max_weight <= 0 else max(34.0, 100.0 * (weight / max_weight) ** 0.5)
    total = sum(max(0.0, float(cell.get(state, 0.0))) for state in SCAN_GRID_STATES)
    segments = "".join(
        f'<div class="pyrite-grid__seg pyrite-grid__seg--{state}" '
        f'style="height:{100.0 * max(0.0, float(cell.get(state, 0.0))):.4g}%"></div>'
        for state in SCAN_GRID_STATES
    )
    done = min(1.0, total)
    title = escape(f"{done:.0%} done, {weight / max_weight:.0%} of the peak per-cell cost")
    return (
        f'<div class="pyrite-grid__cell" title="{title}">'
        f'<div class="pyrite-grid__marker" style="width:{side:.4g}%;height:{side:.4g}%">'
        f"{segments}</div></div>"
    )


def scan_grid(mo, energies, rows, *, max_weight=None, states=SCAN_GRID_STATES):
    """Compute-weighted scan-state matrix: rows (e.g. polar tilts) x beam energies.

    ``energies`` are the column labels (keV); ``rows`` is a list of
    ``{"label": str, "cells": [cell, ...]}`` with one ``cell`` per energy (see
    :func:`_scan_cell`). Each cell's colored fill shows the fraction of its
    aggregated cases that are cached / done-this-session / running / penetration-
    excluded (remaining = the empty slate track), and its marker size scales with
    that group's relative compute cost so the sweep's heavy corner reads as
    heavy. ``max_weight`` overrides the cost normalization (default = the largest
    cell weight); pass a sweep-wide constant to keep the preview and the live
    grid on one scale."""
    if max_weight is None:
        max_weight = max(
            (float(cell["weight"]) for row in rows for cell in row["cells"] if cell),
            default=0.0,
        )
    columns = f"minmax(5ch, max-content) repeat({len(energies)}, minmax(12px, 1fr))"
    header = '<div class="pyrite-grid__corner">tilt \\ keV</div>' + "".join(
        f'<div class="pyrite-grid__col">{escape(_display(energy))}</div>' for energy in energies
    )
    body = "".join(
        f'<div class="pyrite-grid__row">{escape(str(row["label"]))}</div>'
        + "".join(_scan_cell(cell, max_weight) for cell in row["cells"])
        for row in rows
    )
    legend = "".join(
        f'<span><i class="pyrite-grid__swatch" style="background:{_SCAN_GRID_LEGEND[state][1]}"></i>'
        f"{escape(_SCAN_GRID_LEGEND[state][0])}</span>"
        for state in (*states, "remaining")
        if state in _SCAN_GRID_LEGEND
    )
    return mo.Html(
        f'<div class="pyrite-shell pyrite-grid" style="grid-template-columns:{columns}" '
        f'role="img" aria-label="Scan progress matrix, tilt by beam energy, '
        f'cell size by compute cost">{header}{body}</div>'
        f'<div class="pyrite-shell pyrite-grid__legend">{legend}</div>'
    )


def _display(value: object) -> str:
    if value is None or value == "":
        return "not selected"
    return str(value)
