"""Shared presentation primitives for PyRITE Marimo applications."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape

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
    return f"""
<style>
  html[data-pyrite-theme="dark"],
  body[data-theme="dark"] {{
    --cxr-bg: {dark["bg"]};
    --cxr-surface: {dark["surface"]};
    --cxr-text: {dark["text"]};
    --cxr-muted: {dark["muted"]};
    --cxr-subtle: {dark["subtle"]};
    --cxr-rule: {dark["rule"]};
    --cxr-focus: {dark["focus"]};
    --cxr-compute: {dark["compute"]};
    --cxr-failure: {dark["failure"]};
    --cxr-done: {dark["done"]};
    --background: {dark["bg"]};
    --foreground: {dark["text"]};
    --muted: {dark["surface"]};
    --muted-foreground: {dark["muted"]};
    --popover: {dark["surface"]};
    --popover-foreground: {dark["text"]};
    --card: {dark["surface"]};
    --card-foreground: {dark["text"]};
    --border: {dark["rule"]};
    --input: {dark["surface"]};
    --primary: {dark["focus"]};
    --primary-foreground: {dark["bg"]};
    --secondary: {dark["surface"]};
    --secondary-foreground: {dark["text"]};
    --accent: {dark["rule"]};
    --accent-foreground: {dark["text"]};
    --ring: {dark["focus"]};
    color-scheme: dark;
  }}
  html[data-pyrite-theme="light"],
  body[data-theme="light"] {{
    --cxr-bg: {light["bg"]};
    --cxr-surface: {light["surface"]};
    --cxr-text: {light["text"]};
    --cxr-muted: {light["muted"]};
    --cxr-subtle: {light["subtle"]};
    --cxr-rule: {light["rule"]};
    --cxr-focus: {light["focus"]};
    --cxr-compute: {light["compute"]};
    --cxr-failure: {light["failure"]};
    --cxr-done: {light["done"]};
    --background: {light["bg"]};
    --foreground: {light["text"]};
    --muted: #EEF3F5;
    --muted-foreground: {light["muted"]};
    --popover: {light["surface"]};
    --popover-foreground: {light["text"]};
    --card: {light["surface"]};
    --card-foreground: {light["text"]};
    --border: {light["rule"]};
    --input: {light["surface"]};
    --primary: {light["focus"]};
    --primary-foreground: #FFFFFF;
    --secondary: #E8EFF2;
    --secondary-foreground: {light["text"]};
    --accent: #E3EEF1;
    --accent-foreground: {light["text"]};
    --ring: {light["focus"]};
    color-scheme: light;
  }}
  .cxr-shell {{
    color: var(--cxr-text);
    font-family: {TYPOGRAPHY["body"]};
    max-width: 100%;
  }}
  body:has(.cxr-shell) {{
    background: var(--cxr-bg, {dark["bg"]});
    color: var(--cxr-text, {dark["text"]});
  }}
  body:has(.cxr-shell) :is(button, input, select, [tabindex]):focus-visible {{
    outline: {FOCUS["width"]} solid var(--cxr-focus, {dark["focus"]});
    outline-offset: {FOCUS["offset"]};
  }}
  body:has(.cxr-shell) :is(button, select, input:not([type="hidden"])) {{
    min-height: 44px;
  }}
  .cxr-theme-select {{
    display: inline-flex; align-items: center; gap: .55rem;
    color: var(--cxr-muted); font: 600 .7rem/1.2 {TYPOGRAPHY["data"]};
    letter-spacing: .06em; text-transform: uppercase; white-space: nowrap;
  }}
  .cxr-theme-select__control {{
    min-height: 36px !important; padding: .25rem 1.8rem .25rem .55rem;
    border: 1px solid var(--cxr-rule); border-radius: {BORDERS["radius"]};
    background: var(--cxr-surface); color: var(--cxr-text);
    font: 500 .78rem/1.2 {TYPOGRAPHY["data"]}; letter-spacing: 0;
    text-transform: none;
  }}
  /* Marimo's dropdown/select contents are portalled to document.body, outside
     the notebook subtree.  Target the actual Select/Radix portal surfaces and
     win over Marimo/Tailwind utility classes explicitly. */
  html[data-pyrite-theme="light"] :is(
    [data-slot="select-content"],
    [data-slot="dropdown-menu-content"],
    [data-radix-select-content],
    [role="listbox"],
    [role="menu"],
    [data-radix-popper-content-wrapper] > [data-side]
  ) {{
    background: {light["surface"]} !important;
    background-color: {light["surface"]} !important;
    color: {light["text"]} !important;
    border-color: {light["rule"]} !important;
    color-scheme: light !important;
  }}
  html[data-pyrite-theme="light"] :is(
    [data-slot="select-viewport"],
    [data-radix-select-viewport]
  ) {{
    background: {light["surface"]} !important;
    color: {light["text"]} !important;
  }}
  html[data-pyrite-theme="light"] :is(
    [data-slot="select-item"],
    [data-radix-select-item],
    [role="option"],
    [role="menuitem"]
  ) {{
    background: transparent !important;
    color: {light["text"]} !important;
  }}
  html[data-pyrite-theme="light"] :is(
    [data-slot="select-item"],
    [data-radix-select-item],
    [role="option"],
    [role="menuitem"]
  ):is(:hover, :focus, [data-highlighted], [data-state="checked"]) {{
    background: #E3EEF1 !important;
    color: {light["text"]} !important;
  }}
  html[data-pyrite-theme="light"] :is(
    button[role="combobox"],
    button[aria-haspopup="listbox"]
  ) {{
    background: {light["surface"]} !important;
    color: {light["text"]} !important;
    border-color: {light["rule"]} !important;
  }}
  /* Compatibility with marimo releases that style selects through utility
     classes instead of data-slot attributes, plus native <select> fallbacks. */
  html[data-pyrite-theme="light"] :is(.bg-popover, .bg-background) {{
    background-color: {light["surface"]} !important;
  }}
  html[data-pyrite-theme="light"] :is(.text-popover-foreground, .text-foreground) {{
    color: {light["text"]} !important;
  }}
  html[data-pyrite-theme="light"] select,
  html[data-pyrite-theme="light"] select option,
  html[data-pyrite-theme="light"] select optgroup {{
    background-color: {light["surface"]} !important;
    color: {light["text"]} !important;
    color-scheme: light !important;
  }}
  /* Marimo's tab strip (mo.ui.tabs) renders inline, not portalled, but reads
     the same frozen dark palette as the select surfaces above -- its own
     light/dark detection is captured once at load and never re-derives when
     this switch flips body[data-theme] later. Target the stable ARIA roles
     rather than Tailwind's generated utility classnames, which are not a
     stable contract across marimo releases. */
  html[data-pyrite-theme="light"] :is([role="tablist"], .bg-muted) {{
    background-color: {light["surface"]} !important;
  }}
  html[data-pyrite-theme="light"] :is([role="tab"], .text-muted-foreground) {{
    color: {light["muted"]} !important;
  }}
  html[data-pyrite-theme="light"] [role="tab"][data-state="active"] {{
    background-color: {light["bg"]} !important;
    color: {light["text"]} !important;
  }}
  html[data-pyrite-theme="light"] [role="tabpanel"] {{
    color: {light["text"]} !important;
  }}
  .cxr-title {{ max-width: {WIDTHS["prose"]}; margin: 0 0 1.25rem; }}
  .cxr-title__eyebrow {{
    color: var(--cxr-focus); font: 600 .75rem/1.2 {TYPOGRAPHY["data"]};
    letter-spacing: .14em;
  }}
  .cxr-title h1 {{
    font-family: {TYPOGRAPHY["display"]};
    font-size: clamp(2rem, 4vw, 3.4rem); line-height: .98; margin: .35rem 0 .7rem;
  }}
  .cxr-title p {{ color: var(--cxr-subtle); line-height: 1.55; margin: 0; }}
  .cxr-rail {{
    display: flex; flex-wrap: wrap; align-items: baseline; gap: .3rem 1.4rem;
    border-block: {BORDERS["width"]} solid var(--cxr-rule);
    margin: .6rem 0 .9rem; padding: .5rem 0;
  }}
  .cxr-rail__item {{
    display: flex; align-items: baseline; gap: .5rem; min-width: 0;
    border-left: 2px solid var(--cxr-focus); padding-left: .6rem;
  }}
  .cxr-rail dt {{
    color: var(--cxr-muted); font: 600 .64rem/1.2 {TYPOGRAPHY["data"]};
    letter-spacing: .1em; text-transform: uppercase; white-space: nowrap;
  }}
  .cxr-rail dd {{
    color: var(--cxr-text); font: 500 .85rem/1.2 {TYPOGRAPHY["data"]};
    margin: 0; white-space: nowrap;
  }}
  .cxr-badge {{
    border: 1px solid currentColor; border-radius: 999px; display: inline-block;
    font: 600 .72rem/1 {TYPOGRAPHY["data"]}; padding: .38rem .62rem;
  }}
  .cxr-badge--failed {{ color: var(--cxr-failure); }}
  .cxr-badge--running, .cxr-badge--interpret {{ color: var(--cxr-compute); }}
  .cxr-badge--ready, .cxr-badge--cached, .cxr-badge--passed {{ color: var(--cxr-focus); }}
  .cxr-grid {{ display: grid; gap: .28rem; align-items: end; }}
  .cxr-grid__corner, .cxr-grid__col, .cxr-grid__row {{
    color: var(--cxr-muted); font: 600 .62rem/1.1 {TYPOGRAPHY["data"]};
    letter-spacing: .04em; white-space: nowrap;
  }}
  .cxr-grid__col {{ text-align: center; align-self: end; }}
  .cxr-grid__row {{ text-align: right; align-self: center; padding-right: .3rem; }}
  .cxr-grid__cell {{
    position: relative; aspect-ratio: 1 / 1; width: 100%; min-width: 12px;
    border-radius: 2px; background: var(--cxr-surface);
    box-shadow: inset 0 0 0 1px var(--cxr-rule); overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }}
  .cxr-grid__marker {{
    display: flex; flex-direction: column-reverse; border-radius: 1px;
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--cxr-text) 25%, transparent);
  }}
  .cxr-grid__seg {{ width: 100%; }}
  .cxr-grid__seg--cached {{ background: var(--cxr-focus); }}
  .cxr-grid__seg--done {{ background: var(--cxr-done); }}
  .cxr-grid__seg--running {{ background: var(--cxr-compute); }}
  .cxr-grid__seg--excluded {{ background: var(--cxr-failure); }}
  .cxr-grid__legend {{
    display: flex; flex-wrap: wrap; gap: .8rem; margin-top: .6rem;
    color: var(--cxr-muted); font: 500 .68rem/1.2 {TYPOGRAPHY["data"]};
  }}
  .cxr-grid__legend span {{ display: inline-flex; align-items: center; gap: .35rem; }}
  .cxr-grid__swatch {{
    width: .72rem; height: .72rem; border-radius: 2px;
    box-shadow: inset 0 0 0 1px var(--cxr-rule);
  }}
  .cxr-group {{ border-left: 2px solid var(--cxr-rule); padding: .35rem 0 .35rem 1rem; }}
  .cxr-group__label {{
    color: var(--cxr-muted); font: 600 .72rem/1.2 {TYPOGRAPHY["data"]};
    letter-spacing: .08em; margin-bottom: .65rem; text-transform: uppercase;
  }}
  @media (max-width: {WIDTHS["breakpoint"]}) {{
    .cxr-title h1 {{ font-size: 2rem; }}
    .cxr-theme-select__label {{ display: none; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    body:has(.cxr-shell) *,
    body:has(.cxr-shell) *::before,
    body:has(.cxr-shell) *::after {{
      animation-duration: .01ms !important; transition-duration: .01ms !important;
    }}
  }}
</style>
""".strip()


def style_sheet(mo):
    return mo.Html(notebook_css())


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
    the selected PyRITE appearance. Explicit chart titles are rewritten too,
    because a title-level color beats ``configure_title`` in Vega-Lite.
    """
    if chart is None:
        return None
    mode = theme if theme in THEMES else "light"
    palette = THEMES[mode]
    chart = (
        chart.configure(background=palette["surface"])
        .configure_axis(
            labelColor=palette["text"],
            titleColor=palette["text"],
            gridColor=palette["rule"],
            domainColor=palette["muted"],
            tickColor=palette["muted"],
        )
        .configure_legend(
            labelColor=palette["text"],
            titleColor=palette["text"],
        )
        .configure_header(
            labelColor=palette["text"],
            titleColor=palette["text"],
        )
        .configure_title(
            color=palette["text"],
            subtitleColor=palette["muted"],
        )
        .configure_view(stroke=palette["rule"])
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
    except (ImportError, TypeError, ValueError):
        # The configure_* rules above still cover charts without a serializable
        # top-level title (or environments where Altair is optional).
        pass

    return chart


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
            except (AttributeError, ValueError):
                pass
            if hasattr(energy_owner, "width"):
                try:
                    width = energy_owner.width
                    energy_owner.width = max(4, float(width or 0))
                except (TypeError, ValueError):
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
        '<header class="cxr-shell cxr-title">'
        f'<div class="cxr-title__eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(intro)}</p></header>"
    )


def context_rail(mo, values: Mapping[str, object] | Iterable[tuple[str, object]]):
    items = values.items() if isinstance(values, Mapping) else values
    body = "".join(
        '<div class="cxr-rail__item">'
        f"<dt>{escape(str(label))}</dt><dd>{escape(_display(value))}</dd></div>"
        for label, value in items
    )
    return mo.Html(f'<dl class="cxr-shell cxr-rail">{body}</dl>')


def status_badge(mo, state: str):
    kind = STATUS_KINDS.get(state, "skipped")
    return mo.Html(f'<span class="cxr-shell cxr-badge cxr-badge--{kind}">{escape(state)}</span>')


def control_group(mo, label: str, controls):
    return mo.vstack(
        [mo.Html(f'<div class="cxr-shell cxr-group__label">{escape(label)}</div>'), controls]
    ).style({"border-left": "2px solid var(--cxr-rule)", "padding-left": "1rem"})


def directional_state(mo, title: str, detail: str, action: str, *, kind: str = "info"):
    return mo.callout(
        mo.md(f"**{title}**\n\n{detail}\n\nNext: `{action}`"),
        kind=kind,
    )


#: Cell fill segments, drawn bottom-to-top, mapped to the palette in CSS
#: (``.cxr-grid__seg--<state>``). Order is the reading order of the stack.
SCAN_GRID_STATES = ("done", "cached", "running", "excluded")
_SCAN_GRID_LEGEND = {
    "done": ("This session", "var(--cxr-done)"),
    "cached": ("Cached", "var(--cxr-focus)"),
    "running": ("Running", "var(--cxr-compute)"),
    "excluded": ("Penetration-excluded", "var(--cxr-failure)"),
    "remaining": ("Remaining", "var(--cxr-surface)"),
}


def _scan_cell(cell: Mapping[str, float] | None, max_weight: float) -> str:
    """One grid cell: a cost-sized marker whose vertical fill stacks the state
    fractions (see :data:`SCAN_GRID_STATES`). ``cell`` is ``None`` for an absent
    (unrequested) case, or a mapping of state -> fraction in [0, 1] plus
    ``weight`` (relative compute cost). The marker side scales with
    ``sqrt(weight / max_weight)`` so the heavy high-energy corner reads as heavy,
    with a floor so light cells stay legible."""
    if cell is None:
        return '<div class="cxr-grid__cell" style="opacity:.28"></div>'
    weight = float(cell.get("weight", 0.0))
    side = 100.0 if max_weight <= 0 else max(34.0, 100.0 * (weight / max_weight) ** 0.5)
    total = sum(max(0.0, float(cell.get(state, 0.0))) for state in SCAN_GRID_STATES)
    segments = "".join(
        f'<div class="cxr-grid__seg cxr-grid__seg--{state}" '
        f'style="height:{100.0 * max(0.0, float(cell.get(state, 0.0))):.4g}%"></div>'
        for state in SCAN_GRID_STATES
    )
    done = min(1.0, total)
    title = escape(f"{done:.0%} done, {weight / max_weight:.0%} of the peak per-cell cost")
    return (
        f'<div class="cxr-grid__cell" title="{title}">'
        f'<div class="cxr-grid__marker" style="width:{side:.4g}%;height:{side:.4g}%">'
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
    header = '<div class="cxr-grid__corner">tilt \\ keV</div>' + "".join(
        f'<div class="cxr-grid__col">{escape(_display(energy))}</div>' for energy in energies
    )
    body = "".join(
        f'<div class="cxr-grid__row">{escape(str(row["label"]))}</div>'
        + "".join(_scan_cell(cell, max_weight) for cell in row["cells"])
        for row in rows
    )
    legend = "".join(
        f'<span><i class="cxr-grid__swatch" style="background:{_SCAN_GRID_LEGEND[state][1]}"></i>'
        f"{escape(_SCAN_GRID_LEGEND[state][0])}</span>"
        for state in (*states, "remaining")
        if state in _SCAN_GRID_LEGEND
    )
    return mo.Html(
        f'<div class="cxr-shell cxr-grid" style="grid-template-columns:{columns}" '
        f'role="img" aria-label="Scan progress matrix, tilt by beam energy, '
        f'cell size by compute cost">{header}{body}</div>'
        f'<div class="cxr-shell cxr-grid__legend">{legend}</div>'
    )


def _display(value: object) -> str:
    if value is None or value == "":
        return "not selected"
    return str(value)
