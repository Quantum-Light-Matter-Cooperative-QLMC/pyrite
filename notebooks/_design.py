"""Shared presentation primitives for cxr-mc Marimo applications."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape

COLORS = {
    "beamline_navy": "#12202B",
    "instrument_slate": "#1D303D",
    "xray_ice": "#DCEEF2",
    "detector_cyan": "#72C7D5",
    "tungsten_amber": "#D8A657",
    "validation_rose": "#D8787E",
}

SPACING = {"xs": ".35rem", "sm": ".65rem", "md": "1rem", "lg": "1.5rem"}
TYPOGRAPHY = {
    "display": '"IBM Plex Sans Condensed", "Arial Narrow", sans-serif',
    "body": '"IBM Plex Sans", "Aptos", "Segoe UI", sans-serif',
    "data": '"IBM Plex Mono", monospace',
}
BORDERS = {"rule": "#41606f", "width": "1px", "radius": ".35rem"}
FOCUS = {"color": COLORS["detector_cyan"], "width": "3px", "offset": "3px"}
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
    """Return scoped, export-safe CSS with no network dependencies."""
    return f"""
<style>
  .cxr-shell {{
    --cxr-bg: {COLORS["beamline_navy"]};
    --cxr-surface: {COLORS["instrument_slate"]};
    --cxr-text: {COLORS["xray_ice"]};
    --cxr-focus: {COLORS["detector_cyan"]};
    --cxr-compute: {COLORS["tungsten_amber"]};
    --cxr-failure: {COLORS["validation_rose"]};
    color: var(--cxr-text);
    font-family: {TYPOGRAPHY["body"]};
    max-width: 100%;
  }}
  body:has(.cxr-shell) {{
    background: var(--cxr-bg, #12202B);
    color: var(--cxr-text, #DCEEF2);
  }}
  body:has(.cxr-shell) :is(button, input, select, [tabindex]):focus-visible {{
    outline: {FOCUS["width"]} solid var(--cxr-focus, {FOCUS["color"]});
    outline-offset: {FOCUS["offset"]};
  }}
  body:has(.cxr-shell) :is(button, select, input:not([type="hidden"])) {{
    min-height: 44px;
  }}
  .cxr-title {{ max-width: {WIDTHS["prose"]}; margin: 0 0 1.25rem; }}
  .cxr-title__eyebrow {{
    color: var(--cxr-focus); font: 600 .75rem/1.2 "IBM Plex Mono", monospace;
    letter-spacing: .14em; text-transform: uppercase;
  }}
  .cxr-title h1 {{
    font-family: {TYPOGRAPHY["display"]};
    font-size: clamp(2rem, 4vw, 3.4rem); line-height: .98; margin: .35rem 0 .7rem;
  }}
  .cxr-title p {{ color: #b9d0d6; line-height: 1.55; margin: 0; }}
  .cxr-rail {{
    display: flex; flex-wrap: wrap; align-items: baseline; gap: .3rem 1.4rem;
    border-block: {BORDERS["width"]} solid {BORDERS["rule"]};
    margin: .6rem 0 .9rem; padding: .5rem 0;
  }}
  .cxr-rail__item {{
    display: flex; align-items: baseline; gap: .5rem; min-width: 0;
    border-left: 2px solid var(--cxr-focus); padding-left: .6rem;
  }}
  .cxr-rail dt {{
    color: #91aeb7; font: 600 .64rem/1.2 "IBM Plex Mono", monospace;
    letter-spacing: .1em; text-transform: uppercase; white-space: nowrap;
  }}
  .cxr-rail dd {{
    color: var(--cxr-text); font: 500 .85rem/1.2 "IBM Plex Mono", monospace;
    margin: 0; white-space: nowrap;
  }}
  .cxr-badge {{
    border: 1px solid currentColor; border-radius: 999px; display: inline-block;
    font: 600 .72rem/1 "IBM Plex Mono", monospace; padding: .38rem .62rem;
  }}
  .cxr-badge--failed {{ color: var(--cxr-failure); }}
  .cxr-badge--running, .cxr-badge--interpret {{ color: var(--cxr-compute); }}
  .cxr-badge--ready, .cxr-badge--cached, .cxr-badge--passed {{ color: var(--cxr-focus); }}
  .cxr-grid {{ display: grid; gap: .28rem; align-items: end; }}
  .cxr-grid__corner, .cxr-grid__col, .cxr-grid__row {{
    color: #91aeb7; font: 600 .62rem/1.1 "IBM Plex Mono", monospace;
    letter-spacing: .04em; white-space: nowrap;
  }}
  .cxr-grid__col {{ text-align: center; align-self: end; }}
  .cxr-grid__row {{ text-align: right; align-self: center; padding-right: .3rem; }}
  .cxr-grid__cell {{
    position: relative; aspect-ratio: 1 / 1; width: 100%; min-width: 12px;
    border-radius: 2px; background: {COLORS["instrument_slate"]};
    box-shadow: inset 0 0 0 1px {BORDERS["rule"]}; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }}
  .cxr-grid__marker {{
    display: flex; flex-direction: column-reverse; border-radius: 1px;
    box-shadow: inset 0 0 0 1px rgba(18, 32, 43, .55);
  }}
  .cxr-grid__seg {{ width: 100%; }}
  .cxr-grid__seg--cached {{ background: {COLORS["detector_cyan"]}; }}
  .cxr-grid__seg--done {{ background: #A6E4EF; }}
  .cxr-grid__seg--running {{ background: {COLORS["tungsten_amber"]}; }}
  .cxr-grid__seg--excluded {{ background: {COLORS["validation_rose"]}; }}
  .cxr-grid__legend {{
    display: flex; flex-wrap: wrap; gap: .8rem; margin-top: .6rem;
    color: #91aeb7; font: 500 .68rem/1.2 "IBM Plex Mono", monospace;
  }}
  .cxr-grid__legend span {{ display: inline-flex; align-items: center; gap: .35rem; }}
  .cxr-grid__swatch {{
    width: .72rem; height: .72rem; border-radius: 2px;
    box-shadow: inset 0 0 0 1px {BORDERS["rule"]};
  }}
  .cxr-group {{ border-left: 2px solid #41606f; padding: .35rem 0 .35rem 1rem; }}
  .cxr-group__label {{
    color: #91aeb7; font: 600 .72rem/1.2 "IBM Plex Mono", monospace;
    letter-spacing: .08em; margin-bottom: .65rem; text-transform: uppercase;
  }}
  @media (max-width: {WIDTHS["breakpoint"]}) {{
    .cxr-title h1 {{ font-size: 2rem; }}
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


def page_title(mo, title: str, intro: str, *, eyebrow: str = "CXR instrument"):
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
    ).style({"border-left": "2px solid #41606f", "padding-left": "1rem"})


def directional_state(mo, title: str, detail: str, action: str, *, kind: str = "info"):
    return mo.callout(
        mo.md(f"**{title}**\n\n{detail}\n\nNext: `{action}`"),
        kind=kind,
    )


#: Cell fill segments, drawn bottom-to-top, mapped to the palette in CSS
#: (``.cxr-grid__seg--<state>``). Order is the reading order of the stack.
SCAN_GRID_STATES = ("done", "cached", "running", "excluded")
_SCAN_GRID_LEGEND = {
    "done": ("This session", "#A6E4EF"),
    "cached": ("Cached", COLORS["detector_cyan"]),
    "running": ("Running", COLORS["tungsten_amber"]),
    "excluded": ("Penetration-excluded", COLORS["validation_rose"]),
    "remaining": ("Remaining", COLORS["instrument_slate"]),
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
