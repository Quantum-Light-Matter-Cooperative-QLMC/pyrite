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
WIDTHS = {"prose": "72ch", "rail_item": "8.5rem", "breakpoint": "768px"}

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
    --cxr-bg: {COLORS['beamline_navy']};
    --cxr-surface: {COLORS['instrument_slate']};
    --cxr-text: {COLORS['xray_ice']};
    --cxr-focus: {COLORS['detector_cyan']};
    --cxr-compute: {COLORS['tungsten_amber']};
    --cxr-failure: {COLORS['validation_rose']};
    color: var(--cxr-text);
    font-family: {TYPOGRAPHY['body']};
    max-width: 100%;
  }}
  body:has(.cxr-shell) {{
    background: var(--cxr-bg, #12202B);
    color: var(--cxr-text, #DCEEF2);
  }}
  body:has(.cxr-shell) :is(button, input, select, [tabindex]):focus-visible {{
    outline: {FOCUS['width']} solid var(--cxr-focus, {FOCUS['color']});
    outline-offset: {FOCUS['offset']};
  }}
  body:has(.cxr-shell) :is(button, select, input:not([type="hidden"])) {{
    min-height: 44px;
  }}
  .cxr-title {{ max-width: {WIDTHS['prose']}; margin: 0 0 1.25rem; }}
  .cxr-title__eyebrow {{
    color: var(--cxr-focus); font: 600 .75rem/1.2 "IBM Plex Mono", monospace;
    letter-spacing: .14em; text-transform: uppercase;
  }}
  .cxr-title h1 {{
    font-family: {TYPOGRAPHY['display']};
    font-size: clamp(2rem, 4vw, 3.4rem); line-height: .98; margin: .35rem 0 .7rem;
  }}
  .cxr-title p {{ color: #b9d0d6; line-height: 1.55; margin: 0; }}
  .cxr-rail {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax({WIDTHS['rail_item']}, 1fr));
    gap: 0; border-block: {BORDERS['width']} solid {BORDERS['rule']}; margin: 1rem 0 1.5rem;
  }}
  .cxr-rail__item {{ padding: .8rem 1rem; position: relative; min-width: 0; }}
  .cxr-rail__item::before {{
    background: var(--cxr-focus); content: ""; height: .42rem; left: 1rem;
    position: absolute; top: -.24rem; width: 2px;
  }}
  .cxr-rail dt {{
    color: #91aeb7; font: 600 .68rem/1.2 "IBM Plex Mono", monospace;
    letter-spacing: .1em; text-transform: uppercase;
  }}
  .cxr-rail dd {{ font: 500 .9rem/1.35 "IBM Plex Mono", monospace; margin: .3rem 0 0; }}
  .cxr-badge {{
    border: 1px solid currentColor; border-radius: 999px; display: inline-block;
    font: 600 .72rem/1 "IBM Plex Mono", monospace; padding: .38rem .62rem;
  }}
  .cxr-badge--failed {{ color: var(--cxr-failure); }}
  .cxr-badge--running, .cxr-badge--interpret {{ color: var(--cxr-compute); }}
  .cxr-badge--ready, .cxr-badge--cached, .cxr-badge--passed {{ color: var(--cxr-focus); }}
  .cxr-group {{ border-left: 2px solid #41606f; padding: .35rem 0 .35rem 1rem; }}
  .cxr-group__label {{
    color: #91aeb7; font: 600 .72rem/1.2 "IBM Plex Mono", monospace;
    letter-spacing: .08em; margin-bottom: .65rem; text-transform: uppercase;
  }}
  @media (max-width: {WIDTHS['breakpoint']}) {{
    .cxr-rail {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
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
    return mo.Html(
        f'<span class="cxr-shell cxr-badge cxr-badge--{kind}">{escape(state)}</span>'
    )


def control_group(mo, label: str, controls):
    return mo.vstack(
        [mo.Html(f'<div class="cxr-shell cxr-group__label">{escape(label)}</div>'), controls]
    ).style({"border-left": "2px solid #41606f", "padding-left": "1rem"})


def directional_state(mo, title: str, detail: str, action: str, *, kind: str = "info"):
    return mo.callout(
        mo.md(f"**{title}**\n\n{detail}\n\nNext: `{action}`"),
        kind=kind,
    )


def _display(value: object) -> str:
    if value is None or value == "":
        return "not selected"
    return str(value)
