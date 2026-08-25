"""mpl._common

Matplotlib-only figure plumbing shared across the ``mpl`` backend's ``plot_*``
wrappers. Split out of :mod:`pyrite.plots._common` (which stays backend-neutral:
no matplotlib import) because this helper builds a ``matplotlib.figure.Figure``
directly -- Altair and Plotly never call it.
"""

import matplotlib.pyplot as plt


# ---- per-tilt figure helper --------------------------------------------------
def _per_tilt_figs(recs, settings, draw, figsize, *, empty_msg="no results yet", **kw):
    """Shared body of every ``plot_*`` wrapper: one freshly-drawn figure PER POLAR
    TILT. ``draw(fig, tilt_recs, settings, **kw)`` renders a single tilt onto a
    cleared figure (the same ``_draw_*`` the interactive ``browse`` uses), so the
    wrappers and the slider stay in lockstep. Handles the empty-records check, the
    per-tilt grouping, and collecting the figure list."""
    if not recs:
        print(empty_msg)
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=figsize)
        draw(fig, [r for r in recs if r["case"]["tilt_deg"] == t], settings, **kw)
        figs.append(fig)
    return figs
