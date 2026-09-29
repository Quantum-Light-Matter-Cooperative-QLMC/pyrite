"""View navigation for the analysis app.

The nav widget is built before any heavy cell so each view's controls and
widgets can be gated with ``mo.stop(nav.value != VIEW)``; only the selected
view computes on load and on material/emission changes.
"""

EXPLORE = "Explore"
OPTIMIZE = "Optimize"
INSTRUMENTS = "Instruments"

VIEW_NAMES = (EXPLORE, OPTIMIZE, INSTRUMENTS)


def make_view_nav(mo):
    """Tab bar whose panes are empty; the selected view renders in its own cell below."""
    return mo.ui.tabs({name: mo.md("") for name in VIEW_NAMES}, value=EXPLORE)
