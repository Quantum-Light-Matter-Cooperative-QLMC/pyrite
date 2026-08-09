"""Thin ``python -m`` entry-point shims the remote box invokes on a uv-synced
checkout (``python -m cxr_mc._entry.scan``, ``python -m cxr_mc._entry.reproduce_zhai``).

Each module is a minimal ``__main__`` wrapper -- the heavy logic lives elsewhere
(cxr_mc.runs.scan, src/cxr_mc/apps/anchor_figures.py) so the spawn/forkserver worker re-import
stays cheap. Prefer the installed CLI (``cxr run [PROFILE] -m <material>``) for local use.
"""
