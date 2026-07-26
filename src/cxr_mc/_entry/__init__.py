"""Thin ``python -m`` entry-point shims the remote box invokes on a uv-synced
checkout (``python -m cxr_mc._entry.scan``, ``python -m cxr_mc._entry.reproduce_zhai``).

Each module is a minimal ``__main__`` wrapper -- the heavy logic lives elsewhere
(cxr_mc.scan, checks/anchor_figures.py) so the spawn/forkserver worker re-import
stays cheap. Prefer the installed CLI (``cxr scan <material>``) for local use.
"""
